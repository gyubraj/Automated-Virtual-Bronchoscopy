from pathlib import Path
import argparse
import importlib
import json

import numpy as np
import torch
from scipy import ndimage as ndi

try:
    import SimpleITK as sitk
except ImportError:
    sitk = None

try:
    from skimage.morphology import skeletonize_3d
except ImportError:
    from skimage.morphology import skeletonize

    def skeletonize_3d(volume):
        return skeletonize(volume)


def parse_tuple(value):
    values = tuple(int(part.strip()) for part in value.split(","))
    if len(values) != 3:
        raise argparse.ArgumentTypeError("Expected format: 128,128,128")
    return values


def normalize_ct(ct):
    ct = ct.astype(np.float32)
    ct = np.clip(ct, -1000, 400)
    ct = (ct + 1000) / 1400
    return ct.astype(np.float32)


def should_normalize_ct(ct):
    finite = ct[np.isfinite(ct)]
    if finite.size == 0:
        return True

    min_value = float(finite.min())
    max_value = float(finite.max())

    # Processed AirRC .npy files are already normalized to roughly [0, 1].
    return not (-0.1 <= min_value <= 1.1 and -0.1 <= max_value <= 1.1)


def load_dicom_series(ct_dir):
    if sitk is None:
        raise RuntimeError("SimpleITK is required to read DICOM folders.")

    reader = sitk.ImageSeriesReader()
    dicom_files = reader.GetGDCMSeriesFileNames(str(ct_dir))

    if not dicom_files:
        raise RuntimeError(f"No DICOM files found in {ct_dir}")

    reader.SetFileNames(dicom_files)
    image = reader.Execute()
    volume = sitk.GetArrayFromImage(image)
    return volume, image


def load_volume(path, normalize=False):
    path = Path(path)
    reference_img = None

    if path.is_dir():
        volume, reference_img = load_dicom_series(path)
    elif path.suffix == ".npy":
        volume = np.load(path)
    elif path.name.endswith(".nii.gz") or path.suffix == ".nii":
        if sitk is None:
            raise RuntimeError("SimpleITK is required to read NIfTI files.")
        reference_img = sitk.ReadImage(str(path))
        volume = sitk.GetArrayFromImage(reference_img)
    else:
        raise ValueError(f"Unsupported input: {path}")

    volume = np.asarray(volume)

    if volume.ndim == 4:
        if volume.shape[0] == 1:
            volume = volume[0]
        elif volume.shape[-1] == 1:
            volume = volume[..., 0]
        else:
            raise ValueError(f"Expected a single-channel 3D volume, got {volume.shape}")

    if volume.ndim != 3:
        raise ValueError(f"Expected shape (D, H, W), got {volume.shape}")

    volume = volume.astype(np.float32)
    if normalize and should_normalize_ct(volume):
        volume = normalize_ct(volume)
    elif normalize:
        print("  CT appears already normalized; skipping HU normalization.")

    return volume, reference_img


def load_target(path):
    target = np.load(path)

    if target.ndim != 4 or target.shape[0] < 2:
        raise ValueError(f"Expected target shape (2, D, H, W), got {target.shape}")

    return (target[0] > 0), (target[1] > 0)


def import_model(model_module, model_class, model_kwargs):
    module = importlib.import_module(model_module)
    model_cls = getattr(module, model_class)
    return model_cls(**model_kwargs)


def clean_state_dict(state_dict):
    cleaned = {}

    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[len("module."):]
        cleaned[key] = value

    return cleaned


def load_model(checkpoint_path, model_module, model_class, model_kwargs, device):
    model = import_model(model_module, model_class, model_kwargs)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "net", "model"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                checkpoint = checkpoint[key]
                break

    if not isinstance(checkpoint, dict):
        raise RuntimeError("Checkpoint does not contain a state_dict.")

    missing, unexpected = model.load_state_dict(clean_state_dict(checkpoint), strict=False)
    if missing:
        print("[WARN] Missing checkpoint keys:", len(missing))
    if unexpected:
        print("[WARN] Unexpected checkpoint keys:", len(unexpected))

    model.to(device)
    model.eval()
    return model


def get_model_output(output):
    if isinstance(output, (tuple, list)):
        return output[-1]

    if isinstance(output, dict):
        for key in ("logits", "pred", "output", "out"):
            if key in output:
                return output[key]
        return output[next(iter(output))]

    return output


def compute_starts(length, patch_size, stride):
    if length <= patch_size:
        return [0]

    starts = list(range(0, length - patch_size + 1, stride))
    if starts[-1] != length - patch_size:
        starts.append(length - patch_size)
    return starts


def pad_to_patch(volume, patch_size):
    d, h, w = volume.shape
    pd, ph, pw = patch_size

    pad_d = max(0, pd - d)
    pad_h = max(0, ph - h)
    pad_w = max(0, pw - w)

    pads = (
        (pad_d // 2, pad_d - pad_d // 2),
        (pad_h // 2, pad_h - pad_h // 2),
        (pad_w // 2, pad_w - pad_w // 2),
    )

    if pad_d == 0 and pad_h == 0 and pad_w == 0:
        return volume, pads

    return np.pad(volume, pads, mode="constant", constant_values=0), pads


def unpad(volume, pads):
    z_pad, y_pad, x_pad = pads
    z1 = volume.shape[0] - z_pad[1]
    y1 = volume.shape[1] - y_pad[1]
    x1 = volume.shape[2] - x_pad[1]
    return volume[z_pad[0]:z1, y_pad[0]:y1, x_pad[0]:x1]


def make_weight_window(patch_size):
    axes = []

    for size in patch_size:
        axis = np.hanning(size).astype(np.float32)
        axes.append(np.maximum(axis, 0.05))

    return axes[0][:, None, None] * axes[1][None, :, None] * axes[2][None, None, :]


def sliding_window_prediction(model, ct, patch_size, stride, device, amp=False):
    ct, pads = pad_to_patch(ct, patch_size)
    d, h, w = ct.shape
    pd, ph, pw = patch_size
    sd, sh, sw = stride

    z_starts = compute_starts(d, pd, sd)
    y_starts = compute_starts(h, ph, sh)
    x_starts = compute_starts(w, pw, sw)

    pred_sum = np.zeros((2, d, h, w), dtype=np.float32)
    count_sum = np.zeros((d, h, w), dtype=np.float32)
    weight_window = make_weight_window(patch_size)

    total = len(z_starts) * len(y_starts) * len(x_starts)
    done = 0

    with torch.no_grad():
        for z in z_starts:
            for y in y_starts:
                for x in x_starts:
                    patch = ct[z:z + pd, y:y + ph, x:x + pw]
                    patch = torch.from_numpy(patch[None, None]).to(device=device, dtype=torch.float32)

                    with torch.cuda.amp.autocast(enabled=amp and device.type == "cuda"):
                        logits = get_model_output(model(patch))

                    if logits.ndim != 5 or logits.shape[1] < 2:
                        raise RuntimeError(f"Expected model output (B, 2, D, H, W), got {tuple(logits.shape)}")

                    prob = torch.sigmoid(logits[:, :2]).detach().cpu().numpy()[0]
                    pred_sum[:, z:z + pd, y:y + ph, x:x + pw] += prob * weight_window[None]
                    count_sum[z:z + pd, y:y + ph, x:x + pw] += weight_window

                    done += 1
                    if done == 1 or done % 25 == 0 or done == total:
                        print(f"  predicted patches: {done}/{total}")

    pred = pred_sum / np.maximum(count_sum[None], 1e-6)
    return unpad(pred[0], pads), unpad(pred[1], pads)


def dice(pred, target):
    pred = pred.astype(bool)
    target = target.astype(bool)
    denom = pred.sum(dtype=np.float64) + target.sum(dtype=np.float64)

    if denom == 0:
        return 1.0

    intersection = np.logical_and(pred, target).sum(dtype=np.float64)
    return float(2.0 * intersection / denom)


def precision(pred, target):
    pred = pred.astype(bool)
    target = target.astype(bool)
    tp = np.logical_and(pred, target).sum(dtype=np.float64)
    fp = np.logical_and(pred, np.logical_not(target)).sum(dtype=np.float64)

    if tp + fp == 0:
        return 1.0 if target.sum() == 0 else 0.0

    return float(tp / (tp + fp))


def recall(pred, target):
    pred = pred.astype(bool)
    target = target.astype(bool)
    tp = np.logical_and(pred, target).sum(dtype=np.float64)
    fn = np.logical_and(np.logical_not(pred), target).sum(dtype=np.float64)

    if tp + fn == 0:
        return 1.0

    return float(tp / (tp + fn))


def largest_component(mask):
    labels, count = ndi.label(mask)
    if count <= 1:
        return mask.astype(bool)

    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    return labels == int(np.argmax(sizes))


def skeleton_metrics(pred_lumen, target_lumen):
    pred_lumen = largest_component(pred_lumen)
    target_lumen = target_lumen.astype(bool)

    skeleton = skeletonize_3d(target_lumen).astype(bool)
    skeleton_voxels = skeleton.sum(dtype=np.float64)

    if skeleton_voxels == 0:
        length_recall = 0.0
    else:
        length_recall = float(np.logical_and(pred_lumen, skeleton).sum(dtype=np.float64) / skeleton_voxels)

    pred_voxels = pred_lumen.sum(dtype=np.float64)
    if pred_voxels == 0:
        airway_precision = 0.0
    else:
        airway_precision = float(np.logical_and(pred_lumen, target_lumen).sum(dtype=np.float64) / pred_voxels)

    return {
        "target_skeleton_voxels": int(skeleton_voxels),
        "length_recall": length_recall,
        "airway_precision": airway_precision,
    }


def save_nifti(array, reference_img, path):
    if sitk is None or reference_img is None:
        return False

    image = sitk.GetImageFromArray(array.astype(np.float32))
    image.CopyInformation(reference_img)
    sitk.WriteImage(image, str(path))
    return True


def evaluate_case(args):
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ct, reference_img = load_volume(args.ct, normalize=not args.no_normalize)
    print("CT shape:", ct.shape)

    model = load_model(
        checkpoint_path=args.checkpoint,
        model_module=args.model_module,
        model_class=args.model_class,
        model_kwargs=json.loads(args.model_kwargs),
        device=device,
    )

    pred_lumen_prob, pred_wall_prob = sliding_window_prediction(
        model=model,
        ct=ct,
        patch_size=args.patch_size,
        stride=args.stride,
        device=device,
        amp=args.amp,
    )

    pred_lumen = pred_lumen_prob >= args.threshold
    pred_wall = pred_wall_prob >= args.threshold

    np.save(output_dir / "pred_lumen.npy", pred_lumen_prob.astype(np.float32))
    np.save(output_dir / "pred_wall.npy", pred_wall_prob.astype(np.float32))
    np.save(output_dir / "pred_lumen_mask.npy", pred_lumen.astype(np.uint8))
    np.save(output_dir / "pred_wall_mask.npy", pred_wall.astype(np.uint8))

    save_nifti(pred_lumen_prob, reference_img, output_dir / "pred_lumen.nii.gz")
    save_nifti(pred_wall_prob, reference_img, output_dir / "pred_wall.nii.gz")
    save_nifti(pred_lumen.astype(np.uint8), reference_img, output_dir / "pred_lumen_mask.nii.gz")
    save_nifti(pred_wall.astype(np.uint8), reference_img, output_dir / "pred_wall_mask.nii.gz")

    summary = {
        "checkpoint": str(args.checkpoint),
        "ct": str(args.ct),
        "output_dir": str(output_dir),
        "threshold": args.threshold,
        "shape": list(pred_lumen.shape),
        "pred_lumen_voxels": int(pred_lumen.sum()),
        "pred_wall_voxels": int(pred_wall.sum()),
    }

    if args.target:
        target_lumen, target_wall = load_target(args.target)

        if target_lumen.shape != pred_lumen.shape:
            raise ValueError(f"Lumen shape mismatch: pred={pred_lumen.shape}, target={target_lumen.shape}")
        if target_wall.shape != pred_wall.shape:
            raise ValueError(f"Wall shape mismatch: pred={pred_wall.shape}, target={target_wall.shape}")

        summary["lumen"] = {
            "dice": dice(pred_lumen, target_lumen),
            "precision": precision(pred_lumen, target_lumen),
            "recall": recall(pred_lumen, target_lumen),
            "target_voxels": int(target_lumen.sum()),
        }
        summary["wall"] = {
            "dice": dice(pred_wall, target_wall),
            "precision": precision(pred_wall, target_wall),
            "recall": recall(pred_wall, target_wall),
            "target_voxels": int(target_wall.sum()),
        }
        summary["mean_dice"] = float((summary["lumen"]["dice"] + summary["wall"]["dice"]) / 2.0)

        if args.skeleton_metrics:
            summary["skeleton"] = skeleton_metrics(pred_lumen, target_lumen)

    with open(output_dir / "evaluation_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\nSaved predictions to:", output_dir)
    print("  pred_lumen.npy")
    print("  pred_wall.npy")
    print("  pred_lumen_mask.npy")
    print("  pred_wall_mask.npy")

    if "lumen" in summary:
        print("\nFull-case metrics")
        print("  lumen dice: %.4f, precision: %.4f, recall: %.4f" % (
            summary["lumen"]["dice"],
            summary["lumen"]["precision"],
            summary["lumen"]["recall"],
        ))
        print("  wall  dice: %.4f, precision: %.4f, recall: %.4f" % (
            summary["wall"]["dice"],
            summary["wall"]["precision"],
            summary["wall"]["recall"],
        ))
        print("  mean dice: %.4f" % summary["mean_dice"])

    if "skeleton" in summary:
        print("\nSkeleton-style lumen metrics")
        print("  length recall: %.4f" % summary["skeleton"]["length_recall"])
        print("  airway precision: %.4f" % summary["skeleton"]["airway_precision"])

    print("  metrics json:", output_dir / "evaluation_metrics.json")


def main():
    parser = argparse.ArgumentParser(
        description="Project-specific WingsNet full-volume inference and evaluation."
    )
    parser.add_argument("--checkpoint", default="saved_model/wingsnet_best.pth")
    parser.add_argument("--ct", required=True, help="Full CT: processed .npy, NIfTI, or DICOM folder")
    parser.add_argument("--target", default=None, help="Optional full target .npy with shape (2, D, H, W)")
    parser.add_argument("--output-dir", default="datasets/predictions/evaluation_case")
    parser.add_argument("--model-module", default="WingsNet")
    parser.add_argument("--model-class", default="WingsNet")
    parser.add_argument("--model-kwargs", default='{"in_channel": 1, "n_classes": 2}')
    parser.add_argument("--patch-size", type=parse_tuple, default=(128, 128, 128))
    parser.add_argument("--stride", type=parse_tuple, default=(64, 64, 64))
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-normalize", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--skeleton-metrics", action="store_true")
    args = parser.parse_args()

    evaluate_case(args)


if __name__ == "__main__":
    main()
