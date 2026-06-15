from pathlib import Path
import argparse
import importlib
import json

import numpy as np
import torch

try:
    import SimpleITK as sitk
except ImportError:
    sitk = None


def parse_tuple(value):
    parts = [int(part.strip()) for part in value.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("Expected 3 comma-separated integers, e.g. 128,128,128")
    return tuple(parts)


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


def load_ct(input_path, normalize=True):
    input_path = Path(input_path)
    reference_img = None

    if input_path.is_dir():
        ct, reference_img = load_dicom_series(input_path)
    elif input_path.suffix == ".npy":
        ct = np.load(input_path)
    elif input_path.name.endswith(".nii.gz") or input_path.suffix == ".nii":
        if sitk is None:
            raise RuntimeError("SimpleITK is required to read NIfTI files.")
        reference_img = sitk.ReadImage(str(input_path))
        ct = sitk.GetArrayFromImage(reference_img)
    else:
        raise ValueError(f"Unsupported CT input: {input_path}")

    if ct.ndim == 4:
        if ct.shape[0] == 1:
            ct = ct[0]
        elif ct.shape[-1] == 1:
            ct = ct[..., 0]
        else:
            raise ValueError(f"Expected single-channel CT volume, got shape {ct.shape}")

    if ct.ndim != 3:
        raise ValueError(f"Expected CT shape (D, H, W), got {ct.shape}")

    ct = ct.astype(np.float32)
    if normalize and should_normalize_ct(ct):
        ct = normalize_ct(ct)
    elif normalize:
        print("  CT appears already normalized; skipping HU normalization.")

    return ct, reference_img


def import_model(model_module, model_class, model_kwargs):
    module = importlib.import_module(model_module)
    cls = getattr(module, model_class)
    return cls(**model_kwargs)


def clean_state_dict(state_dict):
    cleaned = {}

    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[len("module."):]
        cleaned[key] = value

    return cleaned


def load_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "net", "model"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                checkpoint = checkpoint[key]
                break

    if not isinstance(checkpoint, dict):
        raise RuntimeError("Checkpoint does not contain a valid state_dict.")

    model.load_state_dict(clean_state_dict(checkpoint), strict=False)
    model.to(device)
    model.eval()
    return model


def compute_starts(length, patch, stride):
    if length <= patch:
        return [0]

    starts = list(range(0, length - patch + 1, stride))
    if starts[-1] != length - patch:
        starts.append(length - patch)
    return starts


def pad_volume_to_patch(volume, patch_size):
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


def unpad_volume(volume, pads):
    z_pad, y_pad, x_pad = pads

    z0, z1 = z_pad[0], volume.shape[0] - z_pad[1]
    y0, y1 = y_pad[0], volume.shape[1] - y_pad[1]
    x0, x1 = x_pad[0], volume.shape[2] - x_pad[1]

    return volume[z0:z1, y0:y1, x0:x1]


def make_weight_window(patch_size):
    axes = []
    for size in patch_size:
        if size == 1:
            axes.append(np.ones((1,), dtype=np.float32))
        else:
            axis = np.hanning(size).astype(np.float32)
            axis = np.maximum(axis, 0.05)
            axes.append(axis)

    window = axes[0][:, None, None] * axes[1][None, :, None] * axes[2][None, None, :]
    return window.astype(np.float32)


def get_model_output_tensor(output):
    if isinstance(output, (tuple, list)):
        output = output[0]
    if isinstance(output, dict):
        for key in ("out", "output", "pred", "logits"):
            if key in output:
                return output[key]
        first_key = next(iter(output))
        return output[first_key]
    return output


def sliding_window_predict(model, ct, patch_size, stride, device, use_amp=False):
    padded_ct, pads = pad_volume_to_patch(ct, patch_size)
    d, h, w = padded_ct.shape
    pd, ph, pw = patch_size
    sd, sh, sw = stride

    z_starts = compute_starts(d, pd, sd)
    y_starts = compute_starts(h, ph, sh)
    x_starts = compute_starts(w, pw, sw)

    output_sum = np.zeros((2, d, h, w), dtype=np.float32)
    weight_sum = np.zeros((d, h, w), dtype=np.float32)
    weight_window = make_weight_window(patch_size)

    total = len(z_starts) * len(y_starts) * len(x_starts)
    done = 0

    with torch.no_grad():
        for z in z_starts:
            for y in y_starts:
                for x in x_starts:
                    patch = padded_ct[z:z + pd, y:y + ph, x:x + pw]
                    patch_tensor = torch.from_numpy(patch[None, None]).to(device=device, dtype=torch.float32)

                    with torch.cuda.amp.autocast(enabled=use_amp and device.type == "cuda"):
                        logits = get_model_output_tensor(model(patch_tensor))

                    if logits.ndim != 5 or logits.shape[1] < 2:
                        raise RuntimeError(f"Expected model output shape (B, 2, D, H, W), got {tuple(logits.shape)}")

                    probs = torch.sigmoid(logits[:, :2]).detach().cpu().numpy()[0]

                    output_sum[:, z:z + pd, y:y + ph, x:x + pw] += probs * weight_window[None]
                    weight_sum[z:z + pd, y:y + ph, x:x + pw] += weight_window

                    done += 1
                    if done == 1 or done % 25 == 0 or done == total:
                        print(f"  sliding-window patches: {done}/{total}")

    pred = output_sum / np.maximum(weight_sum[None], 1e-6)

    pred_lumen = unpad_volume(pred[0], pads)
    pred_wall = unpad_volume(pred[1], pads)

    return pred_lumen.astype(np.float32), pred_wall.astype(np.float32)


def save_nifti_like(array, reference_img, output_path):
    if sitk is None or reference_img is None:
        return

    out_img = sitk.GetImageFromArray(array.astype(np.float32))
    out_img.CopyInformation(reference_img)
    sitk.WriteImage(out_img, str(output_path))


def main():
    parser = argparse.ArgumentParser(description="Run WingsNet full-volume sliding-window inference.")
    parser.add_argument("--checkpoint", default="saved_model/wingsnet_best.pth", help="Path to wingsnet_best.pth")
    parser.add_argument("--ct", required=True, help="Full CT input: processed .npy, NIfTI, or DICOM folder")
    parser.add_argument("--output-dir", default="datasets/predictions", help="Folder for prediction outputs")
    parser.add_argument("--model-module", default="WingsNet", help="Python module containing the model class")
    parser.add_argument("--model-class", default="WingsNet", help="Model class name")
    parser.add_argument("--model-kwargs", default="{}", help="JSON dict passed to model constructor")
    parser.add_argument("--patch-size", type=parse_tuple, default=(128, 128, 128), help="D,H,W patch size")
    parser.add_argument("--stride", type=parse_tuple, default=(64, 64, 64), help="D,H,W sliding-window stride")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-normalize", action="store_true", help="Skip CT HU clipping/normalization")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for optional binary masks")
    parser.add_argument("--save-binary", action="store_true", help="Also save thresholded binary masks")
    parser.add_argument("--save-nifti", action="store_true", help="Also save NIfTI outputs when CT metadata is available")
    parser.add_argument("--amp", action="store_true", help="Use CUDA mixed precision during inference")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    model_kwargs = json.loads(args.model_kwargs)

    print("Loading CT:", args.ct)
    ct, reference_img = load_ct(args.ct, normalize=not args.no_normalize)
    print("  CT shape:", ct.shape)

    print("Loading model:", args.model_module, args.model_class)
    model = import_model(args.model_module, args.model_class, model_kwargs)
    model = load_checkpoint(model, args.checkpoint, device)

    print("Running inference")
    print("  checkpoint:", args.checkpoint)
    print("  patch size:", args.patch_size)
    print("  stride:", args.stride)
    print("  device:", device)

    pred_lumen, pred_wall = sliding_window_predict(
        model=model,
        ct=ct,
        patch_size=args.patch_size,
        stride=args.stride,
        device=device,
        use_amp=args.amp,
    )

    lumen_path = output_dir / "pred_lumen.npy"
    wall_path = output_dir / "pred_wall.npy"

    np.save(lumen_path, pred_lumen)
    np.save(wall_path, pred_wall)

    print("Saved:")
    print("  ", lumen_path, pred_lumen.shape, pred_lumen.dtype)
    print("  ", wall_path, pred_wall.shape, pred_wall.dtype)

    if args.save_binary:
        lumen_mask = (pred_lumen >= args.threshold).astype(np.uint8)
        wall_mask = (pred_wall >= args.threshold).astype(np.uint8)
        np.save(output_dir / "pred_lumen_mask.npy", lumen_mask)
        np.save(output_dir / "pred_wall_mask.npy", wall_mask)
        print("  ", output_dir / "pred_lumen_mask.npy")
        print("  ", output_dir / "pred_wall_mask.npy")

    if args.save_nifti:
        save_nifti_like(pred_lumen, reference_img, output_dir / "pred_lumen.nii.gz")
        save_nifti_like(pred_wall, reference_img, output_dir / "pred_wall.nii.gz")
        if reference_img is not None:
            print("  ", output_dir / "pred_lumen.nii.gz")
            print("  ", output_dir / "pred_wall.nii.gz")


if __name__ == "__main__":
    main()
