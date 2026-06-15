import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk


PROJECT_ROOT = Path(__file__).resolve().parent

AIRRC_DIR = PROJECT_ROOT / "datasets_new" / "airrc"
LABEL_DIR = AIRRC_DIR / "labelsTr"

DICOM_ROOT = PROJECT_ROOT / "datasets_new" / "lidc" / "lidc_idri"

OUT_DIR = PROJECT_ROOT / "datasets_new" / "processed_airrc"
IMAGE_OUT = OUT_DIR / "images"
TARGET_OUT = OUT_DIR / "targets"
META_OUT = OUT_DIR / "metadata"


for folder in [IMAGE_OUT, TARGET_OUT, META_OUT]:
    folder.mkdir(parents=True, exist_ok=True)


def find_ct_dir(uid):
    matches = list(DICOM_ROOT.rglob(f"CT_{uid}"))
    if not matches:
        return None
    return matches[0]


def load_dicom_series(ct_dir):
    reader = sitk.ImageSeriesReader()
    dicom_files = reader.GetGDCMSeriesFileNames(str(ct_dir))

    if not dicom_files:
        raise RuntimeError(f"No DICOM files found in {ct_dir}")

    reader.SetFileNames(dicom_files)
    return reader.Execute()


def resample_ct_to_mask(ct_img, mask_img):
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(mask_img)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(-1000)
    resampler.SetTransform(sitk.Transform())
    return resampler.Execute(ct_img)


def normalize_ct(ct):
    ct = ct.astype(np.float32)
    ct = np.clip(ct, -1000, 400)
    ct = (ct + 1000) / 1400
    return ct.astype(np.float32)


def preprocess_case(label_path):
    uid = label_path.name.replace(".nii.gz", "")

    ct_dir = find_ct_dir(uid)
    if ct_dir is None:
        print(f"[SKIP] No CT folder found for {uid}")
        return

    print(f"[PROCESS] {uid}")

    ct_img = load_dicom_series(ct_dir)
    mask_img = sitk.ReadImage(str(label_path))

    ct_resampled_img = resample_ct_to_mask(ct_img, mask_img)

    ct = sitk.GetArrayFromImage(ct_resampled_img)      # z, y, x
    mask = sitk.GetArrayFromImage(mask_img)            # z, y, x

    lumen = (mask == 1).astype(np.uint8)
    wall = (mask == 2).astype(np.uint8)

    target = np.stack([lumen, wall], axis=0)           # 2, z, y, x
    ct = normalize_ct(ct)                              # z, y, x

    np.save(IMAGE_OUT / f"{uid}_ct.npy", ct)
    np.save(TARGET_OUT / f"{uid}_target.npy", target)

    metadata = {
        "uid": uid,
        "ct_dir": str(ct_dir),
        "label_path": str(label_path),
        "ct_shape": list(ct.shape),
        "target_shape": list(target.shape),
        "mask_spacing_xyz": list(mask_img.GetSpacing()),
        "mask_origin_xyz": list(mask_img.GetOrigin()),
        "mask_direction": list(mask_img.GetDirection()),
        "labels_present": [int(x) for x in np.unique(mask)],
        "lumen_voxels": int(lumen.sum()),
        "wall_voxels": int(wall.sum()),
    }

    with open(META_OUT / f"{uid}_meta.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("  CT:", ct.shape)
    print("  target:", target.shape)
    print("  labels:", metadata["labels_present"])
    print("  lumen voxels:", metadata["lumen_voxels"])
    print("  wall voxels:", metadata["wall_voxels"])


def main():
    label_files = sorted(LABEL_DIR.glob("*.nii.gz"))

    print(f"Found {len(label_files)} label files")

    for label_path in label_files:
        preprocess_case(label_path)


if __name__ == "__main__":
    main()