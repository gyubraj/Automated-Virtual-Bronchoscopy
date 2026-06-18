from pathlib import Path
import argparse
import json

import numpy as np

try:
    import SimpleITK as sitk
except ImportError:
    sitk = None


def parse_spacing(value):
    values = tuple(float(part.strip()) for part in value.split(","))
    if len(values) != 3:
        raise argparse.ArgumentTypeError("Expected spacing as z,y,x, for example: 1,1,1")
    return values


def read_dicom_series(dicom_dir):
    if sitk is None:
        raise RuntimeError("SimpleITK is required. Install it with: pip install SimpleITK")

    dicom_dir = Path(dicom_dir)
    if not dicom_dir.is_dir():
        raise FileNotFoundError(f"DICOM folder not found: {dicom_dir}")

    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(str(dicom_dir))

    if not series_ids:
        files = reader.GetGDCMSeriesFileNames(str(dicom_dir))
    else:
        if len(series_ids) > 1:
            print(f"[WARN] Found {len(series_ids)} DICOM series. Using first series id: {series_ids[0]}")
        files = reader.GetGDCMSeriesFileNames(str(dicom_dir), series_ids[0])

    if not files:
        raise RuntimeError(f"No DICOM files found in {dicom_dir}")

    reader.SetFileNames(files)
    image = reader.Execute()
    return image, files


def resample_image(image, target_spacing_zyx, interpolator=sitk.sitkLinear):
    original_spacing_xyz = image.GetSpacing()
    original_size_xyz = image.GetSize()
    target_spacing_xyz = (
        float(target_spacing_zyx[2]),
        float(target_spacing_zyx[1]),
        float(target_spacing_zyx[0]),
    )

    target_size_xyz = [
        int(round(original_size_xyz[i] * (original_spacing_xyz[i] / target_spacing_xyz[i])))
        for i in range(3)
    ]
    target_size_xyz = [max(1, size) for size in target_size_xyz]

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(target_spacing_xyz)
    resampler.SetSize(target_size_xyz)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(-1000)
    resampler.SetInterpolator(interpolator)

    return resampler.Execute(image)


def normalize_ct_hu(ct, clip_min=-1000.0, clip_max=400.0):
    ct = ct.astype(np.float32)
    ct = np.clip(ct, clip_min, clip_max)
    ct = (ct - clip_min) / (clip_max - clip_min)
    return ct.astype(np.float32)


def save_metadata(path, case_id, dicom_dir, dicom_files, original_image, resampled_image, target_spacing_zyx):
    metadata = {
        "case_id": case_id,
        "dicom_dir": str(Path(dicom_dir)),
        "num_dicom_files": len(dicom_files),
        "target_spacing_zyx": list(target_spacing_zyx),
        "original": {
            "size_xyz": list(original_image.GetSize()),
            "spacing_xyz": list(original_image.GetSpacing()),
            "origin_xyz": list(original_image.GetOrigin()),
            "direction": list(original_image.GetDirection()),
        },
        "resampled": {
            "size_xyz": list(resampled_image.GetSize()),
            "spacing_xyz": list(resampled_image.GetSpacing()),
            "origin_xyz": list(resampled_image.GetOrigin()),
            "direction": list(resampled_image.GetDirection()),
        },
        "intensity": {
            "clip_min_hu": -1000.0,
            "clip_max_hu": 400.0,
            "normalized_min": 0.0,
            "normalized_max": 1.0,
        },
    }

    with open(path, "w") as f:
        json.dump(metadata, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess a raw LIDC DICOM CT series for WingsNet airway inference."
    )
    parser.add_argument("--dicom-dir", required=True, help="Folder containing one CT DICOM series")
    parser.add_argument("--case-id", required=True, help="Case id used for output filenames")
    parser.add_argument("--output-root", required=True, help="Root folder containing images/ and metadata/")
    parser.add_argument("--target-spacing", type=parse_spacing, default=(1.0, 1.0, 1.0), help="Target spacing as z,y,x")
    parser.add_argument("--save-nifti", action="store_true", help="Also save normalized CT as NIfTI")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    image_dir = output_root / "images"
    metadata_dir = output_root / "metadata"
    image_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    print("Reading DICOM series:", args.dicom_dir)
    image, dicom_files = read_dicom_series(args.dicom_dir)
    original_arr = sitk.GetArrayFromImage(image)
    print("  original shape z,y,x:", original_arr.shape)
    print("  original spacing x,y,z:", image.GetSpacing())
    print("  original HU min/max/mean:", float(original_arr.min()), float(original_arr.max()), float(original_arr.mean()))

    print("Resampling to spacing z,y,x:", args.target_spacing)
    resampled = resample_image(image, target_spacing_zyx=args.target_spacing)
    resampled_arr = sitk.GetArrayFromImage(resampled)
    print("  resampled shape z,y,x:", resampled_arr.shape)
    print("  resampled spacing x,y,z:", resampled.GetSpacing())

    normalized = normalize_ct_hu(resampled_arr)
    print("  normalized min/max/mean:", float(normalized.min()), float(normalized.max()), float(normalized.mean()))

    npy_path = image_dir / f"{args.case_id}_ct.npy"
    meta_path = metadata_dir / f"{args.case_id}_meta.json"
    np.save(npy_path, normalized)
    save_metadata(
        meta_path,
        case_id=args.case_id,
        dicom_dir=args.dicom_dir,
        dicom_files=dicom_files,
        original_image=image,
        resampled_image=resampled,
        target_spacing_zyx=args.target_spacing,
    )

    print("Saved:")
    print("  CT npy:", npy_path)
    print("  metadata:", meta_path)

    if args.save_nifti:
        nifti_path = image_dir / f"{args.case_id}_ct.nii.gz"
        normalized_img = sitk.GetImageFromArray(normalized)
        normalized_img.CopyInformation(resampled)
        sitk.WriteImage(normalized_img, str(nifti_path))
        print("  CT nifti:", nifti_path)


if __name__ == "__main__":
    main()
