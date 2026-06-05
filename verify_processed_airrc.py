from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
PROCESSED_DIR = PROJECT_ROOT / "datasets" / "processed_airrc"

IMAGE_DIR = PROCESSED_DIR / "images"
TARGET_DIR = PROCESSED_DIR / "targets"

def main():
    image_files = sorted(IMAGE_DIR.glob("*_ct.npy"))

    print(f"Found {len(image_files)} processed CT files")

    good = 0
    bad = 0

    for image_path in image_files:
        uid = image_path.name.replace("_ct.npy", "")
        target_path = TARGET_DIR / f"{uid}_target.npy"

        if not target_path.exists():
            print(f"[BAD] Missing target: {uid}")
            bad += 1
            continue

        ct = np.load(image_path, mmap_mode="r")
        target = np.load(target_path, mmap_mode="r")

        issues = []

        if target.ndim != 4:
            issues.append(f"target ndim is {target.ndim}, expected 4")

        if target.shape[0] != 2:
            issues.append(f"target channels is {target.shape[0]}, expected 2")

        if ct.shape != target.shape[1:]:
            issues.append(f"shape mismatch ct={ct.shape}, target={target.shape}")

        lumen_voxels = int(np.sum(target[0] > 0))
        wall_voxels = int(np.sum(target[1] > 0))

        if lumen_voxels == 0:
            issues.append("no lumen voxels")

        if wall_voxels == 0:
            issues.append("no wall voxels")

        if issues:
            print(f"[BAD] {uid}")
            for issue in issues:
                print(f"  - {issue}")
            bad += 1
        else:
            print(f"[OK] {uid} ct={ct.shape} target={target.shape} lumen={lumen_voxels} wall={wall_voxels}")
            good += 1

    print("\nSummary")
    print("  good:", good)
    print("  bad:", bad)

if __name__ == "__main__":
    main()