from pathlib import Path
import json
import random

import numpy as np
from skimage.morphology import binary_dilation, ball


PROJECT_ROOT = Path(__file__).resolve().parent

PROCESSED_DIR = PROJECT_ROOT / "datasets_new" / "processed_airrc"
IMAGE_DIR = PROCESSED_DIR / "images"
TARGET_DIR = PROCESSED_DIR / "targets"

PATCH_DIR = PROJECT_ROOT / "datasets_new" / "airrc_patches"
PATCH_IMAGE_DIR = PATCH_DIR / "images"
PATCH_TARGET_DIR = PATCH_DIR / "targets"
SPLIT_DIR = PATCH_DIR / "splits"

PATCH_SIZE = (128, 128, 128)
PATCHES_PER_CASE = 12
VAL_RATIO = 0.2
SEED = 42

BOUNDARY_PATCH_FRACTION = 0.8
DILATION_RADIUS = 3

random.seed(SEED)
np.random.seed(SEED)

for folder in [PATCH_IMAGE_DIR, PATCH_TARGET_DIR, SPLIT_DIR]:
    folder.mkdir(parents=True, exist_ok=True)


def pad_to_patch_size(ct, target, patch_size):
    d, h, w = ct.shape
    pd, ph, pw = patch_size

    pad_d = max(0, pd - d)
    pad_h = max(0, ph - h)
    pad_w = max(0, pw - w)

    if pad_d == 0 and pad_h == 0 and pad_w == 0:
        return ct, target

    ct_pad = (
        (pad_d // 2, pad_d - pad_d // 2),
        (pad_h // 2, pad_h - pad_h // 2),
        (pad_w // 2, pad_w - pad_w // 2),
    )

    target_pad = (
        (0, 0),
        ct_pad[0],
        ct_pad[1],
        ct_pad[2],
    )

    ct = np.pad(ct, ct_pad, mode="constant", constant_values=0)
    target = np.pad(target, target_pad, mode="constant", constant_values=0)

    return ct, target


def crop_patch(ct, target, center, patch_size):
    z, y, x = center
    pd, ph, pw = patch_size

    z0 = min(max(z - pd // 2, 0), ct.shape[0] - pd)
    y0 = min(max(y - ph // 2, 0), ct.shape[1] - ph)
    x0 = min(max(x - pw // 2, 0), ct.shape[2] - pw)

    z1 = z0 + pd
    y1 = y0 + ph
    x1 = x0 + pw

    ct_patch = ct[z0:z1, y0:y1, x0:x1]
    target_patch = target[:, z0:z1, y0:y1, x0:x1]

    return ct_patch, target_patch


def random_center(shape):
    d, h, w = shape

    return (
        random.randint(0, d - 1),
        random.randint(0, h - 1),
        random.randint(0, w - 1),
    )


def boundary_center(target):
    lumen = target[0] > 0
    wall = target[1] > 0

    # Prefer locations where the lumen expands into the wall region.
    # These are the patches most relevant for learning the lumen-wall boundary.
    dilated_lumen = binary_dilation(lumen, ball(DILATION_RADIUS))
    boundary_region = dilated_lumen & wall

    coords = np.argwhere(boundary_region)

    # Fallback: if the boundary region is empty, sample any airway voxel.
    if coords.size == 0:
        airway = lumen | wall
        coords = np.argwhere(airway)

    if coords.size == 0:
        return None

    idx = random.randint(0, len(coords) - 1)
    return tuple(int(v) for v in coords[idx])


def choose_patch_center(ct_shape, target):
    use_boundary = random.random() < BOUNDARY_PATCH_FRACTION

    if use_boundary:
        center = boundary_center(target)
        if center is not None:
            return center

    return random_center(ct_shape)


def extract_case(uid, image_path, target_path):
    ct = np.load(image_path).astype(np.float32)
    target = np.load(target_path).astype(np.uint8)

    ct, target = pad_to_patch_size(ct, target, PATCH_SIZE)

    saved = []

    for patch_idx in range(PATCHES_PER_CASE):
        center = choose_patch_center(ct.shape, target)

        ct_patch, target_patch = crop_patch(ct, target, center, PATCH_SIZE)

        if ct_patch.shape != PATCH_SIZE:
            raise RuntimeError(f"Bad CT patch shape for {uid}: {ct_patch.shape}")

        if target_patch.shape != (2, *PATCH_SIZE):
            raise RuntimeError(f"Bad target patch shape for {uid}: {target_patch.shape}")

        ct_patch = ct_patch[None, ...].astype(np.float32)
        target_patch = target_patch.astype(np.uint8)

        patch_id = f"{uid}_patch{patch_idx:03d}"

        image_out = PATCH_IMAGE_DIR / f"{patch_id}_ct.npy"
        target_out = PATCH_TARGET_DIR / f"{patch_id}_target.npy"

        np.save(image_out, ct_patch)
        np.save(target_out, target_patch)

        saved.append({
            "patch_id": patch_id,
            "uid": uid,
            "image": str(image_out),
            "target": str(target_out),
            "center_zyx": list(center),
            "ct_shape": list(ct_patch.shape),
            "target_shape": list(target_patch.shape),
            "lumen_voxels": int(target_patch[0].sum()),
            "wall_voxels": int(target_patch[1].sum()),
        })

    return saved


def main():
    image_files = sorted(IMAGE_DIR.glob("*_ct.npy"))

    print(f"Found {len(image_files)} processed CT files")
    print(f"Patch size: {PATCH_SIZE}")
    print(f"Patches per case: {PATCHES_PER_CASE}")
    print(f"Boundary patch fraction: {BOUNDARY_PATCH_FRACTION}")

    all_patches = []

    for image_path in image_files:
        uid = image_path.name.replace("_ct.npy", "")
        target_path = TARGET_DIR / f"{uid}_target.npy"

        if not target_path.exists():
            print(f"[SKIP] Missing target for {uid}")
            continue

        print(f"[PATCH] {uid}")
        case_patches = extract_case(uid, image_path, target_path)
        all_patches.extend(case_patches)

    random.shuffle(all_patches)

    val_count = int(len(all_patches) * VAL_RATIO)
    val_patches = all_patches[:val_count]
    train_patches = all_patches[val_count:]

    with open(SPLIT_DIR / "train.json", "w") as f:
        json.dump(train_patches, f, indent=2)

    with open(SPLIT_DIR / "val.json", "w") as f:
        json.dump(val_patches, f, indent=2)

    print("\nDone")
    print("  total patches:", len(all_patches))
    print("  train patches:", len(train_patches))
    print("  val patches:", len(val_patches))
    print("  output:", PATCH_DIR)


if __name__ == "__main__":
    main()