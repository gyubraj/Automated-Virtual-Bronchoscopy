from pathlib import Path
import argparse
import json
import os
import random

import numpy as np
from scipy import ndimage as ndi

try:
    from skimage.morphology import skeletonize_3d as sk_skeletonize_3d
except ImportError:
    try:
        from skimage.morphology import skeletonize as sk_skeletonize

        def sk_skeletonize_3d(volume):
            return sk_skeletonize(volume)
    except ImportError:
        sk_skeletonize_3d = None


def make_ball(radius):
    coords = np.ogrid[
        -radius: radius + 1,
        -radius: radius + 1,
        -radius: radius + 1,
    ]
    distance_sq = sum(axis ** 2 for axis in coords)
    return distance_sq <= radius ** 2


def skeletonize_lumen_for_sampling(lumen):
    if sk_skeletonize_3d is None:
        return None
    return sk_skeletonize_3d(lumen).astype(bool)

DATA_ROOT = Path(os.environ.get("AVB_DATA_ROOT", Path.home() / "AMS_Project" / "datasets_new"))

PROCESSED_DIR = DATA_ROOT / "processed_airrc"
IMAGE_DIR = PROCESSED_DIR / "images"
TARGET_DIR = PROCESSED_DIR / "targets"

PATCH_DIR = DATA_ROOT / "airrc_patches"
PATCH_IMAGE_DIR = PATCH_DIR / "images"
PATCH_TARGET_DIR = PATCH_DIR / "targets"
SPLIT_DIR = PATCH_DIR / "splits"

PATCH_SIZE = (128, 128, 128)
PATCHES_PER_CASE = 32
VAL_RATIO = 0.2
SEED = 42

DISTAL_PATCH_FRACTION = 0.5
BOUNDARY_PATCH_FRACTION = 0.3
DILATION_RADIUS = 3
DISTAL_RADIUS_PERCENTILE = 35.0

random.seed(SEED)
np.random.seed(SEED)


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
    dilated_lumen = ndi.binary_dilation(lumen, structure=make_ball(DILATION_RADIUS))
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


def pick_coord(coords):
    if coords is None or len(coords) == 0:
        return None
    idx = random.randint(0, len(coords) - 1)
    return tuple(int(v) for v in coords[idx])


def skeleton_endpoint_mask(skeleton):
    structure = np.ones((3, 3, 3), dtype=np.uint8)
    neighbor_count = ndi.convolve(skeleton.astype(np.uint8), structure, mode="constant") * skeleton
    return skeleton & (neighbor_count <= 2)


def distal_center(target):
    lumen = target[0] > 0
    if int(lumen.sum()) == 0:
        return None

    skeleton = skeletonize_lumen_for_sampling(lumen)
    radius = ndi.distance_transform_edt(lumen)
    if skeleton is None or int(skeleton.sum()) == 0:
        radius_values = radius[lumen]
        if len(radius_values) > 0:
            radius_cutoff = np.percentile(radius_values, DISTAL_RADIUS_PERCENTILE)
            coords = np.argwhere(lumen & (radius <= radius_cutoff))
            if coords.size > 0:
                idx = random.randint(0, len(coords) - 1)
                return tuple(int(v) for v in coords[idx])
        coords = np.argwhere(lumen)
        idx = random.randint(0, len(coords) - 1)
        return tuple(int(v) for v in coords[idx])

    skeleton_radius = radius[skeleton]
    if len(skeleton_radius) == 0:
        coords = np.argwhere(skeleton)
    else:
        radius_cutoff = np.percentile(skeleton_radius, DISTAL_RADIUS_PERCENTILE)
        endpoints = skeleton_endpoint_mask(skeleton)
        distal_region = skeleton & ((radius <= radius_cutoff) | endpoints)
        coords = np.argwhere(distal_region)
        if coords.size == 0:
            coords = np.argwhere(skeleton)

    idx = random.randint(0, len(coords) - 1)
    return tuple(int(v) for v in coords[idx])


def build_sampling_pools(target):
    lumen = target[0] > 0
    wall = target[1] > 0
    airway = lumen | wall

    dilated_lumen = ndi.binary_dilation(lumen, structure=make_ball(DILATION_RADIUS))
    boundary_region = dilated_lumen & wall
    boundary_coords = np.argwhere(boundary_region)
    if boundary_coords.size == 0:
        boundary_coords = np.argwhere(airway)

    distal_coords = None
    if int(lumen.sum()) > 0:
        skeleton = skeletonize_lumen_for_sampling(lumen)
        radius = ndi.distance_transform_edt(lumen)
        if skeleton is not None and int(skeleton.sum()) > 0:
            skeleton_radius = radius[skeleton]
            if len(skeleton_radius) > 0:
                radius_cutoff = np.percentile(skeleton_radius, DISTAL_RADIUS_PERCENTILE)
                endpoints = skeleton_endpoint_mask(skeleton)
                distal_region = skeleton & ((radius <= radius_cutoff) | endpoints)
                distal_coords = np.argwhere(distal_region)
            if distal_coords is None or distal_coords.size == 0:
                distal_coords = np.argwhere(skeleton)
        else:
            radius_values = radius[lumen]
            if len(radius_values) > 0:
                radius_cutoff = np.percentile(radius_values, DISTAL_RADIUS_PERCENTILE)
                distal_coords = np.argwhere(lumen & (radius <= radius_cutoff))
            if distal_coords is None or distal_coords.size == 0:
                distal_coords = np.argwhere(lumen)

    return {
        "distal": distal_coords,
        "boundary": boundary_coords,
        "airway": np.argwhere(airway),
    }


def choose_patch_center(ct_shape, sampling_pools):
    sample = random.random()

    if sample < DISTAL_PATCH_FRACTION:
        center = pick_coord(sampling_pools.get("distal"))
        if center is not None:
            return center, "distal"

    if sample < DISTAL_PATCH_FRACTION + BOUNDARY_PATCH_FRACTION:
        center = pick_coord(sampling_pools.get("boundary"))
        if center is not None:
            return center, "boundary"

    center = pick_coord(sampling_pools.get("airway"))
    if center is not None and random.random() < 0.5:
        return center, "airway_fallback"

    return random_center(ct_shape), "random"


def parse_tuple(value):
    values = tuple(int(part.strip()) for part in value.split(","))
    if len(values) != 3:
        raise argparse.ArgumentTypeError("Expected format: d,h,w")
    return values


def configure(args):
    global PATCH_SIZE
    global PATCHES_PER_CASE
    global VAL_RATIO
    global DISTAL_PATCH_FRACTION
    global BOUNDARY_PATCH_FRACTION
    global DISTAL_RADIUS_PERCENTILE

    PATCH_SIZE = args.patch_size
    PATCHES_PER_CASE = args.patches_per_case
    VAL_RATIO = args.val_ratio
    DISTAL_PATCH_FRACTION = args.distal_patch_fraction
    BOUNDARY_PATCH_FRACTION = args.boundary_patch_fraction
    DISTAL_RADIUS_PERCENTILE = args.distal_radius_percentile

    random.seed(args.seed)
    np.random.seed(args.seed)


def validate_sampling_fractions():
    total = DISTAL_PATCH_FRACTION + BOUNDARY_PATCH_FRACTION
    if not 0.0 <= DISTAL_PATCH_FRACTION <= 1.0:
        raise ValueError("DISTAL_PATCH_FRACTION must be in [0, 1].")
    if not 0.0 <= BOUNDARY_PATCH_FRACTION <= 1.0:
        raise ValueError("BOUNDARY_PATCH_FRACTION must be in [0, 1].")
    if total > 1.0:
        raise ValueError("Distal + boundary patch fractions must be <= 1.")


def choose_patch_center_legacy(ct_shape, target):
    use_boundary = random.random() < BOUNDARY_PATCH_FRACTION

    if use_boundary:
        center = boundary_center(target)
        if center is not None:
            return center, "boundary"

    return random_center(ct_shape), "random"


def extract_case(uid, image_path, target_path):
    ct = np.load(image_path).astype(np.float32)
    target = np.load(target_path).astype(np.uint8)

    ct, target = pad_to_patch_size(ct, target, PATCH_SIZE)
    sampling_pools = build_sampling_pools(target)

    saved = []

    for patch_idx in range(PATCHES_PER_CASE):
        center, sample_type = choose_patch_center(ct.shape, sampling_pools)

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
            "sample_type": sample_type,
        })

    return saved


def main():
    parser = argparse.ArgumentParser(description="Extract AirRC training patches with distal airway oversampling.")
    parser.add_argument("--patch-size", type=parse_tuple, default=PATCH_SIZE, help="Patch size as d,h,w")
    parser.add_argument("--patches-per-case", type=int, default=PATCHES_PER_CASE)
    parser.add_argument("--val-ratio", type=float, default=VAL_RATIO)
    parser.add_argument("--distal-patch-fraction", type=float, default=DISTAL_PATCH_FRACTION)
    parser.add_argument("--boundary-patch-fraction", type=float, default=BOUNDARY_PATCH_FRACTION)
    parser.add_argument("--distal-radius-percentile", type=float, default=DISTAL_RADIUS_PERCENTILE)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    configure(args)
    validate_sampling_fractions()

    for folder in [PATCH_IMAGE_DIR, PATCH_TARGET_DIR, SPLIT_DIR]:
        folder.mkdir(parents=True, exist_ok=True)

    image_files = sorted(IMAGE_DIR.glob("*_ct.npy"))

    print(f"Found {len(image_files)} processed CT files")
    print(f"Patch size: {PATCH_SIZE}")
    print(f"Patches per case: {PATCHES_PER_CASE}")
    print(f"Distal patch fraction: {DISTAL_PATCH_FRACTION}")
    print(f"Boundary patch fraction: {BOUNDARY_PATCH_FRACTION}")

    cases = []

    for image_path in image_files:
        uid = image_path.name.replace("_ct.npy", "")
        target_path = TARGET_DIR / f"{uid}_target.npy"

        if not target_path.exists():
            print(f"[SKIP] Missing target for {uid}")
            continue

        cases.append((uid, image_path, target_path))

    random.shuffle(cases)
    val_case_count = int(round(len(cases) * VAL_RATIO))
    if VAL_RATIO > 0 and len(cases) > 1:
        val_case_count = max(1, min(val_case_count, len(cases) - 1))

    val_uids = {uid for uid, _, _ in cases[:val_case_count]}
    train_uids = {uid for uid, _, _ in cases[val_case_count:]}

    train_patches = []
    val_patches = []

    for uid, image_path, target_path in cases:
        print(f"[PATCH] {uid}")
        case_patches = extract_case(uid, image_path, target_path)
        if uid in val_uids:
            val_patches.extend(case_patches)
        else:
            train_patches.extend(case_patches)

    random.shuffle(train_patches)
    random.shuffle(val_patches)
    all_patches = train_patches + val_patches

    with open(SPLIT_DIR / "train.json", "w") as f:
        json.dump(train_patches, f, indent=2)

    with open(SPLIT_DIR / "val.json", "w") as f:
        json.dump(val_patches, f, indent=2)

    with open(SPLIT_DIR / "split_summary.json", "w") as f:
        json.dump({
            "split_method": "case_uid",
            "seed": args.seed,
            "val_ratio": VAL_RATIO,
            "train_case_count": len(train_uids),
            "val_case_count": len(val_uids),
            "train_uids": sorted(train_uids),
            "val_uids": sorted(val_uids),
            "train_patch_count": len(train_patches),
            "val_patch_count": len(val_patches),
        }, f, indent=2)

    print("\nDone")
    print("  total patches:", len(all_patches))
    print("  train cases:", len(train_uids))
    print("  val cases:", len(val_uids))
    print("  train patches:", len(train_patches))
    print("  val patches:", len(val_patches))
    for sample_type in ("distal", "boundary", "airway_fallback", "random"):
        count = sum(1 for item in all_patches if item.get("sample_type") == sample_type)
        print(f"  {sample_type} patches:", count)
    print("  output:", PATCH_DIR)


if __name__ == "__main__":
    main()
