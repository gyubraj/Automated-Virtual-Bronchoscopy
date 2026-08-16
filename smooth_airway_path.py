from pathlib import Path
import argparse
import json

import numpy as np
from scipy import ndimage as ndi
from scipy.interpolate import splprep, splev
from scipy.spatial import cKDTree


def parse_tuple(value):
    values = tuple(float(part.strip()) for part in value.split(","))
    if len(values) != 3:
        raise argparse.ArgumentTypeError("Expected format: z,y,x")
    return values


def load_paths(path):
    with open(path, "r") as f:
        payload = json.load(f)
    paths = payload.get("paths", [])
    if not paths:
        raise ValueError("No paths found in input JSON.")
    return payload, paths


def select_path(paths, path_id=None):
    if path_id is None:
        return paths

    for path in paths:
        if path.get("path_id") == path_id:
            return [path]

    raise ValueError(f"Path id not found: {path_id}")


def load_lumen_mask(path, threshold):
    if path is None:
        return None

    arr = np.load(path)
    if arr.ndim == 5:
        arr = arr[0, 0]
    elif arr.ndim == 4:
        arr = arr[0]
    elif arr.ndim != 3:
        raise ValueError(f"Expected 3D, 4D, or 5D lumen array, got {arr.shape}")

    mask = arr > threshold
    if int(mask.sum()) == 0:
        raise ValueError("Lumen mask is empty after thresholding.")
    return mask


def load_hysteresis_lumen_mask(path, threshold, low_threshold=None):
    if path is None:
        return None

    arr = np.load(path)
    if arr.ndim == 5:
        arr = arr[0, 0]
    elif arr.ndim == 4:
        arr = arr[0]
    elif arr.ndim != 3:
        raise ValueError(f"Expected 3D, 4D, or 5D lumen array, got {arr.shape}")

    high_mask = arr > threshold
    if low_threshold is None or low_threshold >= threshold:
        mask = high_mask
    else:
        low_mask = arr > low_threshold
        labeled, count = ndi.label(low_mask, structure=ndi.generate_binary_structure(3, 1))
        if count == 0:
            mask = high_mask
        else:
            seed_labels = np.unique(labeled[high_mask])
            seed_labels = seed_labels[seed_labels > 0]
            mask = np.isin(labeled, seed_labels) if len(seed_labels) else high_mask

    if int(mask.sum()) == 0:
        raise ValueError("Lumen mask is empty after thresholding.")
    return mask


def remove_duplicate_points(coords):
    if len(coords) <= 1:
        return coords

    keep = [0]
    for idx in range(1, len(coords)):
        if np.linalg.norm(coords[idx] - coords[keep[-1]]) > 1e-6:
            keep.append(idx)
    return coords[keep]


def cumulative_distance(coords):
    if len(coords) == 0:
        return np.array([], dtype=np.float64)
    distances = np.zeros(len(coords), dtype=np.float64)
    if len(coords) > 1:
        steps = np.linalg.norm(np.diff(coords, axis=0), axis=1)
        distances[1:] = np.cumsum(steps)
    return distances


def resample_polyline(coords, step):
    coords = remove_duplicate_points(np.asarray(coords, dtype=np.float64))
    if len(coords) < 2:
        return coords

    distances = cumulative_distance(coords)
    total = float(distances[-1])
    if total <= 0:
        return coords

    count = max(2, int(np.ceil(total / step)) + 1)
    sample_d = np.linspace(0, total, count)
    resampled = np.column_stack([
        np.interp(sample_d, distances, coords[:, axis])
        for axis in range(3)
    ])
    return resampled


def smooth_spline(coords, smoothing, samples, degree=3):
    coords = remove_duplicate_points(np.asarray(coords, dtype=np.float64))
    if len(coords) < 4:
        return coords

    degree = min(degree, len(coords) - 1)
    distances = cumulative_distance(coords)
    total = float(distances[-1])
    if total <= 0:
        return coords

    u = distances / total
    tck, _ = splprep(
        [coords[:, 0], coords[:, 1], coords[:, 2]],
        u=u,
        s=float(smoothing),
        k=degree,
    )
    u_new = np.linspace(0.0, 1.0, int(samples))
    smoothed = np.asarray(splev(u_new, tck)).T

    smoothed[0] = coords[0]
    smoothed[-1] = coords[-1]
    return smoothed


def smooth_path(raw_coords, pre_resample_step, output_step, smoothing, degree):
    resampled = resample_polyline(raw_coords, step=pre_resample_step)
    resampled_length = cumulative_distance(resampled)[-1] if len(resampled) > 1 else 0.0
    output_samples = max(2, int(np.ceil(resampled_length / output_step)) + 1)
    return smooth_spline(
        resampled,
        smoothing=smoothing,
        samples=output_samples,
        degree=degree,
    )


def project_to_lumen(coords, mask, max_distance):
    if mask is None:
        return coords, 0, 1.0

    lumen_coords = np.argwhere(mask)
    tree = cKDTree(lumen_coords)

    projected = np.asarray(coords, dtype=np.float64).copy()
    rounded = np.rint(projected).astype(int)
    inside_count = 0
    changed = 0

    for idx, point in enumerate(projected):
        voxel = rounded[idx]
        inside = (
            0 <= voxel[0] < mask.shape[0]
            and 0 <= voxel[1] < mask.shape[1]
            and 0 <= voxel[2] < mask.shape[2]
            and bool(mask[tuple(voxel)])
        )
        if inside:
            inside_count += 1
            continue

        distance, nearest_idx = tree.query(point, k=1)
        if distance <= max_distance:
            projected[idx] = lumen_coords[int(nearest_idx)]
            changed += 1

    rounded_after = np.rint(projected).astype(int)
    valid = (
        (rounded_after[:, 0] >= 0)
        & (rounded_after[:, 0] < mask.shape[0])
        & (rounded_after[:, 1] >= 0)
        & (rounded_after[:, 1] < mask.shape[1])
        & (rounded_after[:, 2] >= 0)
        & (rounded_after[:, 2] < mask.shape[2])
    )
    inside_after = np.zeros(len(projected), dtype=bool)
    inside_after[valid] = mask[
        rounded_after[valid, 0],
        rounded_after[valid, 1],
        rounded_after[valid, 2],
    ]
    inside_ratio = float(inside_after.mean()) if len(inside_after) else 0.0

    return projected, changed, inside_ratio


def path_curvature_stats(coords):
    coords = np.asarray(coords, dtype=np.float64)
    if len(coords) < 3:
        return {
            "mean_turn_degrees": 0.0,
            "max_turn_degrees": 0.0,
            "turn_count": 0,
        }

    vectors = np.diff(coords, axis=0)
    norms = np.linalg.norm(vectors, axis=1)
    valid = norms > 1e-8
    vectors = vectors[valid]
    norms = norms[valid]
    if len(vectors) < 2:
        return {
            "mean_turn_degrees": 0.0,
            "max_turn_degrees": 0.0,
            "turn_count": 0,
        }

    unit = vectors / norms[:, None]
    cosines = np.sum(unit[:-1] * unit[1:], axis=1)
    angles = np.degrees(np.arccos(np.clip(cosines, -1.0, 1.0)))
    return {
        "mean_turn_degrees": float(angles.mean()),
        "max_turn_degrees": float(angles.max()),
        "turn_count": int(len(angles)),
    }


def zyx_to_xyz(coords_zyx, spacing_zyx):
    spacing = np.asarray(spacing_zyx, dtype=np.float64)
    scaled = np.asarray(coords_zyx, dtype=np.float64) * spacing
    return scaled[:, [2, 1, 0]]


def make_output_record(path, raw_coords, smooth_coords, spacing_zyx, inside_ratio, projected_points):
    length = cumulative_distance(smooth_coords)[-1] if len(smooth_coords) > 1 else 0.0
    raw_stats = path_curvature_stats(raw_coords)
    smooth_stats = path_curvature_stats(smooth_coords)

    return {
        "path_id": f"{path.get('path_id', 'path_000')}_smooth",
        "source_path_id": path.get("path_id"),
        "root_node": path.get("root_node"),
        "target_node": path.get("target_node"),
        "length_voxels": float(length),
        "raw_length_voxels": path.get("length_voxels"),
        "point_count": int(len(smooth_coords)),
        "raw_point_count": int(len(raw_coords)),
        "coordinates_zyx": np.round(smooth_coords, 3).tolist(),
        "coordinates_xyz": np.round(zyx_to_xyz(smooth_coords, spacing_zyx), 3).tolist(),
        "inside_lumen_ratio": float(inside_ratio),
        "projected_points": int(projected_points),
        "raw_curvature": raw_stats,
        "smoothed_curvature": smooth_stats,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Smooth raw airway graph path into a spline trajectory for virtual bronchoscopy camera motion."
    )
    parser.add_argument("--paths-json", required=True, help="Raw paths JSON from skeletonize_airrc_case.py")
    parser.add_argument("--output-json", required=True, help="Output smoothed paths JSON")
    parser.add_argument("--path-id", default=None, help="Specific path_id to smooth; default smooths all paths")
    parser.add_argument("--lumen-mask", default=None, help="Optional pred_lumen.npy/pred_lumen_mask.npy for inside-lumen projection")
    parser.add_argument("--threshold", type=float, default=0.2, help="Threshold for --lumen-mask if probability volume")
    parser.add_argument("--low-threshold", type=float, default=None, help="Optional lower connected threshold for lumen projection")
    parser.add_argument("--spacing", type=parse_tuple, default=(1.0, 1.0, 1.0), help="Voxel spacing as z,y,x")
    parser.add_argument("--pre-resample-step", type=float, default=2.0, help="Raw path resampling step before spline fit")
    parser.add_argument("--output-step", type=float, default=1.0, help="Approximate output point spacing")
    parser.add_argument("--smoothing", type=float, default=25.0, help="Spline smoothing strength; higher is smoother")
    parser.add_argument("--degree", type=int, default=3, help="Spline degree")
    parser.add_argument("--project-to-lumen", action="store_true", help="Project off-lumen smoothed points to nearest lumen voxel")
    parser.add_argument("--max-projection-distance", type=float, default=3.0, help="Maximum correction distance for projection")
    args = parser.parse_args()

    payload, paths = load_paths(args.paths_json)
    mask = load_hysteresis_lumen_mask(args.lumen_mask, args.threshold, low_threshold=args.low_threshold)
    selected_paths = select_path(paths, args.path_id)
    records = []

    for path in selected_paths:
        raw_coords = np.asarray(path.get("coordinates_zyx", []), dtype=np.float64)
        if raw_coords.ndim != 2 or raw_coords.shape[1] != 3 or len(raw_coords) < 2:
            raise ValueError(
                f"Path {path.get('path_id', '<unknown>')} must contain at least two coordinates_zyx points."
            )

        smoothed = smooth_path(
            raw_coords=raw_coords,
            pre_resample_step=args.pre_resample_step,
            output_step=args.output_step,
            smoothing=args.smoothing,
            degree=args.degree,
        )

        projected_points = 0
        inside_ratio = 1.0
        if args.project_to_lumen:
            smoothed, projected_points, inside_ratio = project_to_lumen(
                smoothed,
                mask=mask,
                max_distance=args.max_projection_distance,
            )
        elif mask is not None:
            _, _, inside_ratio = project_to_lumen(
                smoothed,
                mask=mask,
                max_distance=-1.0,
            )

        records.append(
            make_output_record(
                path=path,
                raw_coords=raw_coords,
                smooth_coords=smoothed,
                spacing_zyx=args.spacing,
                inside_ratio=inside_ratio,
                projected_points=projected_points,
            )
        )

    output_payload = {
        "source_paths_json": str(args.paths_json),
        "source_path_count": len(paths),
        "smoothing_method": "scipy.interpolate.splprep",
        "smoothing": float(args.smoothing),
        "threshold": float(args.threshold),
        "low_threshold": args.low_threshold,
        "pre_resample_step": float(args.pre_resample_step),
        "output_step": float(args.output_step),
        "spacing_zyx": list(args.spacing),
        "path_count": len(records),
        "paths": records,
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output_payload, f, indent=2)

    print("Done")
    print("  source:", args.paths_json)
    print("  output:", output_path)
    print("  smoothed paths:", len(records))
    if records:
        first = records[0]
        print("  first raw points:", first["raw_point_count"])
        print("  first smoothed points:", first["point_count"])
        print("  first projected points:", first["projected_points"])
        print("  first inside lumen ratio:", f"{first['inside_lumen_ratio']:.4f}")
        print("  first raw mean turn:", f"{first['raw_curvature']['mean_turn_degrees']:.2f}")
        print("  first smooth mean turn:", f"{first['smoothed_curvature']['mean_turn_degrees']:.2f}")
        print("  first raw max turn:", f"{first['raw_curvature']['max_turn_degrees']:.2f}")
        print("  first smooth max turn:", f"{first['smoothed_curvature']['max_turn_degrees']:.2f}")


if __name__ == "__main__":
    main()
