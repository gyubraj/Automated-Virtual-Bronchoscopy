from pathlib import Path
import argparse
import json

import numpy as np
from scipy import ndimage as ndi

try:
    import SimpleITK as sitk
except ImportError:
    sitk = None

try:
    from skimage.morphology import skeletonize_3d as sk_skeletonize_3d
except ImportError:
    from skimage.morphology import skeletonize as sk_skeletonize

    def sk_skeletonize_3d(volume):
        return sk_skeletonize(volume)


NEIGHBOR_OFFSETS = [
    (dz, dy, dx)
    for dz in (-1, 0, 1)
    for dy in (-1, 0, 1)
    for dx in (-1, 0, 1)
    if not (dz == 0 and dy == 0 and dx == 0)
]


def load_volume(path, channel=0):
    path = Path(path)

    if path.suffix == ".npy":
        arr = np.load(path)
        reference_img = None
    elif path.name.endswith(".nii.gz") or path.suffix == ".nii":
        if sitk is None:
            raise RuntimeError("SimpleITK is required to read NIfTI files.")
        reference_img = sitk.ReadImage(str(path))
        arr = sitk.GetArrayFromImage(reference_img)
    else:
        raise ValueError(f"Unsupported input format: {path}")

    arr = np.asarray(arr)

    if arr.ndim == 5:
        arr = arr[0, channel]
    elif arr.ndim == 4:
        arr = arr[channel]
    elif arr.ndim != 3:
        raise ValueError(f"Expected 3D, 4D, or 5D volume, got shape {arr.shape}")

    return arr, reference_img


def save_nifti_like(array, reference_img, output_path):
    if sitk is None or reference_img is None:
        return

    out_img = sitk.GetImageFromArray(array.astype(np.uint8))
    out_img.CopyInformation(reference_img)
    sitk.WriteImage(out_img, str(output_path))


def keep_largest_component(mask):
    labeled, num_components = ndi.label(mask)

    if num_components <= 1:
        return mask

    counts = np.bincount(labeled.ravel())
    counts[0] = 0
    largest_label = int(np.argmax(counts))

    return labeled == largest_label


def postprocess_mask(mask, keep_largest=True, closing_radius=1):
    mask = mask.astype(bool)

    if closing_radius > 0:
        structure = ndi.generate_binary_structure(rank=3, connectivity=1)
        for _ in range(closing_radius):
            mask = ndi.binary_closing(mask, structure=structure)

    if keep_largest:
        mask = keep_largest_component(mask)

    return mask


def skeletonize_lumen(mask):
    skeleton = sk_skeletonize_3d(mask.astype(bool))
    return skeleton.astype(np.uint8)


def build_voxel_graph(skeleton):
    coords = np.argwhere(skeleton > 0)
    coord_to_id = {tuple(coord): idx for idx, coord in enumerate(map(tuple, coords))}

    nodes = []
    edges = []

    for idx, coord in enumerate(coords):
        z, y, x = map(int, coord)
        degree = 0

        for dz, dy, dx in NEIGHBOR_OFFSETS:
            neighbor = (z + dz, y + dy, x + dx)
            neighbor_id = coord_to_id.get(neighbor)

            if neighbor_id is None:
                continue

            degree += 1
            if idx < neighbor_id:
                edges.append([idx, neighbor_id])

        nodes.append({
            "id": idx,
            "zyx": [z, y, x],
            "degree": degree,
        })

    endpoints = [node["id"] for node in nodes if node["degree"] == 1]
    branchpoints = [node["id"] for node in nodes if node["degree"] >= 3]

    return {
        "nodes": nodes,
        "edges": edges,
        "endpoints": endpoints,
        "branchpoints": branchpoints,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Skeletonize a predicted airway lumen mask into a 1-voxel-wide centerline."
    )
    parser.add_argument("--input", required=True, help="Predicted lumen mask/probability volume: .npy, .nii, or .nii.gz")
    parser.add_argument("--output-dir", default="datasets/centerlines", help="Output folder")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for probability volumes")
    parser.add_argument("--channel", type=int, default=0, help="Channel to read if input has channels")
    parser.add_argument("--keep-largest", action="store_true", help="Keep only largest connected component")
    parser.add_argument("--closing-radius", type=int, default=1, help="Binary closing iterations before skeletonization")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    volume, reference_img = load_volume(input_path, channel=args.channel)

    mask = volume > args.threshold
    mask = postprocess_mask(
        mask,
        keep_largest=args.keep_largest,
        closing_radius=args.closing_radius,
    )

    skeleton = skeletonize_lumen(mask)
    graph = build_voxel_graph(skeleton)

    stem = input_path.name
    if stem.endswith(".nii.gz"):
        stem = stem[:-7]
    else:
        stem = input_path.stem

    skeleton_npy = output_dir / f"{stem}_centerline.npy"
    graph_json = output_dir / f"{stem}_centerline_graph.json"
    skeleton_nii = output_dir / f"{stem}_centerline.nii.gz"

    np.save(skeleton_npy, skeleton.astype(np.uint8))

    with open(graph_json, "w") as f:
        json.dump({
            "input": str(input_path),
            "threshold": args.threshold,
            "channel": args.channel,
            "skeleton_voxels": int(skeleton.sum()),
            "node_count": len(graph["nodes"]),
            "edge_count": len(graph["edges"]),
            "endpoint_count": len(graph["endpoints"]),
            "branchpoint_count": len(graph["branchpoints"]),
            "graph": graph,
        }, f, indent=2)

    save_nifti_like(skeleton, reference_img, skeleton_nii)

    print("Done")
    print("  input:", input_path)
    print("  centerline npy:", skeleton_npy)
    if reference_img is not None:
        print("  centerline nifti:", skeleton_nii)
    print("  graph json:", graph_json)
    print("  skeleton voxels:", int(skeleton.sum()))
    print("  graph nodes:", len(graph["nodes"]))
    print("  graph edges:", len(graph["edges"]))
    print("  endpoints:", len(graph["endpoints"]))
    print("  branchpoints:", len(graph["branchpoints"]))


if __name__ == "__main__":
    main()