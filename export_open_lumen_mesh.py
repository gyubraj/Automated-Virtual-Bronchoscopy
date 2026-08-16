from pathlib import Path
import argparse
import json

import numpy as np
import pyvista as pv
from scipy import ndimage as ndi
from scipy.spatial import cKDTree


def parse_tuple(value):
    values = tuple(float(part.strip()) for part in value.split(","))
    if len(values) != 3:
        raise argparse.ArgumentTypeError("Expected format: z,y,x")
    return values


def load_volume(path):
    arr = np.load(path)
    if arr.ndim == 5:
        arr = arr[0, 0]
    elif arr.ndim == 4:
        arr = arr[0]
    elif arr.ndim != 3:
        raise ValueError(f"Expected 3D, 4D, or 5D array, got {arr.shape}")
    return arr


def keep_largest_component(mask):
    structure = ndi.generate_binary_structure(3, 2)
    labeled, count = ndi.label(mask, structure=structure)
    if count <= 1:
        return mask

    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    largest = int(np.argmax(sizes))
    return labeled == largest


def postprocess_mask(mask, keep_largest=False, fill_holes=False, closing_radius=0):
    mask = mask.astype(bool)

    if keep_largest:
        mask = keep_largest_component(mask)

    if fill_holes:
        mask = ndi.binary_fill_holes(mask)

    structure = ndi.generate_binary_structure(3, 1)
    for _ in range(max(0, int(closing_radius))):
        mask = ndi.binary_closing(mask, structure=structure)

    return mask.astype(np.uint8)


def zyx_to_xyz(coords_zyx, spacing_zyx):
    coords = np.asarray(coords_zyx, dtype=np.float32)
    spacing = np.asarray(spacing_zyx, dtype=np.float32)
    scaled = coords * spacing
    return scaled[:, [2, 1, 0]]


def mask_to_surface(mask, spacing_zyx):
    nz, ny, nx = mask.shape
    grid = pv.ImageData()
    grid.dimensions = (nx, ny, nz)
    grid.spacing = (spacing_zyx[2], spacing_zyx[1], spacing_zyx[0])
    grid.point_data["values"] = mask.transpose(2, 1, 0).flatten(order="F")

    surface = grid.contour(isosurfaces=[0.5], scalars="values")
    if surface.n_points == 0:
        raise ValueError("Surface extraction produced an empty mesh.")
    return surface.triangulate()


def load_graph(path):
    with open(path, "r") as f:
        payload = json.load(f)
    return payload.get("graph", payload)


def graph_endpoint_coords(graph):
    nodes = graph.get("nodes", [])
    endpoints = set(int(node_id) for node_id in graph.get("endpoints", []))
    coords = []

    for node in nodes:
        node_id = int(node["id"])
        if node_id in endpoints:
            coords.append(node["zyx"])

    return coords


def graph_root_coord(path_json):
    if path_json is None:
        return None

    with open(path_json, "r") as f:
        payload = json.load(f)

    paths = payload.get("paths", [])
    if not paths:
        return None

    coords = paths[0].get("coordinates_zyx", [])
    if not coords:
        return None

    return coords[0]


def choose_opening_coords(graph_json, paths_json, spacing_zyx, mode):
    coords = []

    if graph_json is not None:
        graph = load_graph(graph_json)
        endpoint_coords = graph_endpoint_coords(graph)
        if mode == "all-endpoints":
            coords.extend(endpoint_coords)
        elif mode == "root-only":
            root = graph_root_coord(paths_json)
            if root is not None:
                coords.append(root)
        elif mode == "root-and-endpoints":
            root = graph_root_coord(paths_json)
            if root is not None:
                coords.append(root)
            coords.extend(endpoint_coords)
        else:
            raise ValueError(f"Unsupported opening mode: {mode}")

    if not coords:
        return np.empty((0, 3), dtype=np.float32)

    return zyx_to_xyz(coords, spacing_zyx)


def remove_cells_near_points(surface, points_xyz, radius):
    if len(points_xyz) == 0 or radius <= 0:
        return surface, 0

    centers = surface.cell_centers().points
    tree = cKDTree(points_xyz)
    distances, _ = tree.query(centers, k=1)
    keep = distances > radius

    removed = int((~keep).sum())
    if removed == 0:
        return surface, 0

    open_surface = surface.extract_cells(np.where(keep)[0]).extract_surface().triangulate()
    return open_surface, removed


def smooth_surface(surface, iterations=0, relaxation_factor=0.01):
    if iterations <= 0:
        return surface

    return surface.smooth(
        n_iter=iterations,
        relaxation_factor=relaxation_factor,
        feature_smoothing=False,
        boundary_smoothing=True,
    ).triangulate()


def save_obj(polydata, output_path):
    faces = polydata.faces.reshape((-1, 4))[:, 1:]
    points = polydata.points

    with open(output_path, "w") as f:
        f.write("# Open airway lumen surface\n")
        for x, y, z in points:
            f.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
        for a, b, c in faces:
            f.write(f"f {int(a) + 1} {int(b) + 1} {int(c) + 1}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Export an open airway lumen surface for Blender/endoscope simulation."
    )
    parser.add_argument("--mask", required=True, help="Predicted lumen probability/mask .npy")
    parser.add_argument("--output-dir", required=True, help="Output folder")
    parser.add_argument("--threshold", type=float, default=0.2, help="Lumen probability threshold")
    parser.add_argument("--spacing", type=parse_tuple, default=(1.0, 1.0, 1.0), help="Voxel spacing as z,y,x")
    parser.add_argument("--graph-json", default=None, help="Pruned graph JSON used to locate airway endpoints")
    parser.add_argument("--paths-json", default=None, help="Path JSON used to locate trachea/root opening")
    parser.add_argument(
        "--opening-mode",
        choices=("root-only", "all-endpoints", "root-and-endpoints"),
        default="root-and-endpoints",
        help="Which graph points should be cut open",
    )
    parser.add_argument("--opening-radius", type=float, default=5.0, help="Opening radius in physical/voxel units after spacing")
    parser.add_argument("--keep-largest", action="store_true", help="Keep largest lumen component before meshing")
    parser.add_argument("--fill-holes", action="store_true", help="Fill mask holes before meshing")
    parser.add_argument("--closing-radius", type=int, default=0, help="Binary closing iterations before meshing")
    parser.add_argument("--smooth-iterations", type=int, default=10, help="Surface smoothing iterations")
    parser.add_argument("--smooth-relaxation", type=float, default=0.01, help="Surface smoothing relaxation")
    parser.add_argument("--flip-normals", action="store_true", help="Flip normals inward for inside-airway viewing")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    arr = load_volume(args.mask)
    mask = arr > args.threshold
    raw_voxels = int(mask.sum())
    if raw_voxels == 0:
        raise ValueError("Mask is empty after thresholding.")

    mask = postprocess_mask(
        mask,
        keep_largest=args.keep_largest,
        fill_holes=args.fill_holes,
        closing_radius=args.closing_radius,
    )
    surface = mask_to_surface(mask, args.spacing)
    opening_points = choose_opening_coords(
        graph_json=args.graph_json,
        paths_json=args.paths_json,
        spacing_zyx=args.spacing,
        mode=args.opening_mode,
    )
    surface, removed_cells = remove_cells_near_points(surface, opening_points, args.opening_radius)
    surface = smooth_surface(
        surface,
        iterations=args.smooth_iterations,
        relaxation_factor=args.smooth_relaxation,
    )

    if args.flip_normals:
        surface.flip_normals()

    stl_path = output_dir / "airway_lumen_open.stl"
    obj_path = output_dir / "airway_lumen_open.obj"
    summary_path = output_dir / "airway_lumen_open_summary.json"

    surface.save(stl_path)
    save_obj(surface, obj_path)

    summary = {
        "mask": args.mask,
        "threshold": args.threshold,
        "spacing_zyx": list(args.spacing),
        "mask_shape_zyx": list(mask.shape),
        "raw_mask_voxels": raw_voxels,
        "postprocessed_mask_voxels": int(mask.sum()),
        "graph_json": args.graph_json,
        "paths_json": args.paths_json,
        "opening_mode": args.opening_mode,
        "opening_radius": args.opening_radius,
        "opening_points": int(len(opening_points)),
        "removed_cells_for_openings": removed_cells,
        "surface_points": int(surface.n_points),
        "surface_cells": int(surface.n_cells),
        "flip_normals": bool(args.flip_normals),
        "stl": str(stl_path),
        "obj": str(obj_path),
    }

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("Done")
    print("  STL:", stl_path)
    print("  OBJ:", obj_path)
    print("  summary:", summary_path)
    print("  opening points:", len(opening_points))
    print("  removed cells for openings:", removed_cells)
    print("  surface points:", surface.n_points)
    print("  surface cells:", surface.n_cells)


if __name__ == "__main__":
    main()
