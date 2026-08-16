from pathlib import Path
import argparse
import json

import numpy as np
import pyvista as pv
from scipy import ndimage as ndi


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


def postprocess_lumen(mask, keep_largest=False, fill_holes=False, closing_radius=0):
    mask = mask.astype(bool)

    if keep_largest:
        mask = keep_largest_component(mask)

    if fill_holes:
        mask = ndi.binary_fill_holes(mask)

    structure = ndi.generate_binary_structure(3, 1)
    for _ in range(max(0, int(closing_radius))):
        mask = ndi.binary_closing(mask, structure=structure)

    return mask


def load_graph(path):
    with open(path, "r") as f:
        payload = json.load(f)
    return payload.get("graph", payload)


def graph_endpoint_coords(graph):
    endpoints = set(int(node_id) for node_id in graph.get("endpoints", []))
    coords = []
    for node in graph.get("nodes", []):
        node_id = int(node["id"])
        if node_id in endpoints:
            coords.append(node["zyx"])
    return coords


def graph_root_coord(paths_json):
    if paths_json is None:
        return None

    with open(paths_json, "r") as f:
        payload = json.load(f)

    paths = payload.get("paths", [])
    if not paths:
        return None

    coords = paths[0].get("coordinates_zyx", [])
    if not coords:
        return None

    return coords[0]


def choose_opening_coords_zyx(graph_json, paths_json, mode):
    coords = []

    if graph_json is None:
        return coords

    graph = load_graph(graph_json)
    endpoint_coords = graph_endpoint_coords(graph)
    root = graph_root_coord(paths_json)

    if mode == "root-only":
        if root is not None:
            coords.append(root)
    elif mode == "all-endpoints":
        coords.extend(endpoint_coords)
    elif mode == "root-and-endpoints":
        if root is not None:
            coords.append(root)
        coords.extend(endpoint_coords)
    else:
        raise ValueError(f"Unsupported opening mode: {mode}")

    return coords


def create_outer_wall_shell(lumen_mask, spacing_zyx, wall_thickness):
    if wall_thickness <= 0:
        raise ValueError("--wall-thickness must be greater than 0.")

    outside_distance = ndi.distance_transform_edt(~lumen_mask, sampling=spacing_zyx)
    shell = (outside_distance <= wall_thickness) & (~lumen_mask)
    return shell


def carve_openings(shell, opening_coords_zyx, spacing_zyx, opening_radius):
    if opening_radius <= 0 or not opening_coords_zyx:
        return shell, 0

    shell = shell.copy()
    removed_total = 0
    spacing = np.asarray(spacing_zyx, dtype=np.float32)
    radius_voxels = np.ceil(opening_radius / spacing).astype(int)

    for coord in opening_coords_zyx:
        center = np.asarray(coord, dtype=int)
        z0 = max(0, center[0] - radius_voxels[0])
        z1 = min(shell.shape[0], center[0] + radius_voxels[0] + 1)
        y0 = max(0, center[1] - radius_voxels[1])
        y1 = min(shell.shape[1], center[1] + radius_voxels[1] + 1)
        x0 = max(0, center[2] - radius_voxels[2])
        x1 = min(shell.shape[2], center[2] + radius_voxels[2] + 1)

        zz, yy, xx = np.mgrid[z0:z1, y0:y1, x0:x1]
        distances = np.sqrt(
            ((zz - center[0]) * spacing[0]) ** 2
            + ((yy - center[1]) * spacing[1]) ** 2
            + ((xx - center[2]) * spacing[2]) ** 2
        )
        remove = distances <= opening_radius
        before = int(shell[z0:z1, y0:y1, x0:x1].sum())
        shell[z0:z1, y0:y1, x0:x1][remove] = False
        after = int(shell[z0:z1, y0:y1, x0:x1].sum())
        removed_total += before - after

    return shell, removed_total


def clean_shell(shell, keep_largest=True, closing_radius=0):
    shell = shell.astype(bool)

    structure = ndi.generate_binary_structure(3, 1)
    for _ in range(max(0, int(closing_radius))):
        shell = ndi.binary_closing(shell, structure=structure)

    if keep_largest:
        shell = keep_largest_component(shell)

    return shell


def volume_to_surface(volume, spacing_zyx):
    if int(volume.sum()) == 0:
        raise ValueError("Printable shell is empty.")

    nz, ny, nx = volume.shape
    grid = pv.ImageData()
    grid.dimensions = (nx, ny, nz)
    grid.spacing = (spacing_zyx[2], spacing_zyx[1], spacing_zyx[0])
    grid.point_data["values"] = volume.astype(np.uint8).transpose(2, 1, 0).flatten(order="F")

    surface = grid.contour(isosurfaces=[0.5], scalars="values")
    if surface.n_points == 0:
        raise ValueError("Surface extraction produced an empty mesh.")

    return surface.triangulate()


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
        f.write("# Print-ready airway shell\n")
        for x, y, z in points:
            f.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
        for a, b, c in faces:
            f.write(f"f {int(a) + 1} {int(b) + 1} {int(c) + 1}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Create a thickened, hollow airway shell for 3D printing."
    )
    parser.add_argument("--mask", required=True, help="Predicted lumen probability/mask .npy")
    parser.add_argument("--output-dir", required=True, help="Output folder")
    parser.add_argument("--threshold", type=float, default=0.2, help="Lumen probability threshold")
    parser.add_argument("--spacing", type=parse_tuple, default=(1.0, 1.0, 1.0), help="Voxel spacing as z,y,x")
    parser.add_argument("--wall-thickness", type=float, default=1.5, help="Printable wall thickness in spacing units/mm")
    parser.add_argument("--graph-json", default=None, help="Pruned graph JSON used to locate airway openings")
    parser.add_argument("--paths-json", default=None, help="Path JSON used to locate trachea/root opening")
    parser.add_argument(
        "--opening-mode",
        choices=("root-only", "all-endpoints", "root-and-endpoints"),
        default="root-and-endpoints",
        help="Which graph points should be cut open",
    )
    parser.add_argument("--opening-radius", type=float, default=5.0, help="Opening carve radius in spacing units/mm")
    parser.add_argument("--keep-largest", action="store_true", help="Keep largest lumen component before shell creation")
    parser.add_argument("--fill-holes", action="store_true", help="Fill lumen holes before shell creation")
    parser.add_argument("--lumen-closing-radius", type=int, default=0, help="Binary closing iterations on lumen before shell creation")
    parser.add_argument("--shell-closing-radius", type=int, default=0, help="Binary closing iterations on printable shell")
    parser.add_argument("--smooth-iterations", type=int, default=20, help="Surface smoothing iterations")
    parser.add_argument("--smooth-relaxation", type=float, default=0.02, help="Surface smoothing relaxation")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    arr = load_volume(args.mask)
    lumen = arr > args.threshold
    raw_lumen_voxels = int(lumen.sum())
    if raw_lumen_voxels == 0:
        raise ValueError("Lumen mask is empty after thresholding.")

    lumen = postprocess_lumen(
        lumen,
        keep_largest=args.keep_largest,
        fill_holes=args.fill_holes,
        closing_radius=args.lumen_closing_radius,
    )
    shell = create_outer_wall_shell(
        lumen_mask=lumen,
        spacing_zyx=args.spacing,
        wall_thickness=args.wall_thickness,
    )
    opening_coords = choose_opening_coords_zyx(
        graph_json=args.graph_json,
        paths_json=args.paths_json,
        mode=args.opening_mode,
    )
    shell, removed_opening_voxels = carve_openings(
        shell=shell,
        opening_coords_zyx=opening_coords,
        spacing_zyx=args.spacing,
        opening_radius=args.opening_radius,
    )
    shell = clean_shell(shell, keep_largest=True, closing_radius=args.shell_closing_radius)

    surface = volume_to_surface(shell, args.spacing)
    surface = smooth_surface(
        surface,
        iterations=args.smooth_iterations,
        relaxation_factor=args.smooth_relaxation,
    )

    stl_path = output_dir / "airway_print_ready_shell.stl"
    obj_path = output_dir / "airway_print_ready_shell.obj"
    summary_path = output_dir / "airway_print_ready_shell_summary.json"

    surface.save(stl_path)
    save_obj(surface, obj_path)

    summary = {
        "mask": args.mask,
        "threshold": args.threshold,
        "spacing_zyx": list(args.spacing),
        "wall_thickness": args.wall_thickness,
        "opening_radius": args.opening_radius,
        "opening_mode": args.opening_mode,
        "opening_points": len(opening_coords),
        "raw_lumen_voxels": raw_lumen_voxels,
        "postprocessed_lumen_voxels": int(lumen.sum()),
        "shell_voxels": int(shell.sum()),
        "removed_opening_voxels": int(removed_opening_voxels),
        "surface_points": int(surface.n_points),
        "surface_cells": int(surface.n_cells),
        "smooth_iterations": args.smooth_iterations,
        "smooth_relaxation": args.smooth_relaxation,
        "stl": str(stl_path),
        "obj": str(obj_path),
    }

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("Done")
    print("  STL:", stl_path)
    print("  OBJ:", obj_path)
    print("  summary:", summary_path)
    print("  wall thickness:", args.wall_thickness)
    print("  shell voxels:", int(shell.sum()))
    print("  opening points:", len(opening_coords))
    print("  removed opening voxels:", removed_opening_voxels)
    print("  surface points:", surface.n_points)
    print("  surface cells:", surface.n_cells)


if __name__ == "__main__":
    main()
