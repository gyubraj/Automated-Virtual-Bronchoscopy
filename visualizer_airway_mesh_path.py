from pathlib import Path
import argparse
import json

import numpy as np
import pyvista as pv
from scipy import ndimage as ndi

try:
    import skimage.measure as measure
except ImportError:
    measure = None


def parse_tuple(value):
    values = tuple(float(part.strip()) for part in value.split(","))
    if len(values) != 3:
        raise argparse.ArgumentTypeError("Expected format: z,y,x")
    return values


def load_lumen_mask(mask_path, threshold=0.5):
    arr = np.load(mask_path)

    if arr.ndim == 5:
        arr = arr[0, 0]
    elif arr.ndim == 4:
        arr = arr[0]
    elif arr.ndim != 3:
        raise ValueError(f"Expected 3D, 4D, or 5D mask/probability array, got {arr.shape}")

    mask = arr > threshold
    if int(mask.sum()) == 0:
        raise ValueError("Mask is empty after thresholding. Recheck inference output or threshold.")

    return mask.astype(np.uint8)


def keep_largest_component(mask, connectivity=2):
    if measure is None:
        structure = ndi.generate_binary_structure(3, connectivity)
        labeled, count = ndi.label(mask, structure=structure)
    else:
        labeled, count = measure.label(mask, return_num=True, connectivity=connectivity)

    if count <= 1:
        return mask.astype(np.uint8)

    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    largest = int(np.argmax(sizes))
    return (labeled == largest).astype(np.uint8)


def postprocess_mesh_mask(
    mask,
    keep_largest=False,
    fill_holes=False,
    closing_radius=0,
    opening_radius=0,
):
    mask = mask.astype(bool)

    if keep_largest:
        mask = keep_largest_component(mask.astype(np.uint8), connectivity=2).astype(bool)

    if fill_holes:
        mask = ndi.binary_fill_holes(mask)

    if closing_radius > 0:
        structure = ndi.generate_binary_structure(3, 1)
        for _ in range(closing_radius):
            mask = ndi.binary_closing(mask, structure=structure)

    if opening_radius > 0:
        structure = ndi.generate_binary_structure(3, 1)
        for _ in range(opening_radius):
            mask = ndi.binary_opening(mask, structure=structure)

    return mask.astype(np.uint8)


def load_path_coordinates(paths_json, path_id=None):
    with open(paths_json, "r") as f:
        payload = json.load(f)

    paths = payload.get("paths", [])
    if not paths:
        raise ValueError("No planned paths found in paths JSON.")

    if path_id is None:
        path = paths[0]
    else:
        matches = [item for item in paths if item.get("path_id") == path_id]
        if not matches:
            raise ValueError(f"Path id not found: {path_id}")
        path = matches[0]

    coords_zyx = np.asarray(path["coordinates_zyx"], dtype=np.float32)
    if coords_zyx.ndim != 2 or coords_zyx.shape[1] != 3 or len(coords_zyx) == 0:
        raise ValueError("Selected path does not contain valid coordinates_zyx.")

    return path, coords_zyx


def zyx_to_xyz(coords_zyx, spacing_zyx=(1.0, 1.0, 1.0)):
    spacing = np.asarray(spacing_zyx, dtype=np.float32)
    coords_scaled = coords_zyx * spacing
    return coords_scaled[:, [2, 1, 0]]


def mask_to_pyvista_surface(mask, spacing_zyx=(1.0, 1.0, 1.0)):
    nz, ny, nx = mask.shape
    spacing_xyz = (spacing_zyx[2], spacing_zyx[1], spacing_zyx[0])

    grid = pv.ImageData()
    grid.dimensions = (nx, ny, nz)
    grid.spacing = spacing_xyz
    grid.point_data["values"] = mask.transpose(2, 1, 0).flatten(order="F")

    surface = grid.contour(isosurfaces=[0.5], scalars="values")
    if surface.n_points == 0:
        raise ValueError("Marching-cubes/contour created an empty surface.")

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


def make_path_polyline(path_xyz):
    line = pv.PolyData(path_xyz)
    line.lines = np.hstack([[len(path_xyz)], np.arange(len(path_xyz))])
    return line


def save_obj(polydata, output_path):
    faces = polydata.faces.reshape((-1, 4))[:, 1:]
    points = polydata.points

    with open(output_path, "w") as f:
        f.write("# Airway lumen mesh\n")
        for x, y, z in points:
            f.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
        for a, b, c in faces:
            f.write(f"f {int(a) + 1} {int(b) + 1} {int(c) + 1}\n")


def try_start_xvfb(enabled=True):
    if not enabled:
        return

    try:
        pv.start_xvfb()
    except Exception as exc:
        print("xvfb not started:", exc)


def make_scene(
    surface,
    path_xyz,
    tube_radius=1.5,
    endpoint_point_size=16,
    window_size=(1400, 1000),
):
    path_line = make_path_polyline(path_xyz)
    path_tube = path_line.tube(radius=tube_radius)

    plotter = pv.Plotter(off_screen=True, window_size=window_size)
    plotter.set_background("white")
    plotter.add_mesh(surface, color="#8ecae6", opacity=0.22, smooth_shading=True)
    plotter.add_mesh(path_tube, color="red", smooth_shading=True)
    if endpoint_point_size > 0:
        plotter.add_points(
            path_xyz[0:1],
            color="green",
            point_size=endpoint_point_size,
            render_points_as_spheres=True,
        )
        plotter.add_points(
            path_xyz[-1:],
            color="black",
            point_size=endpoint_point_size,
            render_points_as_spheres=True,
        )
    plotter.camera_position = "iso"
    return plotter


def render_overlay(
    surface,
    path_xyz,
    output_png,
    tube_radius=1.5,
    endpoint_point_size=16,
    window_size=(1400, 1000),
):
    plotter = make_scene(
        surface,
        path_xyz,
        tube_radius=tube_radius,
        endpoint_point_size=endpoint_point_size,
        window_size=window_size,
    )
    plotter.screenshot(output_png)
    plotter.close()


def render_spin_gif(
    surface,
    path_xyz,
    output_gif,
    tube_radius=1.5,
    endpoint_point_size=16,
    frames=72,
    window_size=(900, 700),
):
    plotter = make_scene(
        surface,
        path_xyz,
        tube_radius=tube_radius,
        endpoint_point_size=endpoint_point_size,
        window_size=window_size,
    )
    plotter.open_gif(output_gif)

    for _ in range(frames):
        plotter.write_frame()
        plotter.camera.Azimuth(360.0 / frames)

    plotter.close()


def main():
    parser = argparse.ArgumentParser(
        description="Create an airway lumen mesh and overlay the planned centerline path using PyVista."
    )
    parser.add_argument("--mask", required=True, help="Predicted lumen mask/probability .npy")
    parser.add_argument("--paths-json", required=True, help="Path JSON from skeletonize_airrc_case.py")
    parser.add_argument("--output-dir", required=True, help="Folder for STL/OBJ/PNG outputs")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold if --mask is a probability volume")
    parser.add_argument("--path-id", default=None, help="Specific path_id to visualize; defaults to first path")
    parser.add_argument("--spacing", type=parse_tuple, default=(1.0, 1.0, 1.0), help="Voxel spacing as z,y,x")
    parser.add_argument("--mesh-keep-largest", action="store_true", help="Keep largest connected mask component for mesh")
    parser.add_argument("--mesh-fill-holes", action="store_true", help="Fill holes in mesh mask before surface extraction")
    parser.add_argument("--mesh-closing-radius", type=int, default=0, help="Binary closing iterations for mesh mask")
    parser.add_argument("--mesh-opening-radius", type=int, default=0, help="Binary opening iterations for mesh mask")
    parser.add_argument("--mesh-smooth-iterations", type=int, default=0, help="PyVista surface smoothing iterations")
    parser.add_argument("--mesh-smooth-relaxation", type=float, default=0.01, help="PyVista smoothing relaxation factor")
    parser.add_argument("--tube-radius", type=float, default=1.5, help="Radius of the rendered centerline path tube")
    parser.add_argument(
        "--endpoint-point-size",
        type=float,
        default=16.0,
        help="Rendered start/target marker size; use 0 to hide markers",
    )
    parser.add_argument("--make-video", action="store_true", help="Also save a rotating GIF validation video")
    parser.add_argument("--video-frames", type=int, default=72, help="Number of frames in the optional GIF")
    parser.add_argument("--no-xvfb", action="store_true", help="Do not try to start xvfb for headless rendering")
    args = parser.parse_args()

    try_start_xvfb(enabled=not args.no_xvfb)

    mask_path = Path(args.mask)
    paths_json = Path(args.paths_json)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mask = load_lumen_mask(mask_path, threshold=args.threshold)
    raw_mask_voxels = int(mask.sum())
    mask = postprocess_mesh_mask(
        mask,
        keep_largest=args.mesh_keep_largest,
        fill_holes=args.mesh_fill_holes,
        closing_radius=args.mesh_closing_radius,
        opening_radius=args.mesh_opening_radius,
    )
    path_record, path_zyx = load_path_coordinates(paths_json, path_id=args.path_id)
    path_xyz = zyx_to_xyz(path_zyx, spacing_zyx=args.spacing)
    surface = mask_to_pyvista_surface(mask, spacing_zyx=args.spacing)
    surface = smooth_surface(
        surface,
        iterations=args.mesh_smooth_iterations,
        relaxation_factor=args.mesh_smooth_relaxation,
    )

    stl_path = output_dir / "airway_lumen_mesh.stl"
    obj_path = output_dir / "airway_lumen_mesh.obj"
    overlay_png = output_dir / "airway_mesh_path_overlay.png"
    overlay_gif = output_dir / "airway_mesh_path_overlay.gif"
    summary_json = output_dir / "airway_mesh_path_summary.json"

    surface.save(stl_path)
    save_obj(surface, obj_path)
    render_overlay(
        surface,
        path_xyz,
        overlay_png,
        tube_radius=args.tube_radius,
        endpoint_point_size=args.endpoint_point_size,
    )
    if args.make_video:
        render_spin_gif(
            surface,
            path_xyz,
            overlay_gif,
            tube_radius=args.tube_radius,
            endpoint_point_size=args.endpoint_point_size,
            frames=args.video_frames,
        )

    summary = {
        "mask": str(mask_path),
        "paths_json": str(paths_json),
        "path_id": path_record.get("path_id"),
        "path_length_voxels": path_record.get("length_voxels"),
        "spacing_zyx": list(args.spacing),
        "mask_shape_zyx": list(mask.shape),
        "raw_mask_voxels": raw_mask_voxels,
        "mask_voxels": int(mask.sum()),
        "mesh_postprocess": {
            "keep_largest": bool(args.mesh_keep_largest),
            "fill_holes": bool(args.mesh_fill_holes),
            "closing_radius": int(args.mesh_closing_radius),
            "opening_radius": int(args.mesh_opening_radius),
            "smooth_iterations": int(args.mesh_smooth_iterations),
            "smooth_relaxation": float(args.mesh_smooth_relaxation),
        },
        "surface_points": int(surface.n_points),
        "surface_cells": int(surface.n_cells),
        "path_points": int(len(path_xyz)),
        "tube_radius": float(args.tube_radius),
        "endpoint_point_size": float(args.endpoint_point_size),
        "stl": str(stl_path),
        "obj": str(obj_path),
        "overlay_png": str(overlay_png),
        "overlay_gif": str(overlay_gif) if args.make_video else None,
    }

    with open(summary_json, "w") as f:
        json.dump(summary, f, indent=2)

    print("Done")
    print("  mask:", mask_path)
    print("  paths json:", paths_json)
    print("  path id:", path_record.get("path_id"))
    print("  mask shape:", mask.shape)
    print("  raw mask voxels:", raw_mask_voxels)
    print("  mask voxels:", int(mask.sum()))
    print("  mesh keep largest:", args.mesh_keep_largest)
    print("  mesh fill holes:", args.mesh_fill_holes)
    print("  mesh closing radius:", args.mesh_closing_radius)
    print("  mesh smooth iterations:", args.mesh_smooth_iterations)
    print("  surface points:", surface.n_points)
    print("  surface cells:", surface.n_cells)
    print("  path points:", len(path_xyz))
    print("  path length voxels:", path_record.get("length_voxels"))
    print("  STL:", stl_path)
    print("  OBJ:", obj_path)
    print("  overlay PNG:", overlay_png)
    if args.make_video:
        print("  overlay GIF:", overlay_gif)
    print("  summary JSON:", summary_json)


if __name__ == "__main__":
    main()
