from pathlib import Path
import argparse
import json

import numpy as np
import pyvista as pv
from scipy import ndimage as ndi


def parse_spacing(value):
    parts = tuple(float(item.strip()) for item in value.split(","))
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("Expected spacing as z,y,x")
    return parts


def load_3d_array(path):
    arr = np.load(path)
    if arr.ndim == 5:
        arr = arr[0, 0]
    elif arr.ndim == 4:
        arr = arr[0]
    elif arr.ndim != 3:
        raise ValueError(f"Expected a 3D, 4D, or 5D array, got shape {arr.shape}")
    return arr


def zyx_to_xyz(coords_zyx, spacing_zyx):
    coords = np.asarray(coords_zyx, dtype=np.float32)
    spacing = np.asarray(spacing_zyx, dtype=np.float32)
    scaled = coords * spacing
    return scaled[:, [2, 1, 0]]


def skeleton_points_from_npy(skeleton_path, spacing_zyx):
    skeleton = load_3d_array(skeleton_path) > 0
    coords_zyx = np.argwhere(skeleton)
    if len(coords_zyx) == 0:
        raise ValueError("Skeleton file has zero voxels.")
    return coords_zyx, zyx_to_xyz(coords_zyx, spacing_zyx), skeleton.shape


def load_graph_edges(graph_json, spacing_zyx):
    with open(graph_json, "r") as f:
        payload = json.load(f)

    graph = payload.get("graph", payload)

    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    if not nodes or not edges:
        raise ValueError("Graph JSON must contain non-empty nodes and edges.")

    node_xyz = {}
    for node in nodes:
        node_id = str(node.get("id"))
        coord = node.get("coord_zyx") or node.get("zyx") or node.get("coordinate_zyx")
        if coord is None:
            raise ValueError("Graph node is missing coord_zyx/zyx/coordinate_zyx.")
        node_xyz[node_id] = zyx_to_xyz([coord], spacing_zyx)[0]

    edge_pairs = []
    for edge in edges:
        if isinstance(edge, dict):
            a = str(edge.get("source", edge.get("u", edge.get("a"))))
            b = str(edge.get("target", edge.get("v", edge.get("b"))))
        else:
            a, b = str(edge[0]), str(edge[1])
        if a in node_xyz and b in node_xyz:
            edge_pairs.append((a, b))

    if not edge_pairs:
        raise ValueError("No valid graph edges found after parsing.")

    points = []
    lines = []
    for a, b in edge_pairs:
        start_index = len(points)
        points.append(node_xyz[a])
        points.append(node_xyz[b])
        lines.extend([2, start_index, start_index + 1])

    polyline = pv.PolyData(np.asarray(points, dtype=np.float32))
    polyline.lines = np.asarray(lines, dtype=np.int64)
    return polyline


def make_skeleton_actor(points_xyz, graph_json, spacing_zyx, tube_radius, point_size):
    if graph_json:
        line = load_graph_edges(graph_json, spacing_zyx)
        return line.tube(radius=tube_radius), "mesh"

    cloud = pv.PolyData(points_xyz)
    return cloud, "points"


def maybe_make_lumen_surface(mask_path, threshold, spacing_zyx):
    if mask_path is None:
        return None

    arr = load_3d_array(mask_path)
    mask = arr > threshold
    if int(mask.sum()) == 0:
        raise ValueError("Context mask is empty after thresholding.")

    nz, ny, nx = mask.shape
    grid = pv.ImageData()
    grid.dimensions = (nx, ny, nz)
    grid.spacing = (spacing_zyx[2], spacing_zyx[1], spacing_zyx[0])
    grid.point_data["values"] = mask.astype(np.uint8).transpose(2, 1, 0).flatten(order="F")

    surface = grid.contour(isosurfaces=[0.5], scalars="values")
    if surface.n_points == 0:
        raise ValueError("Context mask surface is empty.")
    return surface.triangulate()


def keep_largest_component(mask):
    structure = ndi.generate_binary_structure(3, 2)
    labeled, count = ndi.label(mask, structure=structure)
    if count <= 1:
        return mask

    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    largest = int(np.argmax(sizes))
    return labeled == largest


def postprocess_mask(mask, keep_largest=False, fill_holes=False, closing_radius=0, opening_radius=0):
    mask = mask.astype(bool)

    if keep_largest:
        mask = keep_largest_component(mask)

    if fill_holes:
        mask = ndi.binary_fill_holes(mask)

    structure = ndi.generate_binary_structure(3, 1)
    for _ in range(max(0, int(closing_radius))):
        mask = ndi.binary_closing(mask, structure=structure)

    for _ in range(max(0, int(opening_radius))):
        mask = ndi.binary_opening(mask, structure=structure)

    return mask


def mask_to_surface(mask_path, threshold, spacing_zyx, keep_largest, fill_holes, closing_radius, opening_radius):
    arr = load_3d_array(mask_path)
    mask = arr > threshold
    if int(mask.sum()) == 0:
        raise ValueError("Mask is empty after thresholding.")

    raw_voxels = int(mask.sum())
    mask = postprocess_mask(
        mask,
        keep_largest=keep_largest,
        fill_holes=fill_holes,
        closing_radius=closing_radius,
        opening_radius=opening_radius,
    )

    nz, ny, nx = mask.shape
    grid = pv.ImageData()
    grid.dimensions = (nx, ny, nz)
    grid.spacing = (spacing_zyx[2], spacing_zyx[1], spacing_zyx[0])
    grid.point_data["values"] = mask.astype(np.uint8).transpose(2, 1, 0).flatten(order="F")

    surface = grid.contour(isosurfaces=[0.5], scalars="values")
    if surface.n_points == 0:
        raise ValueError("Surface extraction created an empty mesh.")

    return surface.triangulate(), raw_voxels, int(mask.sum()), mask.shape


def smooth_surface(surface, iterations=0, relaxation_factor=0.01):
    if iterations <= 0:
        return surface

    return surface.smooth(
        n_iter=iterations,
        relaxation_factor=relaxation_factor,
        feature_smoothing=False,
        boundary_smoothing=True,
    ).triangulate()


def try_start_xvfb(enabled):
    if not enabled:
        return
    try:
        pv.start_xvfb()
    except Exception as exc:
        print("xvfb not started:", exc)


def render_gif(
    skeleton_actor,
    actor_kind,
    output_gif,
    context_surface=None,
    frames=72,
    point_size=4,
    window_size=(900, 700),
):
    plotter = pv.Plotter(off_screen=True, window_size=window_size)
    plotter.set_background("white")

    if context_surface is not None:
        plotter.add_mesh(
            context_surface,
            color="#8ecae6",
            opacity=0.16,
            smooth_shading=True,
        )

    if actor_kind == "points":
        plotter.add_points(
            skeleton_actor,
            color="red",
            point_size=point_size,
            render_points_as_spheres=True,
        )
    else:
        plotter.add_mesh(skeleton_actor, color="red", smooth_shading=True)

    plotter.camera_position = "iso"
    plotter.open_gif(output_gif)
    for _ in range(frames):
        plotter.write_frame()
        plotter.camera.Azimuth(360.0 / frames)
    plotter.close()


def render_png(
    skeleton_actor,
    actor_kind,
    output_png,
    context_surface=None,
    point_size=4,
    window_size=(1200, 900),
):
    plotter = pv.Plotter(off_screen=True, window_size=window_size)
    plotter.set_background("white")

    if context_surface is not None:
        plotter.add_mesh(
            context_surface,
            color="#8ecae6",
            opacity=0.16,
            smooth_shading=True,
        )

    if actor_kind == "points":
        plotter.add_points(
            skeleton_actor,
            color="red",
            point_size=point_size,
            render_points_as_spheres=True,
        )
    else:
        plotter.add_mesh(skeleton_actor, color="red", smooth_shading=True)

    plotter.camera_position = "iso"
    plotter.screenshot(output_png)
    plotter.close()


def render_mesh_gif(surface, output_gif, frames=72, window_size=(900, 700)):
    plotter = pv.Plotter(off_screen=True, window_size=window_size)
    plotter.set_background("white")
    plotter.add_mesh(surface, color="#8ecae6", opacity=0.22, smooth_shading=True)
    plotter.camera_position = "iso"
    plotter.open_gif(output_gif)
    for _ in range(frames):
        plotter.write_frame()
        plotter.camera.Azimuth(360.0 / frames)
    plotter.close()


def render_mesh_png(surface, output_png, window_size=(1200, 900)):
    plotter = pv.Plotter(off_screen=True, window_size=window_size)
    plotter.set_background("white")
    plotter.add_mesh(surface, color="#8ecae6", opacity=0.22, smooth_shading=True)
    plotter.camera_position = "iso"
    plotter.screenshot(output_png)
    plotter.close()


def load_mesh_file(mesh_file, smooth_iterations=0, smooth_relaxation=0.01):
    mesh = pv.read(mesh_file)
    if mesh.n_points == 0 or mesh.n_cells == 0:
        raise ValueError(f"Mesh file is empty: {mesh_file}")

    surface = mesh.extract_surface().triangulate()
    return smooth_surface(
        surface,
        iterations=smooth_iterations,
        relaxation_factor=smooth_relaxation,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Render a presentation GIF: hollow airway mesh only, or skeleton-only if --skeleton is used."
    )
    parser.add_argument("--mesh-file", default=None, help="Existing STL/OBJ mesh file to render directly.")
    parser.add_argument("--mask", default=None, help="Predicted lumen mask/probability .npy for hollow airway mesh render.")
    parser.add_argument("--skeleton", default=None, help="Skeleton .npy file for centerline render.")
    parser.add_argument("--output-dir", required=True, help="Folder for PNG/GIF output.")
    parser.add_argument(
        "--graph-json",
        default=None,
        help="Optional centerline graph JSON. If provided, renders connected tubes instead of voxel points.",
    )
    parser.add_argument(
        "--context-mask",
        default=None,
        help="Optional lumen probability/mask .npy to render faint mesh context.",
    )
    parser.add_argument("--threshold", type=float, default=0.2, help="Threshold for context mask.")
    parser.add_argument("--spacing", type=parse_spacing, default=(1.0, 1.0, 1.0), help="Voxel spacing as z,y,x.")
    parser.add_argument("--mesh-keep-largest", action="store_true", help="Keep largest connected mask component for mesh render.")
    parser.add_argument("--mesh-fill-holes", action="store_true", help="Fill holes in mask before mesh render.")
    parser.add_argument("--mesh-closing-radius", type=int, default=0, help="Binary closing iterations for mesh mask.")
    parser.add_argument("--mesh-opening-radius", type=int, default=0, help="Binary opening iterations for mesh mask.")
    parser.add_argument("--mesh-smooth-iterations", type=int, default=0, help="PyVista mesh smoothing iterations.")
    parser.add_argument("--mesh-smooth-relaxation", type=float, default=0.01, help="PyVista mesh smoothing relaxation factor.")
    parser.add_argument("--tube-radius", type=float, default=0.45, help="Tube radius when graph JSON is used.")
    parser.add_argument("--point-size", type=float, default=3.0, help="Point size when graph JSON is not used.")
    parser.add_argument("--frames", type=int, default=72, help="Number of GIF frames.")
    parser.add_argument("--no-xvfb", action="store_true", help="Do not try to start Xvfb.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.mesh_file is None and args.mask is None and args.skeleton is None:
        raise ValueError("Provide --mesh-file, --mask, or --skeleton.")

    try_start_xvfb(enabled=not args.no_xvfb)

    if args.mesh_file is not None:
        surface = load_mesh_file(
            args.mesh_file,
            smooth_iterations=args.mesh_smooth_iterations,
            smooth_relaxation=args.mesh_smooth_relaxation,
        )

        output_png = output_dir / "mesh_file_preview.png"
        output_gif = output_dir / "mesh_file_preview.gif"

        render_mesh_png(surface, output_png)
        render_mesh_gif(surface, output_gif, frames=args.frames)

        print("Done")
        print(f"  mesh file: {args.mesh_file}")
        print(f"  surface points: {surface.n_points}")
        print(f"  surface cells: {surface.n_cells}")
        print(f"  PNG: {output_png}")
        print(f"  GIF: {output_gif}")
        return

    if args.mask is not None:
        surface, raw_voxels, mask_voxels, shape = mask_to_surface(
            mask_path=args.mask,
            threshold=args.threshold,
            spacing_zyx=args.spacing,
            keep_largest=args.mesh_keep_largest,
            fill_holes=args.mesh_fill_holes,
            closing_radius=args.mesh_closing_radius,
            opening_radius=args.mesh_opening_radius,
        )
        surface = smooth_surface(
            surface,
            iterations=args.mesh_smooth_iterations,
            relaxation_factor=args.mesh_smooth_relaxation,
        )

        output_png = output_dir / "airway_mesh_only.png"
        output_gif = output_dir / "airway_mesh_only.gif"

        render_mesh_png(surface, output_png)
        render_mesh_gif(surface, output_gif, frames=args.frames)

        print("Done")
        print(f"  mask: {args.mask}")
        print(f"  mask shape: {shape}")
        print(f"  raw mask voxels: {raw_voxels}")
        print(f"  mask voxels: {mask_voxels}")
        print(f"  surface points: {surface.n_points}")
        print(f"  surface cells: {surface.n_cells}")
        print(f"  PNG: {output_png}")
        print(f"  GIF: {output_gif}")
        return

    coords_zyx, points_xyz, shape = skeleton_points_from_npy(args.skeleton, args.spacing)
    skeleton_actor, actor_kind = make_skeleton_actor(
        points_xyz=points_xyz,
        graph_json=args.graph_json,
        spacing_zyx=args.spacing,
        tube_radius=args.tube_radius,
        point_size=args.point_size,
    )
    context_surface = maybe_make_lumen_surface(args.context_mask, args.threshold, args.spacing)

    output_png = output_dir / "skeleton_only.png"
    output_gif = output_dir / "skeleton_only.gif"

    render_png(
        skeleton_actor=skeleton_actor,
        actor_kind=actor_kind,
        output_png=output_png,
        context_surface=context_surface,
        point_size=args.point_size,
    )
    render_gif(
        skeleton_actor=skeleton_actor,
        actor_kind=actor_kind,
        output_gif=output_gif,
        context_surface=context_surface,
        frames=args.frames,
        point_size=args.point_size,
    )

    print("Done")
    print(f"  skeleton: {args.skeleton}")
    print(f"  graph json: {args.graph_json}")
    print(f"  context mask: {args.context_mask}")
    print(f"  skeleton shape: {shape}")
    print(f"  skeleton voxels: {len(coords_zyx)}")
    print(f"  PNG: {output_png}")
    print(f"  GIF: {output_gif}")


if __name__ == "__main__":
    main()
