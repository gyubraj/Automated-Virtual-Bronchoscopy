from pathlib import Path
import argparse
import json

import numpy as np
import pyvista as pv


def hex_to_rgb01(value):
    text = value.strip().lstrip("#")
    if len(text) != 6:
        raise argparse.ArgumentTypeError("Expected color as #RRGGBB")
    return np.array([int(text[i : i + 2], 16) for i in (0, 2, 4)], dtype=np.float64) / 255.0


def parse_spacing(value):
    parts = tuple(float(part.strip()) for part in value.split(","))
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("Expected format: z,y,x")
    return parts


def load_path_zyx(paths_json, path_id=None):
    with open(paths_json, "r") as f:
        payload = json.load(f)

    paths = payload.get("paths", [])
    if not paths:
        raise ValueError("No paths found in paths JSON.")

    if path_id is None:
        path = paths[0]
    else:
        matches = [item for item in paths if item.get("path_id") == path_id]
        if not matches:
            raise ValueError(f"Path id not found: {path_id}")
        path = matches[0]

    coords = np.asarray(path.get("coordinates_zyx", []), dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 3 or len(coords) < 2:
        raise ValueError("Selected path must contain at least two coordinates_zyx points.")

    return path, coords


def zyx_to_xyz(coords_zyx, spacing_zyx):
    spacing = np.asarray(spacing_zyx, dtype=np.float64)
    scaled = coords_zyx * spacing[None, :]
    return scaled[:, [2, 1, 0]]


def resample_polyline(points, count):
    deltas = np.diff(points, axis=0)
    lengths = np.linalg.norm(deltas, axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(lengths)])
    total = cumulative[-1]
    if total <= 0:
        raise ValueError("Path has zero length.")

    samples = np.linspace(0.0, total, count)
    resampled = np.empty((count, 3), dtype=np.float64)
    for axis in range(3):
        resampled[:, axis] = np.interp(samples, cumulative, points[:, axis])
    return resampled


def normalize(vector, fallback):
    norm = float(np.linalg.norm(vector))
    if norm < 1e-8:
        return np.asarray(fallback, dtype=np.float64)
    return np.asarray(vector, dtype=np.float64) / norm


def camera_frames(path_xyz, frame_count, lookahead):
    positions = resample_polyline(path_xyz, frame_count + lookahead + 1)
    frames = []
    up = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    for idx in range(frame_count):
        position = positions[idx]
        focal = positions[min(idx + lookahead, len(positions) - 1)]
        tangent = normalize(focal - position, [1.0, 0.0, 0.0])

        # Keep the camera roll stable by carrying forward an up vector and
        # projecting it onto the plane perpendicular to the current tangent.
        up = up - np.dot(up, tangent) * tangent
        up = normalize(up, [0.0, 0.0, 1.0])
        if abs(float(np.dot(up, tangent))) > 0.95:
            up = normalize(np.cross(tangent, [0.0, 1.0, 0.0]), [0.0, 0.0, 1.0])

        frames.append((position, focal, up.copy()))

    return frames


def add_mucosa_texture(mesh, base_color, texture_strength, seed):
    if texture_strength <= 0:
        return None

    points = np.asarray(mesh.points, dtype=np.float64)
    if len(points) == 0:
        return None

    center = points.mean(axis=0)
    centered = points - center[None, :]
    radius = np.linalg.norm(centered[:, :2], axis=1)
    z = centered[:, 2]

    rng = np.random.default_rng(seed)
    phase = rng.uniform(0, 2 * np.pi, size=4)
    pattern = (
        0.45 * np.sin(0.22 * radius + 0.10 * z + phase[0])
        + 0.28 * np.sin(0.48 * points[:, 0] + 0.19 * points[:, 1] + phase[1])
        + 0.18 * np.sin(0.37 * points[:, 1] - 0.21 * points[:, 2] + phase[2])
        + 0.09 * rng.normal(size=len(points))
    )
    pattern = (pattern - pattern.min()) / max(float(pattern.max() - pattern.min()), 1e-8)

    base = hex_to_rgb01(base_color)
    pale = np.array([1.0, 0.68, 0.58], dtype=np.float64)
    dark = np.array([0.46, 0.18, 0.13], dtype=np.float64)
    wet = np.array([1.0, 0.82, 0.74], dtype=np.float64)

    colors = (
        base[None, :] * (1.0 - 0.35 * texture_strength)
        + pale[None, :] * (0.20 * texture_strength * pattern[:, None])
        + dark[None, :] * (0.30 * texture_strength * (1.0 - pattern)[:, None])
        + wet[None, :] * (0.10 * texture_strength * (pattern > 0.78)[:, None])
    )
    colors = np.clip(colors, 0.0, 1.0)
    mesh.point_data["mucosa_rgb"] = (colors * 255).astype(np.uint8)
    return "mucosa_rgb"


def main():
    parser = argparse.ArgumentParser(
        description="Render a virtual bronchoscopy flythrough along a smoothed airway centerline path."
    )
    parser.add_argument("--mesh", required=True, help="Clean airway lumen mesh STL/OBJ")
    parser.add_argument("--paths-json", required=True, help="Smoothed paths JSON")
    parser.add_argument("--output", required=True, help="Output GIF path")
    parser.add_argument("--path-id", default=None, help="Specific path_id; defaults to first path")
    parser.add_argument("--spacing", type=parse_spacing, default=(1.0, 1.0, 1.0), help="Voxel spacing as z,y,x")
    parser.add_argument("--frames", type=int, default=240, help="Number of GIF frames")
    parser.add_argument("--fps", type=int, default=24, help="Output GIF frames per second")
    parser.add_argument("--lookahead", type=int, default=8, help="Number of resampled points to look ahead")
    parser.add_argument("--window-size", default="900,700", help="Render size as width,height")
    parser.add_argument("--camera-offset", type=float, default=0.0, help="Small backward offset from centerline, in mesh units")
    parser.add_argument("--mesh-opacity", type=float, default=1.0, help="Airway mesh opacity")
    parser.add_argument("--mesh-color", default="#f2c6b6", help="Airway inner surface color")
    parser.add_argument("--mucosa-texture", action="store_true", help="Use procedural mucosa-like color variation")
    parser.add_argument("--texture-strength", type=float, default=0.75, help="Strength of procedural mucosa color variation")
    parser.add_argument("--texture-seed", type=int, default=13, help="Seed for procedural texture")
    parser.add_argument("--mesh-smooth-iterations", type=int, default=0, help="Render-time mesh smoothing iterations")
    parser.add_argument("--mesh-smooth-relaxation", type=float, default=0.01, help="Render-time mesh smoothing relaxation")
    parser.add_argument("--view-angle", type=float, default=82.0, help="Camera field of view angle")
    parser.add_argument("--ambient", type=float, default=0.45, help="Ambient material lighting")
    parser.add_argument("--diffuse", type=float, default=0.65, help="Diffuse material lighting")
    parser.add_argument("--specular", type=float, default=0.12, help="Specular material lighting")
    parser.add_argument("--headlight-intensity", type=float, default=0.45, help="Camera headlight intensity")
    parser.add_argument("--background", default="#050505", help="Background color")
    parser.add_argument("--no-xvfb", action="store_true", help="Do not start xvfb before rendering")
    args = parser.parse_args()

    width, height = (int(part.strip()) for part in args.window_size.split(","))
    if width <= 0 or height <= 0:
        raise ValueError("--window-size must be positive")

    if not args.no_xvfb:
        try:
            pv.start_xvfb()
        except Exception:
            pass

    path_record, path_zyx = load_path_zyx(args.paths_json, path_id=args.path_id)
    path_xyz = zyx_to_xyz(path_zyx, args.spacing)
    frames = camera_frames(path_xyz, args.frames, args.lookahead)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    mesh = pv.read(args.mesh)
    if args.mesh_smooth_iterations > 0:
        mesh = mesh.smooth(
            n_iter=args.mesh_smooth_iterations,
            relaxation_factor=args.mesh_smooth_relaxation,
            boundary_smoothing=False,
            feature_smoothing=True,
        )
    try:
        mesh = mesh.compute_normals(auto_orient_normals=True, consistent_normals=True)
    except Exception:
        pass
    texture_scalars = add_mucosa_texture(
        mesh,
        base_color=args.mesh_color,
        texture_strength=args.texture_strength if args.mucosa_texture else 0.0,
        seed=args.texture_seed,
    )

    plotter = pv.Plotter(off_screen=True, window_size=(width, height))
    plotter.set_background(args.background)
    mesh_kwargs = {
        "opacity": args.mesh_opacity,
        "smooth_shading": True,
        "ambient": args.ambient,
        "diffuse": args.diffuse,
        "specular": args.specular,
        "specular_power": 12,
        "backface_params": {"color": args.mesh_color},
    }
    if texture_scalars is None:
        plotter.add_mesh(mesh, color=args.mesh_color, **mesh_kwargs)
    else:
        plotter.add_mesh(mesh, scalars=texture_scalars, rgb=True, **mesh_kwargs)
    plotter.add_light(pv.Light(light_type="headlight", intensity=args.headlight_intensity))
    plotter.camera.view_angle = args.view_angle
    plotter.camera.clipping_range = (0.01, 10000.0)
    plotter.open_gif(str(output), fps=args.fps)

    for position, focal, up in frames:
        tangent = normalize(focal - position, [1.0, 0.0, 0.0])
        camera_position = position - tangent * args.camera_offset
        plotter.camera_position = (camera_position.tolist(), focal.tolist(), up.tolist())
        plotter.write_frame()

    plotter.close()

    summary = {
        "mesh": str(args.mesh),
        "paths_json": str(args.paths_json),
        "path_id": path_record.get("path_id"),
        "output": str(output),
        "frames": int(args.frames),
        "fps": int(args.fps),
        "duration_seconds": float(args.frames / args.fps),
        "lookahead": int(args.lookahead),
        "spacing_zyx": list(args.spacing),
        "path_points": int(len(path_xyz)),
        "start_zyx": path_zyx[0].tolist(),
        "end_zyx": path_zyx[-1].tolist(),
        "mesh_smooth_iterations": int(args.mesh_smooth_iterations),
        "mucosa_texture": bool(args.mucosa_texture),
        "texture_strength": float(args.texture_strength),
        "view_angle": float(args.view_angle),
    }
    summary_path = output.with_suffix(".json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("Flythrough complete")
    print("  path id:", path_record.get("path_id"))
    print("  start zyx:", summary["start_zyx"])
    print("  end zyx:", summary["end_zyx"])
    print("  GIF:", output)
    print("  summary:", summary_path)


if __name__ == "__main__":
    main()
