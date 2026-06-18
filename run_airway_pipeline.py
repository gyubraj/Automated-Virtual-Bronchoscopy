from pathlib import Path
import argparse
import subprocess
import sys


def parse_bool_flag(command, enabled, flag):
    if enabled:
        command.append(flag)


def case_id_from_ct(ct_path):
    path = Path(ct_path)
    name = path.name

    for suffix in ("_ct.npy", ".nii.gz", ".nii", ".npy"):
        if name.endswith(suffix):
            return name[: -len(suffix)]

    return path.name


def threshold_suffix(threshold):
    text = f"{threshold:.3f}".rstrip("0").rstrip(".")
    return "thr" + text.replace(".", "")


def run_step(name, command, dry_run=False):
    print()
    print("=" * 80)
    print(name)
    print("=" * 80)
    print(" ".join(str(part) for part in command))

    if dry_run:
        return

    subprocess.run(command, check=True)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run CT-to-virtual-bronchoscopy outputs in one command: "
            "inference, masks, centerline graph, planned path, mesh, and validation render."
        )
    )
    parser.add_argument("--ct", required=True, help="Full CT input: processed .npy, NIfTI, or DICOM folder")
    parser.add_argument("--case-id", default=None, help="Case id for output folders; inferred from --ct if omitted")
    parser.add_argument("--data-root", default="/home/opat90op/AMS_Project/datasets_new", help="Dataset/output root")
    parser.add_argument("--checkpoint", default="saved_model/wingsnet_best.pth", help="WingsNet checkpoint")
    parser.add_argument("--model-module", default="WingsNet", help="Python module containing the model class")
    parser.add_argument("--model-class", default="WingsNet", help="Model class name")
    parser.add_argument("--model-kwargs", default='{"in_channel": 1, "n_classes": 2}', help="JSON model kwargs")
    parser.add_argument("--patch-size", default="128,128,128", help="D,H,W inference patch size")
    parser.add_argument("--stride", default="64,64,64", help="D,H,W inference stride")
    parser.add_argument("--device", default="cuda", help="Inference device, e.g. cuda or cpu")
    parser.add_argument("--amp", action="store_true", help="Use CUDA mixed precision during inference")
    parser.add_argument("--no-normalize", action="store_true", help="Skip CT HU normalization")
    parser.add_argument("--save-nifti", action="store_true", help="Save NIfTI predictions when CT metadata is available")
    parser.add_argument("--mask-threshold", type=float, default=0.5, help="Threshold used for saved binary masks")
    parser.add_argument("--skeleton-threshold", type=float, default=0.2, help="Threshold used for centerline/path/mesh")
    parser.add_argument("--prune-length", type=float, default=8.0, help="Remove terminal centerline branches shorter than this")
    parser.add_argument(
        "--root-mode",
        choices=["min-z", "max-z", "center-min-z", "center-max-z", "largest-radius", "timi-trachea"],
        default="center-min-z",
        help="Root selection mode for path planning",
    )
    parser.add_argument(
        "--trachea-root-end",
        choices=["min-z", "max-z"],
        default="min-z",
        help="Which parsed trachea end to use when --root-mode timi-trachea",
    )
    parser.add_argument(
        "--target-mode",
        choices=["farthest-endpoint", "all-endpoints"],
        default="farthest-endpoint",
        help="Path target selection mode",
    )
    parser.add_argument("--spacing", default="1,1,1", help="Voxel spacing as z,y,x for mesh/path rendering")
    parser.add_argument("--tube-radius", type=float, default=1.5, help="Path tube radius in visualization")
    parser.add_argument("--make-video", action="store_true", help="Also create a rotating GIF validation video")
    parser.add_argument("--no-xvfb", action="store_true", help="Do not try to start xvfb for PyVista rendering")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    python = sys.executable

    case_id = args.case_id or case_id_from_ct(args.ct)
    data_root = Path(args.data_root)
    centerline_id = f"{case_id}_{threshold_suffix(args.skeleton_threshold)}"

    pred_dir = data_root / "predictions" / case_id
    centerline_dir = data_root / "centerlines" / centerline_id
    visualization_dir = data_root / "visualizations" / centerline_id

    inference_cmd = [
        python,
        str(script_dir / "infer_wingsnet_full_volume.py"),
        "--checkpoint",
        args.checkpoint,
        "--ct",
        args.ct,
        "--output-dir",
        str(pred_dir),
        "--model-module",
        args.model_module,
        "--model-class",
        args.model_class,
        "--model-kwargs",
        args.model_kwargs,
        "--patch-size",
        args.patch_size,
        "--stride",
        args.stride,
        "--device",
        args.device,
        "--threshold",
        str(args.mask_threshold),
        "--save-binary",
    ]
    parse_bool_flag(inference_cmd, args.amp, "--amp")
    parse_bool_flag(inference_cmd, args.no_normalize, "--no-normalize")
    parse_bool_flag(inference_cmd, args.save_nifti, "--save-nifti")

    pred_lumen = pred_dir / "pred_lumen.npy"
    pred_wall = pred_dir / "pred_wall.npy"
    pred_lumen_mask = pred_dir / "pred_lumen_mask.npy"
    pred_wall_mask = pred_dir / "pred_wall_mask.npy"

    skeleton_cmd = [
        python,
        str(script_dir / "skeletonize_airrc_case.py"),
        "--input",
        str(pred_lumen),
        "--output-dir",
        str(centerline_dir),
        "--threshold",
        str(args.skeleton_threshold),
        "--keep-largest",
        "--prune-length",
        str(args.prune_length),
        "--root-mode",
        args.root_mode,
        "--trachea-root-end",
        args.trachea_root_end,
        "--target-mode",
        args.target_mode,
    ]

    paths_json = centerline_dir / "pred_lumen_paths.json"

    visualization_cmd = [
        python,
        str(script_dir / "visualizer_airway_mesh_path.py"),
        "--mask",
        str(pred_lumen),
        "--paths-json",
        str(paths_json),
        "--output-dir",
        str(visualization_dir),
        "--threshold",
        str(args.skeleton_threshold),
        "--spacing",
        args.spacing,
        "--tube-radius",
        str(args.tube_radius),
    ]
    parse_bool_flag(visualization_cmd, args.make_video, "--make-video")
    parse_bool_flag(visualization_cmd, args.no_xvfb, "--no-xvfb")

    print("Case id:", case_id)
    print("Prediction dir:", pred_dir)
    print("Centerline dir:", centerline_dir)
    print("Visualization dir:", visualization_dir)

    run_step("1. Full-volume WingsNet inference", inference_cmd, dry_run=args.dry_run)
    run_step("2. Skeletonize lumen and plan path", skeleton_cmd, dry_run=args.dry_run)
    run_step("3. Export mesh and validation render", visualization_cmd, dry_run=args.dry_run)

    print()
    print("Pipeline complete")
    print("  pred lumen:", pred_lumen)
    print("  pred wall:", pred_wall)
    print("  pred lumen mask:", pred_lumen_mask)
    print("  pred wall mask:", pred_wall_mask)
    print("  centerline graph:", centerline_dir / "pred_lumen_centerline_graph.json")
    print("  pruned centerline graph:", centerline_dir / "pred_lumen_centerline_pruned_graph.json")
    print("  planned paths:", paths_json)
    print("  mesh STL:", visualization_dir / "airway_lumen_mesh.stl")
    print("  mesh OBJ:", visualization_dir / "airway_lumen_mesh.obj")
    print("  validation image:", visualization_dir / "airway_mesh_path_overlay.png")
    if args.make_video:
        print("  validation video:", visualization_dir / "airway_mesh_path_overlay.gif")


if __name__ == "__main__":
    main()
