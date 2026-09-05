"""Run the complete LIDC case evaluation workflow with one command."""

from pathlib import Path
import argparse
import json
import os
import subprocess
import sys


def parse_spacing(value):
    values = tuple(float(part.strip()) for part in value.split(","))
    if len(values) != 3:
        raise argparse.ArgumentTypeError("Expected format: z,y,x")
    return values


def format_triplet(values):
    return ",".join(str(value) for value in values)


def threshold_suffix(threshold):
    text = f"{threshold:.3f}".rstrip("0").rstrip(".")
    return "thr" + text.replace(".", "")


def run_step(name, command, dry_run=False):
    print()
    print("=" * 80)
    print(name)
    print("=" * 80)
    print(" ".join(str(part) for part in command))
    sys.stdout.flush()
    if not dry_run:
        subprocess.run(command, check=True)


def find_dicom_directory(root, series_uid):
    try:
        import SimpleITK as sitk
    except ImportError as exc:
        raise RuntimeError("SimpleITK is required to locate the requested DICOM series") from exc

    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"DICOM root does not exist: {root}")

    candidates = [root]
    candidates.extend(path for path in root.rglob("*") if path.is_dir())
    for directory in candidates:
        try:
            series_ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(directory)) or []
        except RuntimeError:
            continue
        if series_uid in series_ids:
            return directory

    raise FileNotFoundError(
        f"Could not find DICOM SeriesInstanceUID {series_uid} below {root}"
    )


def find_matching_xml(xml_root, series_uid):
    from extract_lidc_poi_target import extract_xml_series_uids

    xml_root = Path(xml_root).expanduser().resolve()
    if not xml_root.is_dir():
        raise FileNotFoundError(f"XML root does not exist: {xml_root}")

    matches = []
    for path in xml_root.rglob("*.xml"):
        try:
            if series_uid in extract_xml_series_uids(path):
                matches.append(path)
        except Exception:
            continue

    if not matches:
        raise FileNotFoundError(
            f"Could not find an LIDC XML file for SeriesInstanceUID {series_uid} below {xml_root}"
        )
    if len(matches) > 1:
        print(f"[WARN] Found {len(matches)} matching XML files; using {matches[0]}")
    return matches[0]


def load_best_target(audit_json):
    with open(audit_json, "r") as handle:
        payload = json.load(handle)
    ranked = payload.get("ranked_targets", [])
    if not ranked:
        raise RuntimeError(f"No usable LIDC targets were written to {audit_json}")
    return ranked[0]


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Preprocess one LIDC series, infer the airway once, rank all nodule POIs, "
            "plan to the best-supported POI, smooth the route, render it, and audit the result."
        )
    )
    parser.add_argument("--case-id", required=True, help="LIDC patient id, for example LIDC-IDRI-0011")
    parser.add_argument("--series-uid", required=True, help="CT SeriesInstanceUID downloaded with the IDC CLI")
    parser.add_argument(
        "--dicom-root",
        required=True,
        help="Downloaded case/series folder; nested directories are searched automatically",
    )
    xml_group = parser.add_mutually_exclusive_group(required=True)
    xml_group.add_argument("--xml", help="Exact LIDC annotation XML file")
    xml_group.add_argument("--xml-root", help="LIDC XML collection root to search by SeriesInstanceUID")
    parser.add_argument(
        "--data-root",
        default=os.environ.get("AVB_DATA_ROOT", str(Path.home() / "AMS_Project" / "datasets_new")),
    )
    parser.add_argument("--checkpoint", default="saved_model_topology/wingsnet_best.pth")
    parser.add_argument("--run-id", default=None, help="Output id; defaults to <case-id>_topology_auto")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--target-spacing", type=parse_spacing, default=(1.0, 1.0, 1.0))
    parser.add_argument("--mask-threshold", type=float, default=0.2)
    parser.add_argument("--skeleton-threshold", type=float, default=0.2)
    parser.add_argument("--skeleton-low-threshold", type=float, default=0.08)
    parser.add_argument("--prune-length", type=float, default=12.0)
    parser.add_argument("--preserve-generations", type=int, default=8)
    parser.add_argument("--smooth-smoothing", type=float, default=25.0)
    parser.add_argument("--endpoint-point-size", type=float, default=0.0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--make-video", action="store_true", help="Create the rotating external validation GIF")
    parser.add_argument("--make-flythrough", action="store_true", help="Create the intraluminal flythrough GIF")
    parser.add_argument("--no-xvfb", action="store_true", help="Do not start Xvfb for PyVista rendering")
    parser.add_argument("--flythrough-frames", type=int, default=720)
    parser.add_argument("--flythrough-fps", type=int, default=24)
    parser.add_argument(
        "--reuse-prediction",
        action="store_true",
        help="Reuse an existing prediction for this run id instead of running WingsNet again",
    )
    parser.add_argument("--force-preprocess", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    python = sys.executable
    data_root = Path(args.data_root).expanduser().resolve()
    run_id = args.run_id or f"{args.case_id}_topology_auto"
    spacing_text = format_triplet(args.target_spacing)
    centerline_id = f"{run_id}_{threshold_suffix(args.skeleton_threshold)}"

    if args.dry_run:
        dicom_dir = Path(args.dicom_root).expanduser()
        xml_path = Path(args.xml or args.xml_root).expanduser()
    else:
        print("Locating DICOM series", args.series_uid)
        dicom_dir = find_dicom_directory(args.dicom_root, args.series_uid)
        xml_path = Path(args.xml).expanduser().resolve() if args.xml else find_matching_xml(args.xml_root, args.series_uid)
        print("  DICOM directory:", dicom_dir)
        print("  annotation XML:", xml_path)

    processed_root = data_root / "processed_lidc"
    ct_path = processed_root / "images" / f"{args.case_id}_ct.npy"
    metadata_path = processed_root / "metadata" / f"{args.case_id}_meta.json"
    poi_json = processed_root / "metadata" / f"{args.case_id}_poi_targets.json"
    pred_dir = data_root / "predictions" / run_id
    centerline_dir = data_root / "centerlines" / centerline_id
    visualization_dir = data_root / "visualizations" / centerline_id
    pred_lumen = pred_dir / "pred_lumen.npy"
    graph_json = centerline_dir / "pred_lumen_centerline_pruned_graph.json"
    paths_json = centerline_dir / "pred_lumen_paths.json"
    smoothed_paths_json = centerline_dir / "pred_lumen_smoothed_paths.json"
    target_audit_json = centerline_dir / "selected_target_audit.json"
    all_target_audit_json = centerline_dir / "all_poi_target_audit.json"
    all_target_audit_csv = centerline_dir / "all_poi_target_audit.csv"

    if args.force_preprocess or not (ct_path.is_file() and metadata_path.is_file()):
        preprocess_cmd = [
            python,
            str(script_dir / "preprocess_lidc_for_inference.py"),
            "--dicom-dir", str(dicom_dir),
            "--series-uid", args.series_uid,
            "--case-id", args.case_id,
            "--output-root", str(processed_root),
            "--target-spacing", spacing_text,
        ]
        run_step("1. Preprocess the selected LIDC CT series", preprocess_cmd, args.dry_run)
    else:
        print("Reusing processed CT:", ct_path)

    extract_cmd = [
        python,
        str(script_dir / "extract_lidc_poi_target.py"),
        "--xml", str(xml_path),
        "--dicom-dir", str(dicom_dir),
        "--metadata", str(metadata_path),
        "--output-json", str(poi_json),
        "--series-uid", args.series_uid,
    ]
    run_step("2. Extract all LIDC nodule POIs", extract_cmd, args.dry_run)

    common_pipeline = [
        python,
        str(script_dir / "run_airway_pipeline.py"),
        "--ct", str(ct_path),
        "--case-id", run_id,
        "--data-root", str(data_root),
        "--checkpoint", args.checkpoint,
        "--model-module", "WingsNet",
        "--model-class", "WingsNet",
        "--model-kwargs", '{"in_channel": 1, "n_classes": 2}',
        "--device", args.device,
        "--mask-threshold", str(args.mask_threshold),
        "--skeleton-threshold", str(args.skeleton_threshold),
        "--skeleton-low-threshold", str(args.skeleton_low_threshold),
        "--prune-length", str(args.prune_length),
        "--preserve-generations", str(args.preserve_generations),
        "--root-mode", "timi-trachea",
        "--spacing", spacing_text,
        "--smooth-smoothing", str(args.smooth_smoothing),
        "--endpoint-point-size", str(args.endpoint_point_size),
    ]
    if args.amp:
        common_pipeline.append("--amp")
    if args.make_video:
        common_pipeline.append("--make-video")
    if args.no_xvfb:
        common_pipeline.append("--no-xvfb")
    if args.dry_run:
        common_pipeline.append("--dry-run")

    discovery_cmd = list(common_pipeline)
    discovery_cmd.extend(["--target-mode", "farthest-endpoint", "--stop-after", "skeleton"])
    if args.reuse_prediction:
        discovery_cmd.append("--skip-inference")
    run_step("3. Infer the airway and build a target-independent discovery graph", discovery_cmd, args.dry_run)

    audit_all_cmd = [
        python,
        str(script_dir / "audit_lidc_poi_targets.py"),
        "--poi-json", str(poi_json),
        "--pred-lumen", str(pred_lumen),
        "--graph-json", str(graph_json),
        "--output-json", str(all_target_audit_json),
        "--output-csv", str(all_target_audit_csv),
    ]
    run_step("4. Rank every LIDC POI by airway support and graph reachability", audit_all_cmd, args.dry_run)

    if args.dry_run:
        selected = {
            "target_zyx": ["BEST_Z", "BEST_Y", "BEST_X"],
            "support_score": None,
            "nearest_graph_distance_voxels": None,
        }
    else:
        selected = load_best_target(all_target_audit_json)
    target_text = format_triplet(selected["target_zyx"])
    print("Selected target:", target_text)
    print("  support score:", selected.get("support_score"))
    print("  initial graph distance:", selected.get("nearest_graph_distance_voxels"))

    final_pipeline_cmd = list(common_pipeline)
    final_pipeline_cmd.extend(["--skip-inference", "--target-zyx", target_text])
    run_step("5. Replan and render using the automatically selected POI", final_pipeline_cmd, args.dry_run)

    audit_selected_cmd = [
        python,
        str(script_dir / "audit_airway_target.py"),
        "--pred-lumen", str(pred_lumen),
        "--graph-json", str(graph_json),
        "--paths-json", str(paths_json),
        "--target-zyx", target_text,
        "--output-json", str(target_audit_json),
    ]
    run_step("6. Audit the selected target and final planned route", audit_selected_cmd, args.dry_run)

    final_target_audit = None
    if not args.dry_run:
        with open(target_audit_json, "r") as handle:
            final_target_audit = json.load(handle)

    flythrough_path = visualization_dir / "airway_flythrough_bronchoscopy.gif"
    if args.make_flythrough:
        flythrough_cmd = [
            python,
            str(script_dir / "flythrough_airway_path.py"),
            "--mesh", str(visualization_dir / "airway_lumen_mesh.stl"),
            "--paths-json", str(smoothed_paths_json),
            "--output", str(flythrough_path),
            "--spacing", spacing_text,
            "--frames", str(args.flythrough_frames),
            "--fps", str(args.flythrough_fps),
            "--lookahead", "14",
            "--window-size", "900,700",
            "--mesh-color", "#c98270",
            "--mucosa-texture",
            "--texture-strength", "0.85",
            "--mesh-smooth-iterations", "4",
            "--mesh-smooth-relaxation", "0.005",
            "--view-angle", "92",
            "--ambient", "0.38",
            "--diffuse", "0.75",
            "--specular", "0.18",
            "--headlight-intensity", "0.28",
        ]
        if args.no_xvfb:
            flythrough_cmd.append("--no-xvfb")
        run_step("7. Render the intraluminal flythrough", flythrough_cmd, args.dry_run)

    summary_path = visualization_dir / "lidc_case_run_summary.json"
    summary = {
        "case_id": args.case_id,
        "run_id": run_id,
        "series_uid": args.series_uid,
        "dicom_dir": str(dicom_dir),
        "xml": str(xml_path),
        "checkpoint": args.checkpoint,
        "selected_target": selected,
        "final_target_audit": final_target_audit,
        "outputs": {
            "processed_ct": str(ct_path),
            "metadata": str(metadata_path),
            "poi_targets": str(poi_json),
            "predictions": str(pred_dir),
            "centerlines": str(centerline_dir),
            "visualizations": str(visualization_dir),
            "all_target_audit_json": str(all_target_audit_json),
            "all_target_audit_csv": str(all_target_audit_csv),
            "selected_target_audit": str(target_audit_json),
            "flythrough": str(flythrough_path) if args.make_flythrough else None,
        },
    }
    if args.dry_run:
        print("Dry run complete; summary would be written to", summary_path)
    else:
        visualization_dir.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
        print()
        print("LIDC case workflow complete")
        print("  selected target:", target_text)
        print("  predictions:", pred_dir)
        print("  centerlines:", centerline_dir)
        print("  visualizations:", visualization_dir)
        print("  run summary:", summary_path)


if __name__ == "__main__":
    main()
