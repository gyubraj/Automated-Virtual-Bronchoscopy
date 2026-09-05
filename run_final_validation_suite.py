from pathlib import Path
import argparse
import csv
import json
import os
import subprocess
import sys


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def case_id_from_patch_record(record):
    if isinstance(record, str):
        name = Path(record).name
    elif isinstance(record, dict):
        for key in ("case_id", "uid", "series_uid"):
            if record.get(key):
                return str(record[key])
        for key in ("image", "image_path", "ct", "ct_path"):
            if record.get(key):
                name = Path(record[key]).name
                break
        else:
            return None
    else:
        return None

    for suffix in ("_ct.npy", "_target.npy", ".npy"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    parts = name.split("_patch_")
    return parts[0]


def load_cases(cases_file, split_json, split_summary_json, limit):
    cases = []

    if cases_file:
        with open(cases_file, "r") as f:
            cases = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    elif split_summary_json and Path(split_summary_json).exists():
        payload = load_json(split_summary_json)
        for key in ("val_uids", "val_cases", "validation_cases", "val_case_ids"):
            if key in payload:
                cases = [str(value) for value in payload[key]]
                break
    elif split_json:
        payload = load_json(split_json)
        records = payload.get("items", payload.get("patches", payload))
        seen = set()
        for record in records:
            case_id = case_id_from_patch_record(record)
            if case_id and case_id not in seen:
                seen.add(case_id)
                cases.append(case_id)

    if not cases:
        raise ValueError("No validation cases found. Pass --cases-file or a split JSON with case ids.")

    if limit is not None:
        cases = cases[:limit]

    return cases


def threshold_suffix(threshold):
    text = f"{threshold:.3f}".rstrip("0").rstrip(".")
    return "thr" + text.replace(".", "")


def run_command(label, command, dry_run=False):
    print()
    print("=" * 80, flush=True)
    print(label, flush=True)
    print("=" * 80, flush=True)
    print(" ".join(str(part) for part in command), flush=True)
    if dry_run:
        return
    subprocess.run(command, check=True)


def pick(metrics, path, default=None):
    current = metrics
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def summarize_case(case_id, validation_json, centerline_id):
    metrics = load_json(validation_json)
    first_path = metrics.get("paths", [{}])[0] if metrics.get("paths") else {}
    path_summary = metrics.get("path_inside_lumen_summary", {})
    gt_branch_recall = pick(metrics, ["ground_truth", "gt_branch_recall_by_generation"], {}) or {}
    recall_by_generation = gt_branch_recall.get("by_generation", {})

    def generation_recall(generation):
        return (recall_by_generation.get(str(generation)) or {}).get("branch_recall")

    def generation_branch_count(generation):
        return (recall_by_generation.get(str(generation)) or {}).get("gt_branch_count")

    row = {
        "case_id": case_id,
        "centerline_id": centerline_id,
        "lumen_components": pick(metrics, ["pred_lumen_components", "component_count"]),
        "largest_lumen_ratio": pick(metrics, ["pred_lumen_components", "largest_component_ratio"]),
        "skeleton_components": pick(metrics, ["pred_skeleton_components", "component_count"]),
        "graph_components": pick(metrics, ["graph_connectivity", "graph_component_count"]),
        "graph_nodes": pick(metrics, ["graph_connectivity", "node_count"]),
        "graph_edges": pick(metrics, ["graph_connectivity", "edge_count"]),
        "endpoints": pick(metrics, ["graph_degree", "endpoint_count"]),
        "branchpoints": pick(metrics, ["graph_degree", "branchpoint_count"]),
        "graph_length_voxels": metrics.get("graph_total_length_voxels"),
        "short_terminal_branches": pick(metrics, ["terminal_branches", "short_terminal_branch_count"]),
        "path_count": metrics.get("path_count"),
        "first_path_inside_lumen_ratio": first_path.get("inside_lumen_ratio"),
        "mean_path_inside_lumen_ratio": path_summary.get("mean_inside_lumen_ratio"),
        "min_path_inside_lumen_ratio": path_summary.get("min_inside_lumen_ratio"),
        "paths_below_095_inside_lumen": path_summary.get("paths_below_safety_threshold"),
        "timi_parsed_branches": pick(metrics, ["timi_tree_parse", "parsed_branch_count"]),
        "timi_max_generation": pick(metrics, ["timi_tree_parse", "max_generation"]),
        "timi_unreached_branches": pick(metrics, ["timi_tree_parse", "unreached_branch_count"]),
        "lumen_dice": pick(metrics, ["ground_truth", "lumen_overlap", "dice"]),
        "lumen_precision": pick(metrics, ["ground_truth", "lumen_overlap", "precision"]),
        "lumen_recall": pick(metrics, ["ground_truth", "lumen_overlap", "recall"]),
        "gt_symmetric_mean_distance": pick(metrics, ["ground_truth", "symmetric_mean_centerline_distance"]),
        "gt_symmetric_hausdorff_distance": pick(metrics, ["ground_truth", "symmetric_hausdorff_distance"]),
        "gt_length_ratio_pred_over_gt": pick(metrics, ["ground_truth", "length_ratio_pred_over_gt"]),
        "gt_centerline_coverage_1vox": pick(
            metrics,
            ["ground_truth", "gt_to_pred_centerline_coverage", "coverage_within_1_voxels"],
        ),
        "gt_centerline_coverage_2vox": pick(
            metrics,
            ["ground_truth", "gt_to_pred_centerline_coverage", "coverage_within_2_voxels"],
        ),
        "gt_centerline_coverage_3vox": pick(
            metrics,
            ["ground_truth", "gt_to_pred_centerline_coverage", "coverage_within_3_voxels"],
        ),
        "gt_centerline_coverage_5vox": pick(
            metrics,
            ["ground_truth", "gt_to_pred_centerline_coverage", "coverage_within_5_voxels"],
        ),
        "pred_centerline_precision_2vox": pick(
            metrics,
            ["ground_truth", "pred_to_gt_centerline_coverage", "coverage_within_2_voxels"],
        ),
        "gt_branch_recall_3vox_80pct": gt_branch_recall.get("branch_recall"),
        "gt_detected_branch_count": gt_branch_recall.get("detected_branch_count"),
        "gt_reachable_branch_count": gt_branch_recall.get("gt_reachable_branch_count"),
        "gt_branch_recall_gen6": generation_recall(6),
        "gt_branch_count_gen6": generation_branch_count(6),
        "gt_branch_recall_gen7": generation_recall(7),
        "gt_branch_count_gen7": generation_branch_count(7),
        "gt_branch_recall_gen8": generation_recall(8),
        "gt_branch_count_gen8": generation_branch_count(8),
    }
    return row


def main():
    default_data_root = os.environ.get(
        "AVB_DATA_ROOT",
        str(Path.home() / "AMS_Project" / "datasets_new"),
    )
    parser = argparse.ArgumentParser(
        description=(
            "Run final held-out AIRRC validation for the airway pipeline. "
            "This is intended to check real connected airway recovery, not just one LIDC render."
        )
    )
    parser.add_argument("--data-root", default=default_data_root)
    parser.add_argument("--checkpoint", default="saved_model_topology/wingsnet_best.pth")
    parser.add_argument("--model-module", default="WingsNet")
    parser.add_argument("--model-class", default="WingsNet")
    parser.add_argument("--model-kwargs", default='{"in_channel": 1, "n_classes": 2}')
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cases-file", default=None, help="Optional text file with one case id per line")
    parser.add_argument("--split-json", default=None, help="Optional val.json patch split to infer case ids from")
    parser.add_argument(
        "--split-summary-json",
        default=str(Path(default_data_root) / "airrc_patches" / "splits" / "split_summary.json"),
        help="Case-level split summary produced by extract_airrc_patches.py",
    )
    parser.add_argument("--limit", type=int, default=10, help="Number of validation cases to run")
    parser.add_argument("--output-root", default=str(Path(default_data_root) / "final_validation" / "airrc"))
    parser.add_argument("--case-suffix", default="final_eval", help="Suffix used for prediction/centerline output case ids")
    parser.add_argument("--mask-threshold", type=float, default=0.2)
    parser.add_argument("--skeleton-threshold", type=float, default=0.2)
    parser.add_argument("--skeleton-low-threshold", type=float, default=0.08)
    parser.add_argument("--prune-length", type=float, default=12.0)
    parser.add_argument("--preserve-generations", type=int, default=8)
    parser.add_argument("--target-mode", choices=["farthest-endpoint", "all-endpoints", "generation"], default="all-endpoints")
    parser.add_argument("--target-generation", type=int, default=8)
    parser.add_argument("--spacing", default="1,1,1")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    cases = load_cases(args.cases_file, args.split_json, args.split_summary_json, args.limit)

    script_dir = Path(__file__).resolve().parent
    python = sys.executable
    rows = []

    for index, case_id in enumerate(cases, start=1):
        eval_case_id = f"{case_id}_{args.case_suffix}"
        centerline_id = f"{eval_case_id}_{threshold_suffix(args.skeleton_threshold)}"
        pred_lumen = data_root / "predictions" / eval_case_id / "pred_lumen.npy"
        centerline_dir = data_root / "centerlines" / centerline_id
        validation_json = output_root / f"{case_id}_validation_metrics.json"

        ct_path = data_root / "processed_airrc" / "images" / f"{case_id}_ct.npy"
        target_path = data_root / "processed_airrc" / "targets" / f"{case_id}_target.npy"

        print(f"\n[{index}/{len(cases)}] {case_id}", flush=True)
        if not ct_path.exists():
            print(f"Skipping missing CT: {ct_path}", flush=True)
            continue
        if not target_path.exists():
            print(f"Skipping missing target: {target_path}", flush=True)
            continue

        if not args.skip_existing or not validation_json.exists():
            pipeline_cmd = [
                python,
                str(script_dir / "run_airway_pipeline.py"),
                "--ct",
                str(ct_path),
                "--case-id",
                eval_case_id,
                "--data-root",
                str(data_root),
                "--checkpoint",
                args.checkpoint,
                "--model-module",
                args.model_module,
                "--model-class",
                args.model_class,
                "--model-kwargs",
                args.model_kwargs,
                "--device",
                args.device,
                "--mask-threshold",
                str(args.mask_threshold),
                "--skeleton-threshold",
                str(args.skeleton_threshold),
                "--skeleton-low-threshold",
                str(args.skeleton_low_threshold),
                "--prune-length",
                str(args.prune_length),
                "--preserve-generations",
                str(args.preserve_generations),
                "--root-mode",
                "timi-trachea",
                "--target-mode",
                args.target_mode,
                "--target-generation",
                str(args.target_generation),
                "--spacing",
                args.spacing,
                "--endpoint-point-size",
                "0",
            ]
            run_command("Pipeline", pipeline_cmd, dry_run=args.dry_run)

            validate_cmd = [
                python,
                str(script_dir / "validate_centerline_graph.py"),
                "--pred-lumen",
                str(pred_lumen),
                "--skeleton",
                str(centerline_dir / "pred_lumen_centerline_pruned.npy"),
                "--graph-json",
                str(centerline_dir / "pred_lumen_centerline_pruned_graph.json"),
                "--paths-json",
                str(centerline_dir / "pred_lumen_smoothed_paths.json"),
                "--target",
                str(target_path),
                "--threshold",
                str(args.skeleton_threshold),
                "--short-branch-length",
                str(args.prune_length),
                "--timi-tree-parse",
                "--output-json",
                str(validation_json),
            ]
            run_command("Validation", validate_cmd, dry_run=args.dry_run)

        if not args.dry_run and validation_json.exists():
            rows.append(summarize_case(case_id, validation_json, centerline_id))

    if args.dry_run:
        return

    summary_json = output_root / "final_validation_summary.json"
    summary_csv = output_root / "final_validation_summary.csv"
    with open(summary_json, "w") as f:
        json.dump({"cases": rows}, f, indent=2)

    if rows:
        with open(summary_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    print()
    print("Final validation complete")
    print("  cases:", len(rows))
    print("  summary json:", summary_json)
    if rows:
        print("  summary csv:", summary_csv)


if __name__ == "__main__":
    main()
