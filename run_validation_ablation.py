from pathlib import Path
import argparse
import os
import subprocess
import sys


ABLATIONS = [
    {
        "name": "baseline_thr02_prune12_project",
        "checkpoint": "saved_model/wingsnet_best.pth",
        "threshold": 0.2,
        "low_threshold": 0.08,
        "prune_length": 12,
        "preserve_generations": 8,
    },
    {
        "name": "topology_thr02_prune12_project",
        "checkpoint": "saved_model_topology/wingsnet_best.pth",
        "threshold": 0.2,
        "low_threshold": 0.08,
        "prune_length": 12,
        "preserve_generations": 8,
    },
    {
        "name": "topology_thr02_noprune_project",
        "checkpoint": "saved_model_topology/wingsnet_best.pth",
        "threshold": 0.2,
        "low_threshold": 0.08,
        "prune_length": 0,
        "preserve_generations": 99,
    },
    {
        "name": "topology_thr03_prune12_project",
        "checkpoint": "saved_model_topology/wingsnet_best.pth",
        "threshold": 0.3,
        "low_threshold": 0.08,
        "prune_length": 12,
        "preserve_generations": 8,
    },
]


def run(command, dry_run=False):
    print()
    print("=" * 80, flush=True)
    print(" ".join(str(part) for part in command), flush=True)
    print("=" * 80, flush=True)
    if not dry_run:
        subprocess.run(command, check=True)


def main():
    default_data_root = os.environ.get(
        "AVB_DATA_ROOT",
        str(Path.home() / "AMS_Project" / "datasets_new"),
    )
    parser = argparse.ArgumentParser(description="Run final validation ablations for report-ready comparison tables.")
    parser.add_argument("--data-root", default=default_data_root)
    parser.add_argument("--output-root", default=str(Path(default_data_root) / "final_validation" / "ablations"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--cases-file", default=None)
    parser.add_argument(
        "--split-summary-json",
        default=str(Path(default_data_root) / "airrc_patches" / "splits" / "split_summary.json"),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    python = sys.executable

    for config in ABLATIONS:
        command = [
            python,
            str(script_dir / "run_final_validation_suite.py"),
            "--data-root",
            args.data_root,
            "--checkpoint",
            config["checkpoint"],
            "--device",
            args.device,
            "--limit",
            str(args.limit),
            "--mask-threshold",
            str(config["threshold"]),
            "--skeleton-threshold",
            str(config["threshold"]),
            "--skeleton-low-threshold",
            str(config["low_threshold"]),
            "--prune-length",
            str(config["prune_length"]),
            "--preserve-generations",
            str(config["preserve_generations"]),
            "--split-summary-json",
            args.split_summary_json,
            "--output-root",
            str(Path(args.output_root) / config["name"]),
            "--case-suffix",
            f"final_eval_{config['name']}",
        ]
        if args.cases_file:
            command.extend(["--cases-file", args.cases_file])
        run(command, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
