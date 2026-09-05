from pathlib import Path
import argparse
import csv
import json
import statistics as stats


DEFAULT_FIELDS = [
    "lumen_dice",
    "lumen_precision",
    "lumen_recall",
    "gt_centerline_coverage_1vox",
    "gt_centerline_coverage_2vox",
    "gt_centerline_coverage_3vox",
    "gt_centerline_coverage_5vox",
    "gt_branch_recall_3vox_80pct",
    "gt_branch_recall_gen6",
    "gt_branch_recall_gen7",
    "gt_branch_recall_gen8",
    "timi_max_generation",
    "endpoints",
    "branchpoints",
    "min_path_inside_lumen_ratio",
    "paths_below_095_inside_lumen",
]


def read_rows(csv_path):
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


def numeric_values(rows, field):
    values = []
    for row in rows:
        value = row.get(field)
        if value in (None, "", "None"):
            continue
        values.append(float(value))
    return values


def summarize_csv(csv_path, label, fields):
    rows = read_rows(csv_path)
    summary = {
        "label": label,
        "cases": len(rows),
    }

    for field in fields:
        values = numeric_values(rows, field)
        if values:
            summary[f"{field}_mean"] = stats.mean(values)
            summary[f"{field}_min"] = min(values)
            summary[f"{field}_max"] = max(values)
        else:
            summary[f"{field}_mean"] = None
            summary[f"{field}_min"] = None
            summary[f"{field}_max"] = None

    return summary


def find_summary_csvs(paths):
    csvs = []
    for path in paths:
        path = Path(path)
        if path.is_file():
            csvs.append(path)
        elif path.is_dir():
            csvs.extend(sorted(path.glob("**/final_validation_summary.csv")))
    return csvs


def fmt(value):
    if value is None:
        return "NA"
    if abs(value) >= 100:
        return f"{value:.1f}"
    return f"{value:.3f}"


def write_markdown(summaries, fields, output_path):
    lines = []
    headers = ["run", "cases"]
    for field in fields:
        headers.append(f"{field} mean")
        headers.append(f"{field} min")
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join("---" for _ in headers) + " |")

    for summary in summaries:
        row = [summary["label"], str(summary["cases"])]
        for field in fields:
            row.append(fmt(summary.get(f"{field}_mean")))
            row.append(fmt(summary.get(f"{field}_min")))
        lines.append("| " + " | ".join(row) + " |")

    output_path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Create report-ready validation summary tables.")
    parser.add_argument("inputs", nargs="+", help="CSV files or folders containing final_validation_summary.csv")
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--fields", default=",".join(DEFAULT_FIELDS), help="Comma-separated CSV fields to summarize")
    args = parser.parse_args()

    fields = [field.strip() for field in args.fields.split(",") if field.strip()]
    csvs = find_summary_csvs(args.inputs)
    if not csvs:
        raise FileNotFoundError("No final_validation_summary.csv files found.")

    summaries = [
        summarize_csv(csv_path, label=csv_path.parent.name, fields=fields)
        for csv_path in csvs
    ]

    output_md = Path(args.output_md)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    write_markdown(summaries, fields, output_md)

    if args.output_json:
        output_json = Path(args.output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, "w") as f:
            json.dump({"summaries": summaries}, f, indent=2)

    print("Done")
    print("  input tables:", len(csvs))
    print("  markdown:", output_md)
    if args.output_json:
        print("  json:", args.output_json)


if __name__ == "__main__":
    main()
