from pathlib import Path
import argparse
import csv
import json

import numpy as np

from audit_airway_target import nearest_graph_node, target_region_stats, component_stats, load_graph


def load_poi_targets(path):
    with open(path, "r") as f:
        payload = json.load(f)
    targets = payload.get("targets", [])
    records = []
    for index, target in enumerate(targets):
        target_zyx = target.get("target_zyx")
        if target_zyx is None:
            continue
        records.append({
            "poi_index": index,
            "nodule_id": target.get("nodule_id"),
            "reader_index": target.get("reader_index"),
            "nodule_index": target.get("nodule_index"),
            "point_count": target.get("point_count"),
            "target_zyx": [int(value) for value in target_zyx],
        })
    return records


def target_probability(volume, target_zyx):
    z, y, x = [int(value) for value in target_zyx]
    if 0 <= z < volume.shape[0] and 0 <= y < volume.shape[1] and 0 <= x < volume.shape[2]:
        return float(volume[z, y, x])
    return None


def score_record(record):
    distance = record["nearest_graph_distance_voxels"]
    region_max_10 = record["region10_max_probability"]
    region_max_20 = record["region20_max_probability"]
    voxels_10 = record["region10_voxels_gt_006"]
    voxels_20 = record["region20_voxels_gt_006"]

    if distance is None:
        return -1e9

    support = 0.0
    support += 20.0 * min(region_max_10, 1.0)
    support += 10.0 * min(region_max_20, 1.0)
    support += min(voxels_10, 200) / 20.0
    support += min(voxels_20, 1000) / 100.0
    return float(support - distance)


def audit_target(volume, graph, poi, thresholds):
    target_zyx = poi["target_zyx"]
    nearest = nearest_graph_node(graph, target_zyx)
    regions = target_region_stats(volume, target_zyx, radii=[5, 10, 20, 30], thresholds=thresholds)
    components = component_stats(volume, target_zyx, thresholds=thresholds)

    record = dict(poi)
    record["target_probability"] = target_probability(volume, target_zyx)
    record["nearest_graph_node"] = nearest["id"] if nearest else None
    record["nearest_graph_zyx"] = nearest["zyx"] if nearest else None
    record["nearest_graph_generation"] = nearest["generation"] if nearest else None
    record["nearest_graph_distance_voxels"] = nearest["distance_voxels"] if nearest else None

    for radius in (5, 10, 20, 30):
        region = regions[str(radius)]
        record[f"region{radius}_max_probability"] = region["max_probability"]
        record[f"region{radius}_mean_probability"] = region["mean_probability"]
        record[f"region{radius}_voxels_gt_006"] = region.get("voxels_gt_0.06", 0)
        record[f"region{radius}_voxels_gt_02"] = region.get("voxels_gt_0.2", 0)

    for threshold in thresholds:
        item = components[str(threshold)]
        key = str(threshold).replace(".", "")
        record[f"target_in_lumen_thr{key}"] = item["target_in_lumen"]
        record[f"component_count_thr{key}"] = item["component_count"]

    record["support_score"] = score_record(record)
    return record


def compact_csv_row(record):
    return {
        "rank": record["rank"],
        "poi_index": record["poi_index"],
        "nodule_id": record["nodule_id"],
        "reader_index": record["reader_index"],
        "point_count": record["point_count"],
        "target_zyx": ",".join(str(value) for value in record["target_zyx"]),
        "target_probability": record["target_probability"],
        "nearest_graph_distance_voxels": record["nearest_graph_distance_voxels"],
        "nearest_graph_generation": record["nearest_graph_generation"],
        "nearest_graph_zyx": ",".join(str(value) for value in record["nearest_graph_zyx"])
        if record["nearest_graph_zyx"] else None,
        "region10_max_probability": record["region10_max_probability"],
        "region20_max_probability": record["region20_max_probability"],
        "region10_voxels_gt_006": record["region10_voxels_gt_006"],
        "region20_voxels_gt_006": record["region20_voxels_gt_006"],
        "target_in_lumen_thr02": record["target_in_lumen_thr02"],
        "support_score": record["support_score"],
    }


def main():
    parser = argparse.ArgumentParser(description="Rank LIDC nodule POIs by predicted airway support/reachability.")
    parser.add_argument("--poi-json", required=True, help="Output from extract_lidc_poi_target.py")
    parser.add_argument("--pred-lumen", required=True, help="Predicted lumen probability .npy")
    parser.add_argument("--graph-json", required=True, help="Pruned centerline graph JSON")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", required=True)
    args = parser.parse_args()

    volume = np.load(args.pred_lumen)
    graph = load_graph(args.graph_json)
    pois = load_poi_targets(args.poi_json)
    thresholds = [0.2, 0.15, 0.1, 0.06]

    records = [audit_target(volume, graph, poi, thresholds) for poi in pois]
    records.sort(key=lambda item: item["support_score"], reverse=True)
    for rank, record in enumerate(records, start=1):
        record["rank"] = rank

    output_json = Path(args.output_json)
    output_csv = Path(args.output_csv)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with open(output_json, "w") as f:
        json.dump({"poi_count": len(records), "ranked_targets": records}, f, indent=2)

    rows = [compact_csv_row(record) for record in records]
    if rows:
        with open(output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    print("Done")
    print("  POIs:", len(records))
    print("  JSON:", output_json)
    print("  CSV:", output_csv)
    if records:
        best = records[0]
        print("  best target zyx:", ",".join(str(value) for value in best["target_zyx"]))
        print("  nearest graph distance:", best["nearest_graph_distance_voxels"])
        print("  region10 max probability:", best["region10_max_probability"])
        print("  support score:", best["support_score"])


if __name__ == "__main__":
    main()
