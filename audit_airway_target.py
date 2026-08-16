from pathlib import Path
import argparse
import json

import numpy as np
from scipy import ndimage as ndi


def parse_zyx(value):
    values = tuple(int(part.strip()) for part in value.split(","))
    if len(values) != 3:
        raise argparse.ArgumentTypeError("Expected format: z,y,x")
    return values


def load_graph(graph_json):
    with open(graph_json, "r") as f:
        payload = json.load(f)
    return payload.get("graph", payload)


def graph_degrees(graph):
    degrees = {int(node["id"]): 0 for node in graph.get("nodes", [])}
    for a, b in graph.get("edges", []):
        degrees[int(a)] = degrees.get(int(a), 0) + 1
        degrees[int(b)] = degrees.get(int(b), 0) + 1
    return degrees


def target_region_stats(volume, target_zyx, radii, thresholds):
    target = np.asarray(target_zyx, dtype=int)
    z, y, x = target
    stats = {}
    for radius in radii:
        region = volume[
            max(0, z - radius): min(volume.shape[0], z + radius + 1),
            max(0, y - radius): min(volume.shape[1], y + radius + 1),
            max(0, x - radius): min(volume.shape[2], x + radius + 1),
        ]
        item = {
            "max_probability": float(region.max()) if region.size else 0.0,
            "mean_probability": float(region.mean()) if region.size else 0.0,
        }
        for threshold in thresholds:
            item[f"voxels_gt_{threshold:g}"] = int((region > threshold).sum())
        stats[str(radius)] = item
    return stats


def component_stats(volume, target_zyx, thresholds):
    target = np.asarray(target_zyx, dtype=int)
    z, y, x = target
    structure = ndi.generate_binary_structure(3, 1)
    stats = {}
    in_bounds = (
        0 <= z < volume.shape[0]
        and 0 <= y < volume.shape[1]
        and 0 <= x < volume.shape[2]
    )
    for threshold in thresholds:
        mask = volume > threshold
        labels, count = ndi.label(mask, structure=structure)
        sizes = np.bincount(labels.ravel()) if count else np.asarray([0])
        sizes[0] = 0
        stats[str(threshold)] = {
            "component_count": int(count),
            "largest_component_voxels": int(sizes.max()) if len(sizes) else 0,
            "target_label": int(labels[z, y, x]) if in_bounds else None,
            "target_in_lumen": bool(labels[z, y, x] > 0) if in_bounds else False,
            "voxels": int(mask.sum()),
        }
    return stats


def nearest_graph_node(graph, target_zyx):
    nodes = graph.get("nodes", [])
    if not nodes:
        return None

    target = np.asarray(target_zyx, dtype=np.float64)
    coords = np.asarray([node["zyx"] for node in nodes], dtype=np.float64)
    distances = np.linalg.norm(coords - target, axis=1)
    index = int(np.argmin(distances))
    node = nodes[index]
    return {
        "id": int(node["id"]),
        "zyx": [int(value) for value in node["zyx"]],
        "generation": node.get("generation"),
        "distance_voxels": float(distances[index]),
    }


def load_first_path(paths_json):
    if paths_json is None:
        return None
    with open(paths_json, "r") as f:
        payload = json.load(f)
    paths = payload.get("paths", [])
    if not paths:
        return {
            "target_mode": payload.get("target_mode"),
            "path_count": 0,
        }
    path = paths[0]
    return {
        "target_mode": payload.get("target_mode"),
        "effective_target_mode": payload.get("effective_target_mode"),
        "requested_target_zyx": payload.get("requested_target_zyx") or path.get("requested_target_zyx"),
        "selected_target_node": path.get("selected_target_node", path.get("target_node")),
        "selected_target_zyx": path.get("selected_target_zyx"),
        "target_distance_voxels": path.get("target_distance_voxels"),
        "target_generation": path.get("target_generation"),
        "path_length_voxels": path.get("length_voxels"),
        "path_count": payload.get("path_count", len(paths)),
    }


def main():
    parser = argparse.ArgumentParser(description="Audit predicted airway connectivity near a target z,y,x coordinate.")
    parser.add_argument("--pred-lumen", required=True, help="Predicted lumen probability .npy")
    parser.add_argument("--graph-json", required=True, help="Pruned centerline graph JSON")
    parser.add_argument("--paths-json", default=None, help="Optional planned paths JSON")
    parser.add_argument("--target-zyx", required=True, type=parse_zyx, help="Target as z,y,x")
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    volume = np.load(args.pred_lumen)
    graph = load_graph(args.graph_json)
    degrees = graph_degrees(graph)
    generations = [
        node.get("generation")
        for node in graph.get("nodes", [])
        if node.get("generation") is not None
    ]
    target = np.asarray(args.target_zyx, dtype=int)
    z, y, x = target
    target_probability = (
        float(volume[z, y, x])
        if 0 <= z < volume.shape[0] and 0 <= y < volume.shape[1] and 0 <= x < volume.shape[2]
        else None
    )

    thresholds = [0.2, 0.15, 0.1, 0.06]
    result = {
        "pred_lumen": str(Path(args.pred_lumen)),
        "graph_json": str(Path(args.graph_json)),
        "paths_json": str(Path(args.paths_json)) if args.paths_json else None,
        "target_zyx": [int(value) for value in args.target_zyx],
        "volume_shape_zyx": list(volume.shape),
        "target_probability": target_probability,
        "graph": {
            "nodes": len(graph.get("nodes", [])),
            "edges": len(graph.get("edges", [])),
            "endpoints": int(sum(value == 1 for value in degrees.values())),
            "branchpoints": int(sum(value >= 3 for value in degrees.values())),
            "max_generation": int(max(generations)) if generations else None,
            "nearest_target_node": nearest_graph_node(graph, args.target_zyx),
        },
        "path": load_first_path(args.paths_json),
        "target_region": target_region_stats(volume, args.target_zyx, radii=[5, 10, 20, 30], thresholds=thresholds),
        "components": component_stats(volume, args.target_zyx, thresholds=thresholds),
    }

    text = json.dumps(result, indent=2)
    print(text)
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n")


if __name__ == "__main__":
    main()
