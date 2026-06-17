from pathlib import Path
import argparse
import json
import math

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

try:
    from skimage.morphology import skeletonize_3d as sk_skeletonize_3d
except ImportError:
    try:
        from skimage.morphology import skeletonize as sk_skeletonize

        def sk_skeletonize_3d(volume):
            return sk_skeletonize(volume)
    except ImportError:
        sk_skeletonize_3d = None


NEIGHBOR_OFFSETS = [
    (dz, dy, dx)
    for dz in (-1, 0, 1)
    for dy in (-1, 0, 1)
    for dx in (-1, 0, 1)
    if not (dz == 0 and dy == 0 and dx == 0)
]


def load_volume(path, channel=None, threshold=0.5):
    arr = np.load(path)
    arr = np.asarray(arr)

    if arr.ndim == 5:
        channel = 0 if channel is None else channel
        arr = arr[0, channel]
    elif arr.ndim == 4:
        channel = 0 if channel is None else channel
        arr = arr[channel]
    elif arr.ndim != 3:
        raise ValueError(f"Expected 3D, 4D, or 5D array at {path}, got {arr.shape}")

    return arr > threshold


def load_target_lumen(path, channel=0):
    target = np.load(path)
    if target.ndim != 4 or target.shape[0] <= channel:
        raise ValueError(f"Expected target shape (C,D,H,W), got {target.shape}")
    return target[channel] > 0


def load_graph(path):
    with open(path, "r") as f:
        payload = json.load(f)
    return payload.get("graph", payload)


def load_paths(path):
    with open(path, "r") as f:
        payload = json.load(f)
    return payload.get("paths", [])


def component_stats(mask, connectivity=1):
    structure = ndi.generate_binary_structure(3, connectivity)
    labeled, count = ndi.label(mask, structure=structure)
    if count == 0:
        return {
            "component_count": 0,
            "largest_component_voxels": 0,
            "total_voxels": int(mask.sum()),
            "largest_component_ratio": 0.0,
        }

    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    largest = int(sizes.max())
    total = int(mask.sum())
    return {
        "component_count": int(count),
        "connectivity": int(connectivity),
        "largest_component_voxels": largest,
        "total_voxels": total,
        "largest_component_ratio": float(largest / total) if total else 0.0,
    }


def find_bounding_box_3d(mask, padding=4):
    coords = np.argwhere(mask > 0)
    if len(coords) == 0:
        return None

    mins = coords.min(axis=0)
    maxs = coords.max(axis=0) + 1
    mins = np.maximum(mins - padding, 0)
    maxs = np.minimum(maxs + padding, mask.shape)
    return tuple(int(v) for pair in zip(mins, maxs) for v in pair)


def timi_skeleton_parsing(skeleton, min_branch_voxels=5):
    neighbor_filter = ndi.generate_binary_structure(3, 3)
    skeleton_filtered = ndi.convolve(skeleton.astype(np.uint8), neighbor_filter) * skeleton
    skeleton_parse = skeleton.copy().astype(np.uint8)
    skeleton_parse[skeleton_filtered > 3] = 0

    con_filter = ndi.generate_binary_structure(3, 3)
    labeled, num = ndi.label(skeleton_parse, structure=con_filter)

    for label_id in range(1, num + 1):
        if int((labeled == label_id).sum()) < min_branch_voxels:
            skeleton_parse[labeled == label_id] = 0

    labeled, num = ndi.label(skeleton_parse, structure=con_filter)
    return skeleton_parse.astype(np.uint8), labeled.astype(np.uint16), int(num)


def timi_tree_parsing(skeleton_parse, lumen_mask, branch_labels):
    _, indices = ndi.distance_transform_edt(1 - skeleton_parse, return_indices=True)
    tree_parsing = branch_labels[indices[0], indices[1], indices[2]] * lumen_mask
    return tree_parsing.astype(np.uint16)


def timi_loc_trachea(tree_parsing, num):
    if num == 0:
        return None

    volumes = np.zeros(num, dtype=np.int64)
    for index in range(num):
        volumes[index] = int((tree_parsing == (index + 1)).sum())

    if int(volumes.max()) == 0:
        return None

    return int(np.argmax(volumes) + 1)


def timi_adjacent_map(tree_parsing, num):
    adjacency = np.zeros((num, num), dtype=np.uint8)

    for index in range(num):
        region = tree_parsing == (index + 1)
        bbox = find_bounding_box_3d(region)
        if bbox is None:
            continue

        z0, z1, y0, y1, x0, x1 = bbox
        cropped_region = region[z0:z1, y0:y1, x0:x1].astype(np.uint8)
        cropped_tree = tree_parsing[z0:z1, y0:y1, x0:x1]

        dilation_filter = ndi.generate_binary_structure(3, 1)
        boundary = ndi.binary_dilation(cropped_region, structure=dilation_filter).astype(np.uint8) - cropped_region
        adjacent_labels = np.unique(boundary * cropped_tree)
        adjacent_labels = adjacent_labels[adjacent_labels > 0]

        for label_id in adjacent_labels:
            adjacency[index, int(label_id) - 1] = 1

    return adjacency


def timi_parent_children_map(adjacency, trachea_label):
    num = adjacency.shape[0]
    parent_map = np.zeros((num, num), dtype=np.uint8)
    children_map = np.zeros((num, num), dtype=np.uint8)
    generation = np.zeros(num, dtype=np.uint16)
    processed = np.zeros(num, dtype=np.uint8)

    root = int(trachea_label) - 1
    queue = [root]
    parent_map[root, root] = 1

    while queue:
        current_level = queue
        processed[current_level] = 1
        queue = []

        while current_level:
            current = current_level.pop()
            children = np.where(adjacency[current] > 0)[0]
            for child in children:
                if parent_map[child].sum() == 0:
                    parent_map[child, current] = 1
                    children_map[current, child] = 1
                    generation[child] = generation[current] + 1
                    queue.append(int(child))
                elif generation[current] + 1 == generation[child]:
                    parent_map[child, current] = 1
                    children_map[current, child] = 1

    return parent_map, children_map, generation, processed


def timi_tree_parse_metrics(lumen_mask, skeleton, min_branch_voxels=5):
    skeleton_parse, branch_labels, branch_count = timi_skeleton_parsing(
        skeleton,
        min_branch_voxels=min_branch_voxels,
    )

    if branch_count == 0:
        return {
            "parsed_branch_count": 0,
            "trachea_label": None,
            "adjacency_edge_count": 0,
            "max_generation": 0,
            "unreached_branch_count": 0,
            "multi_parent_branch_count": 0,
            "single_child_parent_count": 0,
            "parsed_skeleton_voxels": int(skeleton_parse.sum()),
            "min_branch_voxels": int(min_branch_voxels),
        }

    tree_parsing = timi_tree_parsing(skeleton_parse, lumen_mask.astype(np.uint8), branch_labels)
    trachea_label = timi_loc_trachea(tree_parsing, branch_count)
    adjacency = timi_adjacent_map(tree_parsing, branch_count)

    if trachea_label is None:
        return {
            "parsed_branch_count": int(branch_count),
            "trachea_label": None,
            "adjacency_edge_count": int(adjacency.sum()),
            "max_generation": 0,
            "unreached_branch_count": int(branch_count),
            "multi_parent_branch_count": 0,
            "single_child_parent_count": 0,
            "parsed_skeleton_voxels": int(skeleton_parse.sum()),
            "min_branch_voxels": int(min_branch_voxels),
        }

    parent_map, children_map, generation, processed = timi_parent_children_map(adjacency, trachea_label)
    parent_counts = parent_map.sum(axis=1)
    child_counts = children_map.sum(axis=1)

    return {
        "parsed_branch_count": int(branch_count),
        "trachea_label": int(trachea_label),
        "adjacency_edge_count": int(adjacency.sum()),
        "max_generation": int(generation.max()) if len(generation) else 0,
        "unreached_branch_count": int((processed == 0).sum()),
        "multi_parent_branch_count": int((parent_counts > 1).sum()),
        "single_child_parent_count": int((child_counts == 1).sum()),
        "parsed_skeleton_voxels": int(skeleton_parse.sum()),
        "min_branch_voxels": int(min_branch_voxels),
    }


def graph_adjacency(graph):
    adjacency = {int(node["id"]): [] for node in graph.get("nodes", [])}
    coords = {int(node["id"]): tuple(node["zyx"]) for node in graph.get("nodes", [])}

    for edge in graph.get("edges", []):
        a, b = int(edge[0]), int(edge[1])
        if a not in adjacency or b not in adjacency:
            continue
        za, ya, xa = coords[a]
        zb, yb, xb = coords[b]
        weight = math.sqrt((za - zb) ** 2 + (ya - yb) ** 2 + (xa - xb) ** 2)
        adjacency[a].append((b, weight))
        adjacency[b].append((a, weight))

    return adjacency


def graph_component_stats(graph):
    adjacency = graph_adjacency(graph)
    unseen = set(adjacency)
    component_sizes = []

    while unseen:
        start = unseen.pop()
        stack = [start]
        size = 0
        while stack:
            node = stack.pop()
            size += 1
            for neighbor, _ in adjacency[node]:
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    stack.append(neighbor)
        component_sizes.append(size)

    component_sizes.sort(reverse=True)
    node_count = len(adjacency)
    edge_count = len(graph.get("edges", []))
    isolated_nodes = sum(1 for neighbors in adjacency.values() if len(neighbors) == 0)

    return {
        "graph_component_count": len(component_sizes),
        "graph_component_sizes": component_sizes,
        "largest_graph_component_nodes": component_sizes[0] if component_sizes else 0,
        "largest_graph_component_ratio": (
            float(component_sizes[0] / node_count) if component_sizes and node_count else 0.0
        ),
        "node_count": int(node_count),
        "edge_count": int(edge_count),
        "isolated_node_count": int(isolated_nodes),
    }


def graph_degree_stats(graph):
    nodes = graph.get("nodes", [])
    degrees = [int(node.get("degree", 0)) for node in nodes]

    endpoints = [node for node in nodes if int(node.get("degree", 0)) == 1]
    branchpoints = [node for node in nodes if int(node.get("degree", 0)) >= 3]

    return {
        "endpoint_count": int(len(endpoints)),
        "branchpoint_count": int(len(branchpoints)),
        "min_degree": int(min(degrees)) if degrees else 0,
        "max_degree": int(max(degrees)) if degrees else 0,
        "mean_degree": float(np.mean(degrees)) if degrees else 0.0,
    }


def graph_total_length(graph):
    coords = {int(node["id"]): np.asarray(node["zyx"], dtype=np.float64) for node in graph.get("nodes", [])}
    length = 0.0

    for edge in graph.get("edges", []):
        a, b = int(edge[0]), int(edge[1])
        if a in coords and b in coords:
            length += float(np.linalg.norm(coords[a] - coords[b]))

    return length


def short_terminal_branch_stats(graph, max_length=8.0):
    adjacency = graph_adjacency(graph)
    endpoints = [node_id for node_id, neighbors in adjacency.items() if len(neighbors) == 1]
    short_count = 0
    lengths = []

    for endpoint in endpoints:
        previous = None
        current = endpoint
        length = 0.0

        while True:
            neighbors = [(node, weight) for node, weight in adjacency[current] if node != previous]

            if previous is not None and len(adjacency[current]) != 2:
                break

            if not neighbors:
                break

            next_node, weight = neighbors[0]
            previous = current
            current = next_node
            length += weight

        lengths.append(float(length))
        if length < max_length:
            short_count += 1

    return {
        "terminal_branch_count": int(len(endpoints)),
        "short_terminal_branch_count": int(short_count),
        "short_terminal_branch_threshold": float(max_length),
        "min_terminal_branch_length": float(min(lengths)) if lengths else 0.0,
        "mean_terminal_branch_length": float(np.mean(lengths)) if lengths else 0.0,
    }


def skeletonize(mask):
    if sk_skeletonize_3d is None:
        raise RuntimeError("scikit-image is required for ground-truth skeleton metrics.")
    return sk_skeletonize_3d(mask.astype(bool)).astype(bool)


def skeleton_length(skeleton):
    coords = {tuple(coord) for coord in np.argwhere(skeleton)}
    length = 0.0

    for z, y, x in coords:
        for dz, dy, dx in NEIGHBOR_OFFSETS:
            neighbor = (z + dz, y + dy, x + dx)
            if neighbor not in coords:
                continue
            if (z, y, x) < neighbor:
                length += math.sqrt(dz * dz + dy * dy + dx * dx)

    return float(length)


def point_distance_metrics(source_mask, target_mask):
    source = np.argwhere(source_mask)
    target = np.argwhere(target_mask)

    if len(source) == 0 or len(target) == 0:
        return {
            "mean_distance": None,
            "median_distance": None,
            "p95_distance": None,
            "hausdorff_distance": None,
        }

    tree = cKDTree(target)
    distances, _ = tree.query(source, k=1)

    return {
        "mean_distance": float(np.mean(distances)),
        "median_distance": float(np.median(distances)),
        "p95_distance": float(np.percentile(distances, 95)),
        "hausdorff_distance": float(np.max(distances)),
    }


def path_inside_lumen_stats(paths, lumen_mask):
    records = []

    for path in paths:
        coords = np.asarray(path.get("coordinates_zyx", []), dtype=np.int64)
        inside = 0
        valid = 0

        for z, y, x in coords:
            if (
                0 <= z < lumen_mask.shape[0]
                and 0 <= y < lumen_mask.shape[1]
                and 0 <= x < lumen_mask.shape[2]
            ):
                valid += 1
                if lumen_mask[z, y, x]:
                    inside += 1

        records.append({
            "path_id": path.get("path_id"),
            "point_count": int(len(coords)),
            "valid_point_count": int(valid),
            "inside_lumen_point_count": int(inside),
            "inside_lumen_ratio": float(inside / valid) if valid else 0.0,
            "length_voxels": path.get("length_voxels"),
        })

    return records


def main():
    parser = argparse.ArgumentParser(description="Validate airway centerline graph/path quality.")
    parser.add_argument("--pred-lumen", required=True, help="Predicted lumen mask/probability .npy")
    parser.add_argument("--skeleton", required=True, help="Predicted centerline skeleton .npy")
    parser.add_argument("--graph-json", required=True, help="Predicted/pruned graph JSON")
    parser.add_argument("--paths-json", required=True, help="Planned paths JSON")
    parser.add_argument("--output-json", required=True, help="Output metrics JSON")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for --pred-lumen")
    parser.add_argument("--short-branch-length", type=float, default=8.0, help="Short terminal branch threshold")
    parser.add_argument("--target", default=None, help="Optional target.npy with lumen in channel 0")
    parser.add_argument("--target-lumen-channel", type=int, default=0, help="Target lumen channel")
    parser.add_argument("--timi-tree-parse", action="store_true", help="Add TIMI-style branch parsing metrics")
    parser.add_argument("--timi-min-branch-voxels", type=int, default=5, help="Minimum parsed branch size")
    args = parser.parse_args()

    pred_lumen = load_volume(args.pred_lumen, threshold=args.threshold)
    pred_skeleton = load_volume(args.skeleton, threshold=0.5)
    graph = load_graph(args.graph_json)
    paths = load_paths(args.paths_json)

    metrics = {
        "inputs": {
            "pred_lumen": str(Path(args.pred_lumen)),
            "skeleton": str(Path(args.skeleton)),
            "graph_json": str(Path(args.graph_json)),
            "paths_json": str(Path(args.paths_json)),
            "target": str(Path(args.target)) if args.target else None,
            "threshold": args.threshold,
        },
        "pred_lumen_components": component_stats(pred_lumen, connectivity=1),
        "pred_skeleton_components": component_stats(pred_skeleton, connectivity=3),
        "graph_connectivity": graph_component_stats(graph),
        "graph_degree": graph_degree_stats(graph),
        "graph_total_length_voxels": graph_total_length(graph),
        "terminal_branches": short_terminal_branch_stats(graph, max_length=args.short_branch_length),
        "path_count": int(len(paths)),
        "paths": path_inside_lumen_stats(paths, pred_lumen),
    }

    if args.timi_tree_parse:
        metrics["timi_tree_parse"] = timi_tree_parse_metrics(
            lumen_mask=pred_lumen,
            skeleton=pred_skeleton,
            min_branch_voxels=args.timi_min_branch_voxels,
        )

    if args.target:
        gt_lumen = load_target_lumen(args.target, channel=args.target_lumen_channel)
        gt_skeleton = skeletonize(gt_lumen)
        pred_to_gt = point_distance_metrics(pred_skeleton, gt_skeleton)
        gt_to_pred = point_distance_metrics(gt_skeleton, pred_skeleton)
        gt_length = skeleton_length(gt_skeleton)
        pred_length = graph_total_length(graph)

        metrics["ground_truth"] = {
            "gt_lumen_components": component_stats(gt_lumen, connectivity=1),
            "gt_skeleton_components": component_stats(gt_skeleton, connectivity=3),
            "gt_centerline_length_voxels": gt_length,
            "pred_to_gt_centerline_distance": pred_to_gt,
            "gt_to_pred_centerline_distance": gt_to_pred,
            "symmetric_mean_centerline_distance": (
                float((pred_to_gt["mean_distance"] + gt_to_pred["mean_distance"]) / 2.0)
                if pred_to_gt["mean_distance"] is not None and gt_to_pred["mean_distance"] is not None
                else None
            ),
            "symmetric_hausdorff_distance": (
                float(max(pred_to_gt["hausdorff_distance"], gt_to_pred["hausdorff_distance"]))
                if pred_to_gt["hausdorff_distance"] is not None and gt_to_pred["hausdorff_distance"] is not None
                else None
            ),
            "length_ratio_pred_over_gt": float(pred_length / gt_length) if gt_length > 0 else None,
        }

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(metrics, f, indent=2)

    print("Done")
    print("  metrics:", output_json)
    print("  lumen components:", metrics["pred_lumen_components"]["component_count"])
    print("  skeleton components:", metrics["pred_skeleton_components"]["component_count"])
    print("  graph components:", metrics["graph_connectivity"]["graph_component_count"])
    print("  endpoints:", metrics["graph_degree"]["endpoint_count"])
    print("  branchpoints:", metrics["graph_degree"]["branchpoint_count"])
    print("  graph length:", "%.2f" % metrics["graph_total_length_voxels"])
    print("  paths:", metrics["path_count"])
    if metrics["paths"]:
        print("  first path inside lumen ratio:", "%.3f" % metrics["paths"][0]["inside_lumen_ratio"])
    if "timi_tree_parse" in metrics:
        print("  timi parsed branches:", metrics["timi_tree_parse"]["parsed_branch_count"])
        print("  timi max generation:", metrics["timi_tree_parse"]["max_generation"])
        print("  timi unreached branches:", metrics["timi_tree_parse"]["unreached_branch_count"])
        print("  timi multi-parent branches:", metrics["timi_tree_parse"]["multi_parent_branch_count"])
    if "ground_truth" in metrics:
        print("  gt length:", "%.2f" % metrics["ground_truth"]["gt_centerline_length_voxels"])
        print("  length ratio:", metrics["ground_truth"]["length_ratio_pred_over_gt"])
        print("  symmetric mean distance:", metrics["ground_truth"]["symmetric_mean_centerline_distance"])
        print("  symmetric hausdorff:", metrics["ground_truth"]["symmetric_hausdorff_distance"])


if __name__ == "__main__":
    main()
