from pathlib import Path
import argparse
import heapq
import json
import math

import numpy as np
from scipy import ndimage as ndi

try:
    import SimpleITK as sitk
except ImportError:
    sitk = None

try:
    from skimage.morphology import skeletonize_3d as sk_skeletonize_3d
except ImportError:
    try:
        from skimage.morphology import skeletonize as sk_skeletonize

        def sk_skeletonize_3d(volume):
            return sk_skeletonize(volume)
    except ImportError:
        def sk_skeletonize_3d(volume):
            raise RuntimeError(
                "scikit-image is required for skeletonization. Install it with: pip install scikit-image"
            )


NEIGHBOR_OFFSETS = [
    (dz, dy, dx)
    for dz in (-1, 0, 1)
    for dy in (-1, 0, 1)
    for dx in (-1, 0, 1)
    if not (dz == 0 and dy == 0 and dx == 0)
]


def parse_zyx(value):
    values = tuple(int(part.strip()) for part in value.split(","))
    if len(values) != 3:
        raise argparse.ArgumentTypeError("Expected format: z,y,x")
    return values


def load_volume(path, channel=0):
    path = Path(path)

    if path.suffix == ".npy":
        arr = np.load(path)
        reference_img = None
    elif path.name.endswith(".nii.gz") or path.suffix == ".nii":
        if sitk is None:
            raise RuntimeError("SimpleITK is required to read NIfTI files.")
        reference_img = sitk.ReadImage(str(path))
        arr = sitk.GetArrayFromImage(reference_img)
    else:
        raise ValueError(f"Unsupported input format: {path}")

    arr = np.asarray(arr)

    if arr.ndim == 5:
        arr = arr[0, channel]
    elif arr.ndim == 4:
        arr = arr[channel]
    elif arr.ndim != 3:
        raise ValueError(f"Expected 3D, 4D, or 5D volume, got shape {arr.shape}")

    return arr, reference_img


def save_nifti_like(array, reference_img, output_path):
    if sitk is None or reference_img is None:
        return

    out_img = sitk.GetImageFromArray(array.astype(np.uint8))
    out_img.CopyInformation(reference_img)
    sitk.WriteImage(out_img, str(output_path))


def keep_largest_component(mask):
    labeled, num_components = ndi.label(mask)

    if num_components <= 1:
        return mask

    counts = np.bincount(labeled.ravel())
    counts[0] = 0
    largest_label = int(np.argmax(counts))

    return labeled == largest_label


def hysteresis_lumen_mask(volume, high_threshold, low_threshold=None):
    high_mask = volume > high_threshold
    if low_threshold is None or low_threshold >= high_threshold:
        return high_mask

    low_mask = volume > low_threshold
    structure = ndi.generate_binary_structure(rank=3, connectivity=1)
    labeled, count = ndi.label(low_mask, structure=structure)
    if count == 0:
        return high_mask

    seed_labels = np.unique(labeled[high_mask])
    seed_labels = seed_labels[seed_labels > 0]
    if len(seed_labels) == 0:
        return high_mask

    return np.isin(labeled, seed_labels)


def postprocess_mask(mask, keep_largest=True, closing_radius=1):
    mask = mask.astype(bool)

    if closing_radius > 0:
        structure = ndi.generate_binary_structure(rank=3, connectivity=1)
        for _ in range(closing_radius):
            mask = ndi.binary_closing(mask, structure=structure)

    if keep_largest:
        mask = keep_largest_component(mask)

    return mask


def skeletonize_lumen(mask):
    skeleton = sk_skeletonize_3d(mask.astype(bool))
    return skeleton.astype(np.uint8)


def build_voxel_graph(skeleton):
    coords = np.argwhere(skeleton > 0)
    coord_to_id = {tuple(coord): idx for idx, coord in enumerate(map(tuple, coords))}

    nodes = []
    edges = []

    for idx, coord in enumerate(coords):
        z, y, x = map(int, coord)
        degree = 0

        for dz, dy, dx in NEIGHBOR_OFFSETS:
            neighbor = (z + dz, y + dy, x + dx)
            neighbor_id = coord_to_id.get(neighbor)

            if neighbor_id is None:
                continue

            degree += 1
            if idx < neighbor_id:
                edges.append([idx, neighbor_id])

        nodes.append({
            "id": idx,
            "zyx": [z, y, x],
            "degree": degree,
        })

    endpoints = [node["id"] for node in nodes if node["degree"] == 1]
    branchpoints = [node["id"] for node in nodes if node["degree"] >= 3]

    return {
        "nodes": nodes,
        "edges": edges,
        "endpoints": endpoints,
        "branchpoints": branchpoints,
    }


def build_adjacency(graph):
    adjacency = {node["id"]: [] for node in graph["nodes"]}
    coords = {node["id"]: tuple(node["zyx"]) for node in graph["nodes"]}

    for a, b in graph["edges"]:
        za, ya, xa = coords[a]
        zb, yb, xb = coords[b]
        weight = math.sqrt((za - zb) ** 2 + (ya - yb) ** 2 + (xa - xb) ** 2)
        adjacency[a].append((b, weight))
        adjacency[b].append((a, weight))

    return adjacency


def build_active_adjacency(active_nodes, coords, edges):
    adjacency = {node_id: [] for node_id in active_nodes}

    for a, b in edges:
        if a not in active_nodes or b not in active_nodes:
            continue
        za, ya, xa = coords[a]
        zb, yb, xb = coords[b]
        weight = math.sqrt((za - zb) ** 2 + (ya - yb) ** 2 + (xa - xb) ** 2)
        adjacency[a].append((b, weight))
        adjacency[b].append((a, weight))

    return adjacency


def trace_terminal_branch(endpoint, adjacency):
    path = [endpoint]
    previous = None
    current = endpoint
    length = 0.0

    while True:
        neighbors = [(node, weight) for node, weight in adjacency[current] if node != previous]

        if previous is not None and len(adjacency[current]) != 2:
            return path, current, length

        if not neighbors:
            return path, current, length

        next_node, weight = neighbors[0]
        previous = current
        current = next_node
        path.append(current)
        length += weight


def prune_short_terminal_branches(graph, min_branch_length, protected_node_ids=None):
    if min_branch_length <= 0:
        return graph, []

    protected_node_ids = set() if protected_node_ids is None else set(protected_node_ids)
    active_nodes = {node["id"] for node in graph["nodes"]}
    coords = {node["id"]: node["zyx"] for node in graph["nodes"]}
    edges = [tuple(edge) for edge in graph["edges"]]
    removed = []

    changed = True
    while changed:
        changed = False
        adjacency = build_active_adjacency(active_nodes, coords, edges)
        endpoints = [node_id for node_id, neighbors in adjacency.items() if len(neighbors) == 1]

        for endpoint in endpoints:
            if endpoint not in adjacency:
                continue

            path, junction, length = trace_terminal_branch(endpoint, adjacency)

            if junction == endpoint or length >= min_branch_length:
                continue

            to_remove = [node_id for node_id in path if node_id != junction]

            if not to_remove:
                continue
            if protected_node_ids.intersection(to_remove):
                continue

            for node_id in to_remove:
                active_nodes.discard(node_id)

            removed.append({
                "endpoint": int(endpoint),
                "junction": int(junction),
                "length_voxels": float(length),
                "removed_nodes": [int(node_id) for node_id in to_remove],
            })
            changed = True
            break

    return graph_from_active_nodes(active_nodes, coords, edges), removed


def graph_from_active_nodes(active_nodes, coords, edges):
    old_to_new = {old_id: new_id for new_id, old_id in enumerate(sorted(active_nodes))}
    nodes = []
    new_edges = []
    degree = {old_id: 0 for old_id in active_nodes}

    for a, b in edges:
        if a not in active_nodes or b not in active_nodes:
            continue
        degree[a] += 1
        degree[b] += 1
        new_edges.append([old_to_new[a], old_to_new[b]])

    for old_id in sorted(active_nodes):
        nodes.append({
            "id": old_to_new[old_id],
            "zyx": [int(v) for v in coords[old_id]],
            "degree": int(degree[old_id]),
        })

    endpoints = [node["id"] for node in nodes if node["degree"] == 1]
    branchpoints = [node["id"] for node in nodes if node["degree"] >= 3]

    return {
        "nodes": nodes,
        "edges": new_edges,
        "endpoints": endpoints,
        "branchpoints": branchpoints,
    }


def graph_to_skeleton(graph, shape):
    skeleton = np.zeros(shape, dtype=np.uint8)
    for node in graph["nodes"]:
        z, y, x = node["zyx"]
        skeleton[z, y, x] = 1
    return skeleton


def parse_skeleton_branches(skeleton, min_branch_voxels=5):
    neighbor_filter = ndi.generate_binary_structure(3, 3)
    neighbor_count = ndi.convolve(skeleton.astype(np.uint8), neighbor_filter) * skeleton
    skeleton_parse = skeleton.copy().astype(np.uint8)
    skeleton_parse[neighbor_count > 3] = 0

    structure = ndi.generate_binary_structure(3, 3)
    branch_labels, num_branches = ndi.label(skeleton_parse, structure=structure)

    for branch_id in range(1, num_branches + 1):
        if int((branch_labels == branch_id).sum()) < min_branch_voxels:
            skeleton_parse[branch_labels == branch_id] = 0

    branch_labels, num_branches = ndi.label(skeleton_parse, structure=structure)
    return skeleton_parse.astype(np.uint8), branch_labels.astype(np.uint16), int(num_branches)


def assign_lumen_to_skeleton_branches(mask, skeleton_parse, branch_labels):
    _, nearest = ndi.distance_transform_edt(1 - skeleton_parse, return_indices=True)
    tree_labels = branch_labels[nearest[0], nearest[1], nearest[2]] * mask.astype(np.uint8)
    return tree_labels.astype(np.uint16)


def locate_trachea_branch(tree_labels, num_branches):
    if num_branches == 0:
        return None

    volumes = np.zeros(num_branches, dtype=np.int64)
    for branch_id in range(1, num_branches + 1):
        volumes[branch_id - 1] = int((tree_labels == branch_id).sum())

    if int(volumes.max()) == 0:
        return None

    return int(np.argmax(volumes) + 1)


def choose_timi_trachea_root(graph, mask, skeleton, root_end="min-z", min_branch_voxels=5):
    skeleton_parse, branch_labels, num_branches = parse_skeleton_branches(
        skeleton,
        min_branch_voxels=min_branch_voxels,
    )
    tree_labels = assign_lumen_to_skeleton_branches(mask, skeleton_parse, branch_labels)
    trachea_branch = locate_trachea_branch(tree_labels, num_branches)

    if trachea_branch is None:
        return None, {
            "trachea_branch": None,
            "parsed_branch_count": int(num_branches),
            "candidate_count": 0,
        }

    node_by_id = {node["id"]: node for node in graph["nodes"]}
    coords = np.array([node["zyx"] for node in graph["nodes"]], dtype=np.float64)
    center_yx = coords[:, 1:].mean(axis=0)

    endpoints = graph["endpoints"] or [node["id"] for node in graph["nodes"]]
    endpoint_candidates = []
    node_candidates = []

    for node in graph["nodes"]:
        z, y, x = node["zyx"]
        if branch_labels[z, y, x] != trachea_branch:
            continue
        node_candidates.append(node["id"])
        if node["id"] in endpoints:
            endpoint_candidates.append(node["id"])

    candidates = endpoint_candidates or node_candidates
    if not candidates:
        branch_coords = np.argwhere(branch_labels == trachea_branch)
        if len(branch_coords) == 0:
            return None, {
                "trachea_branch": int(trachea_branch),
                "parsed_branch_count": int(num_branches),
                "candidate_count": 0,
            }

        branch_center = branch_coords.mean(axis=0)
        candidates = [min(
            node_by_id,
            key=lambda node_id: float(np.linalg.norm(np.array(node_by_id[node_id]["zyx"]) - branch_center)),
        )]

    def score(node_id):
        z, y, x = node_by_id[node_id]["zyx"]
        center_distance = float(np.linalg.norm(np.array([y, x], dtype=np.float64) - center_yx))
        if root_end == "min-z":
            return (z, center_distance)
        if root_end == "max-z":
            return (-z, center_distance)
        raise ValueError(f"Unsupported trachea root end: {root_end}")

    root_id = min(candidates, key=score)
    return root_id, {
        "trachea_branch": int(trachea_branch),
        "parsed_branch_count": int(num_branches),
        "candidate_count": int(len(candidates)),
        "used_endpoint_candidates": bool(endpoint_candidates),
        "trachea_root_end": root_end,
    }


def choose_root(graph, mode, manual_root=None, radius_map=None, mask=None, skeleton=None, trachea_root_end="min-z"):
    if not graph["nodes"]:
        raise RuntimeError("Cannot choose root from an empty graph.")

    candidates = graph["endpoints"] or [node["id"] for node in graph["nodes"]]
    node_by_id = {node["id"]: node for node in graph["nodes"]}
    coords = np.array([node["zyx"] for node in graph["nodes"]], dtype=np.float64)
    center_yx = coords[:, 1:].mean(axis=0)

    if manual_root is not None:
        target = np.array(manual_root, dtype=np.float64)
        return min(
            node_by_id,
            key=lambda node_id: float(np.linalg.norm(np.array(node_by_id[node_id]["zyx"]) - target)),
        ), {"manual_root_zyx": [int(v) for v in manual_root]}

    if mode == "timi-trachea":
        if mask is None or skeleton is None:
            raise ValueError("timi-trachea root mode requires mask and skeleton.")
        root_id, root_info = choose_timi_trachea_root(
            graph,
            mask=mask,
            skeleton=skeleton,
            root_end=trachea_root_end,
        )
        if root_id is not None:
            return root_id, root_info
        print("[WARN] TIMI trachea root detection failed; falling back to center-min-z.")
        mode = "center-min-z"

    def endpoint_score(node_id):
        z, y, x = node_by_id[node_id]["zyx"]
        center_distance = float(np.linalg.norm(np.array([y, x], dtype=np.float64) - center_yx))

        if mode == "largest-radius":
            if radius_map is None:
                raise ValueError("largest-radius root mode requires a radius map.")
            radius = float(radius_map[z, y, x])
            return (-radius, center_distance)
        if mode == "min-z":
            return (z, center_distance)
        if mode == "max-z":
            return (-z, center_distance)
        if mode == "center-min-z":
            return (center_distance, z)
        if mode == "center-max-z":
            return (center_distance, -z)

        raise ValueError(f"Unsupported root mode: {mode}")

    return min(candidates, key=endpoint_score), {}


def dijkstra(adjacency, start):
    distances = {node_id: float("inf") for node_id in adjacency}
    previous = {node_id: None for node_id in adjacency}
    distances[start] = 0.0
    heap = [(0.0, start)]

    while heap:
        distance, node_id = heapq.heappop(heap)

        if distance > distances[node_id]:
            continue

        for neighbor, weight in adjacency[node_id]:
            new_distance = distance + weight
            if new_distance >= distances[neighbor]:
                continue

            distances[neighbor] = new_distance
            previous[neighbor] = node_id
            heapq.heappush(heap, (new_distance, neighbor))

    return distances, previous


def assign_airway_generations(graph, root_id):
    adjacency = build_adjacency(graph)
    node_by_id = {node["id"]: node for node in graph["nodes"]}
    generations = {node_id: None for node_id in adjacency}
    generations[root_id] = 0
    queue = [(root_id, None)]

    while queue:
        current, previous = queue.pop(0)
        current_generation = generations[current]

        for neighbor, _ in adjacency[current]:
            if neighbor == previous or generations[neighbor] is not None:
                continue

            increment = 1 if int(node_by_id[current].get("degree", 0)) >= 3 else 0
            generations[neighbor] = int(current_generation + increment)
            queue.append((neighbor, current))

    return generations


def reconstruct_path(previous, start, target):
    if target == start:
        return [start]

    path = []
    current = target

    while current is not None:
        path.append(current)
        if current == start:
            path.reverse()
            return path
        current = previous[current]

    return []


def choose_target_nodes(
    graph,
    root_id,
    distances,
    target_node=None,
    target_zyx=None,
    mode="farthest-endpoint",
    generations=None,
    target_generation=6,
):
    node_by_id = {node["id"]: node for node in graph["nodes"]}

    if target_node is not None:
        if target_node not in node_by_id:
            raise ValueError(f"Target node {target_node} is not in graph.")
        return [target_node]

    if target_zyx is not None:
        target = np.array(target_zyx, dtype=np.float64)
        nearest = min(
            node_by_id,
            key=lambda node_id: float(np.linalg.norm(np.array(node_by_id[node_id]["zyx"]) - target)),
        )
        return [nearest]

    endpoints = [node_id for node_id in graph["endpoints"] if node_id != root_id]

    if not endpoints:
        return []

    reachable = [node_id for node_id in endpoints if np.isfinite(distances[node_id])]

    if mode == "all-endpoints":
        return reachable

    if mode == "farthest-endpoint":
        return [max(reachable, key=lambda node_id: distances[node_id])] if reachable else []

    if mode == "generation":
        if generations is None:
            raise ValueError("generation target mode requires computed airway generations.")

        reachable_by_generation = [
            node_id
            for node_id in reachable
            if generations.get(node_id) is not None
        ]
        if not reachable_by_generation:
            return []

        at_or_beyond_target = [
            node_id
            for node_id in reachable_by_generation
            if generations[node_id] >= target_generation
        ]
        if at_or_beyond_target:
            selected_generation = min(generations[node_id] for node_id in at_or_beyond_target)
            candidates = [
                node_id
                for node_id in at_or_beyond_target
                if generations[node_id] == selected_generation
            ]
            return [max(candidates, key=lambda node_id: distances[node_id])]

        deepest_generation = max(generations[node_id] for node_id in reachable_by_generation)
        candidates = [
            node_id
            for node_id in reachable_by_generation
            if generations[node_id] == deepest_generation
        ]
        print(
            "[WARN] No endpoint reached target generation "
            f"{target_generation}; using deepest available generation {deepest_generation}."
        )
        return [max(candidates, key=lambda node_id: distances[node_id])]

    raise ValueError(f"Unsupported target mode: {mode}")


def path_to_record(
    path_id,
    path_nodes,
    graph,
    root_id,
    distances,
    generations=None,
    requested_target_zyx=None,
    effective_target_mode=None,
):
    node_by_id = {node["id"]: node for node in graph["nodes"]}
    target_id = path_nodes[-1] if path_nodes else None
    selected_target_zyx = node_by_id[target_id]["zyx"] if target_id is not None else None
    path_generations = [
        generations.get(node_id)
        for node_id in path_nodes
        if generations is not None and generations.get(node_id) is not None
    ]

    record = {
        "path_id": path_id,
        "root_node": int(root_id),
        "target_node": int(target_id) if target_id is not None else None,
        "selected_target_node": int(target_id) if target_id is not None else None,
        "selected_target_zyx": selected_target_zyx,
        "length_voxels": float(distances[target_id]) if target_id is not None else 0.0,
        "node_ids": [int(node_id) for node_id in path_nodes],
        "coordinates_zyx": [node_by_id[node_id]["zyx"] for node_id in path_nodes],
    }
    if effective_target_mode is not None:
        record["effective_target_mode"] = effective_target_mode
    if requested_target_zyx is not None:
        requested = [int(value) for value in requested_target_zyx]
        record["requested_target_zyx"] = requested
        record["target_distance_voxels"] = (
            float(np.linalg.norm(np.asarray(selected_target_zyx, dtype=np.float64) - np.asarray(requested, dtype=np.float64)))
            if selected_target_zyx is not None
            else None
        )
    if generations is not None:
        record["target_generation"] = (
            int(generations[target_id])
            if target_id is not None and generations.get(target_id) is not None
            else None
        )
        record["max_generation"] = int(max(path_generations)) if path_generations else None
        record["node_generations"] = [
            int(generations[node_id]) if generations.get(node_id) is not None else None
            for node_id in path_nodes
        ]
    return record


def main():
    parser = argparse.ArgumentParser(
        description="Skeletonize a predicted airway lumen mask and plan centerline paths."
    )
    parser.add_argument("--input", required=True, help="Predicted lumen mask/probability volume: .npy, .nii, or .nii.gz")
    parser.add_argument("--output-dir", default="datasets/centerlines", help="Output folder")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for probability volumes")
    parser.add_argument(
        "--low-threshold",
        type=float,
        default=None,
        help="Optional lower connected threshold for weak distal airway extraction",
    )
    parser.add_argument("--channel", type=int, default=0, help="Channel to read if input has channels")
    parser.add_argument("--keep-largest", action="store_true", help="Keep only largest connected component")
    parser.add_argument("--closing-radius", type=int, default=1, help="Binary closing iterations before skeletonization")
    parser.add_argument("--prune-length", type=float, default=8.0, help="Remove terminal branches shorter than this voxel length")
    parser.add_argument(
        "--preserve-generations",
        type=int,
        default=6,
        help=(
            "Protect skeleton graph nodes up to this airway generation from terminal-branch pruning; "
            "use 0 to disable generation-aware preservation"
        ),
    )
    parser.add_argument(
        "--root-mode",
        choices=["min-z", "max-z", "center-min-z", "center-max-z", "largest-radius", "timi-trachea"],
        default="timi-trachea",
        help="Automatic root selection heuristic",
    )
    parser.add_argument(
        "--trachea-root-end",
        choices=["min-z", "max-z"],
        default="min-z",
        help="Which end of the parsed trachea branch to use as root for timi-trachea mode",
    )
    parser.add_argument("--root-zyx", type=parse_zyx, default=None, help="Manual root coordinate as z,y,x")
    parser.add_argument("--target-node", type=int, default=None, help="Plan to a specific graph node id")
    parser.add_argument("--target-zyx", type=parse_zyx, default=None, help="Plan to nearest node to z,y,x")
    parser.add_argument(
        "--target-mode",
        choices=["farthest-endpoint", "all-endpoints", "generation"],
        default="farthest-endpoint",
        help="Target endpoint selection when no explicit target is given",
    )
    parser.add_argument(
        "--target-generation",
        type=int,
        default=6,
        help="Airway generation target when --target-mode generation",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    volume, reference_img = load_volume(input_path, channel=args.channel)

    mask = hysteresis_lumen_mask(volume, args.threshold, low_threshold=args.low_threshold)
    mask = postprocess_mask(
        mask,
        keep_largest=args.keep_largest,
        closing_radius=args.closing_radius,
    )

    skeleton = skeletonize_lumen(mask)
    graph = build_voxel_graph(skeleton)
    radius_map = ndi.distance_transform_edt(mask)
    protected_node_ids = set()
    pre_prune_root_info = {}

    if graph["nodes"] and args.preserve_generations > 0:
        try:
            pre_prune_root_id, pre_prune_root_info = choose_root(
                graph,
                mode=args.root_mode,
                manual_root=args.root_zyx,
                radius_map=radius_map,
                mask=mask,
                skeleton=skeleton,
                trachea_root_end=args.trachea_root_end,
            )
            pre_prune_generations = assign_airway_generations(graph, pre_prune_root_id)
            protected_node_ids = {
                node_id
                for node_id, generation in pre_prune_generations.items()
                if generation is not None and generation <= args.preserve_generations
            }
        except Exception as exc:
            print("[WARN] Generation-aware branch preservation failed:", exc)
            protected_node_ids = set()

    pruned_graph, pruned_branches = prune_short_terminal_branches(
        graph,
        args.prune_length,
        protected_node_ids=protected_node_ids,
    )
    pruned_skeleton = graph_to_skeleton(pruned_graph, skeleton.shape)

    effective_target_mode = args.target_mode
    if args.target_node is not None:
        effective_target_mode = "target-node"
    elif args.target_zyx is not None:
        effective_target_mode = "target-zyx"

    if pruned_graph["nodes"]:
        root_id, root_info = choose_root(
            pruned_graph,
            mode=args.root_mode,
            manual_root=args.root_zyx,
            radius_map=radius_map,
            mask=mask,
            skeleton=pruned_skeleton,
            trachea_root_end=args.trachea_root_end,
        )
        adjacency = build_adjacency(pruned_graph)
        distances, previous = dijkstra(adjacency, root_id)
        generations = assign_airway_generations(pruned_graph, root_id)
        for node in pruned_graph["nodes"]:
            generation = generations.get(node["id"])
            node["generation"] = int(generation) if generation is not None else None
        target_nodes = choose_target_nodes(
            graph=pruned_graph,
            root_id=root_id,
            distances=distances,
            target_node=args.target_node,
            target_zyx=args.target_zyx,
            mode=args.target_mode,
            generations=generations,
            target_generation=args.target_generation,
        )
        paths = []
        for index, target_id in enumerate(target_nodes):
            node_path = reconstruct_path(previous, root_id, target_id)
            if not node_path:
                continue
            paths.append(
                path_to_record(
                    f"path_{index:03d}",
                    node_path,
                    pruned_graph,
                    root_id,
                    distances,
                    generations=generations,
                    requested_target_zyx=args.target_zyx,
                    effective_target_mode=effective_target_mode,
                )
            )
    else:
        root_id = None
        root_info = {}
        generations = {}
        paths = []

    stem = input_path.name
    if stem.endswith(".nii.gz"):
        stem = stem[:-7]
    else:
        stem = input_path.stem

    skeleton_npy = output_dir / f"{stem}_centerline.npy"
    pruned_skeleton_npy = output_dir / f"{stem}_centerline_pruned.npy"
    graph_json = output_dir / f"{stem}_centerline_graph.json"
    pruned_graph_json = output_dir / f"{stem}_centerline_pruned_graph.json"
    paths_json = output_dir / f"{stem}_paths.json"
    skeleton_nii = output_dir / f"{stem}_centerline.nii.gz"
    pruned_skeleton_nii = output_dir / f"{stem}_centerline_pruned.nii.gz"

    np.save(skeleton_npy, skeleton.astype(np.uint8))
    np.save(pruned_skeleton_npy, pruned_skeleton.astype(np.uint8))

    with open(graph_json, "w") as f:
        json.dump({
            "input": str(input_path),
            "threshold": args.threshold,
            "low_threshold": args.low_threshold,
            "channel": args.channel,
            "skeleton_voxels": int(skeleton.sum()),
            "node_count": len(graph["nodes"]),
            "edge_count": len(graph["edges"]),
            "endpoint_count": len(graph["endpoints"]),
            "branchpoint_count": len(graph["branchpoints"]),
            "graph": graph,
        }, f, indent=2)

    with open(pruned_graph_json, "w") as f:
        json.dump({
            "input": str(input_path),
            "threshold": args.threshold,
            "low_threshold": args.low_threshold,
            "channel": args.channel,
            "prune_length": args.prune_length,
            "skeleton_voxels": int(pruned_skeleton.sum()),
            "node_count": len(pruned_graph["nodes"]),
            "edge_count": len(pruned_graph["edges"]),
            "endpoint_count": len(pruned_graph["endpoints"]),
            "branchpoint_count": len(pruned_graph["branchpoints"]),
            "root_node": int(root_id) if root_id is not None else None,
            "root_mode": args.root_mode,
            "root_zyx": (
                pruned_graph["nodes"][root_id]["zyx"]
                if root_id is not None and root_id < len(pruned_graph["nodes"])
                else None
            ),
            "root_info": root_info,
            "generation_method": "branchpoint_count_from_root",
            "max_generation": (
                int(max(value for value in generations.values() if value is not None))
                if generations
                else 0
            ),
            "preserve_generations": int(args.preserve_generations),
            "pre_prune_root_info": pre_prune_root_info,
            "generation_protected_node_count": int(len(protected_node_ids)),
            "removed_terminal_branches": pruned_branches,
            "graph": pruned_graph,
        }, f, indent=2)

    with open(paths_json, "w") as f:
        json.dump({
            "input": str(input_path),
            "threshold": args.threshold,
            "low_threshold": args.low_threshold,
            "root_node": int(root_id) if root_id is not None else None,
            "root_mode": args.root_mode,
            "root_info": root_info,
            "target_mode": effective_target_mode,
            "effective_target_mode": effective_target_mode,
            "fallback_target_mode": args.target_mode,
            "requested_target_node": int(args.target_node) if args.target_node is not None else None,
            "requested_target_zyx": [int(value) for value in args.target_zyx] if args.target_zyx is not None else None,
            "target_generation": int(args.target_generation) if effective_target_mode == "generation" else None,
            "preserve_generations": int(args.preserve_generations),
            "path_count": len(paths),
            "paths": paths,
        }, f, indent=2)

    save_nifti_like(skeleton, reference_img, skeleton_nii)
    save_nifti_like(pruned_skeleton, reference_img, pruned_skeleton_nii)

    print("Done")
    print("  input:", input_path)
    print("  centerline npy:", skeleton_npy)
    print("  pruned centerline npy:", pruned_skeleton_npy)
    if reference_img is not None:
        print("  centerline nifti:", skeleton_nii)
        print("  pruned centerline nifti:", pruned_skeleton_nii)
    print("  graph json:", graph_json)
    print("  pruned graph json:", pruned_graph_json)
    print("  paths json:", paths_json)
    print("  threshold:", args.threshold)
    print("  low threshold:", args.low_threshold)
    print("  skeleton voxels:", int(skeleton.sum()))
    print("  graph nodes:", len(graph["nodes"]))
    print("  graph edges:", len(graph["edges"]))
    print("  endpoints:", len(graph["endpoints"]))
    print("  branchpoints:", len(graph["branchpoints"]))
    print("  pruned skeleton voxels:", int(pruned_skeleton.sum()))
    print("  pruned graph nodes:", len(pruned_graph["nodes"]))
    print("  pruned graph edges:", len(pruned_graph["edges"]))
    print("  pruned endpoints:", len(pruned_graph["endpoints"]))
    print("  pruned branchpoints:", len(pruned_graph["branchpoints"]))
    print("  preserve generations:", args.preserve_generations)
    print("  generation protected nodes:", len(protected_node_ids))
    print("  root node:", root_id)
    if generations:
        print("  max generation:", max(value for value in generations.values() if value is not None))
    print("  planned paths:", len(paths))
    if paths:
        print("  first path length:", "%.2f" % paths[0]["length_voxels"])
        if paths[0].get("target_generation") is not None:
            print("  first path target generation:", paths[0]["target_generation"])


if __name__ == "__main__":
    main()
