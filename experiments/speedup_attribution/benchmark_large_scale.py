"""License-safe paired K=5/K=10 large-scale solver diagnostics."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import networkx as nx

from .benchmark_solver import solve_level
from .schemas import (
    BenchmarkInstance,
    SafetyLimits,
    assert_k5_prefix_of_k10,
    estimate_lp,
)


TOPOLOGY_NODES = (16, 32, 48, 66, 96, 128, 160, 192)
TRAFFIC_FLOWS = (10, 20, 40, 80, 120, 160, 180)
DENSE_NODES = (6, 8, 10, 12, 14)
STRESS_SCALES = (1.0, 1.5, 2.0, 4.0)
METHODS = {"default": -1, "dual_simplex": 1, "barrier": 2}


def _pair_from_index(index: int, nodes: int) -> Tuple[int, int]:
    """Map an integer to an ordered non-self satellite pair without N^2 storage."""
    source, target = divmod(index, nodes - 1)
    if target >= source:
        target += 1
    return source, target


def _satellite_graph(nodes: int) -> nx.DiGraph:
    """Create a deterministic degree-four topology-shaped diagnostic graph."""
    if nodes < 6:
        raise ValueError("nodes must be at least 6")
    width = max(2, round(math.sqrt(nodes)))
    graph = nx.DiGraph()
    graph.add_nodes_from(range(nodes))
    for source in range(nodes):
        for delta in (1, -1, width, -width):
            target = (source + delta) % nodes
            if source != target:
                graph.add_edge(source, target)
    return graph


def constellation_instance(
    nodes: int,
    active_flows: int,
    k: int = 10,
    seed: int = 42,
    demand_scale: float = 1.0,
) -> BenchmarkInstance:
    """Build a sparse CODE_CONFIG constellation-shaped instance with unique paths."""
    if k not in (5, 10):
        raise ValueError("formal paired diagnostics support only K=5 or K=10")
    graph = _satellite_graph(nodes)
    rng = random.Random(seed)
    possible = nodes * (nodes - 1)
    flow_count = min(active_flows, possible)
    satellite_pairs = [_pair_from_index(i, nodes) for i in rng.sample(range(possible), flow_count)]

    # SaTE's Starlink representation includes one user node per satellite.
    physical_edges = list(graph.edges())
    for satellite in range(nodes):
        physical_edges.extend(((satellite, satellite + nodes), (satellite + nodes, satellite)))
    capacities = {
        edge: 200.0 if edge[0] < nodes and edge[1] < nodes else 800.0
        for edge in physical_edges
    }
    paths: Dict[Tuple[int, int], Tuple[Tuple[int, ...], ...]] = {}
    demands = {}
    for satellite_source, satellite_target in satellite_pairs:
        generated = nx.shortest_simple_paths(graph, satellite_source, satellite_target)
        pair = (satellite_source + nodes, satellite_target + nodes)
        middle = [tuple(next(generated)) for _ in range(k)]
        paths[pair] = tuple((pair[0], *path, pair[1]) for path in middle)
        demands[pair] = float((5 + rng.randrange(16)) * demand_scale)
    return BenchmarkInstance(
        snapshot_id=f"diagnostic-scale-n{nodes}-f{flow_count}-seed{seed}-d{demand_scale:g}",
        nodes=tuple(range(nodes * 2)),
        physical_edges=tuple(sorted(set(physical_edges))),
        capacities=capacities,
        demands=demands,
        candidate_paths=paths,
        evidence_label="CONTROLLED_STRESS_DIAGNOSTIC" if demand_scale > 1 else "DIAGNOSTIC_SCALE_CODE_CONFIG",
    )


def equal_split_proxy(instance: BenchmarkInstance) -> Dict[str, float]:
    """Compute congestion selection solely from inputs, before any solver call."""
    loads = {edge: 0.0 for edge in instance.physical_edges}
    for pair in instance.active_pairs:
        paths = instance.candidate_paths[pair]
        share = instance.demands[pair] / len(paths)
        for path in paths:
            for edge in zip(path[:-1], path[1:]):
                loads[edge] += share
    utilizations = [loads[edge] / instance.capacities[edge] for edge in instance.physical_edges]
    binding = sum(value >= 1.0 for value in utilizations)
    return {
        "equal_split_max_utilization": max(utilizations, default=0.0),
        "equal_split_binding_edges": binding,
        "equal_split_binding_edge_ratio": binding / len(utilizations) if utilizations else 0.0,
    }


def controlled_stress_instance(nodes: int, active_flows: int, seed: int = 42) -> Tuple[BenchmarkInstance, Dict[str, float]]:
    """Select the first preregistered input scale with proxy congestion."""
    selected = None
    selected_scale = None
    proxy = None
    for scale in STRESS_SCALES:
        candidate = constellation_instance(nodes, active_flows, 10, seed, scale)
        candidate_proxy = equal_split_proxy(candidate)
        selected, selected_scale, proxy = candidate, scale, candidate_proxy
        if candidate_proxy["equal_split_binding_edges"] > 0:
            break
    assert selected is not None and selected_scale is not None and proxy is not None
    # Regime C is an input-side controlled-stress protocol even if scale 1.0 is already binding.
    selected = BenchmarkInstance(
        selected.snapshot_id,
        selected.nodes,
        selected.physical_edges,
        dict(selected.capacities),
        dict(selected.demands),
        dict(selected.candidate_paths),
        "CONTROLLED_STRESS_DIAGNOSTIC",
    )
    proxy = {**proxy, "demand_scale": selected_scale}
    return selected, proxy


def family_points(family: str, nodes: Sequence[int] = TOPOLOGY_NODES) -> List[Tuple[int, int]]:
    """Keep topology, traffic, and coupled axes explicitly separate."""
    if family == "topology":
        return [(node_count, 40) for node_count in nodes]
    if family == "traffic":
        return [(192, flow_count) for flow_count in TRAFFIC_FLOWS]
    if family == "coupled":
        return [(node_count, min(node_count, 180)) for node_count in nodes]
    raise ValueError(f"unknown scaling family: {family}")


def license_safe_instance(
    nodes: int,
    requested_flows: int,
    regime: str,
    seed: int,
    limits: SafetyLimits,
) -> Tuple[BenchmarkInstance, Dict[str, float], int]:
    """Reduce only flow count before construction when the restricted limit requires it."""
    flow_count = min(requested_flows, int(limits.max_vars // 10))
    while flow_count > 0:
        if regime == "congested":
            instance, proxy = controlled_stress_instance(nodes, flow_count, seed)
        else:
            instance = constellation_instance(nodes, flow_count, 10, seed, 1.0)
            proxy = {**equal_split_proxy(instance), "demand_scale": 1.0}
        estimate = estimate_lp(instance, "L4")
        if estimate["estimated_constraints"] <= limits.max_constraints:
            return instance, proxy, flow_count
        flow_count -= 1
    raise ResourceWarning("SKIPPED_RESOURCE_BOUND")


def paired_problem_hash(instance_k10: BenchmarkInstance) -> str:
    """Bind shared inputs and the full ten-path pool into one auditable pair identity."""
    payload = {**instance_k10.shared_input_hashes(), "k10_path_hash": instance_k10.hashes()["path_hash"]}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _append_jsonl(path: Path, row: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def _resume_keys(path: Path) -> set:
    if not path.is_file():
        return set()
    keys = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            keys.add((row.get("family"), row.get("regime"), row.get("topology_nodes"), row.get("requested_active_flows"), row.get("k"), row.get("baseline"), row.get("repeat")))
    return keys


def run_sparse(
    output: Path,
    families: Sequence[str],
    regimes: Sequence[str],
    nodes: Sequence[int],
    repeats: int,
    seed: int,
    limits: SafetyLimits,
    resume: bool,
) -> List[Dict[str, object]]:
    """Run paired sparse scaling and retain every point, including failures/skips."""
    existing = _resume_keys(output) if resume else set()
    rows: List[Dict[str, object]] = []
    for family in families:
        for node_count, requested_flows in family_points(family, nodes):
            for regime in regimes:
                try:
                    k10, proxy, actual_flows = license_safe_instance(node_count, requested_flows, regime, seed, limits)
                except ResourceWarning:
                    row = {"family": family, "regime": regime, "topology_nodes": node_count, "requested_active_flows": requested_flows, "status": "SKIPPED_RESOURCE_BOUND"}
                    _append_jsonl(output, row)
                    rows.append(row)
                    continue
                k5 = k10.paths_for_k(5)
                assert_k5_prefix_of_k10(k5, k10)
                pair_hash = paired_problem_hash(k10)
                for k, instance in ((5, k5), (10, k10)):
                    baselines = (("sparse_presolve_cold", "L3", -1),)
                    for baseline, level, method in baselines:
                        for repeat in range(repeats):
                            key = (family, regime, node_count, requested_flows, k, baseline, repeat)
                            if key in existing:
                                continue
                            result = solve_level(instance, level, limits, method=method)
                            row = {
                                **result,
                                **proxy,
                                "family": family,
                                "regime": regime,
                                "topology_nodes": node_count,
                                "requested_active_flows": requested_flows,
                                "actual_active_flows": actual_flows,
                                "k": k,
                                "baseline": baseline,
                                "repeat": repeat,
                                "paired_problem_hash": pair_hash,
                                "k10_path_pool_hash": k10.hashes()["path_hash"],
                            }
                            _append_jsonl(output, row)
                            rows.append(row)
                # Calibrate commercial methods on K=10 without using test outcomes other than time.
                calibration = []
                for name, method in METHODS.items():
                    result = solve_level(k10, "L4", limits, method=method)
                    calibration.append((name, method, result))
                measured = [item for item in calibration if item[2].get("status") == "MEASURED"]
                if measured:
                    selected_name, selected_method, _ = min(measured, key=lambda item: item[2]["total_s"])
                    for repeat in range(repeats):
                        key = (family, regime, node_count, requested_flows, 10, "best_cold_commercial", repeat)
                        if key in existing:
                            continue
                        result = solve_level(k10, "L4", limits, method=selected_method)
                        row = {
                            **result,
                            **proxy,
                            "family": family,
                            "regime": regime,
                            "topology_nodes": node_count,
                            "requested_active_flows": requested_flows,
                            "actual_active_flows": actual_flows,
                            "k": 10,
                            "baseline": "best_cold_commercial",
                            "selected_method": selected_name,
                            "repeat": repeat,
                            "paired_problem_hash": pair_hash,
                            "k10_path_pool_hash": k10.hashes()["path_hash"],
                        }
                        _append_jsonl(output, row)
                        rows.append(row)
    return rows


def run_dense(output: Path, repeats: int, seed: int, limits: SafetyLimits) -> List[Dict[str, object]]:
    """Run dense K=10 only after an analytical resource-bound check."""
    rows = []
    for node_count in DENSE_NODES:
        instance = constellation_instance(node_count, node_count * (node_count - 1), 10, seed)
        for repeat in range(repeats):
            result = solve_level(instance, "L1", limits)
            row = {**result, "family": "dense", "regime": "uncongested", "topology_nodes": node_count, "actual_active_flows": len(instance.active_pairs), "k": 10, "repeat": repeat}
            _append_jsonl(output, row)
            rows.append(row)
    return rows


def write_pairs(rows: Iterable[Mapping[str, object]], output: Path) -> None:
    """Write measured paired medians without silently dropping unfavorable ratios."""
    groups: Dict[Tuple[object, ...], Dict[int, List[Mapping[str, object]]]] = {}
    for row in rows:
        if row.get("status") != "MEASURED" or row.get("baseline") != "sparse_presolve_cold":
            continue
        key = (row["family"], row["regime"], row["topology_nodes"], row["actual_active_flows"], row["paired_problem_hash"])
        groups.setdefault(key, {}).setdefault(int(row["k"]), []).append(row)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["family", "regime", "nodes", "active_flows", "paired_problem_hash", "k5_vars", "k10_vars", "k5_nonzeros", "k10_nonzeros", "k5_build_median_s", "k10_build_median_s", "k5_optimize_median_s", "k10_optimize_median_s", "k5_total_median_s", "k10_total_median_s", "build_slowdown", "optimize_slowdown", "total_slowdown", "k5_objective", "k10_objective"]
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for key, by_k in sorted(groups.items()):
            if 5 not in by_k or 10 not in by_k:
                continue
            k5, k10 = by_k[5], by_k[10]
            median = lambda values: statistics.median(values)
            values = {
                "family": key[0], "regime": key[1], "nodes": key[2], "active_flows": key[3], "paired_problem_hash": key[4],
                "k5_vars": k5[0]["num_vars"], "k10_vars": k10[0]["num_vars"], "k5_nonzeros": k5[0]["num_nonzeros"], "k10_nonzeros": k10[0]["num_nonzeros"],
                "k5_build_median_s": median([x["build_s"] for x in k5]), "k10_build_median_s": median([x["build_s"] for x in k10]),
                "k5_optimize_median_s": median([x["optimize_wall_s"] for x in k5]), "k10_optimize_median_s": median([x["optimize_wall_s"] for x in k10]),
                "k5_total_median_s": median([x["total_s"] for x in k5]), "k10_total_median_s": median([x["total_s"] for x in k10]),
                "k5_objective": k5[0]["objective"], "k10_objective": k10[0]["objective"],
            }
            values["build_slowdown"] = values["k10_build_median_s"] / values["k5_build_median_s"]
            values["optimize_slowdown"] = values["k10_optimize_median_s"] / values["k5_optimize_median_s"]
            values["total_slowdown"] = values["k10_total_median_s"] / values["k5_total_median_s"]
            writer.writerow(values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--families", nargs="+", choices=("topology", "traffic", "coupled"), default=["topology", "traffic", "coupled"])
    parser.add_argument("--regime", nargs="+", choices=("uncongested", "congested"), default=["uncongested", "congested"])
    parser.add_argument("--nodes", nargs="+", type=int, default=list(TOPOLOGY_NODES))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-vars", type=int, default=1900)
    parser.add_argument("--max-constraints", type=int, default=1900)
    parser.add_argument("--max-estimated-memory-gb", type=float, default=0.25)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("output/speedup_attribution/large_scale"))
    args = parser.parse_args()
    limits = SafetyLimits(args.max_vars, args.max_constraints, args.max_estimated_memory_gb)
    solver_output = args.output_dir / "solver_scaling.jsonl"
    if solver_output.exists() and not args.resume:
        solver_output.unlink()
    rows = run_sparse(solver_output, args.families, args.regime, args.nodes, args.repeats, args.seed, limits, args.resume)
    dense_output = args.output_dir / "dense_scaling.jsonl"
    if dense_output.exists() and not args.resume:
        dense_output.unlink()
    run_dense(dense_output, args.repeats, args.seed, limits)
    all_rows = [json.loads(line) for line in solver_output.read_text(encoding="utf-8").splitlines() if line.strip()]
    write_pairs(all_rows, args.output_dir / "k5_k10_pairs.csv")
    print(json.dumps({"new_rows": len(rows), "solver_output": str(solver_output), "status": "COMPLETE"}, sort_keys=True))


if __name__ == "__main__":
    main()
