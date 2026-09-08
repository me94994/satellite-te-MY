"""Safe, exact-equivalent Gurobi attribution benchmark.

The default workload is a deterministic reduced diagnostic because the paper
dataset is not distributed in this worktree. Results must therefore retain the
DIAGNOSTIC_CODE_NATIVE evidence label.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import platform
import random
import statistics
import time
from typing import Dict, Iterable, List, Tuple

import networkx as nx

from .schemas import BenchmarkInstance, SafetyLimits, enforce_safety, estimate_lp


LEVELS = {
    "L0": {"dense": True, "presolve": 0, "active_subgraph": False},
    "L1": {"dense": True, "presolve": -1, "active_subgraph": False},
    "L2": {"dense": False, "presolve": 0, "active_subgraph": False},
    "L3": {"dense": False, "presolve": -1, "active_subgraph": False},
    "L4": {"dense": False, "presolve": -1, "active_subgraph": True},
}


def diagnostic_instance(nodes: int = 16, active_flows: int = 20, k: int = 5, seed: int = 42) -> BenchmarkInstance:
    """Build a deterministic reduced topology with genuine unique paths."""
    if nodes < 6:
        raise ValueError("nodes must be at least 6")
    graph = nx.DiGraph()
    for u in range(nodes):
        for delta in (1, 2, -1, -2):
            graph.add_edge(u, (u + delta) % nodes)
    capacities = {edge: 200.0 for edge in graph.edges()}
    all_pairs = [(s, t) for s in range(nodes) for t in range(nodes) if s != t]
    paths = {}
    for pair in all_pairs:
        generated = nx.shortest_simple_paths(graph, pair[0], pair[1])
        paths[pair] = tuple(tuple(next(generated)) for _ in range(k))
    rng = random.Random(seed)
    active = rng.sample(all_pairs, min(active_flows, len(all_pairs)))
    demands = {pair: float(5 + rng.randrange(16)) for pair in active}
    return BenchmarkInstance(
        snapshot_id=f"diagnostic-n{nodes}-seed{seed}",
        nodes=tuple(range(nodes)),
        physical_edges=tuple(sorted(graph.edges())),
        capacities=capacities,
        demands=demands,
        candidate_paths=paths,
    )


def iridium_shared_instance(active_flows: int = 20, k: int = 5, seed: int = 42) -> BenchmarkInstance:
    """Create the 66-satellite/66-user shape expected by the shipped checkpoint.

    This is a current-hardware latency diagnostic, not a reconstruction of the
    missing paper dataset. Both Gurobi and SaTE consume this same instance.
    """
    satellites = 66
    graph = nx.DiGraph()
    graph.add_nodes_from(range(satellites * 2))
    satellite_edges = []
    for u in range(satellites):
        for delta in (1, 2, -1, -2):
            edge = (u, (u + delta) % satellites)
            graph.add_edge(*edge)
            satellite_edges.append(edge)
        graph.add_edge(u, u + satellites)
        graph.add_edge(u + satellites, u)
    capacities = {
        edge: (25.0 if edge[0] < satellites and edge[1] < satellites else 100.0)
        for edge in graph.edges()
    }
    rng = random.Random(seed)
    user_pairs = [(s + satellites, t + satellites) for s in range(satellites) for t in range(satellites) if s != t]
    active = rng.sample(user_pairs, active_flows)
    paths = {}
    demands = {}
    for pair in active:
        generated = nx.shortest_simple_paths(graph, *pair)
        paths[pair] = tuple(tuple(next(generated)) for _ in range(k))
        demands[pair] = float(5 + rng.randrange(16))
    return BenchmarkInstance(
        snapshot_id=f"diagnostic-iridium-shape-seed{seed}",
        nodes=tuple(graph.nodes()), physical_edges=tuple(sorted(graph.edges())),
        capacities=capacities, demands=demands, candidate_paths=paths,
        evidence_label="DIAGNOSTIC_SYNTHETIC_IRIDIUM_SHAPE",
    )


def _load_gurobi():
    try:
        import gurobipy as gp
        return gp
    except Exception as exc:  # pragma: no cover - environment-dependent
        raise RuntimeError(f"GUROBI_NOT_AVAILABLE: {exc}") from exc


def build_model(instance: BenchmarkInstance, level: str, method: int = -1, threads: int = 1):
    """Build one LP level and return model plus timing metadata."""
    if level not in LEVELS:
        raise ValueError(f"unknown level: {level}")
    gp = _load_gurobi()
    cfg = LEVELS[level]
    parse_start = time.perf_counter()
    pairs = instance.all_pairs if cfg["dense"] else instance.active_pairs
    edges = instance.active_path_edges if cfg["active_subgraph"] else instance.physical_edges
    path_rows = [(pair, i, path) for pair in pairs for i, path in enumerate(instance.candidate_paths[pair])]
    edge_to_vars = {edge: [] for edge in edges}
    parse_s = time.perf_counter() - parse_start

    build_start = time.perf_counter()
    model = gp.Model(f"speedup-{level}")
    model.Params.OutputFlag = 0
    model.Params.Threads = threads
    model.Params.Presolve = cfg["presolve"]
    model.Params.Method = method
    variables = {}
    for pair, index, path in path_rows:
        variable = model.addVar(lb=0.0, name=f"x_{pair[0]}_{pair[1]}_{index}")
        variables[(pair, index)] = variable
        for edge in zip(path[:-1], path[1:]):
            if edge in edge_to_vars:
                edge_to_vars[edge].append(variable)
    model.update()
    for pair in pairs:
        model.addConstr(
            gp.quicksum(variables[(pair, i)] for i in range(len(instance.candidate_paths[pair])))
            <= instance.demands.get(pair, 0.0),
            name=f"d_{pair[0]}_{pair[1]}",
        )
    for edge in edges:
        model.addConstr(gp.quicksum(edge_to_vars[edge]) <= instance.capacities[edge], name=f"c_{edge[0]}_{edge[1]}")
    model.setObjective(gp.quicksum(variables.values()), gp.GRB.MAXIMIZE)
    model.update()
    build_s = time.perf_counter() - build_start
    return model, variables, {"parse_s": parse_s, "build_s": build_s}


def solve_level(instance: BenchmarkInstance, level: str, limits: SafetyLimits, method: int = -1) -> Dict[str, object]:
    estimate = estimate_lp(instance, level)
    try:
        enforce_safety(estimate, limits)
    except ResourceWarning:
        return {"level": level, "status": "SKIPPED_RESOURCE_BOUND", **estimate}
    try:
        model, variables, timing = build_model(instance, level, method=method)
    except Exception as exc:
        return {"level": level, "status": str(exc).splitlines()[0], **estimate}
    before = {"num_vars": model.NumVars, "num_constraints": model.NumConstrs, "num_nonzeros": model.NumNZs}
    presolved = {"presolved_vars": "NOT_AVAILABLE", "presolved_constraints": "NOT_AVAILABLE", "presolved_nonzeros": "NOT_AVAILABLE"}
    if LEVELS[level]["presolve"] != 0 and hasattr(model, "presolve"):
        try:
            pm = model.presolve()
            presolved = {"presolved_vars": pm.NumVars, "presolved_constraints": pm.NumConstrs, "presolved_nonzeros": pm.NumNZs}
            pm.dispose()
        except Exception:
            pass
    start = time.perf_counter()
    model.optimize()
    optimize_wall_s = time.perf_counter() - start
    gp = _load_gurobi()
    ok = model.Status == gp.GRB.OPTIMAL
    result = {
        "snapshot_id": instance.snapshot_id,
        "evidence_label": instance.evidence_label,
        "level": level,
        "status": "MEASURED" if ok else f"GUROBI_STATUS_{model.Status}",
        **instance.hashes(), **estimate, **before, **presolved, **timing,
        "optimize_wall_s": optimize_wall_s,
        "gurobi_runtime_s": model.Runtime,
        "total_s": timing["parse_s"] + timing["build_s"] + optimize_wall_s,
        "objective": model.ObjVal if ok else None,
        "num_active_flows": len(instance.active_pairs),
        "num_all_flows": len(instance.all_pairs),
        "num_paths": len(variables),
        "physical_edges": len(instance.physical_edges),
        "active_path_edges": len(instance.active_path_edges),
        "threads": 1,
        "method": method,
    }
    model.dispose()
    return result


def run(args: argparse.Namespace) -> List[Dict[str, object]]:
    instance = diagnostic_instance(args.nodes, args.active_flows, args.k, args.seed)
    limits = SafetyLimits(args.max_vars, args.max_constraints, args.max_estimated_memory_gb)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for repeat in range(args.repeats):
        for level in LEVELS:
            row = solve_level(instance, level, limits)
            row["repeat"] = repeat
            rows.append(row)
            # Incremental JSONL is the resume/audit artifact; each row is independently valid.
            with output.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes", type=int, default=16)
    parser.add_argument("--active-flows", type=int, default=20)
    parser.add_argument("--k", type=int, default=5, choices=(1, 3, 5, 10))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--max-vars", type=int, default=2000)
    parser.add_argument("--max-constraints", type=int, default=2000)
    parser.add_argument("--max-estimated-memory-gb", type=float, default=0.25)
    parser.add_argument("--output", default="output/speedup_attribution/solver_results.jsonl")
    args = parser.parse_args()
    rows = run(args)
    print(json.dumps({"rows": len(rows), "output": args.output, "status_counts": {s: sum(r["status"] == s for r in rows) for s in sorted({r["status"] for r in rows})}}, sort_keys=True))


if __name__ == "__main__":
    main()
