"""Causal same-model and previous-basis online solver diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time
from typing import Dict, Mapping

from .benchmark_large_scale import controlled_stress_instance, paired_problem_hash
from .benchmark_solver import _load_gurobi, build_model
from .schemas import SafetyLimits, enforce_safety, estimate_lp


def _set_demands(model, demands: Mapping, factor: float) -> float:
    start = time.perf_counter()
    for pair, demand in demands.items():
        model.getConstrByName(f"d_{pair[0]}_{pair[1]}").RHS = demand * factor
    model.update()
    return time.perf_counter() - start


def benchmark_online(nodes: int, active_flows: int, repeats: int, limits: SafetyLimits) -> Dict[str, object]:
    """Compare model reuse with and without a causal previous simplex basis."""
    instance, proxy = controlled_stress_instance(nodes, active_flows)
    enforce_safety(estimate_lp(instance, "L4"), limits)
    gp = _load_gurobi()
    methods = {"same_model_reuse_no_basis": [], "previous_basis_reuse": []}
    for repeat in range(repeats):
        for name in methods:
            model, _, _ = build_model(instance, "L4", method=1, threads=1)
            # The previous demand is fixed a priori and never selected from solver output.
            _set_demands(model, instance.demands, 0.95)
            model.optimize()
            update_s = _set_demands(model, instance.demands, 1.0)
            if name == "same_model_reuse_no_basis":
                # Clear the solution/basis while retaining the already-built model matrix.
                model.reset(1)
            start = time.perf_counter()
            model.optimize()
            optimize_s = time.perf_counter() - start
            methods[name].append({
                "repeat": repeat,
                "status": "MEASURED" if model.Status == gp.GRB.OPTIMAL else f"GUROBI_STATUS_{model.Status}",
                "rhs_update_s": update_s,
                "optimize_wall_s": optimize_s,
                "total_s": update_s + optimize_s,
                "iter_count": model.IterCount,
                "objective": model.ObjVal if model.Status == gp.GRB.OPTIMAL else None,
            })
            model.dispose()
    summaries = {}
    for name, rows in methods.items():
        measured = [row for row in rows if row["status"] == "MEASURED"]
        summaries[name] = {
            "rows": rows,
            "median_rhs_update_s": statistics.median(row["rhs_update_s"] for row in measured),
            "median_optimize_s": statistics.median(row["optimize_wall_s"] for row in measured),
            "median_total_s": statistics.median(row["total_s"] for row in measured),
            "median_iter_count": statistics.median(row["iter_count"] for row in measured),
        }
    return {
        "status": "MEASURED_CONTROLLED_SEQUENTIAL_DIAGNOSTIC",
        "evidence_label": "CONTROLLED_SEQUENTIAL_DIAGNOSTIC",
        "topology_nodes": nodes,
        "active_flows": len(instance.active_pairs),
        "k": 10,
        "paired_problem_hash": paired_problem_hash(instance),
        "sequence_rule": "previous demand = 0.95 * current demand; fixed before solver calls",
        **proxy,
        "methods": summaries,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes", type=int, nargs="+", default=[66, 128, 192])
    parser.add_argument("--active-flows", type=int, default=180)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path("output/speedup_attribution/large_scale/solver_online.jsonl"))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        args.output.unlink()
    for nodes in args.nodes:
        try:
            result = benchmark_online(nodes, min(nodes, args.active_flows), args.repeats, SafetyLimits(1900, 1900, 0.25))
        except (ResourceWarning, RuntimeError) as exc:
            result = {"status": str(exc).splitlines()[0], "topology_nodes": nodes, "k": 10}
        with args.output.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(result, sort_keys=True) + "\n")
        print(json.dumps({"nodes": nodes, "status": result["status"]}, sort_keys=True))


if __name__ == "__main__":
    main()
