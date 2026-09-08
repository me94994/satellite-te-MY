"""Best-practice cold and causal online commercial-solver diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time

from .benchmark_solver import build_model, iridium_shared_instance, solve_level
from .schemas import SafetyLimits


METHODS = {"default": -1, "dual_simplex": 1, "barrier": 2}


def _median(values):
    return statistics.median(values)


def benchmark(calibration_repeats=5, formal_repeats=30, online_repeats=30):
    instance = iridium_shared_instance()
    limits = SafetyLimits(max_vars=2000, max_constraints=2000)
    calibration = {}
    for name, method in METHODS.items():
        rows = [solve_level(instance, "L4", limits, method=method) for _ in range(calibration_repeats)]
        if all(row["status"] == "MEASURED" for row in rows):
            calibration[name] = _median([row["total_s"] for row in rows])
    if not calibration:
        return {"status": "BLOCKED_GUROBI_NOT_AVAILABLE"}
    selected_name = min(calibration, key=calibration.get)
    selected_method = METHODS[selected_name]
    cold_rows = [solve_level(instance, "L4", limits, method=selected_method) for _ in range(formal_repeats)]

    online_rows = []
    for repeat in range(online_repeats):
        model, _, timing = build_model(instance, "L4", method=1, threads=1)
        # Previous snapshot differs only in causal demand RHS; no future pair universe is used.
        for pair in instance.active_pairs:
            model.getConstrByName(f"d_{pair[0]}_{pair[1]}").RHS = instance.demands[pair] * (0.90 + 0.01 * (repeat % 3))
        model.optimize()
        update_start = time.perf_counter()
        for pair in instance.active_pairs:
            model.getConstrByName(f"d_{pair[0]}_{pair[1]}").RHS = instance.demands[pair]
        model.update()
        parse_update_s = time.perf_counter() - update_start
        optimize_start = time.perf_counter()
        model.optimize()
        optimize_wall_s = time.perf_counter() - optimize_start
        online_rows.append({
            "repeat": repeat,
            "status": "MEASURED",
            "parse_update_s": parse_update_s,
            "optimize_wall_s": optimize_wall_s,
            "gurobi_runtime_s": model.Runtime,
            "online_total_s": parse_update_s + optimize_wall_s,
            "objective": model.ObjVal,
        })
        model.dispose()

    def formal_summary(rows):
        return {
            "repeats": len(rows),
            "median_build_s": _median([row["build_s"] for row in rows]),
            "median_optimize_s": _median([row["optimize_wall_s"] for row in rows]),
            "median_total_s": _median([row["total_s"] for row in rows]),
            "objective_min": min(row["objective"] for row in rows),
            "objective_max": max(row["objective"] for row in rows),
        }
    return {
        "status": "MEASURED_DIAGNOSTIC_SYNTHETIC_IRIDIUM_SHAPE",
        "evidence_label": instance.evidence_label,
        "hashes": instance.hashes(),
        "calibration_rule": "minimum median L4 total time over fixed default/dual-simplex/barrier candidates",
        "calibration_repeats": calibration_repeats,
        "calibration_median_total_s": calibration,
        "selected_cold_method": selected_name,
        "best_cold": formal_summary(cold_rows),
        "best_online": {
            "method": "dual_simplex_same_model_previous_basis",
            "causality": "previous RHS only; no future pair universe",
            "repeats": len(online_rows),
            "median_parse_update_s": _median([row["parse_update_s"] for row in online_rows]),
            "median_optimize_s": _median([row["optimize_wall_s"] for row in online_rows]),
            "median_total_s": _median([row["online_total_s"] for row in online_rows]),
            "objective_min": min(row["objective"] for row in online_rows),
            "objective_max": max(row["objective"] for row in online_rows),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("output/speedup_attribution/commercial_results.json"))
    args = parser.parse_args()
    result = benchmark()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
