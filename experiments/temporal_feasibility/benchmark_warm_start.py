"""Benchmark cold rebuild, model reuse, and explicit simplex-basis restore."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from .extract_primal_dual import calibrate_dual_sign
from .sequence_schema import Snapshot, load_dataset
from .sequential_lp import (
    CapacityPolicy,
    GurobiSequentialLP,
    LPUniverse,
    ScipySequentialLP,
    SolveMode,
    SolveState,
    choose_backend,
)

MODES: tuple[SolveMode, ...] = ("cold_rebuild", "reuse_model", "explicit_basis")


def _windows(snapshots: list[Snapshot], window_size: int, stride: int) -> list[list[Snapshot]]:
    if window_size <= 1 or stride <= 0:
        raise ValueError("window_size must exceed 1 and stride must be positive")
    if len(snapshots) <= window_size:
        return [snapshots]
    starts = list(range(0, len(snapshots) - window_size + 1, stride))
    final_start = len(snapshots) - window_size
    if starts[-1] != final_start:
        starts.append(final_start)
    return [snapshots[start: start + window_size] for start in starts]


def _percentile(values: list[float], quantile: float) -> float | None:
    return float(np.percentile(values, quantile)) if values else None


def _bootstrap_median_ci(values: list[float], seed: int = 42, samples: int = 2000) -> list[float] | None:
    if not values:
        return None
    rng = np.random.default_rng(seed)
    array = np.asarray(values, dtype=float)
    medians = np.median(rng.choice(array, size=(samples, len(array)), replace=True), axis=1)
    return [float(np.percentile(medians, 2.5)), float(np.percentile(medians, 97.5))]


def _paired_reduction_ci(records: list[dict[str, Any]], mode: str, seed: int = 42) -> list[float] | None:
    """Bootstrap the ratio-of-medians reduction on paired snapshot solves."""

    key = lambda row: (int(row["window_id"]), int(row["window_local_index"]))
    cold = {
        key(row): float(row["optimize_wall_time"])
        for row in records if row["solve_mode"] == "cold_rebuild" and not row["is_first_in_window"]
    }
    warm = {
        key(row): float(row["optimize_wall_time"])
        for row in records if row["solve_mode"] == mode and not row["is_first_in_window"]
    }
    paired_keys = sorted(set(cold).intersection(warm))
    if not paired_keys:
        return None
    cold_values = np.asarray([cold[item] for item in paired_keys])
    warm_values = np.asarray([warm[item] for item in paired_keys])
    rng = np.random.default_rng(seed)
    reductions = []
    for _ in range(2000):
        indices = rng.integers(0, len(paired_keys), size=len(paired_keys))
        cold_median = float(np.median(cold_values[indices]))
        warm_median = float(np.median(warm_values[indices]))
        if cold_median > 0:
            reductions.append(1.0 - warm_median / cold_median)
    return [float(np.percentile(reductions, 2.5)), float(np.percentile(reductions, 97.5))]


def _summarize(records: list[dict[str, Any]], backend: str) -> dict[str, Any]:
    summary: dict[str, Any] = {"actual_backend": backend, "modes": {}}
    for mode in MODES:
        # First snapshot of every window has no history and is excluded from warm statistics.
        rows = [row for row in records if row["solve_mode"] == mode and not row["is_first_in_window"]]
        optimize = [float(row["optimize_wall_time"]) for row in rows]
        total = [float(row["total_wall_time"]) for row in rows]
        iterations = [float(row["iter_count"]) for row in rows if row["iter_count"] not in (None, "")]
        summary["modes"][mode] = {
            "sample_count_excluding_first": len(rows),
            "optimize_time": {
                "median": statistics.median(optimize) if optimize else None,
                "mean": statistics.mean(optimize) if optimize else None,
                "p90": _percentile(optimize, 90),
                "p95": _percentile(optimize, 95),
                "p99": _percentile(optimize, 99),
                "bootstrap_median_95_ci": _bootstrap_median_ci(optimize),
            },
            "total_time": {
                "median": statistics.median(total) if total else None,
                "mean": statistics.mean(total) if total else None,
                "p90": _percentile(total, 90),
                "p95": _percentile(total, 95),
                "p99": _percentile(total, 99),
            },
            "iter_count_median": statistics.median(iterations) if iterations else None,
        }
    cold = summary["modes"]["cold_rebuild"]["optimize_time"]["median"]
    reductions: dict[str, float | None] = {}
    reduction_intervals: dict[str, list[float] | None] = {}
    for mode in ("reuse_model", "explicit_basis"):
        warm = summary["modes"][mode]["optimize_time"]["median"]
        reductions[mode] = None if cold in (None, 0) or warm is None else 1.0 - warm / cold
        reduction_intervals[mode] = _paired_reduction_ci(records, mode)
    summary["median_optimize_time_reduction_vs_cold"] = reductions
    summary["paired_bootstrap_reduction_95_ci"] = reduction_intervals
    qualified = [
        value >= 0.20 and reduction_intervals[mode] is not None and reduction_intervals[mode][0] > 0.0
        for mode, value in reductions.items() if value is not None
    ]
    summary["gate_c"] = (
        "PASS" if backend == "gurobi" and any(qualified)
        else "FAIL" if backend == "gurobi"
        else "BLOCKED_NO_GUROBI_FORMAL_BENCHMARK"
    )
    return summary


def benchmark(
    snapshots: list[Snapshot], backend: str, policy: CapacityPolicy,
    window_size: int, stride: int, tolerance: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run multiple windows and enforce cross-mode objective/feasibility parity."""

    records: list[dict[str, Any]] = []
    windows = _windows(snapshots, window_size, stride)
    calibration = calibrate_dual_sign(backend)
    for window_id, window in enumerate(windows):
        universe = LPUniverse.from_snapshots(window)
        states_by_mode: dict[SolveMode, list[SolveState]] = {}
        for mode in MODES:
            if backend == "gurobi":
                solver: Any = GurobiSequentialLP(universe, policy, calibration["sign_multiplier"])
            else:
                solver = ScipySequentialLP(universe, policy, calibration["sign_multiplier"])
            states_by_mode[mode] = [solver.solve(snapshot, mode=mode) for snapshot in window]
        for local_index, snapshot in enumerate(window):
            cold = states_by_mode["cold_rebuild"][local_index]
            for mode in MODES:
                state = states_by_mode[mode][local_index]
                objective_delta = abs(state.objective - cold.objective)
                feasible = max(
                    state.max_capacity_violation,
                    state.max_demand_violation,
                    state.max_path_availability_violation,
                ) <= tolerance
                parity = objective_delta <= tolerance * max(1.0, abs(cold.objective))
                if not feasible or not parity or state.status != "OPTIMAL":
                    raise AssertionError(
                        f"Parity/feasibility failure window={window_id} position={snapshot.position} mode={mode}"
                    )
                row = state.csv_record()
                row.update(
                    {
                        "window_id": window_id,
                        "window_local_index": local_index,
                        "is_first_in_window": local_index == 0,
                        "objective_delta_vs_cold": objective_delta,
                        "objective_parity": parity,
                        "feasibility_pass": feasible,
                    }
                )
                records.append(row)
    summary = _summarize(records, backend)
    summary.update(
        {
            "window_count": len(windows),
            "window_size_requested": window_size,
            "stride": stride,
            "objective_feasibility_parity": "PASS",
            "dual_sign_calibration": calibration,
            "fallback_warning": None if backend == "gurobi" else "SciPy modes do not reuse or restore a solver basis; timing is not Gate C evidence",
        }
    )
    return records, summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--backend", choices=("auto", "gurobi", "scipy"), default="auto")
    parser.add_argument("--allow-scipy-fallback", action="store_true")
    parser.add_argument("--window-size", type=int, default=40)
    parser.add_argument("--stride", type=int, default=20)
    parser.add_argument("--network-edge-capacity", type=float, default=200.0)
    parser.add_argument("--path-only-edge-capacity", type=float, default=800.0)
    parser.add_argument("--tolerance", type=float, default=1e-7)
    parser.add_argument("--evidence-source", choices=("real_adapter_pickle", "synthetic_fixture"), default="real_adapter_pickle")
    parser.add_argument("--output-dir", default="output/temporal_feasibility")
    args = parser.parse_args()

    snapshots = load_dataset(args.dataset, limit=args.limit)
    backend, detail = choose_backend(args.backend, args.allow_scipy_fallback)
    records, summary = benchmark(
        snapshots=snapshots,
        backend=backend,
        policy=CapacityPolicy(args.network_edge_capacity, args.path_only_edge_capacity),
        window_size=args.window_size,
        stride=args.stride,
        tolerance=args.tolerance,
    )
    summary.update({"dataset": args.dataset, "backend_resolution": detail, "evidence_source": args.evidence_source})
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "solve_records.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    with (output_dir / "warm_start_summary.json").open("w", encoding="utf-8-sig") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False, allow_nan=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
