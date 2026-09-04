"""Compare adjacent optimal states with matched random controls and lag curves."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .extract_primal_dual import calibrate_dual_sign
from .semantic_alignment import (
    classify_change,
    correlations,
    initialization_baselines,
    jaccard,
    normalized_l1,
    normalized_l2,
    path_birth_death_mass,
    topology_delta,
    transport_edge_prices,
)
from .sequence_schema import Snapshot, load_dataset
from .sequential_lp import CapacityPolicy, GurobiSequentialLP, LPUniverse, ScipySequentialLP, SolveState, choose_backend

LAGS = (1, 2, 5, 10)


def _finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(value) else None


def _pair_metrics(
    previous: Snapshot, current: Snapshot, previous_state: SolveState,
    current_state: SolveState, pair_type: str, lag: int,
) -> dict[str, Any]:
    persistent, new_edges, deleted_edges = topology_delta(previous, current)
    common_paths = sorted(set(previous.paths).intersection(current.paths))
    path_l1 = normalized_l1(previous_state.path_flow, current_state.path_flow, common_paths)
    path_l2 = normalized_l2(previous_state.path_flow, current_state.path_flow, common_paths)
    path_pearson, path_spearman = correlations(previous_state.path_flow, current_state.path_flow, common_paths)
    persistent_keys = sorted(persistent)
    load_pearson, load_spearman = correlations(previous_state.edge_load, current_state.edge_load, persistent_keys)
    dual_pearson, dual_spearman = correlations(previous_state.congestion_price, current_state.congestion_price, persistent_keys)
    birth_mass, death_mass = path_birth_death_mass(previous_state.path_flow, current_state.path_flow)
    previous_binding = {edge for edge, active in previous_state.binding_capacity.items() if active}
    current_binding = {edge for edge, active in current_state.binding_capacity.items() if active}
    edge_union_size = max(1, len(set(previous.graph_edges).union(current.graph_edges)))
    return {
        "pair_type": pair_type,
        "lag": lag,
        "previous_position": previous.position,
        "current_position": current.position,
        "previous_data_idx": previous.data_idx,
        "current_data_idx": current.data_idx,
        "change_type": classify_change(previous, current),
        "edge_jaccard": jaccard(previous.graph_edges, current.graph_edges),
        "normalized_traffic_l1_drift": normalized_l1(previous.demands, current.demands),
        "semantic_path_overlap": jaccard(previous.paths, current.paths),
        "path_flow_common_normalized_l1": path_l1,
        "path_flow_common_normalized_l2": path_l2,
        "path_flow_common_pearson": _finite_or_none(path_pearson),
        "path_flow_common_spearman": _finite_or_none(path_spearman),
        "path_birth_mass": birth_mass,
        "path_death_mass": death_mass,
        "edge_load_persistent_normalized_l1": normalized_l1(previous_state.edge_load, current_state.edge_load, persistent_keys),
        "edge_load_persistent_normalized_l2": normalized_l2(previous_state.edge_load, current_state.edge_load, persistent_keys),
        "edge_load_persistent_pearson": _finite_or_none(load_pearson),
        "edge_load_persistent_spearman": _finite_or_none(load_spearman),
        "dual_persistent_normalized_l1": normalized_l1(previous_state.congestion_price, current_state.congestion_price, persistent_keys),
        "dual_persistent_normalized_l2": normalized_l2(previous_state.congestion_price, current_state.congestion_price, persistent_keys),
        "dual_persistent_pearson": _finite_or_none(dual_pearson),
        "dual_persistent_spearman": _finite_or_none(dual_spearman),
        "binding_edge_jaccard": jaccard(previous_binding, current_binding),
        "new_edge_ratio": len(new_edges) / edge_union_size,
        "deleted_edge_ratio": len(deleted_edges) / edge_union_size,
        "dual_objective_normalized_delta": abs(previous_state.dual_objective - current_state.dual_objective) / max(abs(previous_state.dual_objective), abs(current_state.dual_objective), 1e-12),
    }


def _control_target(
    snapshots: list[Snapshot], base_index: int, adjacent_index: int, kind: str, rng: random.Random,
) -> int:
    candidates = [index for index in range(len(snapshots)) if index != base_index and abs(index - base_index) > 1]
    if not candidates:
        candidates = [index for index in range(len(snapshots)) if index != base_index]
    if kind == "random_unrestricted":
        return rng.choice(candidates)
    adjacent = snapshots[adjacent_index]
    base = snapshots[base_index]
    if kind == "random_demand_matched":
        target_delta = abs(base.total_demand - adjacent.total_demand) / max(base.total_demand, adjacent.total_demand, 1e-12)
        return min(
            candidates,
            key=lambda index: (
                abs(abs(base.total_demand - snapshots[index].total_demand) / max(base.total_demand, snapshots[index].total_demand, 1e-12) - target_delta),
                rng.random(),
            ),
        )
    if kind == "random_topology_matched":
        target_similarity = jaccard(base.graph_edges, adjacent.graph_edges)
        return min(candidates, key=lambda index: (abs(jaccard(base.graph_edges, snapshots[index].graph_edges) - target_similarity), rng.random()))
    raise ValueError(f"Unknown control kind: {kind}")


def _bootstrap_ratio(adjacent: list[float], control: list[float], seed: int = 42) -> list[float] | None:
    if not adjacent or not control:
        return None
    rng = np.random.default_rng(seed)
    left, right = np.asarray(adjacent), np.asarray(control)
    ratios = []
    for _ in range(2000):
        lm = float(np.median(rng.choice(left, size=len(left), replace=True)))
        rm = float(np.median(rng.choice(right, size=len(right), replace=True)))
        if rm > 1e-12:
            ratios.append(lm / rm)
    return [float(np.percentile(ratios, 2.5)), float(np.percentile(ratios, 97.5))] if ratios else None


def _distribution_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = (
        "path_flow_common_normalized_l1",
        "edge_load_persistent_normalized_l1",
        "dual_persistent_normalized_l1",
        "binding_edge_jaccard",
    )
    result: dict[str, Any] = {}
    pair_types = sorted({str(row["pair_type"]) for row in records})
    for metric in metrics:
        result[metric] = {}
        for pair_type in pair_types:
            values = [float(row[metric]) for row in records if row["pair_type"] == pair_type and row[metric] is not None]
            result[metric][pair_type] = {
                "count": len(values),
                "median": float(np.median(values)) if values else None,
                "mean": float(np.mean(values)) if values else None,
            }
        adjacent = [float(row[metric]) for row in records if row["pair_type"] == "adjacent" and row[metric] is not None]
        control = [float(row[metric]) for row in records if row["pair_type"] == "random_demand_matched" and row[metric] is not None]
        if adjacent and control and float(np.median(control)) > 1e-12:
            result[metric]["adjacent_to_demand_matched_median_ratio"] = float(np.median(adjacent) / np.median(control))
        else:
            result[metric]["adjacent_to_demand_matched_median_ratio"] = None
        result[metric]["ratio_bootstrap_95_ci"] = _bootstrap_ratio(adjacent, control)
    qualified = []
    for name in metrics[:3]:
        ratio = result[name]["adjacent_to_demand_matched_median_ratio"]
        interval = result[name]["ratio_bootstrap_95_ci"]
        if ratio is not None and interval is not None:
            # A favorable point estimate is insufficient when uncertainty crosses the gate.
            qualified.append(ratio <= 0.7 and interval[1] <= 0.7)
    result["gate_b"] = "PASS" if len(qualified) >= 2 and all(qualified) else "FAIL_OR_INCONCLUSIVE"
    return result


def _transport_records(snapshots: list[Snapshot], states: list[SolveState]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index in range(1, len(snapshots)):
        previous, current = snapshots[index - 1], snapshots[index]
        previous_state, current_state = states[index - 1], states[index]
        baselines = initialization_baselines(current.graph_edges, previous_state.congestion_price, seed=42 + index)
        candidates = {
            f"transport_{policy}": transport_edge_prices(
                previous_state.congestion_price, previous.graph_edges, current.graph_edges, policy
            )
            for policy in ("zero", "global_median", "neighbor_median")
        }
        candidates.update({f"default_{name}": value for name, value in baselines.items()})
        keys = sorted(current.graph_edges)
        for name, candidate in candidates.items():
            pearson, spearman = correlations(candidate, current_state.congestion_price, keys)
            records.append(
                {
                    "previous_position": previous.position,
                    "current_position": current.position,
                    "change_type": classify_change(previous, current),
                    "initialization": name,
                    "normalized_l1": normalized_l1(candidate, current_state.congestion_price, keys),
                    "normalized_l2": normalized_l2(candidate, current_state.congestion_price, keys),
                    "pearson": _finite_or_none(pearson),
                    "spearman": _finite_or_none(spearman),
                }
            )
    return records


def _transport_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in records:
        grouped[str(row["change_type"])][str(row["initialization"])].append(float(row["normalized_l1"]))
    summary: dict[str, Any] = {}
    for change_type, methods in grouped.items():
        summary[change_type] = {name: float(np.median(values)) for name, values in methods.items()}
    dynamic_rows = [row for row in records if row["change_type"] in {"topology_only", "both_change"}]
    method_medians = {
        name: float(np.median([float(row["normalized_l1"]) for row in dynamic_rows if row["initialization"] == name]))
        for name in sorted({str(row["initialization"]) for row in dynamic_rows})
    } if dynamic_rows else {}
    transports = [value for name, value in method_medians.items() if name.startswith("transport_")]
    controls = [value for name, value in method_medians.items() if name.startswith("default_")]
    best_transport = min(
        ((name, value) for name, value in method_medians.items() if name.startswith("transport_")),
        key=lambda item: item[1], default=None,
    )
    best_control = min(
        ((name, value) for name, value in method_medians.items() if name.startswith("default_")),
        key=lambda item: item[1], default=None,
    )
    paired_interval = None
    if best_transport is not None and best_control is not None:
        by_method = {
            name: {int(row["current_position"]): float(row["normalized_l1"]) for row in dynamic_rows if row["initialization"] == name}
            for name in (best_transport[0], best_control[0])
        }
        keys = sorted(set(by_method[best_transport[0]]).intersection(by_method[best_control[0]]))
        if keys:
            differences = np.asarray([
                by_method[best_transport[0]][key] - by_method[best_control[0]][key] for key in keys
            ])
            rng = np.random.default_rng(42)
            boot = [
                float(np.median(rng.choice(differences, size=len(differences), replace=True)))
                for _ in range(2000)
            ]
            paired_interval = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
    summary["dynamic_method_medians"] = method_medians
    summary["best_transport_vs_best_control"] = {
        "best_transport": best_transport,
        "best_control": best_control,
        "paired_median_difference_95_ci": paired_interval,
    }
    summary["gate_d"] = (
        "PASS" if transports and controls and min(transports) < min(controls)
        and paired_interval is not None and paired_interval[1] < 0.0
        else "FAIL_OR_INCONCLUSIVE" if dynamic_rows
        else "BLOCKED_NO_DYNAMIC_TRANSITIONS"
    )
    summary["claim_limit"] = "Price distance only; no primal recovery claim"
    return summary


def analyze(
    snapshots: list[Snapshot], backend: str, policy: CapacityPolicy, seed: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], list[SolveState]]:
    """Solve snapshots once and construct adjacent, lagged, and matched controls."""

    universe = LPUniverse.from_snapshots(snapshots)
    calibration = calibrate_dual_sign(backend)
    solver: Any = (
        GurobiSequentialLP(universe, policy, calibration["sign_multiplier"])
        if backend == "gurobi"
        else ScipySequentialLP(universe, policy, calibration["sign_multiplier"])
    )
    states = [solver.solve(snapshot, mode="cold_rebuild") for snapshot in snapshots]
    records: list[dict[str, Any]] = []
    rng = random.Random(seed)
    for lag in LAGS:
        for index in range(len(snapshots) - lag):
            pair_type = "adjacent" if lag == 1 else f"lag_{lag}"
            records.append(_pair_metrics(snapshots[index], snapshots[index + lag], states[index], states[index + lag], pair_type, lag))
    for index in range(len(snapshots) - 1):
        for kind in ("random_unrestricted", "random_demand_matched", "random_topology_matched"):
            target = _control_target(snapshots, index, index + 1, kind, rng)
            records.append(_pair_metrics(snapshots[index], snapshots[target], states[index], states[target], kind, abs(target - index)))
    transport = _transport_records(snapshots, states)
    summary = {
        "backend": backend,
        "snapshot_count": len(snapshots),
        "dual_sign_calibration": calibration,
        "continuity": _distribution_summary(records),
        "transport": _transport_summary(transport),
        "degeneracy_warning": "Capacity duals may be non-unique; edge load, binding set, and dual objective are reported alongside them",
    }
    return records, transport, summary, states


def _write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    if not records:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--backend", choices=("auto", "gurobi", "scipy"), default="auto")
    parser.add_argument("--allow-scipy-fallback", action="store_true")
    parser.add_argument("--network-edge-capacity", type=float, default=200.0)
    parser.add_argument("--path-only-edge-capacity", type=float, default=800.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--evidence-source", choices=("real_adapter_pickle", "synthetic_fixture"), default="real_adapter_pickle")
    parser.add_argument("--output-dir", default="output/temporal_feasibility")
    args = parser.parse_args()
    snapshots = load_dataset(args.dataset, limit=args.limit)
    backend, detail = choose_backend(args.backend, args.allow_scipy_fallback)
    records, transport, summary, states = analyze(
        snapshots, backend, CapacityPolicy(args.network_edge_capacity, args.path_only_edge_capacity), args.seed
    )
    summary.update({"dataset": args.dataset, "backend_resolution": detail, "evidence_source": args.evidence_source})
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "continuity_records.csv", records)
    _write_csv(output / "transport_records.csv", transport)
    with (output / "summary.json").open("w", encoding="utf-8-sig") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False, allow_nan=False)
    with (output / "state_records.jsonl").open("w", encoding="utf-8-sig") as handle:
        for state in states:
            handle.write(json.dumps(state.json_record(), ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
