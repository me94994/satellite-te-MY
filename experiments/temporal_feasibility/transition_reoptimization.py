"""Run the preregistered topology-transition reoptimization qualification."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import subprocess
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.stats import spearmanr, wilcoxon

from .extract_primal_dual import calibrate_dual_sign
from .gurobi_probe import probe_gurobi
from .semantic_alignment import jaccard, normalized_l1
from .sequence_schema import EdgeKey, PathKey, Snapshot
from .sequential_lp import CapacityPolicy, GurobiSequentialLP, LPUniverse, SolveState
from .state_transport import (
    direct_previous_path_copy,
    repair_path_flow,
    transport_edge_state,
    transported_advanced_start,
    zero_advanced_start,
)
from .transition_dataset import (
    SLICE_TYPES, TransitionEvent, build_event_slices, load_raw_indices,
    event_path_severity, read_transition_events, slice_manifest_digest,
)


CLASSICAL_METHODS = ("AUTO_MODEL_REUSE", "DIRECT_PREVIOUS_BASIS", "BASIS_PRESOLVE")
BOOTSTRAP_SAMPLES = 5000
PIPELINE_VERSION = "topology_transition_v3_counterbalanced_pair_order"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
    temporary.replace(path)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty required output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    ).stdout.strip()


def _manifest_rows(path: Path) -> tuple[list[dict[str, str]], dict[tuple[str, int], dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return rows, {(row["sequence_id"], int(row["source_record_index"])): row for row in rows}


def _pair_universe(pair: Sequence[Snapshot], constraint_budget: int) -> LPUniverse:
    """Build and audit the fixed union of pair flows, paths, and edges."""

    if len(pair) != 2:
        raise ValueError("Pair-universe requires exactly two snapshots")
    universe = LPUniverse.from_snapshots(pair)
    constraints = len(universe.edges) + len(universe.flows) + len(universe.paths)
    if constraints > constraint_budget:
        raise RuntimeError(f"Pair universe exceeds restricted-license budget: {constraints}>{constraint_budget}")
    return universe


def _state_row(
    state: SolveState, *, event: TransitionEvent, slice_type: str, pair_kind: str,
    pair_start: int, method: str, universe: LPUniverse, preparation_time: float = 0.0,
) -> dict[str, Any]:
    row = state.csv_record()
    feasible = max(
        state.max_capacity_violation, state.max_demand_violation,
        state.max_path_availability_violation,
    ) <= 1e-7
    row.update(
        {
            "event_id": event.event_id, "sequence_id": event.sequence_id,
            "transition_index": event.transition_index, "pair_start_index": pair_start,
            "pair_kind": pair_kind, "slice_type": slice_type, "method": method,
            "preparation_wall_time": preparation_time,
            "operational_total_wall_time": state.total_wall_time + preparation_time,
            "flow_count": len(universe.flows), "path_count": len(universe.paths),
            "edge_count": len(universe.edges), "variable_count": len(universe.paths),
            "constraint_count": len(universe.edges) + len(universe.flows) + len(universe.paths),
            "basis_validity": "VALID" if state.warm_start_effective else "NOT_APPLICABLE_OR_RESET",
            "feasibility_pass": feasible,
        }
    )
    return row


def _solve_pair(
    pair: Sequence[Snapshot], event: TransitionEvent, slice_type: str, pair_kind: str,
    policy: CapacityPolicy, dual_sign: float, constraint_budget: int,
) -> tuple[list[dict[str, Any]], SolveState, SolveState, dict[str, Any]]:
    """Run all classical and transported methods on one fixed pair universe."""

    previous, current = pair
    universe = _pair_universe(pair, constraint_budget)
    records: list[dict[str, Any]] = []

    cold = GurobiSequentialLP(universe, policy, dual_sign)
    previous_cold = cold.solve(previous, "cold_rebuild")
    current_cold = cold.solve(current, "cold_rebuild")
    records.append(_state_row(
        current_cold, event=event, slice_type=slice_type, pair_kind=pair_kind,
        pair_start=previous.position, method="COLD", universe=universe,
    ))
    cold.close()

    classical_modes = (
        ("AUTO_MODEL_REUSE", "reuse_model_auto"),
        ("DIRECT_PREVIOUS_BASIS", "explicit_basis"),
        ("BASIS_PRESOLVE", "explicit_basis_presolve"),
        ("RESET", "reset_basis"),
    )
    for method, solve_mode in classical_modes:
        solver = GurobiSequentialLP(universe, policy, dual_sign)
        solver.solve(previous, solve_mode)
        state = solver.solve(current, solve_mode)
        records.append(_state_row(
            state, event=event, slice_type=slice_type, pair_kind=pair_kind,
            pair_start=previous.position, method=method, universe=universe,
        ))
        solver.close()

    prepare_start = time.perf_counter()
    transported, transport_audit = transported_advanced_start(previous_cold, current, universe, policy)
    preparation_time = time.perf_counter() - prepare_start
    for method, mode, start in (
        ("TRANSPORTED_PSTART", "transported_pstart", transported),
        ("TRANSPORTED_PSTART_DSTART", "transported_pstart_dstart", transported),
        ("ZERO_START", "zero_start", zero_advanced_start(universe)),
    ):
        solver = GurobiSequentialLP(universe, policy, dual_sign)
        solver.solve(previous, "reuse_model_auto")
        state = solver.solve(current, mode, advanced_start=start)
        records.append(_state_row(
            state, event=event, slice_type=slice_type, pair_kind=pair_kind,
            pair_start=previous.position, method=method, universe=universe,
            preparation_time=preparation_time if method.startswith("TRANSPORTED") else 0.0,
        ))
        solver.close()

    tolerance = 1e-7 * max(1.0, abs(current_cold.objective))
    for row in records:
        row["objective_delta_vs_cold"] = abs(float(row["objective"]) - current_cold.objective)
        row["objective_parity"] = row["objective_delta_vs_cold"] <= tolerance
        if not row["objective_parity"] or not row["feasibility_pass"] or row["status"] != "OPTIMAL":
            raise AssertionError(
                f"Solver parity failure event={event.event_id} slice={slice_type} method={row['method']}"
            )
    return records, previous_cold, current_cold, transport_audit


def _path_edge_load(values: Mapping[PathKey, float], universe: LPUniverse) -> dict[EdgeKey, float]:
    loads = {edge: 0.0 for edge in universe.edges}
    for path, value in values.items():
        for edge in universe.path_to_edges[path]:
            loads[edge] += float(value)
    return loads


def _equal_split_start(current: Snapshot, universe: LPUniverse) -> dict[PathKey, float]:
    """Input-only no-history initialization from current demand/path availability."""

    paths_by_flow: dict[tuple[int, int], list[PathKey]] = {flow: [] for flow in universe.flows}
    for path in universe.paths:
        if path in current.paths:
            paths_by_flow[(path[0], path[1])].append(path)
    result = {path: 0.0 for path in universe.paths}
    for flow, paths in paths_by_flow.items():
        if paths:
            for path in paths:
                result[path] = current.demands.get(flow, 0.0) / len(paths)
    return result


def _transport_state_rows(
    pair: Sequence[Snapshot], previous: SolveState, current: SolveState,
    event: TransitionEvent, slice_type: str, policy: CapacityPolicy,
) -> list[dict[str, Any]]:
    """Compare transported state against strong input-only/history controls."""

    universe = LPUniverse.from_snapshots(pair)
    direct = direct_previous_path_copy(previous, pair[1], universe)
    repaired, _ = repair_path_flow(direct, pair[1], universe, policy)
    previous_nonzero = [value for value in previous.path_flow.values() if value > 0.0]
    global_value = float(np.mean(previous_nonzero)) if previous_nonzero else 0.0
    candidates = {
        "transport_repaired": repaired,
        "direct_copy_no_repair": direct,
        "zero": {path: 0.0 for path in universe.paths},
        "global_mean": {path: global_value if path in pair[1].paths else 0.0 for path in universe.paths},
        "no_history_equal_split": _equal_split_start(pair[1], universe),
    }
    actual_path = current.path_flow
    actual_load = current.edge_load
    actual_util = current.edge_utilization
    actual_binding = {edge for edge, value in current.binding_capacity.items() if value}
    rows: list[dict[str, Any]] = []
    for initialization, path_values in candidates.items():
        loads = _path_edge_load(path_values, universe)
        utilization = {
            edge: loads[edge] / current.edge_capacity[edge] if current.edge_capacity[edge] > 0.0 else 0.0
            for edge in universe.edges
        }
        predicted_binding = {
            edge for edge in universe.edges
            if current.edge_capacity[edge] > 0.0
            and current.edge_capacity[edge] - loads[edge] <= 1e-7 * max(1.0, current.edge_capacity[edge])
        }
        metrics = {
            "path_flow": normalized_l1(path_values, actual_path, list(universe.paths)),
            "edge_load": normalized_l1(loads, actual_load, list(universe.edges)),
            "utilization": normalized_l1(utilization, actual_util, list(universe.edges)),
            "binding": 1.0 - jaccard(predicted_binding, actual_binding),
        }
        for state_name, distance in metrics.items():
            rows.append(
                {
                    "event_id": event.event_id, "transition_index": event.transition_index,
                    "slice_type": slice_type, "evidence_label": pair[1].meta["evidence_label"],
                    "demand_scale": pair[1].meta["demand_scale"],
                    "state": state_name, "initialization": initialization,
                    "normalized_l1": distance,
                    "informative": bool(actual_binding) if state_name == "binding" else True,
                }
            )

    current_edges = frozenset(edge for edge, cap in current.edge_capacity.items() if cap > 0.0)
    previous_edges = frozenset(edge for edge, cap in previous.edge_capacity.items() if cap > 0.0)
    # Also preserve the explicitly preregistered edge-state policies.  These do
    # not alter the primal start; they isolate how newly born edge state is set.
    for state_name, previous_values, current_values in (
        ("edge_load", previous.edge_load, current.edge_load),
        ("utilization", previous.edge_utilization, current.edge_utilization),
    ):
        for new_edge_policy in ("zero", "neighbor_mean", "global_mean"):
            candidate = transport_edge_state(
                previous_values, previous_edges, current_edges, new_edge_policy
            )
            rows.append(
                {
                    "event_id": event.event_id, "transition_index": event.transition_index,
                    "slice_type": slice_type, "evidence_label": pair[1].meta["evidence_label"],
                    "demand_scale": pair[1].meta["demand_scale"], "state": state_name,
                    "initialization": f"edge_transport_{new_edge_policy}_new",
                    "normalized_l1": normalized_l1(candidate, current_values, list(current_edges)),
                }
            )
    dual_informative = any(abs(value) > 1e-12 for value in current.congestion_price.values())
    for initialization, policy_name in (
        ("dual_transport_zero_new", "zero"),
        ("dual_transport_global_median_new", "global_median"),
        ("dual_transport_neighbor_median_new", "neighbor_median"),
        ("zero", "zero"),
    ):
        candidate = (
            {edge: 0.0 for edge in current_edges}
            if initialization == "zero"
            else transport_edge_state(previous.congestion_price, previous_edges, current_edges, policy_name)
        )
        rows.append(
            {
                "event_id": event.event_id, "transition_index": event.transition_index,
                "slice_type": slice_type, "evidence_label": pair[1].meta["evidence_label"],
                "demand_scale": pair[1].meta["demand_scale"],
                "state": "dual", "initialization": initialization,
                "normalized_l1": normalized_l1(candidate, current.congestion_price, list(current_edges)),
                "informative": dual_informative,
            }
        )
    return rows


def _scale_snapshots(snapshots: Sequence[Snapshot], scale: float) -> list[Snapshot]:
    return [
        replace(
            snapshot, demands={flow: demand * scale for flow, demand in snapshot.demands.items()},
            meta={**snapshot.meta, "demand_scale": scale,
                  "evidence_label": "OFFICIAL_REAL_WORKLOAD" if scale == 1.0 else "REAL_TOPOLOGY_REAL_TRAFFIC_SCALED_STRESS"},
        )
        for snapshot in snapshots
    ]


def _congestion_row(
    state: SolveState, event: TransitionEvent, slice_type: str, scale: float,
) -> dict[str, Any]:
    positive_edges = [edge for edge, cap in state.edge_capacity.items() if cap > 0.0]
    binding = sum(bool(state.binding_capacity[edge]) for edge in positive_edges)
    nonzero_dual = sum(abs(state.congestion_price[edge]) > 1e-12 for edge in positive_edges)
    return {
        "event_id": event.event_id, "transition_index": event.transition_index,
        "slice_type": slice_type, "demand_scale": scale,
        "evidence_label": "OFFICIAL_REAL_WORKLOAD" if scale == 1.0 else "REAL_TOPOLOGY_REAL_TRAFFIC_SCALED_STRESS",
        "objective": state.objective, "binding_edge_count": binding,
        "positive_capacity_edge_count": len(positive_edges),
        "binding_rate": binding / len(positive_edges) if positive_edges else 0.0,
        "nonzero_capacity_dual_count": nonzero_dual,
        "nonzero_capacity_dual_rate": nonzero_dual / len(positive_edges) if positive_edges else 0.0,
        # Filled by the caller from the current Snapshot; never infer demand from an optimum.
        "total_demand": 0.0, "unsatisfied_demand": 0.0,
    }


def _checkpoint_path(output_dir: Path, event_id: int, scale: float) -> Path:
    return output_dir / "checkpoints" / f"event_{event_id:04d}_scale_{scale:g}.json"


def _event_payload(
    event: TransitionEvent, event_slices: Mapping[str, Sequence[Snapshot]],
    manifests: Mapping[str, Mapping[str, Any]], severity: Mapping[str, Any],
    raw_lookup: Mapping[tuple[str, int], Mapping[str, str]], policy: CapacityPolicy,
    dual_sign: float, constraint_budget: int, scale: float,
) -> dict[str, Any]:
    solver_rows: list[dict[str, Any]] = []
    state_rows: list[dict[str, Any]] = []
    congestion_rows: list[dict[str, Any]] = []
    stable_rows: list[dict[str, Any]] = []
    for slice_type in sorted(event_slices, key=SLICE_TYPES.index):
        snapshots = _scale_snapshots(event_slices[slice_type], scale)
        transition_pair = [
            snapshot for snapshot in snapshots
            if snapshot.position in (event.transition_index - 1, event.transition_index)
        ]
        # Retain at most one same-event stable control.  Its topology label is
        # taken from the raw routing signature, not selected-path artifacts.
        adjacent_pairs = list(zip(snapshots[:-1], snapshots[1:]))
        stable_pair = next(
            (
                pair for pair in adjacent_pairs
                if pair[1].position != event.transition_index
                and raw_lookup[(event.sequence_id, pair[0].position)]["routing_graph_hash"]
                == raw_lookup[(event.sequence_id, pair[1].position)]["routing_graph_hash"]
            ),
            None,
        )

        def solve_stable() -> None:
            if stable_pair is None:
                return
            stable_solve_rows, _, _, _ = _solve_pair(
                stable_pair, event, slice_type, "STABLE", policy, dual_sign, constraint_budget
            )
            for row in stable_solve_rows:
                row["demand_scale"] = scale
                row["evidence_label"] = stable_pair[1].meta["evidence_label"]
            stable_rows.extend(stable_solve_rows)

        # Counterbalance pair order by event ID so neither regime receives a
        # systematic process/cache warm-up advantage in microsecond timings.
        if event.event_id % 2 == 0:
            solve_stable()
        rows, previous, current, transport_audit = _solve_pair(
            transition_pair, event, slice_type, "TRANSITION", policy, dual_sign, constraint_budget
        )
        pair_severity = event_path_severity(transition_pair)
        for row in rows:
            row.update(severity)
            row.update(pair_severity)
            row["demand_scale"] = scale
            row["evidence_label"] = transition_pair[1].meta["evidence_label"]
        solver_rows.extend(rows)
        state_rows.extend(_transport_state_rows(
            transition_pair, previous, current, event, slice_type, policy
        ))
        congestion = _congestion_row(current, event, slice_type, scale)
        congestion["total_demand"] = transition_pair[1].total_demand
        congestion["unsatisfied_demand"] = transition_pair[1].total_demand - current.objective
        congestion_rows.append(congestion)
        if event.event_id % 2 == 1:
            solve_stable()
    return {
        "event": asdict(event), "scale": scale,
        "manifests": {name: dict(value) for name, value in manifests.items()},
        "severity": dict(severity), "solver_rows": solver_rows,
        "state_rows": state_rows, "congestion_rows": congestion_rows,
        "stable_rows": stable_rows,
    }


def _load_or_run_event(
    output_dir: Path, event: TransitionEvent, event_slices: Mapping[str, Sequence[Snapshot]],
    manifests: Mapping[str, Mapping[str, Any]], severity: Mapping[str, Any],
    raw_lookup: Mapping[tuple[str, int], Mapping[str, str]], policy: CapacityPolicy,
    dual_sign: float, constraint_budget: int, scale: float, resume: bool,
) -> dict[str, Any]:
    checkpoint = _checkpoint_path(output_dir, event.event_id, scale)
    signature = {
        "event_id": event.event_id, "scale": scale,
        "pipeline_version": PIPELINE_VERSION,
        "slice_digests": {name: slice_manifest_digest(value) for name, value in manifests.items()},
    }
    if resume and checkpoint.is_file():
        with checkpoint.open("r", encoding="utf-8-sig") as handle:
            payload = json.load(handle)
        if payload.get("resume_signature") == signature:
            return payload
    payload = _event_payload(
        event, event_slices, manifests, severity, raw_lookup, policy,
        dual_sign, constraint_budget, scale,
    )
    payload["resume_signature"] = signature
    _write_json(checkpoint, payload)
    return payload


def _median(values: Iterable[float]) -> float | None:
    materialized = list(values)
    return float(np.median(materialized)) if materialized else None


def _method_summary(
    rows: Sequence[Mapping[str, Any]], pair_kind: str, scale: float = 1.0,
    slice_type: str | None = None,
) -> dict[str, Any]:
    selected = [
        row for row in rows
        if row["pair_kind"] == pair_kind and float(row["demand_scale"]) == scale
        and (slice_type is None or row["slice_type"] == slice_type)
    ]
    result: dict[str, Any] = {}
    for method in sorted({str(row["method"]) for row in selected}):
        method_rows = [row for row in selected if row["method"] == method]
        result[method] = {
            "count": len(method_rows),
            "iter_count": _distribution([float(row["iter_count"]) for row in method_rows]),
            "optimize_time": _distribution([float(row["optimize_wall_time"]) for row in method_rows]),
            "total_time": _distribution([float(row["operational_total_wall_time"]) for row in method_rows]),
        }
    return result


def _distribution(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {"median": None, "mean": None, "p90": None, "p95": None}
    return {
        "median": float(np.median(values)), "mean": float(np.mean(values)),
        "p90": float(np.percentile(values, 90)), "p95": float(np.percentile(values, 95)),
    }


def _paired_stats(
    rows: Sequence[Mapping[str, Any]], baseline: str, method: str, metric: str,
    *, scale: float = 1.0, slice_type: str = "HIGH_PRESSURE_SLICE",
) -> dict[str, Any]:
    key = lambda row: (int(row["event_id"]), str(row["slice_type"]), int(row["pair_start_index"]))
    selected = [
        row for row in rows
        if row["pair_kind"] == "TRANSITION" and float(row["demand_scale"]) == scale
        and row["slice_type"] == slice_type
    ]
    left = {key(row): float(row[metric]) for row in selected if row["method"] == baseline}
    right = {key(row): float(row[metric]) for row in selected if row["method"] == method}
    keys = sorted(set(left) & set(right))
    baseline_values = np.asarray([left[item] for item in keys], dtype=float)
    method_values = np.asarray([right[item] for item in keys], dtype=float)
    differences = baseline_values - method_values
    if len(keys) and not np.allclose(differences, 0.0):
        pvalue = float(wilcoxon(baseline_values, method_values, alternative="greater").pvalue)
    else:
        pvalue = 1.0
    rng = np.random.default_rng(42)
    boot = np.asarray([], dtype=float)
    if len(keys):
        indices = rng.integers(0, len(keys), size=(BOOTSTRAP_SAMPLES, len(keys)))
        boot = np.median(differences[indices], axis=1)
    ci = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))] if len(boot) else None
    nonzero_baseline = float(np.median(baseline_values)) if len(keys) else 0.0
    reduction = None if nonzero_baseline == 0.0 else float(np.median(differences) / nonzero_baseline)
    return {
        "count": len(keys), "median_paired_difference": float(np.median(differences)) if len(keys) else None,
        "bootstrap_median_difference_95_ci": ci, "wilcoxon_one_sided_pvalue": pvalue,
        "median_reduction": reduction,
        "paired_rank_biserial_effect": (
            float((np.sum(differences > 0) - np.sum(differences < 0)) / len(keys)) if len(keys) else None
        ),
    }


def _paired_log_time_stats(
    rows: Sequence[Mapping[str, Any]], baseline: str, method: str, *, scale: float = 1.0,
    slice_type: str = "HIGH_PRESSURE_SLICE", metric: str = "optimize_wall_time",
) -> dict[str, Any]:
    key = lambda row: (int(row["event_id"]), str(row["slice_type"]), int(row["pair_start_index"]))
    selected = [
        row for row in rows
        if row["pair_kind"] == "TRANSITION" and float(row["demand_scale"]) == scale
        and row["slice_type"] == slice_type
    ]
    left = {key(row): float(row[metric]) for row in selected if row["method"] == baseline}
    right = {key(row): float(row[metric]) for row in selected if row["method"] == method}
    keys = sorted(set(left) & set(right))
    logs = np.asarray([math.log(left[item] / right[item]) for item in keys], dtype=float)
    if len(logs) and not np.allclose(logs, 0.0):
        pvalue = float(wilcoxon(logs, alternative="greater").pvalue)
    else:
        pvalue = 1.0
    rng = np.random.default_rng(42)
    ci = None
    if len(logs):
        indices = rng.integers(0, len(logs), size=(BOOTSTRAP_SAMPLES, len(logs)))
        boot = np.median(logs[indices], axis=1)
        ci = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
    median_log = float(np.median(logs)) if len(logs) else None
    return {
        "count": len(keys), "median_log_speedup": median_log,
        "metric": metric,
        "median_time_reduction": None if median_log is None else 1.0 - math.exp(-median_log),
        "bootstrap_median_log_speedup_95_ci": ci, "wilcoxon_one_sided_pvalue": pvalue,
        "paired_rank_biserial_effect": (
            float((np.sum(logs > 0) - np.sum(logs < 0)) / len(logs)) if len(logs) else None
        ),
    }


def _state_gate(rows: Sequence[Mapping[str, Any]], scale: float) -> tuple[str, dict[str, Any]]:
    selected = [
        row for row in rows
        if row["slice_type"] == "HIGH_PRESSURE_SLICE" and float(row.get("demand_scale", scale)) == scale
    ]
    details: dict[str, Any] = {}
    passed: list[str] = []
    for state in ("edge_load", "utilization", "path_flow"):
        state_rows = [row for row in selected if row["state"] == state]
        transport = {int(row["event_id"]): float(row["normalized_l1"]) for row in state_rows if row["initialization"] == "transport_repaired"}
        comparisons: dict[str, Any] = {}
        all_pass = True
        for control in ("zero", "global_mean", "no_history_equal_split", "direct_copy_no_repair"):
            baseline = {int(row["event_id"]): float(row["normalized_l1"]) for row in state_rows if row["initialization"] == control}
            keys = sorted(set(transport) & set(baseline))
            benefit = np.asarray([baseline[key] - transport[key] for key in keys])
            pvalue = (
                float(wilcoxon(benefit, alternative="greater").pvalue)
                if len(benefit) and not np.allclose(benefit, 0.0) else 1.0
            )
            rng = np.random.default_rng(42)
            ci = None
            if len(benefit):
                indices = rng.integers(0, len(benefit), size=(BOOTSTRAP_SAMPLES, len(benefit)))
                boot = np.median(benefit[indices], axis=1)
                ci = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
            baseline_median = float(np.median([baseline[key] for key in keys])) if keys else 0.0
            relative = None if baseline_median == 0.0 else float(np.median(benefit) / baseline_median)
            comparison_pass = pvalue < 0.05 and ci is not None and ci[0] > 0.0 and relative is not None and relative >= 0.20
            all_pass &= comparison_pass
            comparisons[control] = {
                "count": len(keys), "median_benefit": float(np.median(benefit)) if len(benefit) else None,
                "relative_median_benefit": relative, "wilcoxon_pvalue": pvalue,
                "bootstrap_95_ci": ci, "pass": comparison_pass,
            }
        details[state] = {
            "median_distance_by_initialization": {
                initialization: _median(
                    float(row["normalized_l1"]) for row in state_rows
                    if row["initialization"] == initialization
                )
                for initialization in sorted({str(row["initialization"]) for row in state_rows})
            },
            "comparisons": comparisons,
        }
        if all_pass:
            passed.append(state)
    auxiliary = {}
    for state in ("binding", "dual"):
        state_rows = [row for row in selected if row["state"] == state]
        auxiliary[state] = {
            "median_distance_by_initialization": {
                initialization: _median(
                    float(row["normalized_l1"]) for row in state_rows
                    if row["initialization"] == initialization
                )
                for initialization in sorted({str(row["initialization"]) for row in state_rows})
            },
            "informative_event_fraction": (
                sum(str(row.get("informative", "True")).lower() == "true" for row in state_rows)
                / len(state_rows) if state_rows else None
            ),
        }
    return ("PASS" if passed else "FAIL"), {
        "qualified_states": passed, "states": details, "auxiliary_states": auxiliary,
    }


def _severity_analysis(solver_rows: Sequence[Mapping[str, Any]], best_classical: str) -> list[dict[str, Any]]:
    rows = [
        row for row in solver_rows
        if row["pair_kind"] == "TRANSITION" and row["slice_type"] == "HIGH_PRESSURE_SLICE"
        and float(row["demand_scale"]) == 1.0 and row["method"] == best_classical
    ]
    invalidation = np.asarray([float(row["candidate_path_invalidated_fraction"]) for row in rows])
    iterations = np.asarray([float(row["iter_count"]) for row in rows])
    optimize = np.asarray([float(row["optimize_wall_time"]) for row in rows])
    change = np.asarray([float(row.get("edge_change_count", 0.0)) for row in rows])
    tertiles = np.quantile(invalidation, [1 / 3, 2 / 3]) if len(invalidation) else [0.0, 0.0]
    output: list[dict[str, Any]] = []
    for row, value in zip(rows, invalidation):
        level = "small" if value <= tertiles[0] else "medium" if value <= tertiles[1] else "large"
        output.append({
            "event_id": row["event_id"], "severity_stratum": level,
            "edge_change_count": row.get("edge_change_count", 0),
            "candidate_path_invalidated_fraction": value,
            "classical_iter_count": row["iter_count"],
            "classical_optimize_wall_time": row["optimize_wall_time"],
        })
    if output:
        invalidation_iter = spearmanr(invalidation, iterations) if np.ptp(invalidation) > 0 and np.ptp(iterations) > 0 else None
        invalidation_time = spearmanr(invalidation, optimize) if np.ptp(invalidation) > 0 and np.ptp(optimize) > 0 else None
        edge_iter = spearmanr(change, iterations) if np.ptp(change) > 0 and np.ptp(iterations) > 0 else None
        for row in output:
            row["path_invalidation_vs_iteration_spearman"] = invalidation_iter.statistic if invalidation_iter else "UNDEFINED"
            row["path_invalidation_vs_iteration_pvalue"] = invalidation_iter.pvalue if invalidation_iter else "UNDEFINED"
            row["path_invalidation_vs_optimize_time_spearman"] = invalidation_time.statistic if invalidation_time else "UNDEFINED"
            row["path_invalidation_vs_optimize_time_pvalue"] = invalidation_time.pvalue if invalidation_time else "UNDEFINED"
            row["edge_change_vs_iteration_spearman"] = edge_iter.statistic if edge_iter else "UNDEFINED"
            row["edge_change_vs_iteration_pvalue"] = edge_iter.pvalue if edge_iter else "UNDEFINED"
    return output


def _regime_paired_comparison(
    rows: Sequence[Mapping[str, Any]], method: str,
    slice_type: str = "HIGH_PRESSURE_SLICE",
) -> dict[str, Any]:
    """Pair each event's stable control with its topology-transition solve."""

    selected = [
        row for row in rows
        if row["method"] == method and row["slice_type"] == slice_type
        and float(row["demand_scale"]) == 1.0
    ]
    stable = {int(row["event_id"]): row for row in selected if row["pair_kind"] == "STABLE"}
    transition = {int(row["event_id"]): row for row in selected if row["pair_kind"] == "TRANSITION"}
    keys = sorted(set(stable) & set(transition))
    iteration_delta = np.asarray([
        float(transition[key]["iter_count"]) - float(stable[key]["iter_count"]) for key in keys
    ])
    time_log_increase = np.asarray([
        math.log(float(transition[key]["optimize_wall_time"]) / float(stable[key]["optimize_wall_time"]))
        for key in keys
    ])

    def paired(values: np.ndarray) -> dict[str, Any]:
        pvalue = (
            float(wilcoxon(values, alternative="greater").pvalue)
            if len(values) and not np.allclose(values, 0.0) else 1.0
        )
        ci = None
        if len(values):
            rng = np.random.default_rng(42)
            indices = rng.integers(0, len(values), size=(BOOTSTRAP_SAMPLES, len(values)))
            boot = np.median(values[indices], axis=1)
            ci = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
        return {
            "count": len(values), "median": float(np.median(values)) if len(values) else None,
            "bootstrap_median_95_ci": ci, "wilcoxon_one_sided_pvalue": pvalue,
        }

    return {
        "iteration_increase_transition_minus_stable": paired(iteration_delta),
        "log_optimize_time_increase_transition_over_stable": paired(time_log_increase),
    }


def _congestion_control_comparison(
    rows: Sequence[Mapping[str, Any]], metric: str,
) -> dict[str, Any]:
    """Compare high-pressure selection with each demand-matched control by event."""

    official = [row for row in rows if float(row["demand_scale"]) == 1.0]
    high = {
        int(row["event_id"]): float(row[metric]) for row in official
        if row["slice_type"] == "HIGH_PRESSURE_SLICE"
    }
    result: dict[str, Any] = {}
    for control in ("PERSISTENCE_SLICE", "DEMAND_MATCHED_RANDOM_SLICE"):
        baseline = {
            int(row["event_id"]): float(row[metric]) for row in official
            if row["slice_type"] == control
        }
        keys = sorted(set(high) & set(baseline))
        differences = np.asarray([high[key] - baseline[key] for key in keys])
        pvalue = (
            float(wilcoxon(differences, alternative="greater").pvalue)
            if len(differences) and not np.allclose(differences, 0.0) else 1.0
        )
        rng = np.random.default_rng(42)
        ci = None
        if len(differences):
            indices = rng.integers(0, len(differences), size=(BOOTSTRAP_SAMPLES, len(differences)))
            boot = np.median(differences[indices], axis=1)
            ci = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
        result[control] = {
            "count": len(keys),
            "median_high_pressure_minus_control": float(np.median(differences)) if len(differences) else None,
            "bootstrap_median_95_ci": ci, "wilcoxon_one_sided_pvalue": pvalue,
        }
    return result


def _build_summary(
    all_solver: list[dict[str, Any]], all_state: list[dict[str, Any]],
    congestion: list[dict[str, Any]], scanner_summary: Mapping[str, Any],
    gurobi: Mapping[str, Any], event_count: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    stable_by_slice = {
        slice_type: _method_summary(all_solver, "STABLE", slice_type=slice_type)
        for slice_type in SLICE_TYPES
    }
    transition_by_slice = {
        slice_type: _method_summary(all_solver, "TRANSITION", slice_type=slice_type)
        for slice_type in SLICE_TYPES
    }
    # Gates use one real event once.  Selection controls are never pooled as
    # independent replicates of the same topology transition.
    stable = stable_by_slice["HIGH_PRESSURE_SLICE"]
    transition = transition_by_slice["HIGH_PRESSURE_SLICE"]
    best_classical = min(
        CLASSICAL_METHODS,
        key=lambda method: (
            transition[method]["iter_count"]["median"],
            transition[method]["optimize_time"]["median"],
        ),
    )
    stable_iter = stable.get(best_classical, {}).get("iter_count", {}).get("median")
    transition_iter = transition[best_classical]["iter_count"]["median"]
    stable_time = stable.get(best_classical, {}).get("optimize_time", {}).get("median")
    transition_time = transition[best_classical]["optimize_time"]["median"]
    time_increase = (
        None if stable_time in (None, 0.0) else transition_time / stable_time - 1.0
    )
    t1 = (
        "NO_DEGRADATION"
        if stable_iter is not None and transition_iter <= 1.0 and time_increase is not None and time_increase < 0.20
        else "DEGRADATION"
    )
    t2, t2_details = _state_gate(all_state, 1.0)
    iter_stats = _paired_stats(all_solver, best_classical, "TRANSPORTED_PSTART_DSTART", "iter_count")
    time_stats = _paired_log_time_stats(all_solver, best_classical, "TRANSPORTED_PSTART_DSTART")
    total_time_stats = _paired_log_time_stats(
        all_solver, best_classical, "TRANSPORTED_PSTART_DSTART",
        metric="operational_total_wall_time",
    )
    iter_pass = (
        iter_stats["median_reduction"] is not None and iter_stats["median_reduction"] >= 0.20
        and iter_stats["bootstrap_median_difference_95_ci"] is not None
        and iter_stats["bootstrap_median_difference_95_ci"][0] > 0.0
    )
    time_pass = (
        time_stats["median_time_reduction"] is not None
        and time_stats["median_time_reduction"] >= 0.20
        and time_stats["bootstrap_median_log_speedup_95_ci"] is not None
        and time_stats["bootstrap_median_log_speedup_95_ci"][0] > 0.0
    )
    parity = all(bool(row["objective_parity"]) and bool(row["feasibility_pass"]) for row in all_solver)
    t3 = "PASS" if (iter_pass or time_pass) and parity else "FAIL"
    official = [row for row in congestion if float(row["demand_scale"]) == 1.0]
    official_informative = any(
        int(row["binding_edge_count"]) > 0 or int(row["nonzero_capacity_dual_count"]) > 0
        or float(row["unsatisfied_demand"]) > 1e-7 for row in official
    )
    stress = [row for row in congestion if float(row["demand_scale"]) > 1.0]
    stress_informative = any(
        int(row["binding_edge_count"]) > 0 or int(row["nonzero_capacity_dual_count"]) > 0
        or float(row["unsatisfied_demand"]) > 1e-7 for row in stress
    )
    t4 = "PASS_OFFICIAL_REAL_WORKLOAD" if official_informative else "STRESS_ONLY" if stress_informative else "FAIL_UNINFORMATIVE"
    t0 = "STRONG" if event_count >= 50 else "PASS" if event_count >= 30 else "BLOCKED"
    if t0 == "BLOCKED":
        decision = "NOT_AUTHORIZED_T0_BLOCKED"
    elif t1 == "NO_DEGRADATION":
        decision = "STOP_TOPOLOGY_TRANSITION_TRACKING"
    elif t2 == "PASS" and t3 == "PASS":
        decision = "CONTINUE_TOPOLOGY_TRANSITION_STATE_TRANSPORT"
    else:
        decision = "STOP_TOPOLOGY_TRANSITION_TRACKING"
    severity_rows = _severity_analysis(all_solver, best_classical)
    regime_paired = _regime_paired_comparison(all_solver, best_classical)
    congestion_by_slice = {
        slice_type: {
            "count": len(slice_rows),
            "binding_rate": _distribution([float(row["binding_rate"]) for row in slice_rows]),
            "nonzero_dual_rate": _distribution([float(row["nonzero_capacity_dual_rate"]) for row in slice_rows]),
            "unsatisfied_event_fraction": (
                sum(float(row["unsatisfied_demand"]) > 1e-7 for row in slice_rows) / len(slice_rows)
                if slice_rows else None
            ),
            "median_unsatisfied_demand": _median(float(row["unsatisfied_demand"]) for row in slice_rows),
        }
        for slice_type in SLICE_TYPES
        for slice_rows in [[row for row in official if row["slice_type"] == slice_type]]
    }
    severity_summary = {
        "edge_change_count": _distribution([float(row["edge_change_count"]) for row in severity_rows]),
        "candidate_path_invalidated_fraction": _distribution(
            [float(row["candidate_path_invalidated_fraction"]) for row in severity_rows]
        ),
        "path_invalidation_vs_iteration_spearman": (
            severity_rows[0]["path_invalidation_vs_iteration_spearman"] if severity_rows else None
        ),
        "path_invalidation_vs_optimize_time_spearman": (
            severity_rows[0]["path_invalidation_vs_optimize_time_spearman"] if severity_rows else None
        ),
        "path_invalidation_vs_iteration_pvalue": (
            severity_rows[0]["path_invalidation_vs_iteration_pvalue"] if severity_rows else None
        ),
        "path_invalidation_vs_optimize_time_pvalue": (
            severity_rows[0]["path_invalidation_vs_optimize_time_pvalue"] if severity_rows else None
        ),
    }
    summary = {
        "raw_record_count": scanner_summary["raw_record_count"],
        "available_transition_count": scanner_summary["transition_count"],
        "evaluated_transition_count": event_count,
        "regime_statistics": scanner_summary["regime_statistics"],
        "official_workload_intensity": 25,
        "gurobi": dict(gurobi), "best_classical_warm_start": best_classical,
        "stable_warm_start_by_slice": stable_by_slice,
        "transition_warm_start_by_slice": transition_by_slice,
        "gate_primary_slice": "HIGH_PRESSURE_SLICE",
        "classical_transition_degradation": {
            "stable_median_iter_count": stable_iter,
            "transition_median_iter_count": transition_iter,
            "stable_median_optimize_time": stable_time,
            "transition_median_optimize_time": transition_time,
            "transition_optimize_time_increase": time_increase,
            "paired_statistics": regime_paired,
        },
        "transport_state": t2_details,
        "transport_solver": {
            "comparison_method": "TRANSPORTED_PSTART_DSTART",
            "iteration": iter_stats, "optimize_time_log_ratio": time_stats,
            "end_to_end_time_log_ratio": total_time_stats,
            "objective_feasibility_parity": "PASS" if parity else "FAIL",
        },
        "congestion": {
            "official_real_informative": official_informative,
            "official_binding_rate_median": _median(float(row["binding_rate"]) for row in official),
            "official_nonzero_dual_rate_median": _median(float(row["nonzero_capacity_dual_rate"]) for row in official),
            "official_unsatisfied_demand_rate": sum(float(row["unsatisfied_demand"]) > 1e-7 for row in official) / len(official) if official else None,
            "stress_informative": stress_informative,
            "stress_scales_run": sorted({float(row["demand_scale"]) for row in stress}),
            "official_by_slice": congestion_by_slice,
            "selection_control_statistics": {
                "binding_rate": _congestion_control_comparison(congestion, "binding_rate"),
                "unsatisfied_demand": _congestion_control_comparison(congestion, "unsatisfied_demand"),
            },
        },
        "transition_severity": severity_summary,
        "gates": {"T0": t0, "T1": t1, "T2": t2, "T3": t3, "T4": t4},
        "final_decision": decision,
        "oracle_future_topology_label": "ORACLE_RECORDED_SEQUENCE",
        "ephemeris_claim": "NOT_EVALUATED_NO_PHYSICAL_CADENCE_OR_EPHEMERIS",
    }
    return summary, severity_rows


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Execute scan-qualified events serially with per-event resume checkpoints."""

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.transition_manifest)
    scanner_summary_path = Path(args.transition_summary)
    with scanner_summary_path.open("r", encoding="utf-8-sig") as handle:
        scanner_summary = json.load(handle)
    manifest_rows, raw_lookup = _manifest_rows(manifest_path)
    events = read_transition_events(manifest_path, args.max_events, args.window_radius)
    if not events:
        raise RuntimeError("No real routing topology transitions were selected")
    raw_path = Path(args.raw_path)
    required_indices = {index for event in events for index in event.window_indices}
    raw_records = load_raw_indices(raw_path, required_indices)
    gurobi = probe_gurobi(check_size_limit=True)
    if not gurobi.get("available"):
        raise RuntimeError("Gurobi formal evidence is required")
    if gurobi.get("license_size_restriction") != "RESTRICTED_SIZE_LIMIT_CONFIRMED":
        raise RuntimeError("This run requires the probed restricted-license contract")
    dual_sign = float(calibrate_dual_sign("gurobi")["sign_multiplier"])
    policy = CapacityPolicy(args.network_edge_capacity, args.path_only_edge_capacity)

    payloads: list[dict[str, Any]] = []
    for event in events:
        slices, manifests, _ = build_event_slices(
            event, raw_records, intensity=args.intensity, policy=policy,
            constraint_budget=args.constraint_budget, random_seed=args.seed,
        )
        raw_event = raw_lookup[(event.sequence_id, event.transition_index)]
        severity = {
            "edge_birth_count": int(raw_event["added_edges"]),
            "edge_death_count": int(raw_event["deleted_edges"]),
            "edge_change_count": int(raw_event["edge_change_count"]),
            "edge_jaccard": float(raw_event["edge_jaccard"]),
            "degree_change_l1": int(raw_event["degree_change_l1"]),
            "raw_active_flow_churn": float(raw_event["active_flow_churn"]),
            "raw_total_demand_delta": float(raw_event["total_demand_delta"]),
        }
        manifest_dir = output_dir / "slice_manifests"
        for name, manifest in manifests.items():
            _write_json(manifest_dir / f"event_{event.event_id:04d}_{name.lower()}.json", manifest)
        payload = _load_or_run_event(
            output_dir, event, slices, manifests, severity, raw_lookup, policy,
            dual_sign, args.constraint_budget, 1.0, args.resume,
        )
        payloads.append(payload)
        # Write progress after every event; output remains useful after interruption.
        _write_json(output_dir / "progress.json", {
            "completed_events": len(payloads), "target_events": len(events),
            "last_transition_index": event.transition_index,
        })

    congestion_official = [row for payload in payloads for row in payload["congestion_rows"]]
    official_pressure_informative = any(
        row["slice_type"] == "HIGH_PRESSURE_SLICE"
        and (int(row["binding_edge_count"]) > 0 or int(row["nonzero_capacity_dual_count"]) > 0
             or float(row["unsatisfied_demand"]) > 1e-7)
        for row in congestion_official
    )
    if not official_pressure_informative and not args.skip_stress:
        # Stress is a mechanism diagnostic only.  Stop at the first scale that
        # makes the high-pressure slice informative across at least one event.
        for scale in (1.5, 2.0, 4.0):
            scale_payloads: list[dict[str, Any]] = []
            for event in events:
                slices, manifests, _ = build_event_slices(
                    event, raw_records, intensity=args.intensity, policy=policy,
                    constraint_budget=args.constraint_budget, random_seed=args.seed,
                )
                raw_event = raw_lookup[(event.sequence_id, event.transition_index)]
                severity = {
                    "edge_birth_count": int(raw_event["added_edges"]),
                    "edge_death_count": int(raw_event["deleted_edges"]),
                    "edge_change_count": int(raw_event["edge_change_count"]),
                    "edge_jaccard": float(raw_event["edge_jaccard"]),
                    "degree_change_l1": int(raw_event["degree_change_l1"]),
                    "raw_active_flow_churn": float(raw_event["active_flow_churn"]),
                    "raw_total_demand_delta": float(raw_event["total_demand_delta"]),
                }
                # Scaled stress is evaluated on high-pressure selection only;
                # the official three-way selection control remains scale 1.0.
                slices = {"HIGH_PRESSURE_SLICE": slices["HIGH_PRESSURE_SLICE"]}
                manifests = {"HIGH_PRESSURE_SLICE": {**manifests["HIGH_PRESSURE_SLICE"], "demand_scale": scale,
                                                       "evidence_label": "REAL_TOPOLOGY_REAL_TRAFFIC_SCALED_STRESS"}}
                payload = _load_or_run_event(
                    output_dir, event, slices, manifests, severity, raw_lookup, policy,
                    dual_sign, args.constraint_budget, scale, args.resume,
                )
                scale_payloads.append(payload)
            payloads.extend(scale_payloads)
            if any(
                int(row["binding_edge_count"]) > 0 or int(row["nonzero_capacity_dual_count"]) > 0
                or float(row["unsatisfied_demand"]) > 1e-7
                for payload in scale_payloads for row in payload["congestion_rows"]
            ):
                break

    transition_solver = [row for payload in payloads for row in payload["solver_rows"]]
    stable_solver = [row for payload in payloads for row in payload["stable_rows"]]
    all_solver = transition_solver + stable_solver
    state_rows = [row for payload in payloads for row in payload["state_rows"]]
    congestion_rows = [row for payload in payloads for row in payload["congestion_rows"]]
    event_rows = [
        {
            **asdict(event),
            **{key: raw_lookup[(event.sequence_id, event.transition_index)][key] for key in (
                "added_edges", "deleted_edges", "edge_change_count", "edge_jaccard",
                "degree_change_l1", "active_flow_churn", "total_demand_delta",
            )},
        }
        for event in events
    ]
    _write_csv(output_dir / "raw_transition_manifest.csv", manifest_rows)
    _write_csv(output_dir / "transition_events.csv", event_rows)
    _write_json(output_dir / "regime_statistics.json", scanner_summary["regime_statistics"])
    _write_csv(output_dir / "transition_lp_records.csv", [row for row in transition_solver if row["method"] == "COLD"])
    if stable_solver:
        _write_csv(output_dir / "stable_lp_records.csv", [row for row in stable_solver if row["method"] == "COLD"])
    _write_csv(output_dir / "warm_start_transition.csv", [row for row in transition_solver if row["method"] in {"COLD", *CLASSICAL_METHODS, "RESET"}])
    _write_csv(output_dir / "transport_state_records.csv", state_rows)
    _write_csv(output_dir / "transported_solver_records.csv", [row for row in transition_solver if row["method"] in {"TRANSPORTED_PSTART", "TRANSPORTED_PSTART_DSTART", "ZERO_START"}])
    _write_csv(output_dir / "congestion_statistics.csv", congestion_rows)
    summary, severity_rows = _build_summary(
        all_solver, state_rows, congestion_rows, scanner_summary, gurobi, len(events)
    )
    _write_csv(output_dir / "severity_analysis.csv", severity_rows)
    provenance = {
        "source_file": str(raw_path), "source_sha256": _sha256(raw_path),
        "pipeline_version": PIPELINE_VERSION,
        "source_sequence_id": events[0].sequence_id, "official_workload_intensity": args.intensity,
        "raw_record_count": scanner_summary["raw_record_count"],
        "available_transition_count": scanner_summary["transition_count"],
        "evaluated_transition_count": len(events), "selection": "EARLIEST_TRANSITIONS_NO_SOLVER_OUTCOME_FILTER",
        "starting_git_head": _git_head(), "threads": 1, "seed": args.seed,
        "capacity_policy": asdict(policy), "constraint_budget": args.constraint_budget,
        "timestamp_or_orbit_epoch_present": False, "physical_cadence_claim": "ORDERED_ONLY",
        "future_topology_label": "ORACLE_RECORDED_SEQUENCE", "gurobi": gurobi,
    }
    _write_json(output_dir / "provenance.json", provenance)
    summary["provenance"] = provenance
    _write_json(output_dir / "summary.json", summary)
    _plot_results(output_dir / "figures", event_rows, all_solver, state_rows, congestion_rows, summary)
    return summary


def _sha256(path: Path) -> str:
    digest = __import__("hashlib").sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _plot_results(
    output_dir: Path, events: Sequence[Mapping[str, Any]], solver: Sequence[Mapping[str, Any]],
    states: Sequence[Mapping[str, Any]], congestion: Sequence[Mapping[str, Any]], summary: Mapping[str, Any],
) -> None:
    """Generate compact audit figures without changing any statistical decision."""

    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)

    def save(name: str) -> None:
        plt.tight_layout()
        plt.savefig(output_dir / name, dpi=160)
        plt.close()

    regime = summary["regime_statistics"]
    plt.bar(["P10", "P50", "P90", "max"], [regime["p10"], regime["p50"], regime["p90"], regime["maximum"]])
    plt.ylabel("records")
    save("figure_01_regime_length.png")

    edge_changes = [int(row["edge_change_count"]) for row in events]
    plt.hist(edge_changes, bins=min(20, max(1, len(set(edge_changes)))))
    plt.xlabel("edge change count")
    save("figure_02_transition_severity.png")

    best = summary["best_classical_warm_start"]
    official = [row for row in solver if float(row["demand_scale"]) == 1.0 and row["method"] == best]
    for metric, name, ylabel in (
        ("iter_count", "figure_03_stable_vs_transition_iterations.png", "IterCount"),
        ("optimize_wall_time", "figure_04_stable_vs_transition_optimize_time.png", "seconds"),
    ):
        groups = [[float(row[metric]) for row in official if row["pair_kind"] == kind] for kind in ("STABLE", "TRANSITION")]
        plt.boxplot(groups, tick_labels=["stable", "transition"])
        plt.ylabel(ylabel)
        save(name)

    hp = [row for row in official if row["slice_type"] == "HIGH_PRESSURE_SLICE" and row["pair_kind"] == "TRANSITION"]
    plt.scatter([float(row.get("edge_change_count", 0)) for row in hp], [float(row["iter_count"]) for row in hp])
    plt.xlabel("edge change count"); plt.ylabel("IterCount")
    save("figure_05_severity_vs_iterations.png")
    plt.scatter([float(row["candidate_path_invalidated_fraction"]) for row in hp], [float(row["optimize_wall_time"]) for row in hp])
    plt.xlabel("path invalidation fraction"); plt.ylabel("optimize seconds")
    save("figure_06_path_invalidation_vs_cost.png")

    load_rows = [row for row in states if row["state"] == "edge_load" and row["slice_type"] == "HIGH_PRESSURE_SLICE"]
    init_names = ["transport_repaired", "zero", "global_mean", "no_history_equal_split", "direct_copy_no_repair"]
    plt.boxplot(
        [[float(row["normalized_l1"]) for row in load_rows if row["initialization"] == name] for name in init_names],
        tick_labels=["transport", "zero", "global", "no-history", "copy"],
    )
    plt.ylabel("normalized L1")
    save("figure_07_transport_state_distance.png")

    for metric, name, ylabel in (
        ("iter_count", "figure_08_classical_vs_transport_iterations.png", "IterCount"),
        ("optimize_wall_time", "figure_09_classical_vs_transport_time.png", "seconds"),
    ):
        groups = [
            [float(row[metric]) for row in solver if row["pair_kind"] == "TRANSITION" and float(row["demand_scale"]) == 1.0 and row["method"] == method]
            for method in (best, "TRANSPORTED_PSTART_DSTART")
        ]
        plt.boxplot(groups, tick_labels=["best classical", "transport"])
        plt.ylabel(ylabel)
        save(name)

    official_congestion = [row for row in congestion if float(row["demand_scale"]) == 1.0]
    slice_names = list(SLICE_TYPES)
    plt.bar(
        range(len(slice_names)),
        [_median(float(row["binding_rate"]) for row in official_congestion if row["slice_type"] == name) or 0.0 for name in slice_names],
    )
    plt.xticks(range(len(slice_names)), ["persistence", "high-pressure", "random"], rotation=15)
    plt.ylabel("median binding rate")
    save("figure_10_official_congestion.png")

    stress = [row for row in congestion if float(row["demand_scale"]) > 1.0]
    if stress:
        scales = sorted({float(row["demand_scale"]) for row in congestion})
        plt.plot(scales, [_median(float(row["binding_rate"]) for row in congestion if float(row["demand_scale"]) == scale) or 0.0 for scale in scales], marker="o")
        plt.xlabel("demand scale"); plt.ylabel("median binding rate")
        save("figure_s1_scaling_vs_binding.png")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-path", required=True)
    parser.add_argument("--transition-manifest", default="output/temporal_feasibility/transitions/raw_transition_manifest.csv")
    parser.add_argument("--transition-summary", default="output/temporal_feasibility/transitions/transition_summary.json")
    parser.add_argument("--output-dir", default="output/temporal_feasibility/topology_transition")
    parser.add_argument("--max-events", type=int, default=50)
    parser.add_argument("--window-radius", type=int, default=2)
    parser.add_argument("--intensity", type=int, default=25)
    parser.add_argument("--constraint-budget", type=int, default=1900)
    parser.add_argument("--network-edge-capacity", type=float, default=200.0)
    parser.add_argument("--path-only-edge-capacity", type=float, default=800.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-stress", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    summary = run(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
