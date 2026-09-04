"""Qualification tests for topology-transition scanning and transport."""

from __future__ import annotations

import csv
import inspect
import json
import pickle
from pathlib import Path

import pytest

from experiments.temporal_feasibility.extract_primal_dual import calibrate_dual_sign
from experiments.temporal_feasibility.scan_transitions import RawSource, scan_sources
from experiments.temporal_feasibility.semantic_alignment import normalized_l1
from experiments.temporal_feasibility.sequential_lp import (
    CapacityPolicy, LPUniverse, ScipySequentialLP, capacities_for_snapshot,
    gurobi_available,
)
from experiments.temporal_feasibility.state_transport import (
    canonical_secondary_costs, direct_previous_path_copy, repair_path_flow,
    transport_edge_state, transported_advanced_start,
)
from experiments.temporal_feasibility.synthetic_fixture import make_fixture, write_fixture
from experiments.temporal_feasibility.sequence_schema import load_dataset
from experiments.temporal_feasibility.transition_dataset import (
    TransitionEvent, build_event_slices, event_path_severity, load_raw_indices,
    read_transition_events, slice_manifest_digest,
)
from experiments.temporal_feasibility.transition_reoptimization import (
    PIPELINE_VERSION, _load_or_run_event, _method_summary, _paired_stats, _scale_snapshots,
    _solve_pair,
)


def _raw_record(link: int, demand: float = 10.0) -> dict:
    return {
        "InterShell_GrdRelay": [-1.0] * 4,
        "InterShell_ISL": [[link], [], []],
        "FlowSet": [[0, 1, demand], [2, 3, demand], [4, 5, demand], [6, 7, demand]],
    }


def _write_stream(path: Path, records: list[dict]) -> Path:
    with path.open("wb") as handle:
        for record in records:
            pickle.dump(record, handle)
    return path


def _write_manifest(path: Path, transitions: list[int]) -> Path:
    fields = ["sequence_id", "source_file", "source_record_index", "transition"]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index in range(max(transitions) + 2):
            writer.writerow({
                "sequence_id": "seq", "source_file": "raw.pkl",
                "source_record_index": index, "transition": index in transitions,
            })
    return path


def test_full_stream_transition_scanner_reads_until_eof(tmp_path: Path) -> None:
    source = _write_stream(tmp_path / "raw.pkl", [_raw_record(0), _raw_record(0), _raw_record(1)])
    rows, summary = scan_sources([RawSource("seq", source)])
    assert len(rows) == 3
    assert summary["raw_record_count"] == 3
    assert summary["transition_count"] == 1


def test_scanner_never_compares_across_sequence_boundary(tmp_path: Path) -> None:
    first = _write_stream(tmp_path / "a.pkl", [_raw_record(0)])
    second = _write_stream(tmp_path / "b.pkl", [_raw_record(1)])
    rows, summary = scan_sources([RawSource("A", first), RawSource("B", second)])
    assert len(rows) == 2
    assert summary["transition_count"] == 0
    assert summary["sequence_boundary_comparisons"] == 0
    assert rows[1]["previous_routing_graph_hash"] == ""


def test_event_window_extraction_is_centered_and_bounded(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path / "manifest.csv", [2, 5])
    events = read_transition_events(manifest, max_events=2, radius=2)
    assert events[0].window_indices == (0, 1, 2, 3, 4)
    assert events[1].window_indices == (3, 4, 5, 6, 7)


def test_load_raw_indices_uses_requested_event_window(tmp_path: Path) -> None:
    source = _write_stream(tmp_path / "raw.pkl", [_raw_record(index % 2) for index in range(8)])
    records = load_raw_indices(source, {1, 3, 7})
    assert sorted(records) == [1, 3, 7]


def test_high_pressure_selection_is_deterministic_and_control_matched(tmp_path: Path) -> None:
    records = {index: _raw_record(0 if index < 2 else 1, 10.0 + index) for index in range(5)}
    event = TransitionEvent(0, "seq", "raw.pkl", 2, (0, 1, 2, 3, 4))
    left = build_event_slices(event, records, intensity=25, policy=CapacityPolicy())
    right = build_event_slices(event, records, intensity=25, policy=CapacityPolicy())
    left_manifests, right_manifests = left[1], right[1]
    assert left_manifests == right_manifests
    assert len({manifest["flow_count"] for manifest in left_manifests.values()}) == 1
    assert left_manifests["DEMAND_MATCHED_RANDOM_SLICE"]["demand_match_exact_to_high_pressure"]
    assert 0.8 <= left_manifests["DEMAND_MATCHED_RANDOM_SLICE"]["path_count_ratio_to_high_pressure"] <= 1.25


def test_high_pressure_selection_has_no_solver_output_input(tmp_path: Path) -> None:
    records = {index: _raw_record(0 if index < 2 else 1) for index in range(5)}
    event = TransitionEvent(0, "seq", "raw.pkl", 2, (0, 1, 2, 3, 4))
    _, manifests, _ = build_event_slices(event, records, intensity=25, policy=CapacityPolicy())
    assert all(not manifest["selection_uses_solver_output"] for manifest in manifests.values())
    parameters = set(inspect.signature(build_event_slices).parameters)
    assert not parameters.intersection({"objective", "pi", "binding", "iter_count", "optimal_path_flow"})


def test_restricted_license_budget_is_enforced(tmp_path: Path) -> None:
    records = {index: _raw_record(0 if index < 2 else 1) for index in range(5)}
    event = TransitionEvent(0, "seq", "raw.pkl", 2, (0, 1, 2, 3, 4))
    _, manifests, _ = build_event_slices(
        event, records, intensity=25, policy=CapacityPolicy(), constraint_budget=1900,
    )
    assert all(manifest["constraint_count"] <= 1900 for manifest in manifests.values())


def test_pair_universe_and_path_availability_are_fixed(tmp_path: Path) -> None:
    snapshots = load_dataset(write_fixture(tmp_path / "fixture.pkl", 16))[3:5]
    universe = LPUniverse.from_snapshots(snapshots)
    assert universe.paths == tuple(sorted(snapshots[0].paths | snapshots[1].paths))
    solver = ScipySequentialLP(universe, CapacityPolicy())
    state = solver.solve(snapshots[1], "cold_rebuild")
    assert all(value <= 1e-8 for path, value in state.path_flow.items() if path not in snapshots[1].paths)
    assert all(capacities_for_snapshot(snapshots[0], universe, CapacityPolicy())[edge] == 0.0
               for edge in universe.edges if edge not in snapshots[0].edges)


def test_path_transport_handles_persistent_new_deleted_and_repairs(tmp_path: Path) -> None:
    snapshots = load_dataset(write_fixture(tmp_path / "fixture.pkl", 16))[3:5]
    universe = LPUniverse.from_snapshots(snapshots)
    previous = ScipySequentialLP(universe, CapacityPolicy()).solve(snapshots[0], "cold_rebuild")
    direct = direct_previous_path_copy(previous, snapshots[1], universe)
    assert all(direct[path] == 0.0 for path in universe.paths if path not in snapshots[1].paths)
    repaired, audit = repair_path_flow(direct, snapshots[1], universe, CapacityPolicy())
    assert max(audit.values()) <= 1e-7
    assert set(repaired) == set(universe.paths)


def test_edge_transport_handles_persistent_new_deleted() -> None:
    result = transport_edge_state(
        {(0, 1): 2.0, (1, 2): 4.0}, frozenset({(0, 1), (1, 2)}),
        frozenset({(1, 2), (2, 3)}), "neighbor_mean",
    )
    assert (0, 1) not in result
    assert result[(1, 2)] == 4.0
    assert result[(2, 3)] == 4.0


def test_transported_pstart_dstart_and_best_classical_have_parity(tmp_path: Path) -> None:
    available, detail = gurobi_available()
    if not available:
        pytest.skip(detail)
    pair = load_dataset(write_fixture(tmp_path / "fixture.pkl", 16))[3:5]
    event = TransitionEvent(0, "synthetic", "fixture.pkl", 4, (2, 3, 4, 5))
    rows, previous, _, audit = _solve_pair(
        pair, event, "HIGH_PRESSURE_SLICE", "TRANSITION", CapacityPolicy(),
        float(calibrate_dual_sign("gurobi")["sign_multiplier"]), 1900,
    )
    assert max(audit["feasibility"].values()) <= 1e-7
    assert all(row["objective_parity"] and row["feasibility_pass"] for row in rows)
    start, _ = transported_advanced_start(previous, pair[1], LPUniverse.from_snapshots(pair), CapacityPolicy())
    assert len(start.path_flow) == len(LPUniverse.from_snapshots(pair).paths)
    assert start.capacity_dual is not None and start.demand_dual is not None


def test_stable_transition_conditioned_metrics_are_not_pooled() -> None:
    rows = [
        {"pair_kind": kind, "demand_scale": 1.0, "slice_type": slice_type,
         "method": "AUTO_MODEL_REUSE", "iter_count": value,
         "optimize_wall_time": 0.1, "operational_total_wall_time": 0.2}
        for kind, slice_type, value in (
            ("STABLE", "HIGH_PRESSURE_SLICE", 0),
            ("TRANSITION", "HIGH_PRESSURE_SLICE", 3),
            ("TRANSITION", "PERSISTENCE_SLICE", 99),
        )
    ]
    summary = _method_summary(rows, "TRANSITION", slice_type="HIGH_PRESSURE_SLICE")
    assert summary["AUTO_MODEL_REUSE"]["iter_count"]["median"] == 3


def test_path_churn_severity_calculation(tmp_path: Path) -> None:
    pair = load_dataset(write_fixture(tmp_path / "fixture.pkl", 16))[3:5]
    severity = event_path_severity(pair)
    assert 0.0 <= severity["candidate_path_survival_ratio"] <= 1.0
    assert severity["candidate_path_invalidated_fraction"] >= 0.0
    assert severity["number_of_flows_affected_by_transition"] >= 0


def test_bootstrap_paired_test_uses_event_keys_once() -> None:
    rows = []
    for event in range(4):
        for method, value in (("BASE", 10.0), ("METHOD", 5.0)):
            rows.append({
                "event_id": event, "slice_type": "HIGH_PRESSURE_SLICE", "pair_start_index": event,
                "pair_kind": "TRANSITION", "demand_scale": 1.0, "method": method,
                "iter_count": value,
            })
    stats = _paired_stats(rows, "BASE", "METHOD", "iter_count")
    assert stats["count"] == 4
    assert stats["median_reduction"] == pytest.approx(0.5)
    assert stats["bootstrap_median_difference_95_ci"] == pytest.approx([5.0, 5.0])


def test_canonical_diagnostic_has_no_previous_solution_dependency(tmp_path: Path) -> None:
    snapshots = load_dataset(write_fixture(tmp_path / "fixture.pkl", 16))
    universe = LPUniverse.from_snapshots(snapshots)
    assert "previous" not in inspect.signature(canonical_secondary_costs).parameters
    assert canonical_secondary_costs(universe) == canonical_secondary_costs(universe)


def test_transition_output_resume_uses_manifest_identity(tmp_path: Path) -> None:
    event = TransitionEvent(0, "seq", "raw.pkl", 2, (0, 1, 2, 3, 4))
    manifest = {"HIGH_PRESSURE_SLICE": {"event_id": 0, "flows": [[0, 1]]}}
    signature = {"event_id": 0, "scale": 1.0, "pipeline_version": PIPELINE_VERSION,
                 "slice_digests": {name: slice_manifest_digest(value) for name, value in manifest.items()}}
    checkpoint = tmp_path / "checkpoints" / "event_0000_scale_1.json"
    checkpoint.parent.mkdir()
    expected = {"resume_signature": signature, "sentinel": "resumed"}
    checkpoint.write_text(json.dumps(expected), encoding="utf-8-sig")
    actual = _load_or_run_event(
        tmp_path, event, {}, manifest, {}, {}, CapacityPolicy(), 1.0, 1900, 1.0, True,
    )
    assert actual == expected


def test_real_and_stress_labels_propagate(tmp_path: Path) -> None:
    snapshots = load_dataset(write_fixture(tmp_path / "fixture.pkl", 16))
    snapshots = [
        snapshot.__class__(**{**snapshot.__dict__, "meta": {"evidence_label": "OFFICIAL_REAL_WORKLOAD"}})
        for snapshot in snapshots
    ]
    official = _scale_snapshots(snapshots, 1.0)
    stress = _scale_snapshots(snapshots, 2.0)
    assert all(snapshot.meta["evidence_label"] == "OFFICIAL_REAL_WORKLOAD" for snapshot in official)
    assert all(snapshot.meta["evidence_label"] == "REAL_TOPOLOGY_REAL_TRAFFIC_SCALED_STRESS" for snapshot in stress)
    assert stress[0].total_demand == pytest.approx(2.0 * official[0].total_demand)
