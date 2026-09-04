"""CPU-sized falsification tests for schema, LP parity, and dual semantics."""

from __future__ import annotations

from pathlib import Path
from argparse import Namespace
import json
import pickle

import pytest

from experiments.temporal_feasibility.extract_primal_dual import calibrate_dual_sign
from experiments.temporal_feasibility.inspect_dataset import inspect_snapshots
from experiments.temporal_feasibility.analyze_continuity import _transport_records, analyze
from experiments.temporal_feasibility.counterfactual import analyze_counterfactuals
from experiments.temporal_feasibility.gurobi_probe import probe_gurobi
from experiments.temporal_feasibility.real_sequence import select_persistent_flows
from experiments.temporal_feasibility.run_real_experiment import run
from experiments.temporal_feasibility.semantic_alignment import jaccard, topology_delta
from experiments.temporal_feasibility.sequence_schema import load_dataset
from experiments.temporal_feasibility.sequential_lp import (
    CapacityPolicy,
    GurobiSequentialLP,
    LPUniverse,
    ScipySequentialLP,
    gurobi_available,
)
from experiments.temporal_feasibility.smoke_test import run_synthetic_smoke
from experiments.temporal_feasibility.synthetic_fixture import make_fixture, write_fixture
from lib.data.starlink.adapter import StarlinkAdapter
import lib.data.starlink.adapter as starlink_adapter_module
from lib.data.starlink.ism import InterShellMode


def _snapshots(tmp_path: Path, disorder: bool = False):
    path = write_fixture(tmp_path / "fixture.pkl", 16, disorder)
    return load_dataset(path)


def test_semantic_path_key_ignores_list_order(tmp_path: Path) -> None:
    snapshots = _snapshots(tmp_path)
    assert snapshots[1].paths == snapshots[2].paths
    assert jaccard(snapshots[1].paths, snapshots[2].paths) == 1.0


def test_dynamic_edge_birth_and_death(tmp_path: Path) -> None:
    snapshots = _snapshots(tmp_path)
    persistent, new_edges, deleted_edges = topology_delta(snapshots[3], snapshots[4])
    assert (1, 2) in new_edges
    assert not deleted_edges
    persistent, new_edges, deleted_edges = topology_delta(snapshots[9], snapshots[10])
    assert (1, 2) in deleted_edges
    assert not new_edges


def test_data_idx_disorder_is_detected(tmp_path: Path) -> None:
    rows, summary = inspect_snapshots(_snapshots(tmp_path, disorder=True))
    assert rows
    assert not summary["data_idx_strictly_monotonic"]
    assert summary["nonmonotonic_positions"]


def test_scipy_modes_have_objective_and_feasibility_parity(tmp_path: Path) -> None:
    snapshots = _snapshots(tmp_path)[:6]
    universe = LPUniverse.from_snapshots(snapshots)
    policy = CapacityPolicy()
    objectives = {}
    for mode in ("cold_rebuild", "reuse_model", "explicit_basis"):
        solver = ScipySequentialLP(universe, policy)
        states = [solver.solve(snapshot, mode=mode) for snapshot in snapshots]
        objectives[mode] = [state.objective for state in states]
        assert all(state.status == "OPTIMAL" for state in states)
        assert all(state.max_capacity_violation <= 1e-7 for state in states)
        assert all(state.max_demand_violation <= 1e-7 for state in states)
        assert all(state.duality_gap <= 1e-7 for state in states)
        assert all(not state.warm_start_effective for state in states)
    assert objectives["cold_rebuild"] == pytest.approx(objectives["reuse_model"])
    assert objectives["cold_rebuild"] == pytest.approx(objectives["explicit_basis"])


def test_shadow_price_sign_matches_finite_difference() -> None:
    evidence = calibrate_dual_sign("scipy")
    assert evidence["loose_capacity_raw_pi"] == pytest.approx(0.0, abs=1e-8)
    assert evidence["congestion_price"] == pytest.approx(evidence["finite_difference_price"], rel=1e-6)


def test_real_sequence_metadata_roundtrip_and_slice(tmp_path: Path) -> None:
    raw = make_fixture(16)
    for index, sample in enumerate(raw):
        sample["meta"] = {
            "source_file": "official.pkl", "source_record_index": index,
            "intensity": 25, "adapter_mode": "test", "source_timestamp": None,
        }
    path = tmp_path / "metadata.pkl"
    with path.open("wb") as handle:
        pickle.dump(raw, handle)
    snapshots = load_dataset(path, start_index=3, limit=4)
    assert [snapshot.position for snapshot in snapshots] == [3, 4, 5, 6]
    assert snapshots[0].meta["source_record_index"] == 3
    rows, summary = inspect_snapshots(snapshots)
    assert rows[0]["source_file"] == "official.pkl"
    assert summary["real_sequence_verdict"] == "ORDERED_ONLY"


def test_restricted_slice_selection_is_deterministic() -> None:
    records = [
        {"FlowSet": [[2, 3, 1], [1, 4, 1]]},
        {"FlowSet": [[1, 4, 1], [2, 3, 1], [5, 6, 1]]},
    ]
    assert select_persistent_flows(records, 2) == [(1, 4), (2, 3)]


def test_official_adapter_start_index_and_limit_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured = []
    monkeypatch.setattr(StarlinkAdapter, "_adapt_topo_file", staticmethod(lambda *args: captured.append(args)))
    adapter = StarlinkAdapter(
        str(tmp_path), "source_{}.pkl", ["A"], 200, InterShellMode.ISL,
        parallel=1, start_index=7, limit=11, intensity=25,
    )
    adapter.adapt(str(tmp_path / "output"))
    assert captured[0][-3:] == (7, 11, 25)


def test_official_adapter_applies_start_index_and_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source_A.pkl"
    with source.open("wb") as handle:
        for index in range(5):
            pickle.dump(
                {"InterShell_GrdRelay": [], "InterShell_ISL": [[], [], []], "FlowSet": [[0, 1, index + 1.0]]},
                handle,
            )
    captured = {}
    monkeypatch.setattr(starlink_adapter_module.MSG, "Inter_Shell_Graph", lambda *args: (None, None, []))
    monkeypatch.setattr(starlink_adapter_module.SPG, "SPOnGrid", lambda *args: [[0, 1]])
    monkeypatch.setattr(starlink_adapter_module, "generate_sat2user", lambda *args: (lambda value: value + 5000))
    monkeypatch.setattr(
        starlink_adapter_module.AssetManager, "save_dataset_",
        lambda output, name, dataset: captured.update({"dataset": dataset}),
    )
    StarlinkAdapter._adapt_topo_file(
        str(source), 5, "A", InterShellMode.ISL, str(tmp_path), 2, 2, 25
    )
    assert [sample["data_idx"] for sample in captured["dataset"]] == [2, 3]
    assert [sample["meta"]["source_record_index"] for sample in captured["dataset"]] == [2, 3]


def test_gurobi_probe_and_no_secret_leakage(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WLSACCESSID", "DO_NOT_LEAK_ACCESS")
    monkeypatch.setenv("WLSSECRET", "DO_NOT_LEAK_SECRET")
    result = probe_gurobi(check_size_limit=False)
    serialized = json.dumps(result)
    assert result["available"]
    assert result["wls_environment_present"]
    assert "DO_NOT_LEAK" not in serialized


def test_gurobi_basis_modes_when_available(tmp_path: Path) -> None:
    available, detail = gurobi_available()
    if not available:
        pytest.skip(f"Gurobi-specific test skipped: {detail}")
    snapshots = _snapshots(tmp_path)[:6]
    universe = LPUniverse.from_snapshots(snapshots)
    sign = calibrate_dual_sign("gurobi")["sign_multiplier"]
    cold = GurobiSequentialLP(universe, CapacityPolicy(), sign)
    reuse = GurobiSequentialLP(universe, CapacityPolicy(), sign)
    basis = GurobiSequentialLP(universe, CapacityPolicy(), sign)
    basis_presolve = GurobiSequentialLP(universe, CapacityPolicy(), sign)
    reset = GurobiSequentialLP(universe, CapacityPolicy(), sign)
    cold_obj = [cold.solve(snapshot, "cold_rebuild").objective for snapshot in snapshots]
    reuse_states = [reuse.solve(snapshot, "reuse_model") for snapshot in snapshots]
    basis_states = [basis.solve(snapshot, "explicit_basis") for snapshot in snapshots]
    presolve_states = [basis_presolve.solve(snapshot, "explicit_basis_presolve") for snapshot in snapshots]
    reset_states = [reset.solve(snapshot, "reset_basis") for snapshot in snapshots]
    assert cold_obj == pytest.approx([state.objective for state in reuse_states])
    assert cold_obj == pytest.approx([state.objective for state in basis_states])
    assert cold_obj == pytest.approx([state.objective for state in presolve_states])
    assert cold_obj == pytest.approx([state.objective for state in reset_states])
    assert all(state.warm_start_effective for state in basis_states[1:])
    assert all(state.lp_warm_start == 2 for state in presolve_states)
    assert all(not state.warm_start_effective for state in reset_states)


def test_counterfactual_and_transport_with_edge_birth_death(tmp_path: Path) -> None:
    snapshots = _snapshots(tmp_path)[:12]
    records, summary = analyze_counterfactuals(snapshots, "scipy", CapacityPolicy())
    assert records
    assert summary["topology_vs_traffic_verdict"] in {
        "TRAFFIC_DOMINANT", "TOPOLOGY_DOMINANT", "COMPARABLE", "STRONG_INTERACTION"
    }
    _, transport, _, _ = analyze(snapshots, "scipy", CapacityPolicy())
    assert transport
    assert any(row["new_edge_count"] for row in transport)
    assert any(row["deleted_edge_count"] for row in transport)
    assert {row["state"] for row in transport} == {
        "edge_load", "utilization", "binding_state", "dual_price"
    }


def test_resume_returns_completed_matching_run(tmp_path: Path) -> None:
    output = tmp_path / "real"
    output.mkdir()
    expected = {"run_signature": [0, 100, 10], "verdicts": {"research_recommendation": "STOP_TEMPORAL_TRACKING"}}
    with (output / "summary.json").open("w", encoding="utf-8-sig") as handle:
        json.dump(expected, handle)
    args = Namespace(output_dir=str(output), resume=True, start_index=0, limit=100, max_flows=10)
    assert run(args) == expected


def test_complete_synthetic_smoke(tmp_path: Path) -> None:
    summary = run_synthetic_smoke(tmp_path / "smoke")
    output = tmp_path / "smoke"
    assert summary["evidence_source"] == "synthetic_fixture"
    assert summary["warm_start"]["objective_feasibility_parity"] == "PASS"
    assert (output / "sequence_manifest.csv").is_file()
    assert (output / "continuity_records.csv").is_file()
    assert (output / "solve_records.csv").is_file()
    assert list((output / "figures").glob("*.png"))
