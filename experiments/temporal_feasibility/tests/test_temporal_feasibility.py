"""CPU-sized falsification tests for schema, LP parity, and dual semantics."""

from __future__ import annotations

from pathlib import Path

import pytest

from experiments.temporal_feasibility.extract_primal_dual import calibrate_dual_sign
from experiments.temporal_feasibility.inspect_dataset import inspect_snapshots
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
from experiments.temporal_feasibility.synthetic_fixture import write_fixture


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
    cold_obj = [cold.solve(snapshot, "cold_rebuild").objective for snapshot in snapshots]
    reuse_states = [reuse.solve(snapshot, "reuse_model") for snapshot in snapshots]
    basis_states = [basis.solve(snapshot, "explicit_basis") for snapshot in snapshots]
    assert cold_obj == pytest.approx([state.objective for state in reuse_states])
    assert cold_obj == pytest.approx([state.objective for state in basis_states])
    assert all(state.warm_start_effective for state in basis_states[1:])


def test_complete_synthetic_smoke(tmp_path: Path) -> None:
    summary = run_synthetic_smoke(tmp_path / "smoke")
    output = tmp_path / "smoke"
    assert summary["evidence_source"] == "synthetic_fixture"
    assert summary["warm_start"]["objective_feasibility_parity"] == "PASS"
    assert (output / "sequence_manifest.csv").is_file()
    assert (output / "continuity_records.csv").is_file()
    assert (output / "solve_records.csv").is_file()
    assert list((output / "figures").glob("*.png"))
