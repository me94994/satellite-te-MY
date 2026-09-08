"""Protocol tests for the K=10 large-scale crossover diagnosis."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from experiments.speedup_attribution.analyze_crossover import detect_measured_crossover, scaling_fit
from experiments.speedup_attribution.benchmark_k10 import validate_checkpoint_dimension
from experiments.speedup_attribution.benchmark_large_scale import (
    _resume_keys,
    constellation_instance,
    controlled_stress_instance,
    equal_split_proxy,
    family_points,
    license_safe_instance,
    paired_problem_hash,
)
from experiments.speedup_attribution.benchmark_solver import solve_level
from experiments.speedup_attribution.schemas import SafetyLimits, assert_k5_prefix_of_k10, estimate_lp


@pytest.fixture(scope="module")
def paired_instances():
    k10 = constellation_instance(8, 10, 10)
    return k10.paths_for_k(5), k10


def test_k5_is_exact_prefix_of_k10(paired_instances):
    assert_k5_prefix_of_k10(*paired_instances)


def test_all_ten_paths_are_unique(paired_instances):
    assert all(len(paths) == len(set(paths)) == 10 for paths in paired_instances[1].candidate_paths.values())


def test_no_duplicated_fake_paths(paired_instances):
    assert all(len(set(paths[5:])) == 5 for paths in paired_instances[1].candidate_paths.values())


def test_k10_objective_not_below_k5(paired_instances):
    limits = SafetyLimits(max_vars=1900, max_constraints=1900)
    k5 = solve_level(paired_instances[0], "L4", limits, method=1)
    k10 = solve_level(paired_instances[1], "L4", limits, method=1)
    if k5["status"] != "MEASURED" or k10["status"] != "MEASURED":
        pytest.skip("Gurobi/license unavailable")
    assert k10["objective"] + 1e-8 >= k5["objective"]


def test_k10_checkpoint_dimension_validation(tmp_path):
    path = tmp_path / "k10.pt"
    torch.save({"mean_linear.weight": torch.zeros(10, 10), "mean_linear.bias": torch.zeros(10)}, path)
    assert validate_checkpoint_dimension(path, 10)["status"] == "VALID"


def test_k5_checkpoint_rejected_by_k10(tmp_path):
    path = tmp_path / "k5.pt"
    torch.save({"mean_linear.weight": torch.zeros(5, 5), "mean_linear.bias": torch.zeros(5)}, path)
    assert validate_checkpoint_dimension(path, 10)["status"] == "REJECTED_CHECKPOINT_DIMENSION_MISMATCH"


def test_license_scale_guard_reduces_flows():
    instance, _, actual = license_safe_instance(16, 40, "uncongested", 42, SafetyLimits(max_vars=100, max_constraints=1900))
    assert actual == 10
    assert estimate_lp(instance, "L4")["estimated_variables"] == 100


def test_sparse_scale_generation():
    instance = constellation_instance(32, 20, 10)
    assert len(instance.active_pairs) == 20
    assert estimate_lp(instance, "L4")["estimated_variables"] == 200


def test_topology_only_scaling():
    assert family_points("topology", (16, 32)) == [(16, 40), (32, 40)]


def test_flow_only_scaling():
    points = family_points("traffic")
    assert {nodes for nodes, _ in points} == {192}
    assert [flows for _, flows in points] == [10, 20, 40, 80, 120, 160, 180]


def test_coupled_scaling():
    assert family_points("coupled", (16, 192)) == [(16, 16), (192, 180)]


def test_congestion_label_propagation():
    instance, proxy = controlled_stress_instance(16, 40)
    if proxy["demand_scale"] > 1:
        assert instance.evidence_label == "CONTROLLED_STRESS_DIAGNOSTIC"
    assert proxy == {**equal_split_proxy(instance), "demand_scale": proxy["demand_scale"]}


def test_controlled_stress_is_not_real_workload():
    instance, _ = controlled_stress_instance(16, 40)
    assert instance.evidence_label != "OFFICIAL_REAL_WORKLOAD"
    assert instance.evidence_label != "PAPER_WORKLOAD"


def test_cuda_timing_has_boundary_synchronization():
    source = Path("experiments/speedup_attribution/benchmark_k10.py").read_text(encoding="utf-8-sig")
    timed = source[source.index("def _timed_cuda"):source.index("def benchmark_instance")]
    assert timed.count("torch.cuda.synchronize()") == 2


def test_measured_extrapolated_paper_evidence_separation():
    source = Path("experiments/speedup_attribution/analyze_crossover.py").read_text(encoding="utf-8-sig")
    assert "PAPER_REPORTED_REFERENCE" in source
    assert "facecolors=\"none\"" in source
    assert "MEASURED_FIT" in source


def test_crossover_detection():
    solver = [{"status": "MEASURED", "topology_nodes": 128, "actual_active_flows": 100, "regime": "congested", "total_s": 0.02}]
    sate = [{"status": "MEASURED", "k": 10, "topology_nodes": 128, "active_flows": 100, "regime": "congested", "t_e2e": {"median_s": 0.01}}]
    assert detect_measured_crossover(solver, sate)["status"] == "OBSERVED_AT_N=128"


def test_no_false_crossover_from_extrapolation():
    solver = [{"status": "EXTRAPOLATED_4236", "topology_nodes": 4236, "actual_active_flows": 180, "regime": "congested", "total_s": 10.0}]
    sate = [{"status": "MEASURED", "k": 10, "topology_nodes": 4236, "active_flows": 180, "regime": "congested", "t_e2e": {"median_s": 0.1}}]
    assert detect_measured_crossover(solver, sate)["status"] == "NOT_OBSERVED"


def test_same_problem_hash_matching(paired_instances):
    assert paired_instances[0].shared_input_hashes() == paired_instances[1].shared_input_hashes()
    assert len(paired_problem_hash(paired_instances[1])) == 64


def test_resume_output(tmp_path):
    path = tmp_path / "rows.jsonl"
    row = {"family": "coupled", "regime": "congested", "topology_nodes": 16, "requested_active_flows": 16, "k": 10, "baseline": "sparse_presolve_cold", "repeat": 0}
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    assert ("coupled", "congested", 16, 16, 10, "sparse_presolve_cold", 0) in _resume_keys(path)


def test_scaling_fit_requires_measured_points_only():
    rows = [{"x": x, "y": x * x} for x in (1, 2, 4, 8)]
    fit = scaling_fit(rows, "x", "y", bootstrap_samples=100)
    assert fit["status"] == "MEASURED_FIT"
    assert fit["slope_b"] == pytest.approx(2.0)


def test_no_large_binary_added_to_experiment_tree():
    files = [path for path in Path("experiments/speedup_attribution").rglob("*") if path.is_file()]
    assert max(path.stat().st_size for path in files) < 2_000_000
