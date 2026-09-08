"""Protocol tests named after the preregistered attribution gates."""

from pathlib import Path

import pytest

from experiments.speedup_attribution.audit_configuration import audit
from experiments.speedup_attribution.benchmark_solver import diagnostic_instance, solve_level
from experiments.speedup_attribution.schemas import (
    BenchmarkInstance, SafetyLimits, assert_same_problem, compatible_speedup,
    enforce_safety, estimate_lp, topology_pruning_accounting,
)


def test_benchmark_instance_hash_consistency():
    a = diagnostic_instance(nodes=8, active_flows=6, k=3)
    b = diagnostic_instance(nodes=8, active_flows=6, k=3)
    assert a.hashes() == b.hashes()
    assert_same_problem(a, b)


def _gurobi_rows():
    instance = diagnostic_instance(nodes=8, active_flows=6, k=3)
    rows = [solve_level(instance, level, SafetyLimits()) for level in ("L0", "L1", "L2", "L3", "L4")]
    if any(row["status"] != "MEASURED" for row in rows):
        pytest.skip("Gurobi/license unavailable")
    return rows


def test_dense_sparse_objective_parity():
    rows = _gurobi_rows()
    assert max(r["objective"] for r in rows) - min(r["objective"] for r in rows) <= 1e-8


def test_zero_demand_pruning_exactness():
    l0, _, l2, *_ = _gurobi_rows()
    assert l0["objective"] == pytest.approx(l2["objective"], rel=1e-8)
    assert l2["num_vars"] < l0["num_vars"]


def test_unused_edge_constraint_pruning_exactness():
    *_, l3, l4 = _gurobi_rows()
    assert l3["objective"] == pytest.approx(l4["objective"], rel=1e-8)
    assert l4["num_constraints"] <= l3["num_constraints"]


def test_presolve_on_off_objective_parity():
    l0, l1, l2, l3, _ = _gurobi_rows()
    assert l0["objective"] == pytest.approx(l1["objective"], rel=1e-8)
    assert l2["objective"] == pytest.approx(l3["objective"], rel=1e-8)


def test_path_count_configuration_audit():
    result = audit(Path.cwd())
    assert result["paper_config"]["candidate_paths_per_sd_pair"] == 10
    assert result["code_config"]["candidate_paths_sate"] == 5
    assert result["status"] == "PAPER_CONFIG_CODE_CONFIG_MISMATCH"


def test_no_fake_duplicated_k10_paths():
    instance = diagnostic_instance(nodes=8, active_flows=6, k=5)
    with pytest.raises(ValueError, match="fake paths"):
        instance.paths_for_k(10)


def test_duplicate_paths_rejected_at_source():
    instance = diagnostic_instance(nodes=8, active_flows=6, k=1)
    pair = instance.all_pairs[0]
    paths = dict(instance.candidate_paths)
    paths[pair] = (paths[pair][0], paths[pair][0])
    with pytest.raises(ValueError, match="duplicate"):
        BenchmarkInstance(instance.snapshot_id, instance.nodes, instance.physical_edges, instance.capacities, instance.demands, paths)


def test_gurobi_timing_decomposition():
    row = _gurobi_rows()[3]
    assert row["parse_s"] >= 0 and row["build_s"] >= 0 and row["optimize_wall_s"] >= 0
    assert row["total_s"] == pytest.approx(row["parse_s"] + row["build_s"] + row["optimize_wall_s"])


def test_cuda_synchronize_timing_correctness():
    source = Path("experiments/speedup_attribution/benchmark_sate.py").read_text(encoding="utf-8-sig")
    assert source.count("torch.cuda.synchronize()") >= 2


def test_sate_gurobi_same_input_verification():
    instance = diagnostic_instance(nodes=8, active_flows=6, k=3)
    assert_same_problem(instance, instance)


def test_dense_size_safety_guard():
    estimate = estimate_lp(diagnostic_instance(nodes=20, active_flows=5, k=5), "L0")
    with pytest.raises(ResourceWarning, match="SKIPPED_RESOURCE_BOUND"):
        enforce_safety(estimate, SafetyLimits(max_vars=10, max_constraints=10))


def test_restricted_license_skip_is_explicit():
    instance = diagnostic_instance(nodes=20, active_flows=5, k=5)
    row = solve_level(instance, "L0", SafetyLimits(max_vars=10, max_constraints=10))
    assert row["status"] == "SKIPPED_RESOURCE_BOUND"


def test_topology_pruning_offline_only_accounting():
    result = topology_pruning_accounting(8000, 512, 0.25)
    assert result["sample_reduction_x"] == 15.625
    assert result["online_speedup_x"] == 1.0


def test_no_multiplication_of_incompatible_speedups():
    with pytest.raises(ValueError, match="incompatible"):
        compatible_speedup(10, 2, same_stage=False)


def test_paper_reference_vs_measured_labels():
    instance = diagnostic_instance(nodes=8, active_flows=6, k=3)
    assert instance.evidence_label.startswith("DIAGNOSTIC_")
    assert instance.evidence_label != "PAPER_REPORTED_REFERENCE"


def test_representation_reduction_not_solver_speedup():
    result = diagnostic_instance(nodes=8, active_flows=6, k=3).representation_bytes()
    assert result["sparse_pickle_bytes"] < result["dense_pickle_bytes"]
