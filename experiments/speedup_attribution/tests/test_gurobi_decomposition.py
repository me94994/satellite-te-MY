"""Round 3.1 timing/reuse evidence gates。"""

from __future__ import annotations

import inspect
import json
from pathlib import Path
import pickle
import subprocess

import pytest

from experiments.speedup_attribution import gurobi_decomposition as gd
from experiments.speedup_attribution import official_workload as official
from experiments.speedup_attribution import run_gurobi_decomposition as runner
from experiments.speedup_attribution.schemas import BenchmarkInstance


def tiny_instance(demand: float = 2.0, add_flow: bool = False) -> BenchmarkInstance:
    demands = {(0, 3): demand}
    paths = {(0, 3): ((0, 1, 3), (0, 2, 3))}
    if add_flow:
        demands[(1, 2)] = 1.0
        paths[(1, 2)] = ((1, 3, 2),)
    edges = tuple(sorted({edge for values in paths.values() for path in values for edge in zip(path[:-1], path[1:])}))
    return BenchmarkInstance("DataSetForSaTE100-A-1-PAPER_LIKE_CONFIG", (0, 1, 2, 3), edges,
                             {edge: 3.0 for edge in edges}, demands, paths, "OFFICIAL_WORKLOAD")


def wrapped(instance: BenchmarkInstance, index: int = 1) -> official.OfficialBenchmarkInstance:
    hashes = instance.hashes()
    provenance = official.SnapshotProvenance("DataSetForSaTE100", "A", "raw.pkl", index,
                                             len(instance.demands), len(instance.demands), sum(instance.demands.values()))
    return official.OfficialBenchmarkInstance(provenance, instance, 10, hashes["topology_hash"],
                                              hashes["demand_hash"], hashes["path_hash"], 1.0, 2.0, 2.0)


def test_timing_components_nonnegative():
    row = {name: 0.1 for name in ("T_driver_read", "T_driver_conversion", "T_driver_setup",
                                  "T_gurobi_optimize", "T_driver_output")}
    row["T_solve_call_wall"] = 0.6
    result = gd.timing_accounting(row)
    assert all(value >= 0 for value in row.values())
    assert result["timing_gate"] == "PASS"


def test_solver_timing_sum_sanity_and_negative_gate():
    row = {name: 1.0 for name in ("T_driver_read", "T_driver_conversion", "T_driver_setup",
                                  "T_gurobi_optimize", "T_driver_output")}
    row["T_solve_call_wall"] = 4.0
    assert gd.timing_accounting(row)["timing_gate"] == "FAIL"


def test_optimize_timing_is_not_subtraction_derived():
    source = inspect.getsource(gd.driver_statistics)
    assert 'float(times["solver"])' in source
    assert "solve_wall -" not in source


def test_same_snapshot_hashes_unchanged():
    path = Path("output/speedup_attribution/ampl_full_scale/instances/A-339-median.pkl")
    if not path.is_file():
        pytest.skip("third-round ignored artifact unavailable")
    with path.open("rb") as handle:
        value = pickle.load(handle)["instance"]
    assert value.hashes() == {name: value.benchmark.hashes()[name] for name in ("topology_hash", "demand_hash", "path_hash")}


def test_same_structure_detector():
    first = gd.structure_record(wrapped(tiny_instance(), 1))
    assert gd.classify_transition(first, dict(first)) == "EXACT_SAME_STRUCTURE"


def test_demand_only_structure_recognized():
    first = gd.structure_record(wrapped(tiny_instance(2.0), 1))
    second = gd.structure_record(wrapped(tiny_instance(2.5), 2))
    assert first["structure_hash"] == second["structure_hash"]
    assert first["demand_hash"] != second["demand_hash"]
    assert gd.classify_transition(first, second) == "EXACT_SAME_STRUCTURE"


def test_structure_change_recognized():
    first = gd.structure_record(wrapped(tiny_instance(), 1))
    second = gd.structure_record(wrapped(tiny_instance(add_flow=True), 2))
    assert gd.classify_transition(first, second) == "SAME_SUPERSET_COMPATIBLE"


def test_fixed_universe_inactive_flow_demand_zero():
    first, second = tiny_instance(), tiny_instance(add_flow=True)
    universe = gd.build_fixed_universe([wrapped(first), wrapped(second)])
    values = gd.fixed_values(universe, first)
    assert values["demand"][universe.flow_ids[(1, 2)]] == 0.0


def test_fixed_universe_inactive_path_ub_zero():
    first, second = tiny_instance(), tiny_instance(add_flow=True)
    universe = gd.build_fixed_universe([wrapped(first), wrapped(second)])
    values = gd.fixed_values(universe, first)
    variable = ((1, 2), (1, 3, 2))
    assert values["upper"][universe.variable_ids[variable]] == 0.0


def test_fixed_universe_active_path_bound_is_redundant_demand_bound():
    instance = tiny_instance(2.5)
    universe = gd.build_fixed_universe([wrapped(instance)])
    values = gd.fixed_values(universe, instance)
    assert all(values["upper"][universe.variable_ids[((0, 3), path)]] == 2.5
               for path in instance.candidate_paths[(0, 3)])


def test_fixed_universe_cold_parity_gate():
    cold = {"objective": 3.0}
    reuse = {"objective": 3.0, "max_demand_violation": 0.0, "max_capacity_violation": 0.0}
    assert gd.verify_fixed_parity(cold, reuse)["status"] == "FIXED_UNIVERSE_REUSE_PASS"


def test_reuse_no_basis_objective_parity_gate():
    cold = {"objective": 3.0}
    reuse = {"objective": 2.9, "max_demand_violation": 0.0, "max_capacity_violation": 0.0}
    assert gd.verify_fixed_parity(cold, reuse)["status"] == "FIXED_UNIVERSE_REUSE_FAIL"


def test_reuse_basis_objective_parity_gate():
    cold = {"objective": 3.0}
    basis = {"objective": 3.0 + 1e-9, "max_demand_violation": 0.0, "max_capacity_violation": 0.0}
    assert gd.verify_fixed_parity(cold, basis)["status"] == "FIXED_UNIVERSE_REUSE_PASS"


def test_reuse_feasibility_parity_gate():
    cold = {"objective": 3.0}
    infeasible = {"objective": 3.0, "max_demand_violation": 0.0, "max_capacity_violation": 1e-7}
    assert gd.verify_fixed_parity(cold, infeasible)["status"] == "FIXED_UNIVERSE_REUSE_FAIL"


def test_explicit_no_basis_mode_disables_all_warm_state():
    source = inspect.getsource(gd.FixedUniverseSession.solve)
    assert "alg:basis=0 alg:start=0 lp:warmstart=0" in source
    assert "_clear_basis()" not in source

    persistent_source = inspect.getsource(gd.PersistentGurobiSession.optimize)
    assert "GRBreset" in persistent_source
    assert 'set_param("LPWarmStart", 0)' in persistent_source


def test_basis_reuse_requires_positive_log_evidence():
    assert gd.basis_log_verified("LP warm-start: use basis")
    assert gd.basis_log_verified("LP warm-start: crush basis")
    assert not gd.basis_log_verified("LP warm-start: discard basis")


def test_basis_export_import_roundtrip():
    amplpy = pytest.importorskip("amplpy")
    ampl = amplpy.AMPL()
    try:
        ampl.eval(gd.FIXED_UNIVERSE_MODEL)
        ampl.getSet("F").setValues([0]); ampl.getSet("E").setValues([0]); ampl.getSet("V").setValues([0])
        ampl.getSet("FV").setValues([(0, 0)]); ampl.getSet("EV").setValues([(0, 0)])
        ampl.getParameter("demand").setValues({0: 1.0}); ampl.getParameter("capacity").setValues({0: 1.0})
        ampl.getParameter("upper").setValues({0: 1.0})
        ampl.setOption("presolve", 0)
        ampl.setOption("gurobi_options", "outlev=0 pre:solve=0 alg:method=1 alg:basis=2")
        ampl.solve(solver="gurobi", verbose=False)
        basis = gd.export_basis(ampl)
        gd.import_basis(ampl, basis)
        assert gd.export_basis(ampl) == basis
    finally:
        ampl.close()


def test_persistent_api_availability_detection():
    assert gd.persistent_api_probe()["status"] in {
        "PERSISTENT_GUROBI_API_IMPORT_AVAILABLE", "PERSISTENT_GUROBI_API_NOT_AVAILABLE"
    }


def test_direct_gurobipy_license_gate_label(monkeypatch):
    result = gd.direct_gurobipy_license_probe(1)
    assert result["status"] in {"DIRECT_GUROBIPY_FULL_SCALE_LICENSED", gd.DIRECT_LICENSE_BLOCKED}


def test_no_license_identifier_persisted(monkeypatch, tmp_path):
    monkeypatch.setenv("AMPL_LICENSE_UUID", "secret-sentinel")
    with pytest.raises(ValueError, match="LICENSE_IDENTIFIER_PERSISTENCE_REJECTED"):
        gd.write_json({"value": "secret-sentinel"}, tmp_path / "unsafe.json")
    assert not (tmp_path / "unsafe.json").exists()


def test_child_timeout_and_resume(monkeypatch, tmp_path):
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: (_ for _ in ()).throw(subprocess.TimeoutExpired("x", 1)))
    assert runner.run_child(["python", "x"], tmp_path / "missing.json", timeout_s=1)["status"] == "TIMEOUT"
    output = tmp_path / "done.json"
    output.write_text('{"status":"MEASURED"}', encoding="utf-8")
    assert runner.run_child(["never-called"], output, resume=True)["status"] == "MEASURED"


def test_solver_output_redaction():
    text = "AMPL_LICENSE_UUID=secret\nLicense identifier=abc\nSolver time = 1.0s\n"
    assert gd.timing_log_lines(text) == ["Solver time = 1.0s"]
    assert "secret" not in gd.redact_text(text)


def test_cold_and_reuse_evidence_labels_are_distinct():
    labels = {gd.EVIDENCE_COLD, gd.EVIDENCE_AMPL_REUSE, gd.EVIDENCE_FIXED,
              gd.EVIDENCE_PERSISTENT_NO_BASIS, gd.EVIDENCE_PERSISTENT_BASIS}
    assert len(labels) == 5
    assert "PERSISTENT" not in gd.EVIDENCE_AMPL_REUSE


def test_persistent_optimize_uses_native_runtime_without_subtraction():
    source = inspect.getsource(gd.PersistentGurobiSession.optimize)
    assert 'get_double_attr("Runtime")' in source
    assert '"T_gurobi_optimize": native_runtime' in source
    assert "optimize_wall -" not in source


def test_no_workload_modification_in_fixed_values():
    source = inspect.getsource(gd.fixed_values)
    assert "PAPER_LIKE_CONFIG" not in source and "candidate_paths" in source


def test_no_path_truncation_or_manufacture():
    source = inspect.getsource(gd.build_fixed_universe)
    assert "paths[:" not in source and "duplicate" not in source and "requested_k" not in source


def test_no_active_sd_truncation():
    source = inspect.getsource(runner.build_sequence)
    assert "active_sd" not in source and "[:180]" not in source and "min(180" not in source
