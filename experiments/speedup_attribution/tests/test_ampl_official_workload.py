"""第三轮 AMPL official-workload 协议测试。"""

from __future__ import annotations

import inspect
import json
from pathlib import Path
import pickle
import subprocess
from types import SimpleNamespace

import pytest

from experiments.speedup_attribution import ampl_environment as env
from experiments.speedup_attribution import ampl_solver
from experiments.speedup_attribution import official_workload as official
from experiments.speedup_attribution.ampl_solver import (
    CODE_PUBLIC_CONFIG, PAPER_LIKE_CONFIG, SolverBackend, parity_check, sparse_rows,
)
from experiments.speedup_attribution.benchmark_solver import diagnostic_instance
from experiments.speedup_attribution.run_ampl_full_scale import append_status
from experiments.speedup_attribution.schemas import assert_k5_prefix_of_k10


def raw_record(flowset=None):
    return {
        "FlowSet": flowset or [[0, 3, 2.0], [0, 3, 4.0], [1, 4, 5.0]],
        "InterShell_ISL": [[], [], []],
        "InterShell_GrdRelay": [],
    }


def test_ampl_env_var_presence(monkeypatch):
    monkeypatch.delenv("AMPL_LICENSE_UUID", raising=False)
    assert env.license_env_present() is False
    monkeypatch.setenv("AMPL_LICENSE_UUID", "secret-sentinel")
    assert env.license_env_present() is True


def test_activation_never_logs_uuid(monkeypatch, capsys):
    monkeypatch.setenv("AMPL_LICENSE_UUID", "secret-sentinel")
    observed = {}

    def fake_run(command, **kwargs):
        observed.update(command=command, kwargs=kwargs)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = env.activate_from_environment()
    captured = capsys.readouterr()
    assert result == {"status": "ACTIVATED", "return_code": 0}
    assert captured.out == captured.err == ""
    assert observed["kwargs"]["stdout"] is subprocess.DEVNULL
    assert observed["kwargs"]["stderr"] is subprocess.DEVNULL


def test_uuid_cannot_be_persisted(tmp_path):
    with pytest.raises(ValueError, match="LICENSE_IDENTIFIER_PERSISTENCE_REJECTED"):
        env.write_environment({"AMPL_LICENSE_UUID": "secret-sentinel"}, tmp_path / "environment.json")
    assert not (tmp_path / "environment.json").exists()


def test_amplpy_import_and_modules_present():
    pytest.importorskip("amplpy")
    assert "base" in env.installed_modules()
    assert "gurobi" in env.installed_modules()


def test_ampl_probe_has_only_safe_failure_type():
    result = env.safe_ampl_probe()
    assert set(result) == {"AMPL_instantiation_success", "failure_type"}


class FakeObjective:
    def __init__(self, value): self._value = value
    def value(self): return self._value


class FakeAmpl:
    def eval(self, text): self.size = int(text.split("1..")[1].split(";")[0])
    def setOption(self, *_): pass
    def solve(self, **_): pass
    def getValue(self, _): return "solved"
    def getObjective(self, _): return FakeObjective(float(self.size))
    def close(self): pass


def test_5000_and_10000_smoke_protocol(monkeypatch):
    import amplpy
    monkeypatch.setattr(amplpy, "AMPL", FakeAmpl)
    rows = env.smoke_test_sizes()
    assert [(row["variables"], row["status"]) for row in rows] == [(5000, "COMPLETED"), (10000, "COMPLETED")]


def test_solver_backend_keeps_direct_reference():
    assert {item.value for item in SolverBackend} == {"DIRECT_GUROBIPY", "AMPL_GUROBI"}


def test_sparse_formulation_counts_match_existing_estimate():
    instance = diagnostic_instance(nodes=16, active_flows=4, k=10)
    rows = sparse_rows(instance)
    assert len(rows["variables"]) == 40
    assert rows["num_nonzeros"] == 40 + len(rows["EV"])


def test_objective_and_feasibility_parity_gate():
    base = {"status": "MEASURED", "objective": 8.0, "num_vars": 10, "num_constraints": 7,
            "num_nonzeros": 30, "max_capacity_violation": 0.0, "max_demand_violation": 0.0}
    assert parity_check(base, dict(base))["status"] == "AMPL_FORMULATION_PARITY_PASS"
    changed = dict(base, objective=7.0)
    assert parity_check(base, changed)["status"] == "AMPL_FORMULATION_PARITY_FAIL"


def test_official_flowset_parsing_list_and_stream(tmp_path):
    list_path = tmp_path / "DataSetForSaTE100" / "StarLink_DataSetForAgent100_5000_A.pkl"
    list_path.parent.mkdir()
    with list_path.open("wb") as handle:
        pickle.dump([raw_record(), raw_record()], handle)
    stream_path = list_path.with_name("StarLink_DataSetForAgent100_5000_B.pkl")
    with stream_path.open("wb") as handle:
        pickle.dump(raw_record(), handle)
        pickle.dump(raw_record(), handle)
    assert official.inspect_pickle(list_path)["readable_pickle_records"] == 2
    assert official.inspect_pickle(stream_path)["readable_pickle_records"] == 2


def test_active_sd_aggregation_and_duplicates():
    demands, stats = official.aggregate_flowset(raw_record()["FlowSet"])
    assert demands == {(0, 3): 6.0, (1, 4): 5.0}
    assert stats["raw_flow_count"] == 3
    assert stats["active_sd_pairs"] == 2
    assert stats["aggregation_ratio"] == 1.5


def test_no_dense_traffic_matrix_construction():
    source = inspect.getsource(official.aggregate_flowset)
    assert "numpy" not in source and "zeros(" not in source and "SATELLITE_COUNT * [" not in source


def test_k10_never_fakes_duplicate_paths(monkeypatch):
    monkeypatch.setattr(official, "SPG", SimpleNamespace(SPOnGrid=lambda *args: [[0, 2], [0, 1, 2], [0, 2]]))
    paths, audit = official.generate_unique_paths(raw_record(), [(0, 2)], requested_k=10)
    assert paths[(0, 2)] == ((0, 2), (0, 1, 2))
    assert audit[0]["status"] == "K10_PATH_SHORTFALL"


def test_k5_is_prefix_of_k10():
    k10 = diagnostic_instance(nodes=16, active_flows=5, k=10)
    k5 = k10.paths_for_k(5)
    assert_k5_prefix_of_k10(k5, k10)


def test_official_hash_stability():
    instance = diagnostic_instance(nodes=16, active_flows=5, k=10)
    assert instance.hashes() == instance.hashes()


def test_capacity_configs_are_separate():
    assert PAPER_LIKE_CONFIG.network_capacity == CODE_PUBLIC_CONFIG.network_capacity == 200.0
    assert PAPER_LIKE_CONFIG.access_capacity == 50.0
    assert CODE_PUBLIC_CONFIG.access_capacity == 800.0
    assert PAPER_LIKE_CONFIG.name != CODE_PUBLIC_CONFIG.name


def test_incremental_status_persistence_and_resume(monkeypatch, tmp_path):
    monkeypatch.setattr("experiments.speedup_attribution.run_ampl_full_scale.OUTPUT_ROOT", tmp_path)
    append_status(4, "STARTED", "snapshot")
    append_status(4, "COMPLETED", "snapshot")
    rows = [json.loads(line) for line in (tmp_path / "stage_status.jsonl").read_text().splitlines()]
    assert [row["status"] for row in rows] == ["STARTED", "COMPLETED"]


def test_ampl_failure_recovery_records_only_type(monkeypatch):
    import amplpy
    class BrokenAmpl:
        def __init__(self): raise RuntimeError("secret-sentinel")
    monkeypatch.setattr(amplpy, "AMPL", BrokenAmpl)
    result = ampl_solver.solve_ampl_gurobi(diagnostic_instance(nodes=16, active_flows=2, k=5))
    assert result["status"] == "FAILED"
    assert result["failure_type"] == "RuntimeError"
    assert "secret-sentinel" not in json.dumps(result)


def test_full_scale_timeout_boundary_is_declared():
    source = inspect.getsource(env.smoke_test_sizes) + inspect.getsource(ampl_solver.solve_ampl_gurobi)
    assert "threads=1" in source


def test_evidence_labels_remain_separate():
    assert diagnostic_instance().evidence_label.startswith("DIAGNOSTIC")
    assert "PAPER_ACCURACY_REPRODUCED" not in inspect.getsource(official)


def test_raw_cache_and_license_paths_are_ignored():
    gitignore = Path(".gitignore").read_text(encoding="utf-8")
    assert "input" in gitignore and ".env" in gitignore
    assert "output" in gitignore


def test_inventory_reports_all_missing_volumes(tmp_path):
    inventory = official.build_inventory([tmp_path])
    assert inventory["status"] == "BENCHMARK_INCOMPLETE"
    assert len(inventory["missing_files"]) == 8


def test_inventory_requires_all_three_benchmark_fields(tmp_path):
    path = tmp_path / "DataSetForSaTE25" / "StarLink_DataSetForAgent25_5000_A.pkl"
    path.parent.mkdir()
    with path.open("wb") as handle:
        pickle.dump({"FlowSet": []}, handle)
    assert official.inspect_pickle(path)["benchmark_status"] == "BENCHMARK_UNUSABLE"


def test_distribution_and_nominal_k10_math():
    values = [10, 20, 30, 40, 50]
    stats = official.distribution(values)
    assert stats["median"] == 30
    assert 10 * stats["P95"] == 480


def test_threads_above_one_are_rejected():
    result = diagnostic_instance(nodes=16, active_flows=2, k=5)
    with pytest.raises(ValueError, match="ONLY_SINGLE_THREAD_ALLOWED"):
        ampl_solver.solve_ampl_gurobi(result, threads=24)


def test_no_paper_reference_is_stored_as_measurement():
    source = inspect.getsource(ampl_solver.solve_ampl_gurobi)
    assert "46" not in source and "2738" not in source
