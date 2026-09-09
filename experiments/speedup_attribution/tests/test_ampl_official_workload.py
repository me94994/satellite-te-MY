"""第三轮 AMPL official-workload 协议测试。"""

from __future__ import annotations

import inspect
import json
from pathlib import Path
import pickle
import subprocess
from types import SimpleNamespace
import zipfile

import pytest

from experiments.speedup_attribution import ampl_environment as env
from experiments.speedup_attribution import ampl_solver
from experiments.speedup_attribution import official_workload as official
from experiments.speedup_attribution.ampl_solver import (
    CODE_PUBLIC_CONFIG, PAPER_LIKE_CONFIG, SolverBackend, ThreadMode, parity_check, sparse_rows,
)
from experiments.speedup_attribution.benchmark_solver import diagnostic_instance
from experiments.speedup_attribution.run_ampl_full_scale import append_status, run_solve_process
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
    assert inventory["BENCHMARK_PRIMARY_READY"] is False
    assert len(inventory["missing_files"]) == 8


def test_primary_one_volume_ready_and_missing_b_does_not_block(tmp_path):
    path = tmp_path / "DataSetForSaTE100" / "StarLink_DataSetForAgent100_5000_A.pkl"
    path.parent.mkdir()
    with path.open("wb") as handle:
        pickle.dump(raw_record(), handle)
    inventory = official.build_inventory([tmp_path])
    assert inventory["status"] == "BENCHMARK_PRIMARY_READY"
    assert inventory["BENCHMARK_PRIMARY_READY"] is True
    assert inventory["BENCHMARK_ALL_INTENSITIES_READY"] is False
    assert inventory["BENCHMARK_A_B_COMPLETE"] is False


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


def test_download_archive_validity(tmp_path):
    archive = tmp_path / "DataSetForSaTE100.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("DataSetForSaTE100/StarLink_DataSetForAgent100_5000_A.pkl", b"data")
    assert official.inspect_download_archive(archive)["status"] == "DOWNLOAD_ARCHIVE_VALID"
    invalid = tmp_path / "quota.html"
    invalid.write_text("<html>quota exceeded</html>")
    assert official.inspect_download_archive(invalid)["status"] == "DOWNLOAD_ARCHIVE_INVALID"


def test_access_user_nodes_and_capacity_values(monkeypatch):
    monkeypatch.setattr(official, "_satellite_to_user", lambda mode: lambda sat: sat + official.SATELLITE_COUNT)
    monkeypatch.setattr(official, "SPG", SimpleNamespace(SPOnGrid=lambda src, dst, *_: [
        [src, 100 + index, dst] for index in range(10)
    ]))
    provenance = official.SnapshotProvenance("DataSetForSaTE100", "A", "raw.pkl", 0, 3, 2, 11.0)
    paper, audit = official.build_official_instance(raw_record(), provenance, PAPER_LIKE_CONFIG)
    code = paper.with_capacity(CODE_PUBLIC_CONFIG)
    access = [edge for edge in paper.benchmark.physical_edges if max(edge) >= official.SATELLITE_COUNT]
    assert access
    assert {paper.benchmark.capacities[edge] for edge in access} == {50.0}
    assert {code.benchmark.capacities[edge] for edge in access} == {800.0}
    assert paper.hashes() == code.hashes()
    assert paper.benchmark.candidate_paths[(4236, 4239)][0][0] == 4236
    assert paper.benchmark.candidate_paths[(4236, 4239)][0][-1] == 4239
    assert all(item["unique_paths"] == 10 for item in audit)


def test_grdstation_uses_official_user_offset_for_access_classification(monkeypatch):
    monkeypatch.setattr(official, "_satellite_to_user", lambda mode: lambda sat: sat + 4236 + 222)
    monkeypatch.setattr(official, "SPG", SimpleNamespace(SPOnGrid=lambda src, dst, *_: [
        [src, 4300, 100 + index, dst] for index in range(10)
    ]))
    provenance = official.SnapshotProvenance("DataSetForSaTE100", "A", "raw.pkl", 0, 3, 2, 11.0)
    instance, _ = official.build_official_instance(raw_record(), provenance, PAPER_LIKE_CONFIG, mode="GrdStation")
    assert instance.user_node_floor == 4458
    assert instance.benchmark.capacities[(0, 4300)] == 200.0
    assert instance.benchmark.capacities[(4458, 0)] == 50.0


def test_path_cache_key_stability_and_provenance():
    key = official.path_cache_key("topology", "ISL", 1, 2, 10)
    assert key == official.path_cache_key("topology", "ISL", 1, 2, 10)
    assert key != official.path_cache_key("other", "ISL", 1, 2, 10)
    assert set(official.SnapshotProvenance.__dataclass_fields__) >= {
        "dataset", "volume", "source_path", "record_index", "raw_flow_count", "active_sd_pairs", "total_demand"
    }


def test_thread_modes_declared_without_forcing_default():
    source = inspect.getsource(ampl_solver.solve_ampl_gurobi)
    assert {item.value for item in ThreadMode} == {
        "GUROBI_DEFAULT", "GUROBI_THREADS_1", "GUROBI_THREADS_24"
    }
    assert 'options = "outlev=0"' in source
    assert 'options += " threads=1"' in source
    assert 'options += " threads=24"' in source


def test_child_process_timeout_is_incremental(monkeypatch, tmp_path):
    monkeypatch.setattr("experiments.speedup_attribution.run_ampl_full_scale.OUTPUT_ROOT", tmp_path)
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: (_ for _ in ()).throw(subprocess.TimeoutExpired("x", 1)))
    result = run_solve_process(tmp_path / "job.pkl", tmp_path / "result.json", ThreadMode.DEFAULT, timeout_s=1)
    assert result["status"] == "TIMEOUT"
    statuses = [json.loads(line)["status"] for line in (tmp_path / "solve_status.jsonl").read_text().splitlines()]
    assert statuses == ["STARTED", "TIMEOUT"]


def test_no_flow_or_active_sd_cap_in_official_pipeline():
    source = inspect.getsource(official.build_official_instance)
    assert "[:180]" not in source and "min(180" not in source


def test_sate_masks_shortfalls_without_fake_k10_or_self_pair_drop():
    source = Path("experiments/speedup_attribution/benchmark_k10.py").read_text(encoding="utf-8-sig")
    env_source = Path("lib/spaceTE/sate_env.py").read_text(encoding="utf-8")
    assert "INACTIVE_MASK_NO_FAKE_DUPLICATES" in source
    assert "active_path_mask" in env_source
    assert "filtered_tm = dict(data['tm'])" in env_source


def test_no_paper_reference_is_stored_as_measurement():
    source = inspect.getsource(ampl_solver.solve_ampl_gurobi)
    assert "46" not in source and "2738" not in source
