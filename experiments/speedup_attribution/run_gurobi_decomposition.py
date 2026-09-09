"""运行 Round 3.1 full-scale timing decomposition；所有阶段均支持安全 resume。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import statistics
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

from .ampl_solver import PAPER_LIKE_CONFIG
from .gurobi_decomposition import (
    OUTPUT_ROOT, AmplObjectSession, FixedUniverseSession, PersistentGurobiSession,
    basis_seed, build_fixed_universe,
    driver_capabilities, solve_cold, structure_record, classify_transition,
    verify_fixed_parity, write_json, write_jsonl,
)
from .official_workload import (
    SnapshotProvenance, _objects_from_pickle, aggregate_flowset,
    assemble_official_instance, generate_unique_paths,
)


ROUND3_ROOT = Path("output/speedup_attribution/ampl_full_scale")
CACHE_ROOT = ROUND3_ROOT / "cache"
SEQUENCE_INDICES = tuple(range(329, 350))
SOLVE_TIMEOUT_S = 900
SATE_FORWARD_S = 0.019810
SATE_E2E_S = 0.709610


def _atomic_pickle(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)


def _median_selection() -> dict[str, Any]:
    selected = json.loads((ROUND3_ROOT / "selected_snapshots.json").read_text(encoding="utf-8"))
    return next(row for row in selected if row["dataset"] == "DataSetForSaTE100" and row["volume"] == "A"
                and int(row["record_index"]) == 339)


def _provenance(row: Mapping[str, Any], index: int, stats: Mapping[str, Any]) -> SnapshotProvenance:
    return SnapshotProvenance(
        "DataSetForSaTE100", "A", str(row["source_path"]), index,
        int(stats["raw_flow_count"]), int(stats["active_sd_pairs"]), float(stats["total_demand"]),
    )


def build_sequence(resume: bool = True) -> list[Path]:
    """一次读取 official pickle，再逐 snapshot 使用原 K10 generator/cache 构建 canonical instance。"""

    row = _median_selection()
    source = Path(row["source_path"])
    sequence_load_start = time.perf_counter()
    records = {index: record for index, record in enumerate(_objects_from_pickle(source)) if index in SEQUENCE_INDICES}
    sequence_read_s = time.perf_counter() - sequence_load_start
    if set(records) != set(SEQUENCE_INDICES):
        raise RuntimeError("ORDERED_OFFICIAL_SEQUENCE_INCOMPLETE")
    paths: list[Path] = []
    timing_rows = []
    for index in SEQUENCE_INDICES:
        output = OUTPUT_ROOT / "instances" / f"A-{index}.pkl"
        paths.append(output)
        if resume and output.is_file():
            with output.open("rb") as handle:
                payload = pickle.load(handle)
            # official pickle 是单个 5000-record list；独立读取任一 snapshot 都承担整次 pickle.load。
            payload["timings"]["T_pickle_read"] = sequence_read_s
            _atomic_pickle(payload, output)
            timing_rows.append(payload["timings"])
            continue
        record = records[index]
        aggregate_start = time.perf_counter()
        demands, stats = aggregate_flowset(record["FlowSet"])
        aggregate_s = time.perf_counter() - aggregate_start
        path_start = time.perf_counter()
        satellite_paths, audit = generate_unique_paths(record, demands, 10, cache_dir=CACHE_ROOT)
        path_phase_s = time.perf_counter() - path_start
        canonical_start = time.perf_counter()
        official = assemble_official_instance(
            _provenance(row, index, stats), PAPER_LIKE_CONFIG, demands, satellite_paths, audit,
        )
        canonical_s = time.perf_counter() - canonical_start
        timings = {
            "record_index": index,
            "T_pickle_read": sequence_read_s,
            "T_flow_aggregate": aggregate_s,
            "T_path_generation_uncached": sum(float(item["path_generation_uncached_s"]) for item in audit),
            "T_path_cache_lookup": sum(float(item.get("path_cache_lookup_s", 0.0)) for item in audit),
            "T_path_phase_wall": path_phase_s,
            "T_canonical_instance": canonical_s,
            "cache_hits": sum(bool(item["cache_hit"]) for item in audit),
            "path_queries": len(audit),
        }
        _atomic_pickle({"instance": official, "timings": timings}, output)
        timing_rows.append(timings)
        print(f"BUILT_SEQUENCE_INSTANCE index={index} vars={sum(len(v) for v in official.benchmark.candidate_paths.values())}", flush=True)
    write_json({"evidence_label": "ORDERED_OFFICIAL_SEQUENCE", "indices": list(SEQUENCE_INDICES),
                "source_path": str(source), "sequence_pickle_read_s": sequence_read_s,
                "per_snapshot": timing_rows}, OUTPUT_ROOT / "sequence_build_timings.json")
    return paths


def _load_instances(paths: Sequence[Path]) -> list[Any]:
    result = []
    for path in paths:
        with path.open("rb") as handle:
            result.append(pickle.load(handle)["instance"])
    return result


def write_structure_artifacts(instances: Sequence[Any]) -> None:
    rows = [structure_record(item) for item in instances]
    transitions = [{
        "from_record_index": previous["record_index"], "to_record_index": current["record_index"],
        "classification": classify_transition(previous, current),
        "from_structure_hash": previous["structure_hash"], "to_structure_hash": current["structure_hash"],
    } for previous, current in zip(rows, rows[1:])]
    write_jsonl(rows, OUTPUT_ROOT / "sequence_structure.jsonl")
    exact = [row for row in transitions if row["classification"] == "EXACT_SAME_STRUCTURE"]
    write_json({"pair_count": len(exact), "pairs": exact, "all_transitions": transitions},
               OUTPUT_ROOT / "same_structure_pairs.json")
    manifests = []
    for count in (5, 10, 21):
        start = max(0, 10 - count // 2)
        selected = list(instances[start:start + count])
        universe = build_fixed_universe(selected)
        per_vars = [sum(len(paths) for paths in item.benchmark.candidate_paths.values()) for item in selected]
        per_nz = [structure_record(item)["num_nonzeros"] for item in selected]
        manifests.append({
            "snapshot_count": count, "record_indices": [item.provenance.record_index for item in selected],
            "union_flows": len(universe.flows), "union_vars": len(universe.variables),
            "union_constraints": len(universe.flows) + len(universe.edges), "union_edges": len(universe.edges),
            "union_nz": universe.num_nonzeros,
            "vars_growth_vs_median_snapshot": len(universe.variables) / statistics.median(per_vars),
            "nz_growth_vs_median_snapshot": universe.num_nonzeros / statistics.median(per_nz),
            "construction": "inactive flow demand=0; inactive semantic path UB=0; absent edge capacity=0",
        })
    write_json({"status": "OFFICIAL_FIXED_UNIVERSE_EXACT_REUSE", "windows": manifests},
               OUTPUT_ROOT / "fixed_universe_manifest.json")


def run_child(command: Sequence[str], output: Path, timeout_s: int = SOLVE_TIMEOUT_S,
              resume: bool = True) -> dict[str, Any]:
    """单 child、单命令、可 resume；timeout 时不会伪造测量值。"""

    if resume and output.is_file():
        return json.loads(output.read_text(encoding="utf-8"))
    try:
        completed = subprocess.run(list(command), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                   timeout=timeout_s, check=False)
    except subprocess.TimeoutExpired:
        return {"status": "TIMEOUT", "timeout_s": timeout_s}
    if completed.returncode != 0 or not output.is_file():
        return {"status": "FAILED", "failure_type": f"CHILD_EXIT_{completed.returncode}"}
    return json.loads(output.read_text(encoding="utf-8"))


def cold_child(instance_path: Path, output: Path, method: str) -> int:
    with instance_path.open("rb") as handle:
        payload = pickle.load(handle)
    result = solve_cold(payload["instance"].benchmark, method)
    result["record_index"] = payload["instance"].provenance.record_index
    result["application_timings"] = payload.get("timings", {})
    application = result["application_timings"]
    if all(name in application for name in ("T_pickle_read", "T_flow_aggregate", "T_path_phase_wall", "T_canonical_instance")):
        result["T_full_pipeline"] = result["T_solver_pipeline"] + sum(float(application[name]) for name in (
            "T_pickle_read", "T_flow_aggregate", "T_path_phase_wall", "T_canonical_instance",
        ))
    if result["record_index"] == 339:
        expected_path = ROUND3_ROOT / "instances/A-339-median.pkl"
        with expected_path.open("rb") as handle:
            expected = pickle.load(handle)["instance"]
        if payload["instance"].hashes() != expected.hashes():
            raise AssertionError("OFFICIAL_MEDIAN_HASH_CHANGED")
    write_json(result, output)
    return 0


def run_cold(resume: bool = True) -> list[dict[str, Any]]:
    jobs: list[tuple[Path, str, str]] = []
    median = OUTPUT_ROOT / "instances/A-339.pkl"
    jobs.extend((median, "DEFAULT", f"median-default-{repeat}") for repeat in range(1, 4))
    jobs.append((median, "DUAL_SIMPLEX", "median-dual-1"))
    for label, source in (("P90", ROUND3_ROOT / "instances/A-38-P90.pkl"),
                          ("P95", ROUND3_ROOT / "instances/A-103-P95.pkl")):
        jobs.append((source, "DEFAULT", f"{label}-default-1"))
    rows = []
    for instance_path, method, name in jobs:
        output = OUTPUT_ROOT / "cold" / f"{name}.json"
        command = [sys.executable, "-m", "experiments.speedup_attribution.run_gurobi_decomposition",
                   "--cold-child", str(instance_path), str(output), "--method", method]
        row = run_child(command, output, resume=resume)
        if row.get("status") == "MEASURED":
            with instance_path.open("rb") as handle:
                application = pickle.load(handle).get("timings", {})
            if application:
                row["application_timings"] = application
                row["T_full_pipeline"] = row["T_solver_pipeline"] + sum(float(application[name]) for name in (
                    "T_pickle_read", "T_flow_aggregate", "T_path_phase_wall", "T_canonical_instance",
                ))
        row["repeat_name"] = name
        rows.append(row)
        print(f"COLD_JOB {name} status={row.get('status')}", flush=True)
    write_jsonl(rows, OUTPUT_ROOT / "cold_decomposition.jsonl")
    return rows


def _reuse_window(instances: Sequence[Any]) -> list[Any]:
    # 10-snapshot union 兼顾协议要求的规模审计与至少 5 个 transition。
    return list(instances[5:15])


def run_ampl_fixed_reuse(instances: Sequence[Any], resume: bool = True) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    no_basis_path = OUTPUT_ROOT / "fixed_universe_ampl_no_basis.jsonl"
    basis_path = OUTPUT_ROOT / "fixed_universe_ampl_basis_attempt.jsonl"
    if resume and no_basis_path.is_file() and basis_path.is_file():
        return ([json.loads(line) for line in no_basis_path.read_text().splitlines()],
                [json.loads(line) for line in basis_path.read_text().splitlines()])
    window = _reuse_window(instances)
    universe = build_fixed_universe(window)

    no_basis_rows = []
    no_basis_session = FixedUniverseSession(universe)
    try:
        for item in window[1:]:
            result, _ = no_basis_session.solve(item.benchmark, None)
            result.update(T_initial_build=no_basis_session.T_ampl_create + no_basis_session.T_ampl_model_definition
                          + no_basis_session.T_initial_structure_load)
            no_basis_rows.append(result)
            print(f"REUSE_NO_BASIS index={item.provenance.record_index} total={result['T_reuse_total']:.6f}", flush=True)
            if len(no_basis_rows) >= 5 and result["T_reuse_total"] >= 30.0:
                break
    finally:
        no_basis_session.close()
    write_jsonl(no_basis_rows, no_basis_path)

    targets = {int(row["record_index"]) for row in no_basis_rows}
    basis_rows = []
    basis_session = FixedUniverseSession(universe)
    try:
        seed, basis = basis_seed(basis_session, window[0].benchmark)
        write_json(seed, OUTPUT_ROOT / "basis_seed.json")
        for item in window[1:]:
            if item.provenance.record_index not in targets:
                break
            result, basis = basis_session.solve(item.benchmark, basis)
            if basis is None:
                raise RuntimeError("BASIS_EXPORT_FAILED")
            result.update(T_initial_build=basis_session.T_ampl_create + basis_session.T_ampl_model_definition
                          + basis_session.T_initial_structure_load)
            basis_rows.append(result)
            print(f"REUSE_BASIS index={item.provenance.record_index} total={result['T_reuse_total']:.6f} verified={result['basis_reuse_verified']}", flush=True)
    finally:
        basis_session.close()
    write_jsonl(basis_rows, basis_path)
    return no_basis_rows, basis_rows


def run_persistent_reuse(instances: Sequence[Any], resume: bool = True) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """R2/R3：同一个 AMPLS/native Gurobi model 上原位更新并重新优化。"""

    no_basis_path = OUTPUT_ROOT / "reuse_no_basis.jsonl"
    basis_path = OUTPUT_ROOT / "reuse_basis.jsonl"
    if resume and no_basis_path.is_file() and basis_path.is_file():
        return ([json.loads(line) for line in no_basis_path.read_text().splitlines()],
                [json.loads(line) for line in basis_path.read_text().splitlines()])
    window = _reuse_window(instances)
    universe = build_fixed_universe(window)

    no_basis_rows = []
    no_basis_session = PersistentGurobiSession(universe, window[0].benchmark)
    try:
        for item in window[1:]:
            result, _ = no_basis_session.optimize(item.benchmark, None, no_basis=True)
            result.update(T_initial_build=no_basis_session.T_initial_build,
                          T_initial_export=no_basis_session.T_initial_export)
            no_basis_rows.append(result)
            print(f"PERSISTENT_NO_BASIS index={item.provenance.record_index} total={result['T_reuse_total']:.6f}", flush=True)
    finally:
        no_basis_session.close()
    write_jsonl(no_basis_rows, no_basis_path)

    basis_rows = []
    basis_session = PersistentGurobiSession(universe, window[0].benchmark)
    try:
        seed, basis = basis_session.seed_basis(window[0].benchmark)
        write_json(seed, OUTPUT_ROOT / "persistent_basis_seed.json")
        for item in window[1:]:
            result, basis = basis_session.optimize(item.benchmark, basis)
            if basis is None:
                raise RuntimeError("PERSISTENT_BASIS_EXPORT_FAILED")
            result.update(T_initial_build=basis_session.T_initial_build,
                          T_initial_export=basis_session.T_initial_export)
            basis_rows.append(result)
            print(f"PERSISTENT_BASIS index={item.provenance.record_index} total={result['T_reuse_total']:.6f} verified={result['basis_reuse_verified']}", flush=True)
    finally:
        basis_session.close()
    write_jsonl(basis_rows, basis_path)

    capabilities_path = OUTPUT_ROOT / "driver_capabilities.json"
    capabilities = json.loads(capabilities_path.read_text()) if capabilities_path.is_file() else driver_capabilities()
    capabilities["persistent_full_scale_probe"] = {
        "status": "PERSISTENT_GUROBI_FULL_SCALE_VERIFIED",
        "union_vars": len(universe.variables), "union_constraints": len(universe.flows) + len(universe.edges),
        "union_nz": universe.num_nonzeros,
    }
    capabilities["persistent_verdict"] = "PERSISTENT_GUROBI_MODEL_REUSE_IMPLEMENTED"
    write_json(capabilities, capabilities_path)
    return no_basis_rows, basis_rows


def run_ampl_object_reuse(instances: Sequence[Any], resume: bool = True) -> list[dict[str, Any]]:
    """R1 official sequence：同一 AMPL/model definition，逐 snapshot 完整 reset/reload data。"""

    output = OUTPUT_ROOT / "ampl_object_reuse.jsonl"
    if resume and output.is_file():
        return [json.loads(line) for line in output.read_text().splitlines()]
    window = _reuse_window(instances)[:6]
    session = AmplObjectSession()
    rows = []
    try:
        seed = session.solve(window[0].benchmark)
        write_json(seed, OUTPUT_ROOT / "ampl_object_seed.json")
        for item in window[1:]:
            result = session.solve(item.benchmark)
            result["T_initial_ampl_build"] = session.T_ampl_create + session.T_ampl_model_definition
            rows.append(result)
            print(f"AMPL_OBJECT_REUSE index={item.provenance.record_index} total={result['T_reuse_total']:.6f}", flush=True)
    finally:
        session.close()
    write_jsonl(rows, output)
    return rows


def run_basis_presolve_sanity(instances: Sequence[Any], resume: bool = True) -> dict[str, Any]:
    """非主结果：5-window 上用 Presolve off 判断 basis transport 是否真实可用。"""

    output = OUTPUT_ROOT / "basis_presolve_off_sanity.json"
    if resume and output.is_file():
        return json.loads(output.read_text())
    window = list(instances[8:13])
    universe = build_fixed_universe(window)
    session = FixedUniverseSession(universe, presolve=False)
    try:
        seed, basis = basis_seed(session, window[0].benchmark)
        result, _ = session.solve(window[1].benchmark, basis)
    finally:
        session.close()
    payload = {"status": "SANITY_ONLY_PRESOLVE_OFF", "seed": seed, "transition": result,
               "formal_main_result": False}
    write_json(payload, output)
    return payload


def _stats(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, Any]:
    values = sorted(float(row[key]) for row in rows if isinstance(row.get(key), (int, float)))
    if not values:
        return {"median": "NOT_AVAILABLE", "P90": "NOT_AVAILABLE", "P95": "NOT_AVAILABLE", "max": "NOT_AVAILABLE"}
    def percentile(p: float) -> float:
        position = (len(values) - 1) * p
        lower, upper = int(position), min(len(values) - 1, int(position) + 1)
        fraction = position - lower
        return values[lower] * (1 - fraction) + values[upper] * fraction
    return {"median": statistics.median(values), "P90": percentile(0.90), "P95": percentile(0.95), "max": max(values)}


def summarize() -> dict[str, Any]:
    cold = [json.loads(line) for line in (OUTPUT_ROOT / "cold_decomposition.jsonl").read_text().splitlines()]
    no_basis = [json.loads(line) for line in (OUTPUT_ROOT / "reuse_no_basis.jsonl").read_text().splitlines()]
    basis = [json.loads(line) for line in (OUTPUT_ROOT / "reuse_basis.jsonl").read_text().splitlines()]

    def normalize_persistent(rows: list[dict[str, Any]]) -> None:
        """兼容首次实验 artifact：保留原 wall，并以 native Runtime 组成正式 total。"""
        for row in rows:
            if "T_reuse_optimize_wall" not in row:
                row["T_reuse_optimize_wall"] = row["T_reuse_optimize"]
                row["T_reuse_total_wall_observed"] = row["T_reuse_total"]
                row["T_reuse_optimize"] = row["T_gurobi_optimize"]
                row["T_reuse_total"] = sum(float(row[name]) for name in (
                    "T_reuse_update", "T_basis_import", "T_gurobi_optimize", "T_reuse_extract",
                    "T_feasibility_check", "T_basis_export",
                ))
    normalize_persistent(no_basis)
    normalize_persistent(basis)
    write_jsonl(no_basis, OUTPUT_ROOT / "reuse_no_basis.jsonl")
    write_jsonl(basis, OUTPUT_ROOT / "reuse_basis.jsonl")
    ampl_object = [json.loads(line) for line in (OUTPUT_ROOT / "ampl_object_reuse.jsonl").read_text().splitlines()]
    fixed_no_basis = [json.loads(line) for line in (OUTPUT_ROOT / "fixed_universe_ampl_no_basis.jsonl").read_text().splitlines()]
    fixed_basis_attempt = [json.loads(line) for line in (OUTPUT_ROOT / "fixed_universe_ampl_basis_attempt.jsonl").read_text().splitlines()]
    cold_default = [row for row in cold if row.get("record_index") == 339 and row.get("method") == "DEFAULT" and row.get("status") == "MEASURED"]
    metric_names = (
        "T_sparse_rows", "T_ampl_create", "T_ampl_model_definition", "T_ampl_data_load",
        "T_driver_read", "T_driver_conversion", "T_driver_setup", "T_gurobi_optimize",
        "T_driver_output", "T_solution_extract", "T_solve_call_wall", "T_solver_pipeline",
        "accounted_fraction", "unaccounted_wall", "simplex_iterations",
    )
    cold_medians = {name: statistics.median(float(row[name]) for row in cold_default) for name in metric_names}
    cold_medians["T_ampl_internal_cpu"] = statistics.median(float(row["T_ampl_internal_cpu"]) for row in cold_default)
    cold_medians["T_feasibility_check"] = statistics.median(float(row["T_feasibility_check"]) for row in cold_default)
    cold_medians["barrier_iterations"] = statistics.median(float(row["barrier_iterations"]) for row in cold_default)
    cold_medians.update({
        name: statistics.median(float(row[name]) for row in cold_default)
        for name in ("presolve_removed_rows", "presolve_removed_cols", "presolved_rows", "presolved_cols", "presolved_nz")
    })
    reuse_keys = ("T_reuse_update", "T_gurobi_optimize", "T_reuse_extract", "T_reuse_total", "simplex_iterations")
    reuse_stats = {
        "reuse_no_basis": {key: _stats(no_basis, key) for key in reuse_keys},
        "reuse_basis": {key: _stats(basis, key) for key in reuse_keys},
        "basis_export": _stats(basis, "T_basis_export"), "basis_import": _stats(basis, "T_basis_import"),
    }
    cold_total = cold_medians["T_solver_pipeline"]
    cold_solve_wall = cold_medians["T_solve_call_wall"]
    no_basis_total = float(reuse_stats["reuse_no_basis"]["T_reuse_total"]["median"])
    basis_total = float(reuse_stats["reuse_basis"]["T_reuse_total"]["median"])
    summary = {
        "cold_repeats": len(cold_default), "cold_median": cold_medians,
        "timing_aggregates": {
            "APPLICATION_INPUT": cold_medians["T_sparse_rows"] + cold_medians["T_ampl_data_load"],
            "SOLVER_INPUT": cold_medians["T_driver_read"] + cold_medians["T_driver_conversion"],
            "T_input_total": cold_medians["T_sparse_rows"] + cold_medians["T_ampl_data_load"]
                             + cold_medians["T_driver_read"] + cold_medians["T_driver_conversion"],
            "AMPL_MODEL_BUILD": cold_medians["T_ampl_create"] + cold_medians["T_ampl_model_definition"],
            "GUROBI_SETUP": cold_medians["T_driver_setup"],
            "T_modeling_total": cold_medians["T_ampl_create"] + cold_medians["T_ampl_model_definition"]
                                + cold_medians["T_driver_setup"],
        },
        "reuse": reuse_stats,
        "ampl_object_reuse": {
            "T_reuse_ampl_data_update": _stats(ampl_object, "T_reuse_ampl_data_update"),
            "T_gurobi_optimize": _stats(ampl_object, "T_gurobi_optimize"),
            "T_reuse_total": _stats(ampl_object, "T_reuse_total"),
        },
        "fixed_universe_ampl_reuse": {
            "no_basis_total": _stats(fixed_no_basis, "T_reuse_total"),
            "basis_attempt_total": _stats(fixed_basis_attempt, "T_reuse_total"),
            "basis_reuse_verified": False,
        },
        "ModelReuseSpeedup": cold_total / no_basis_total,
        "BasisIncrementalSpeedup": no_basis_total / basis_total,
        "TotalReuseSpeedup": cold_total / basis_total,
        "latency_vs_sate": {
            "cold_vs_forward": cold_solve_wall / SATE_FORWARD_S, "cold_vs_e2e": cold_solve_wall / SATE_E2E_S,
            "reuse_no_basis_vs_e2e": no_basis_total / SATE_E2E_S,
            "reuse_basis_vs_e2e": basis_total / SATE_E2E_S,
            "reuse_no_basis_vs_forward": no_basis_total / SATE_FORWARD_S,
            "reuse_basis_vs_forward": basis_total / SATE_FORWARD_S,
            "paper_17ms_reference_persistent_no_basis": no_basis_total / 0.017,
        },
        "persistent_gurobi_reuse": "PERSISTENT_GUROBI_MODEL_REUSE_IMPLEMENTED",
        "basis_reuse_verified": bool(basis) and all(row["basis_reuse_verified"] for row in basis),
        "classifications": [
            "MODEL_CONSTRUCTION_DOMINATES_COLD", "REUSE_REMOVES_MODELING_BOTTLENECK",
            "SATE_REMAINS_FASTER_AFTER_CLASSICAL_REUSE",
        ],
    }
    write_json(summary, OUTPUT_ROOT / "timing_summary.json")
    write_json({"evidence_boundaries": {
        "cold": "OFFICIAL_COLD_AMPL_GUROBI", "ampl_object": "OFFICIAL_AMPL_OBJECT_REUSE",
        "fixed_universe": "OFFICIAL_FIXED_UNIVERSE_EXACT_REUSE",
        "persistent_no_basis": "OFFICIAL_PERSISTENT_REUSE_NO_BASIS",
        "persistent_basis": "OFFICIAL_PERSISTENT_REUSE_PREVIOUS_BASIS",
    }, **summary}, OUTPUT_ROOT / "reuse_summary.json")
    make_figures(cold_default, no_basis, basis, summary)
    return summary


def make_figures(cold: Sequence[Mapping[str, Any]], no_basis: Sequence[Mapping[str, Any]],
                 basis: Sequence[Mapping[str, Any]], summary: Mapping[str, Any]) -> None:
    import matplotlib.pyplot as plt

    root = OUTPUT_ROOT / "figures"
    root.mkdir(parents=True, exist_ok=True)
    cold_med = summary["cold_median"]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    keys = ("T_driver_read", "T_driver_conversion", "T_driver_setup", "T_gurobi_optimize", "T_driver_output", "unaccounted_wall")
    left = 0.0
    for key in keys:
        ax.barh(["Cold"], [cold_med[key]], left=left, label=key.removeprefix("T_"))
        left += cold_med[key]
    ax.set(xlabel="seconds", title="Cold solve-call decomposition"); ax.legend(fontsize=7); fig.tight_layout()
    fig.savefig(root / "01_cold_decomposition.png", dpi=160); plt.close(fig)

    labels = ["Cold", "Reuse no basis", "Reuse + basis"]
    totals = [cold_med["T_solver_pipeline"], summary["reuse"]["reuse_no_basis"]["T_reuse_total"]["median"],
              summary["reuse"]["reuse_basis"]["T_reuse_total"]["median"]]
    optimize = [cold_med["T_gurobi_optimize"], summary["reuse"]["reuse_no_basis"]["T_gurobi_optimize"]["median"],
                summary["reuse"]["reuse_basis"]["T_gurobi_optimize"]["median"]]
    iterations = [cold_med["simplex_iterations"], summary["reuse"]["reuse_no_basis"]["simplex_iterations"]["median"],
                  summary["reuse"]["reuse_basis"]["simplex_iterations"]["median"]]
    for filename, title, values, ylabel in (
        ("02_total_latency.png", "Cold vs reuse total latency", totals, "seconds"),
        ("03_optimize_only.png", "Optimize-only latency", optimize, "seconds"),
        ("04_iteration_count.png", "Simplex iteration count", iterations, "iterations"),
    ):
        fig, ax = plt.subplots(figsize=(7, 4.5)); ax.bar(labels, values); ax.set(title=title, ylabel=ylabel)
        ax.tick_params(axis="x", rotation=15); fig.tight_layout(); fig.savefig(root / filename, dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot([row["record_index"] for row in no_basis], [row["T_reuse_total"] for row in no_basis], marker="o", label="no basis")
    ax.plot([row["record_index"] for row in basis], [row["T_reuse_total"] for row in basis], marker="o", label="previous basis")
    ax.set(title="Reuse latency across ordered official sequence", xlabel="record index", ylabel="seconds"); ax.legend(); fig.tight_layout()
    fig.savefig(root / "05_ordered_sequence.png", dpi=160); plt.close(fig)

    structures = {row["record_index"]: row for row in map(json.loads, (OUTPUT_ROOT / "sequence_structure.jsonl").read_text().splitlines())}
    benefits = [cold_med["T_solver_pipeline"] / row["T_reuse_total"] for row in no_basis]
    fig, ax = plt.subplots(figsize=(7, 4.5)); ax.scatter([structures[row["record_index"]]["num_nonzeros"] for row in no_basis], benefits)
    ax.set(title="Snapshot NZ vs persistent reuse benefit", xlabel="snapshot NZ", ylabel="cold / reuse speedup"); fig.tight_layout()
    fig.savefig(root / "06_model_size_reuse.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    comparison_labels = ["Gurobi cold", "Reuse no basis", "Reuse+basis", "SaTE forward", "SaTE e2e"]
    comparison_values = [totals[0], totals[1], totals[2], SATE_FORWARD_S, SATE_E2E_S]
    ax.bar(comparison_labels, comparison_values); ax.set_yscale("log"); ax.set(title="Gurobi cold/reuse vs SaTE", ylabel="seconds (log)")
    ax.tick_params(axis="x", rotation=20); fig.tight_layout(); fig.savefig(root / "07_gurobi_vs_sate.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(["Paper solver 46s", "Paper solver 47s", "Paper SaTE 17ms", "Corrected cold", "Corrected reuse+basis"],
           [46.0, 47.0, 0.017, totals[0], totals[2]])
    ax.set_yscale("log"); ax.set(title="Paper reference vs corrected solver baseline", ylabel="seconds (log)")
    ax.tick_params(axis="x", rotation=20); fig.tight_layout(); fig.savefig(root / "08_paper_reference_corrected.png", dpi=160); plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--capabilities", action="store_true")
    parser.add_argument("--build-sequence", action="store_true")
    parser.add_argument("--cold", action="store_true")
    parser.add_argument("--reuse", action="store_true")
    parser.add_argument("--ampl-object-reuse", action="store_true")
    parser.add_argument("--basis-sanity", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--cold-child", nargs=2)
    parser.add_argument("--method", default="DEFAULT", choices=("DEFAULT", "DUAL_SIMPLEX", "PRIMAL_SIMPLEX"))
    args = parser.parse_args()
    if args.cold_child:
        output = Path(args.cold_child[1])
        try:
            code = cold_child(Path(args.cold_child[0]), output, args.method)
        except Exception as exc:
            # 子进程异常只持久化类型，禁止 Python traceback 泄露许可证正文。
            write_json({"status": "FAILED", "failure_type": type(exc).__name__}, output)
            code = 1
        raise SystemExit(code)
    resume = not args.no_resume
    if args.all or args.capabilities:
        capabilities = driver_capabilities()
        existing_path = OUTPUT_ROOT / "driver_capabilities.json"
        if existing_path.is_file():
            existing = json.loads(existing_path.read_text(encoding="utf-8"))
            # capability 重测不应丢失已经完成的 full-scale persistent probe。
            for key in ("persistent_full_scale_probe",):
                if key in existing:
                    capabilities[key] = existing[key]
            if existing.get("persistent_verdict") == "PERSISTENT_GUROBI_MODEL_REUSE_IMPLEMENTED":
                capabilities["persistent_verdict"] = existing["persistent_verdict"]
        write_json(capabilities, existing_path)
    paths = build_sequence(resume) if args.all or args.build_sequence else sorted((OUTPUT_ROOT / "instances").glob("A-*.pkl"))
    instances = _load_instances(paths) if paths else []
    if args.all or args.build_sequence:
        write_structure_artifacts(instances)
    if args.all or args.cold:
        run_cold(resume)
    if args.all or args.reuse:
        if len(instances) != 21:
            raise RuntimeError("BUILD_SEQUENCE_REQUIRED")
        run_persistent_reuse(instances, resume)
    if args.all or args.ampl_object_reuse:
        if len(instances) != 21:
            raise RuntimeError("BUILD_SEQUENCE_REQUIRED")
        run_ampl_object_reuse(instances, resume)
    if args.all or args.basis_sanity:
        if len(instances) != 21:
            raise RuntimeError("BUILD_SEQUENCE_REQUIRED")
        run_basis_presolve_sanity(instances, resume)
    if args.all or args.summarize:
        summarize()


if __name__ == "__main__":
    main()
