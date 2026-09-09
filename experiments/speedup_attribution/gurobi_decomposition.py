"""Round 3.1：official full-scale Gurobi timing decomposition 与严格 reuse 工具。"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import metadata
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
from typing import Any, Iterable, Mapping, Sequence

from .ampl_solver import AMPL_MODEL, PAPER_LIKE_CONFIG, sparse_rows
from .official_workload import OfficialBenchmarkInstance
from .schemas import BenchmarkInstance


OUTPUT_ROOT = Path("output/speedup_attribution/gurobi_decomposition")
EVIDENCE_COLD = "OFFICIAL_COLD_AMPL_GUROBI"
EVIDENCE_AMPL_REUSE = "OFFICIAL_AMPL_OBJECT_REUSE"
EVIDENCE_FIXED = "OFFICIAL_FIXED_UNIVERSE_EXACT_REUSE"
EVIDENCE_PERSISTENT_NO_BASIS = "OFFICIAL_PERSISTENT_REUSE_NO_BASIS"
EVIDENCE_PERSISTENT_BASIS = "OFFICIAL_PERSISTENT_REUSE_PREVIOUS_BASIS"
PERSISTENT_UNAVAILABLE = "TRUE_PERSISTENT_GUROBI_REUSE_NOT_AVAILABLE_WITH_CURRENT_LICENSE_PATH"
DIRECT_LICENSE_BLOCKED = "DIRECT_GUROBIPY_FULL_SCALE_NOT_LICENSED"

FIXED_UNIVERSE_MODEL = r"""
set F ordered;
set E ordered;
set V ordered;
set FV within {F, V};
set EV within {E, V};
param demand {F} >= 0;
param capacity {E} >= 0;
param upper {V} >= 0;
param basis_x {V} symbolic default 'none';
param basis_f {F} symbolic default 'none';
param basis_e {E} symbolic default 'none';
var x {v in V} >= 0, <= upper[v];
maximize total: sum {v in V} x[v];
subject to demand_limit {f in F}:
    sum {(ff,v) in FV: ff = f} x[v] <= demand[f];
subject to capacity_limit {e in E}:
    sum {(ee,v) in EV: ee = e} x[v] <= capacity[e];
"""

_UUID_RE = re.compile(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b")
_SAFE_LOG_RE = re.compile(
    r"^(NL model (read|conversion) time|Setup time|Solver time|Output time|Total time|"
    r"Optimize a model|Presolve removed|Presolved:|Solved in|LP warm-start:|\d+ simplex iteration)"
)


def _digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def redact_text(text: str) -> str:
    """只保留可发布文本；许可证 UUID、环境值与 host 标识一律移除。"""

    redacted = _UUID_RE.sub("[REDACTED_UUID]", text)
    secret = os.environ.get("AMPL_LICENSE_UUID")
    if secret:
        redacted = redacted.replace(secret, "[REDACTED_LICENSE]")
    redacted = re.sub(r"(?i)(AMPL_LICENSE_UUID\s*[=:]\s*)\S+", r"\1[REDACTED_LICENSE]", redacted)
    redacted = re.sub(r"(?i)\b(host(id|name)?|license (id|identifier|token))\s*[=:]\s*\S+", r"\1=[REDACTED]", redacted)
    return redacted


def timing_log_lines(text: str) -> list[str]:
    """正式 artifact 仅保留 timing、presolve、iteration 与 basis 证据行。"""

    return [line for line in redact_text(text).splitlines() if _SAFE_LOG_RE.match(line.strip())]


def basis_log_verified(text: str) -> bool:
    """只接受明确 use/crush basis 的日志，discard/未知文本不能通过证据 gate。"""

    return re.search(r"LP warm-start:\s*(use|crush)\b", text, flags=re.IGNORECASE) is not None


def _reject_secrets(value: Any) -> None:
    serialized = json.dumps(value, sort_keys=True, default=str)
    secret = os.environ.get("AMPL_LICENSE_UUID")
    if "AMPL_LICENSE_UUID" in serialized or (secret and secret in serialized) or _UUID_RE.search(serialized):
        raise ValueError("LICENSE_IDENTIFIER_PERSISTENCE_REJECTED")


def write_json(value: Any, path: Path) -> None:
    _reject_secrets(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_jsonl(rows: Iterable[Mapping[str, Any]], path: Path) -> None:
    materialized = [dict(row) for row in rows]
    _reject_secrets(materialized)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in materialized), encoding="utf-8")
    temporary.replace(path)


class CaptureOutput:
    """兼容 amplpy.OutputHandler 的内存捕获器；原始 solver log 不落盘。"""

    def __init__(self) -> None:
        self.parts: list[str] = []

    def output(self, _kind: Any, message: str) -> None:
        self.parts.append(message)

    @property
    def text(self) -> str:
        return "".join(self.parts)


def driver_statistics(ampl: Any) -> dict[str, Any]:
    """从 driver 直接返回的 objective suffix 读取 timing/stats，禁止 wall 相减。"""

    payload = json.loads(str(ampl.getValue("total.stats")))
    times = payload.get("times", {})
    stats = payload.get("stats", {})
    required = ("read", "conversion", "setup", "solver", "output")
    if any(name not in times or float(times[name]) < 0 for name in required):
        raise RuntimeError("DRIVER_TIMING_FIELDS_INVALID")
    return {
        "T_driver_read": float(times["read"]),
        "T_driver_conversion": float(times["conversion"]),
        "T_driver_setup": float(times["setup"]),
        # 该值来自 Gurobi driver 的 time_solver，不由 Python wall 相减得出。
        "T_gurobi_optimize": float(times["solver"]),
        "T_driver_output": float(times["output"]),
        "T_driver_reported_total": float(times.get("total", 0.0)),
        "simplex_iterations": int(stats["simplex_iterations"]) if "simplex_iterations" in stats else "NOT_AVAILABLE",
        "barrier_iterations": int(stats["barrier_iterations"]) if "barrier_iterations" in stats else "NOT_AVAILABLE",
        "node_count": float(stats["node_count"]) if "node_count" in stats else "NOT_AVAILABLE",
    }


def parse_presolve(log_text: str) -> dict[str, Any]:
    """解析明确的 Gurobi log 行；缺字段时 fail-closed 为 NOT_AVAILABLE。"""

    result: dict[str, Any] = {
        "presolve_removed_rows": "NOT_AVAILABLE", "presolve_removed_cols": "NOT_AVAILABLE",
        "presolved_rows": "NOT_AVAILABLE", "presolved_cols": "NOT_AVAILABLE",
        "presolved_nz": "NOT_AVAILABLE",
    }
    removed = re.search(r"Presolve removed (\d+) rows and (\d+) columns", log_text)
    presolved = re.search(r"Presolved:\s*(\d+) rows,\s*(\d+) columns,\s*([\d,]+) nonzeros", log_text)
    if removed:
        result.update(presolve_removed_rows=int(removed.group(1)), presolve_removed_cols=int(removed.group(2)))
    if presolved:
        result.update(presolved_rows=int(presolved.group(1)), presolved_cols=int(presolved.group(2)),
                      presolved_nz=int(presolved.group(3).replace(",", "")))
    return result


def timing_accounting(row: Mapping[str, Any]) -> dict[str, float | str]:
    names = ("T_driver_read", "T_driver_conversion", "T_driver_setup", "T_gurobi_optimize", "T_driver_output")
    if any(not isinstance(row.get(name), (int, float)) for name in names):
        return {"T_driver_components_sum": "NOT_AVAILABLE", "accounted_fraction": "NOT_AVAILABLE",
                "unaccounted_wall": "NOT_AVAILABLE", "timing_gate": "FAIL"}
    component_sum = sum(float(row[name]) for name in names)
    solve_wall = float(row["T_solve_call_wall"])
    unaccounted = solve_wall - component_sum
    return {
        "T_driver_components_sum": component_sum,
        "accounted_fraction": component_sum / solve_wall if solve_wall else 0.0,
        "unaccounted_wall": unaccounted,
        "timing_gate": "PASS" if unaccounted >= -1e-6 else "FAIL",
    }


def _load_ampl_data(ampl: Any, rows: Mapping[str, Any], instance: BenchmarkInstance) -> None:
    ampl.getSet("F").setValues(range(len(rows["flows"])))
    ampl.getSet("E").setValues(range(len(rows["edges"])))
    ampl.getSet("V").setValues(range(len(rows["variables"])))
    ampl.getSet("FV").setValues(rows["FV"])
    ampl.getSet("EV").setValues(rows["EV"])
    ampl.getParameter("demand").setValues({
        rows["flow_ids"][pair]: float(instance.demands[pair]) for pair in rows["flows"]
    })
    ampl.getParameter("capacity").setValues({
        rows["edge_ids"][edge]: float(instance.capacities[edge]) for edge in rows["edges"]
    })


def _extract_solution(rows: Mapping[str, Any], ampl: Any) -> tuple[float, list[float]]:
    """只把 AMPL solution 复制到 Python，不混入 feasibility 遍历。"""

    values = ampl.getVariable("x").getValues().toDict()
    solution = [float(values.get(variable_id, 0.0)) for variable_id in range(len(rows["variables"]))]
    objective = float(ampl.getObjective("total").value())
    return objective, solution


def _feasibility(instance: BenchmarkInstance, rows: Mapping[str, Any], solution: Sequence[float]) -> dict[str, Any]:
    """在 extraction 之后独立重算 demand/capacity loads。"""

    flow_loads = {pair: 0.0 for pair in rows["flows"]}
    edge_loads = {edge: 0.0 for edge in rows["edges"]}
    for value, (pair, _, path) in zip(solution, rows["variables"]):
        flow_loads[pair] += value
        for edge in zip(path[:-1], path[1:]):
            edge_loads[edge] += value
    return {
        "max_demand_violation": max(0.0, max((flow_loads[p] - instance.demands[p] for p in flow_loads), default=0.0)),
        "max_capacity_violation": max(0.0, max((edge_loads[e] - instance.capacities[e] for e in edge_loads), default=0.0)),
    }


def solve_cold(instance: BenchmarkInstance, method: str = "DEFAULT") -> dict[str, Any]:
    """一次独立 cold AMPL/Gurobi solve，并直接读取 driver timing suffix。"""

    from amplpy import AMPL

    sparse_start = time.perf_counter()
    rows = sparse_rows(instance)
    sparse_s = time.perf_counter() - sparse_start
    create_start = time.perf_counter()
    ampl = AMPL()
    create_s = time.perf_counter() - create_start
    capture = CaptureOutput()
    ampl.setOutputHandler(capture)
    try:
        definition_start = time.perf_counter()
        ampl.eval(AMPL_MODEL)
        definition_s = time.perf_counter() - definition_start
        load_start = time.perf_counter()
        _load_ampl_data(ampl, rows, instance)
        load_s = time.perf_counter() - load_start
        options = "outlev=1 tech:timing=2 tech:stats=3 alg:basis=2"
        if method == "DUAL_SIMPLEX":
            options += " alg:method=1"
        elif method == "PRIMAL_SIMPLEX":
            options += " alg:method=0"
        elif method != "DEFAULT":
            raise ValueError("UNSUPPORTED_SOLVER_METHOD")
        ampl.setOption("gurobi_options", options)
        ampl_cpu_before = float(ampl.getValue("_ampl_time"))
        solve_start = time.perf_counter()
        # verbose=True 将 driver log 送入内存 handler；正式 artifact 只保存白名单行。
        ampl.solve(solver="gurobi", verbose=True)
        solve_wall = time.perf_counter() - solve_start
        ampl_internal_cpu = max(0.0, float(ampl.getValue("_ampl_time")) - ampl_cpu_before)
        solve_result = str(ampl.getValue("solve_result"))
        if solve_result != "solved":
            # 失败时只暴露经过白名单过滤的 timing/stat 行，许可证文本不会进入输出。
            raise RuntimeError(f"SOLVE_NOT_OPTIMAL:{solve_result}:{timing_log_lines(capture.text)}")
        driver = driver_statistics(ampl)
        extract_start = time.perf_counter()
        objective, solution = _extract_solution(rows, ampl)
        extract_s = time.perf_counter() - extract_start
        check_start = time.perf_counter()
        checked = _feasibility(instance, rows, solution)
        feasible = checked["max_demand_violation"] <= 1e-8 and checked["max_capacity_violation"] <= 1e-8
        check_s = time.perf_counter() - check_start
        result = {
            "status": "MEASURED", "evidence_label": EVIDENCE_COLD, "method": method,
            **instance.hashes(), "num_vars": len(rows["variables"]),
            "num_constraints": len(rows["flows"]) + len(rows["edges"]), "num_nonzeros": rows["num_nonzeros"],
            "objective": objective, **checked, "feasible": feasible,
            "T_sparse_rows": sparse_s, "T_ampl_create": create_s,
            "T_ampl_model_definition": definition_s, "T_ampl_data_load": load_s,
            **driver, "T_solve_call_wall": solve_wall,
            "T_ampl_internal_cpu": ampl_internal_cpu,
            "T_solution_extract": extract_s, "T_feasibility_check": check_s,
            **parse_presolve(capture.text), "solver_log_evidence": timing_log_lines(capture.text),
        }
        result.update(timing_accounting(result))
        result["T_solver_pipeline"] = sparse_s + create_s + definition_s + load_s + solve_wall + extract_s + check_s
        return result
    finally:
        ampl.close()


def structure_record(official: OfficialBenchmarkInstance) -> dict[str, Any]:
    """对一个 canonical snapshot 生成完整结构签名。"""

    instance = official.benchmark
    flows = list(instance.active_pairs)
    edges = list(instance.active_path_edges)
    variables = [(pair, path) for pair in flows for path in instance.candidate_paths[pair]]
    fv = [(pair, path) for pair, path in variables]
    ev = [(edge, pair, path) for pair, path in variables for edge in zip(path[:-1], path[1:])]
    active_sd_hash = _digest([list(pair) for pair in flows])
    candidate_path_hash = _digest([[list(pair), [list(path) for path in instance.candidate_paths[pair]]] for pair in flows])
    path_incidence_hash = _digest([[list(edge), list(pair), list(path)] for edge, pair, path in ev])
    capacity_structure_hash = _digest([list(edge) for edge in edges])
    structure_hash = _digest({
        "F": [list(pair) for pair in flows],
        "E": [list(edge) for edge in edges],
        "V": [[list(pair), list(path)] for pair, path in variables],
        "FV": [[list(pair), list(path)] for pair, path in fv],
        "EV": [[list(edge), list(pair), list(path)] for edge, pair, path in ev],
    })
    return {
        "record_index": official.provenance.record_index, **official.hashes(),
        "active_sd_hash": active_sd_hash, "candidate_path_hash": candidate_path_hash,
        "path_incidence_hash": path_incidence_hash, "capacity_structure_hash": capacity_structure_hash,
        "structure_hash": structure_hash, "demand_hash": instance.hashes()["demand_hash"],
        "num_flows": len(flows), "num_vars": len(variables), "num_edges": len(edges),
        "num_nonzeros": len(variables) + len(ev),
    }


def classify_transition(previous: Mapping[str, Any], current: Mapping[str, Any]) -> str:
    if previous["structure_hash"] == current["structure_hash"]:
        return "EXACT_SAME_STRUCTURE"
    required = ("active_sd_hash", "candidate_path_hash", "path_incidence_hash", "capacity_structure_hash")
    if all(previous[name] == current[name] for name in required):
        raise AssertionError("STRUCTURE_HASH_DETECTOR_INCONSISTENT")
    return "SAME_SUPERSET_COMPATIBLE"


@dataclass(frozen=True)
class FixedUniverse:
    flows: tuple[tuple[int, int], ...]
    edges: tuple[tuple[int, int], ...]
    variables: tuple[tuple[tuple[int, int], tuple[int, ...]], ...]
    flow_ids: Mapping[tuple[int, int], int]
    edge_ids: Mapping[tuple[int, int], int]
    variable_ids: Mapping[tuple[tuple[int, int], tuple[int, ...]], int]
    fv: tuple[tuple[int, int], ...]
    ev: tuple[tuple[int, int], ...]

    @property
    def num_nonzeros(self) -> int:
        return len(self.fv) + len(self.ev)


def build_fixed_universe(instances: Sequence[OfficialBenchmarkInstance]) -> FixedUniverse:
    """用 semantic (SD,path) variable 构造有限 union；不复制、不补 path。"""

    flows = tuple(sorted({pair for item in instances for pair in item.benchmark.active_pairs}))
    variables = tuple(sorted({
        (pair, path) for item in instances for pair in item.benchmark.active_pairs
        for path in item.benchmark.candidate_paths[pair]
    }))
    edges = tuple(sorted({edge for _, path in variables for edge in zip(path[:-1], path[1:])}))
    flow_ids, edge_ids = ({value: index for index, value in enumerate(values)} for values in (flows, edges))
    variable_ids = {value: index for index, value in enumerate(variables)}
    fv = tuple((flow_ids[pair], variable_ids[(pair, path)]) for pair, path in variables)
    ev = tuple((edge_ids[edge], variable_ids[(pair, path)]) for pair, path in variables for edge in zip(path[:-1], path[1:]))
    return FixedUniverse(flows, edges, variables, flow_ids, edge_ids, variable_ids, fv, ev)


def fixed_values(universe: FixedUniverse, instance: BenchmarkInstance) -> dict[str, dict[int, float]]:
    """inactive flow demand=0、inactive semantic path UB=0；active UB 是冗余 demand 上界。"""

    active_variables = {(pair, path) for pair in instance.active_pairs for path in instance.candidate_paths[pair]}
    return {
        "demand": {universe.flow_ids[pair]: float(instance.demands.get(pair, 0.0)) for pair in universe.flows},
        "capacity": {universe.edge_ids[edge]: float(instance.capacities.get(edge, 0.0)) for edge in universe.edges},
        "upper": {
            universe.variable_ids[(pair, path)]: float(instance.demands[pair]) if (pair, path) in active_variables else 0.0
            for pair, path in universe.variables
        },
    }


def export_basis(ampl: Any) -> dict[str, dict[int, str]]:
    """显式导出 solver 返回的 simplex basis suffix。"""

    return {
        "x": dict(ampl.getVariable("x").getValues(["sstatus"]).toList()),
        "f": dict(ampl.getConstraint("demand_limit").getValues(["sstatus"]).toList()),
        "e": dict(ampl.getConstraint("capacity_limit").getValues(["sstatus"]).toList()),
    }


def import_basis(ampl: Any, basis: Mapping[str, Mapping[int, str]]) -> None:
    """通过 symbolic params 批量恢复 basis suffix，避免逐元素 Python/AMPL 调用。"""

    ampl.getParameter("basis_x").setValues(dict(basis["x"]))
    ampl.getParameter("basis_f").setValues(dict(basis["f"]))
    ampl.getParameter("basis_e").setValues(dict(basis["e"]))
    ampl.eval("let {v in V} x[v].sstatus := basis_x[v];")
    ampl.eval("let {f in F} demand_limit[f].sstatus := basis_f[f];")
    ampl.eval("let {e in E} capacity_limit[e].sstatus := basis_e[e];")


class FixedUniverseSession:
    """复用一个 AMPL object/model；每次 solve 仍由 timing 证据判断 driver 是否 persistent。"""

    def __init__(self, universe: FixedUniverse, presolve: bool = True) -> None:
        from amplpy import AMPL

        self.universe = universe
        self.presolve = presolve
        create_start = time.perf_counter()
        self.ampl = AMPL()
        self.T_ampl_create = time.perf_counter() - create_start
        self.capture = CaptureOutput()
        self.ampl.setOutputHandler(self.capture)
        definition_start = time.perf_counter()
        self.ampl.eval(FIXED_UNIVERSE_MODEL)
        if not presolve:
            # 仅供 basis transport sanity；Presolve off 永不作为正式主结果。
            self.ampl.setOption("presolve", 0)
        self.T_ampl_model_definition = time.perf_counter() - definition_start
        load_start = time.perf_counter()
        self.ampl.getSet("F").setValues(range(len(universe.flows)))
        self.ampl.getSet("E").setValues(range(len(universe.edges)))
        self.ampl.getSet("V").setValues(range(len(universe.variables)))
        self.ampl.getSet("FV").setValues(universe.fv)
        self.ampl.getSet("EV").setValues(universe.ev)
        self.T_initial_structure_load = time.perf_counter() - load_start

    def close(self) -> None:
        self.ampl.close()

    def update(self, instance: BenchmarkInstance) -> float:
        start = time.perf_counter()
        values = fixed_values(self.universe, instance)
        for name in ("demand", "capacity", "upper"):
            self.ampl.getParameter(name).setValues(values[name])
        return time.perf_counter() - start

    def _clear_basis(self) -> None:
        self.ampl.eval("let {v in V} x[v].sstatus := 'none';")
        self.ampl.eval("let {f in F} demand_limit[f].sstatus := 'none';")
        self.ampl.eval("let {e in E} capacity_limit[e].sstatus := 'none';")

    def solve(self, instance: BenchmarkInstance, basis: Mapping[str, Mapping[int, str]] | None) -> tuple[dict[str, Any], dict[str, dict[int, str]] | None]:
        update_s = self.update(instance)
        basis_import_s = 0.0
        if basis is None:
            # 该 session 从未请求返回 basis，三个 option 已显式禁止所有 warm state。
            options = "outlev=1 tech:timing=2 tech:stats=3 alg:method=1 alg:basis=0 alg:start=0 lp:warmstart=0"
            mode = "REUSE_NO_BASIS"
        else:
            import_start = time.perf_counter()
            import_basis(self.ampl, basis)
            basis_import_s = time.perf_counter() - import_start
            # LPWarmStart=2 允许 Gurobi 在默认 presolve 下 crush previous basis。
            warmstart = "lp:warmstart=2" if self.presolve else "lp:warmstart=1 pre:solve=0"
            options = f"outlev=1 tech:timing=2 tech:stats=3 alg:method=1 alg:basis=3 alg:start=0 {warmstart}"
            mode = "REUSE_PREVIOUS_BASIS"
        self.ampl.setOption("gurobi_options", options)
        self.capture.parts.clear()
        solve_start = time.perf_counter()
        self.ampl.solve(solver="gurobi", verbose=True)
        solve_wall = time.perf_counter() - solve_start
        if str(self.ampl.getValue("solve_result")) != "solved":
            raise RuntimeError("SOLVE_NOT_OPTIMAL")
        driver = driver_statistics(self.ampl)
        extract_start = time.perf_counter()
        raw_values = self.ampl.getVariable("x").getValues().toDict()
        solution = [float(raw_values.get(index, 0.0)) for index in range(len(self.universe.variables))]
        objective = float(self.ampl.getObjective("total").value())
        extract_s = time.perf_counter() - extract_start
        check_start = time.perf_counter()
        flow_loads = {pair: 0.0 for pair in instance.active_pairs}
        edge_loads = {edge: 0.0 for edge in instance.active_path_edges}
        for value, (pair, path) in zip(solution, self.universe.variables):
            if value == 0.0:
                continue
            if pair in flow_loads:
                flow_loads[pair] += value
            for edge in zip(path[:-1], path[1:]):
                if edge in edge_loads:
                    edge_loads[edge] += value
        demand_violation = max(0.0, max((flow_loads[p] - instance.demands[p] for p in flow_loads), default=0.0))
        capacity_violation = max(0.0, max((edge_loads[e] - instance.capacities[e] for e in edge_loads), default=0.0))
        check_s = time.perf_counter() - check_start
        basis_export_s = 0.0
        next_basis = None
        if basis is not None:
            export_start = time.perf_counter()
            next_basis = export_basis(self.ampl)
            basis_export_s = time.perf_counter() - export_start
        log = self.capture.text
        result = {
            "status": "MEASURED", "evidence_label": EVIDENCE_FIXED,
            "reuse_layer": EVIDENCE_AMPL_REUSE, "mode": mode,
            "record_index": int(instance.snapshot_id.split("-")[-2]), **instance.hashes(),
            "objective": objective, "max_demand_violation": demand_violation,
            "max_capacity_violation": capacity_violation,
            "T_reuse_update": update_s, "T_basis_import": basis_import_s,
            **driver, "T_reuse_solve_wall": solve_wall, "T_reuse_extract": extract_s,
            "T_feasibility_check": check_s, "T_basis_export": basis_export_s,
            "T_reuse_total": update_s + basis_import_s + solve_wall + extract_s + check_s + basis_export_s,
            "warm_state_disabled": basis is None,
            "basis_reuse_verified": basis is not None and basis_log_verified(log),
            **parse_presolve(log), "solver_log_evidence": timing_log_lines(log),
        }
        result.update(timing_accounting({**result, "T_solve_call_wall": solve_wall}))
        return result, next_basis


def basis_seed(session: FixedUniverseSession, instance: BenchmarkInstance) -> tuple[dict[str, Any], dict[str, dict[int, str]]]:
    """首次 dual solve 只负责产生可导出的 basis，不计入 transition 统计。"""

    session.update(instance)
    options = "outlev=1 tech:timing=2 tech:stats=3 alg:method=1 alg:basis=2 alg:start=0 lp:warmstart=0"
    if not session.presolve:
        options += " pre:solve=0"
    session.ampl.setOption("gurobi_options", options)
    session.capture.parts.clear()
    start = time.perf_counter()
    session.ampl.solve(solver="gurobi", verbose=True)
    solve_wall = time.perf_counter() - start
    export_start = time.perf_counter()
    basis = export_basis(session.ampl)
    export_s = time.perf_counter() - export_start
    return {
        "status": "MEASURED", "mode": "BASIS_SEED", "T_solve_call_wall": solve_wall,
        "T_basis_export": export_s, **driver_statistics(session.ampl),
        "solver_log_evidence": timing_log_lines(session.capture.text),
    }, basis


class AmplObjectSession:
    """R1：仅复用 AMPL object/model definition，每个 snapshot 完整重载 sets/data。"""

    def __init__(self) -> None:
        from amplpy import AMPL

        create_start = time.perf_counter()
        self.ampl = AMPL()
        self.T_ampl_create = time.perf_counter() - create_start
        self.capture = CaptureOutput()
        self.ampl.setOutputHandler(self.capture)
        definition_start = time.perf_counter()
        self.ampl.eval(AMPL_MODEL)
        self.T_ampl_model_definition = time.perf_counter() - definition_start

    def close(self) -> None:
        self.ampl.close()

    def solve(self, instance: BenchmarkInstance) -> dict[str, Any]:
        sparse_start = time.perf_counter()
        rows = sparse_rows(instance)
        sparse_s = time.perf_counter() - sparse_start
        update_start = time.perf_counter()
        self.ampl.eval("reset data;")
        _load_ampl_data(self.ampl, rows, instance)
        update_s = time.perf_counter() - update_start
        self.ampl.setOption(
            "gurobi_options",
            "outlev=1 tech:timing=2 tech:stats=3 alg:method=1 alg:basis=0 alg:start=0 lp:warmstart=0",
        )
        self.capture.parts.clear()
        solve_start = time.perf_counter()
        self.ampl.solve(solver="gurobi", verbose=True)
        solve_wall = time.perf_counter() - solve_start
        if str(self.ampl.getValue("solve_result")) != "solved":
            raise RuntimeError("SOLVE_NOT_OPTIMAL")
        driver = driver_statistics(self.ampl)
        extract_start = time.perf_counter()
        objective, solution = _extract_solution(rows, self.ampl)
        extract_s = time.perf_counter() - extract_start
        check_start = time.perf_counter()
        checked = _feasibility(instance, rows, solution)
        check_s = time.perf_counter() - check_start
        result = {
            "status": "MEASURED", "evidence_label": EVIDENCE_AMPL_REUSE,
            "record_index": int(instance.snapshot_id.split("-")[-2]), **instance.hashes(),
            "objective": objective, **checked, "T_sparse_rows": sparse_s,
            "T_reuse_ampl_data_update": update_s, **driver,
            "T_reuse_solve_wall": solve_wall, "T_reuse_extract": extract_s,
            "T_feasibility_check": check_s,
            "T_reuse_total": sparse_s + update_s + solve_wall + extract_s + check_s,
            "warm_state_disabled": True, **parse_presolve(self.capture.text),
            "solver_log_evidence": timing_log_lines(self.capture.text),
        }
        result.update(timing_accounting({**result, "T_solve_call_wall": solve_wall}))
        return result


class PersistentGurobiSession:
    """R2/R3：AMPLS 持有同一个 native Gurobi model，只原位修改 RHS/UB。"""

    def __init__(self, universe: FixedUniverse, initial: BenchmarkInstance) -> None:
        from amplpy import AMPL
        import amplpy_gurobi as ampls

        self.ampls = ampls
        self.universe = universe
        self.tempdir = tempfile.TemporaryDirectory(prefix="sate-ampls-")
        self.log_path = Path(self.tempdir.name) / "gurobi.log"
        self.log_offset = 0
        build_start = time.perf_counter()
        self.ampl = AMPL()
        self.ampl.eval(FIXED_UNIVERSE_MODEL)
        # persistent union 必须禁止 AMPL presolve删掉当前 inactive variable；Gurobi Presolve 保持默认。
        self.ampl.setOption("presolve", 0)
        self.ampl.getSet("F").setValues(range(len(universe.flows)))
        self.ampl.getSet("E").setValues(range(len(universe.edges)))
        self.ampl.getSet("V").setValues(range(len(universe.variables)))
        self.ampl.getSet("FV").setValues(universe.fv)
        self.ampl.getSet("EV").setValues(universe.ev)
        values = fixed_values(universe, initial)
        for name in ("demand", "capacity", "upper"):
            self.ampl.getParameter(name).setValues(values[name])
        export_start = time.perf_counter()
        self.model = self.ampl.to_ampls("gurobi", ["outlev=0"])
        self.T_initial_export = time.perf_counter() - export_start
        self.T_initial_build = time.perf_counter() - build_start
        self.grb_model = self.model.getGRBmodel()
        self.var_map = dict(self.model.get_var_map())
        self.con_map = dict(self.model.get_con_map())
        self.model.set_param("Method", 1)
        self.model.set_param("Presolve", -1)
        self.model.set_param("LogToConsole", 0)
        self.model.set_param("OutputFlag", 1)
        self.model.set_param("LogFile", str(self.log_path))
        self._validate_maps()

    def _validate_maps(self) -> None:
        expected_vars = {f"x[{index}]" for index in range(len(self.universe.variables))}
        expected_cons = ({f"demand_limit[{index}]" for index in range(len(self.universe.flows))}
                         | {f"capacity_limit[{index}]" for index in range(len(self.universe.edges))})
        # AMPLS con map 还包含 objective `total`，其 index 位于约束范围之外。
        if set(self.var_map) != expected_vars or set(self.con_map) != expected_cons | {"total"}:
            raise RuntimeError("AMPLS_INDEX_MAP_INCOMPLETE")

    def close(self) -> None:
        try:
            self.ampl.close()
        finally:
            self.tempdir.cleanup()

    def _set_double_array(self, attribute: str, values: Sequence[float]) -> None:
        array = self.ampls.dblArray(len(values))
        for index, value in enumerate(values):
            array[index] = float(value)
        code = self.ampls.GRBsetdblattrarray(self.grb_model, attribute, 0, len(values), array)
        if code:
            raise RuntimeError("GUROBI_ATTRIBUTE_UPDATE_FAILED")

    def update(self, instance: BenchmarkInstance) -> float:
        start = time.perf_counter()
        values = fixed_values(self.universe, instance)
        rhs = [0.0] * self.model.get_num_cons()
        for flow, flow_id in self.universe.flow_ids.items():
            rhs[self.con_map[f"demand_limit[{flow_id}]"]] = values["demand"][flow_id]
        for edge, edge_id in self.universe.edge_ids.items():
            rhs[self.con_map[f"capacity_limit[{edge_id}]"]] = values["capacity"][edge_id]
        upper = [0.0] * self.model.get_num_vars()
        for variable_id, value in values["upper"].items():
            upper[self.var_map[f"x[{variable_id}]"]] = value
        self._set_double_array(self.ampls.GRB_DBL_ATTR_RHS, rhs)
        self._set_double_array(self.ampls.GRB_DBL_ATTR_UB, upper)
        if self.ampls.GRBupdatemodel(self.grb_model):
            raise RuntimeError("GUROBI_MODEL_UPDATE_FAILED")
        return time.perf_counter() - start

    def export_basis(self) -> tuple[dict[str, list[int]], float]:
        start = time.perf_counter()
        arrays: dict[str, list[int]] = {}
        for key, attribute, length in (
            ("VBasis", self.ampls.GRB_INT_ATTR_VBASIS, self.model.get_num_vars()),
            ("CBasis", self.ampls.GRB_INT_ATTR_CBASIS, self.model.get_num_cons()),
        ):
            raw = self.ampls.intArray(length)
            code = self.ampls.GRBgetintattrarray(self.grb_model, attribute, 0, length, raw)
            if code:
                raise RuntimeError("GUROBI_BASIS_EXPORT_FAILED")
            arrays[key] = [int(raw[index]) for index in range(length)]
        return arrays, time.perf_counter() - start

    def import_basis(self, basis: Mapping[str, Sequence[int]]) -> float:
        start = time.perf_counter()
        for key, attribute, length in (
            ("VBasis", self.ampls.GRB_INT_ATTR_VBASIS, self.model.get_num_vars()),
            ("CBasis", self.ampls.GRB_INT_ATTR_CBASIS, self.model.get_num_cons()),
        ):
            if len(basis[key]) != length:
                raise RuntimeError("GUROBI_BASIS_SIZE_MISMATCH")
            raw = self.ampls.intArray(length)
            for index, value in enumerate(basis[key]):
                raw[index] = int(value)
            code = self.ampls.GRBsetintattrarray(self.grb_model, attribute, 0, length, raw)
            if code:
                raise RuntimeError("GUROBI_BASIS_IMPORT_FAILED")
        if self.ampls.GRBupdatemodel(self.grb_model):
            raise RuntimeError("GUROBI_BASIS_UPDATE_FAILED")
        return time.perf_counter() - start

    def _new_log_lines(self) -> list[str]:
        if not self.log_path.is_file():
            return []
        raw = self.log_path.read_text(encoding="utf-8", errors="replace")
        new = raw[self.log_offset:]
        self.log_offset = len(raw)
        return timing_log_lines(new)

    def _solution_and_feasibility(self, instance: BenchmarkInstance) -> tuple[float, float, float, float, float]:
        extract_start = time.perf_counter()
        native_solution = list(self.model.get_solution_vector())
        solution = [native_solution[self.var_map[f"x[{index}]"]] for index in range(len(self.universe.variables))]
        objective = float(self.model.get_double_attr("ObjVal"))
        extract_s = time.perf_counter() - extract_start
        check_start = time.perf_counter()
        flow_loads = {pair: 0.0 for pair in instance.active_pairs}
        edge_loads = {edge: 0.0 for edge in instance.active_path_edges}
        for value, (pair, path) in zip(solution, self.universe.variables):
            if value == 0.0:
                continue
            if pair in flow_loads:
                flow_loads[pair] += value
            for edge in zip(path[:-1], path[1:]):
                if edge in edge_loads:
                    edge_loads[edge] += value
        demand_violation = max(0.0, max((flow_loads[p] - instance.demands[p] for p in flow_loads), default=0.0))
        capacity_violation = max(0.0, max((edge_loads[e] - instance.capacities[e] for e in edge_loads), default=0.0))
        return objective, demand_violation, capacity_violation, extract_s, time.perf_counter() - check_start

    def optimize(self, instance: BenchmarkInstance, basis: Mapping[str, Sequence[int]] | None,
                 no_basis: bool = False) -> tuple[dict[str, Any], dict[str, list[int]] | None]:
        update_s = self.update(instance)
        basis_import_s = 0.0
        if no_basis:
            if self.ampls.GRBreset(self.grb_model, 0):
                raise RuntimeError("GUROBI_RESET_FAILED")
            self.model.set_param("LPWarmStart", 0)
            evidence_label = EVIDENCE_PERSISTENT_NO_BASIS
        else:
            if basis is None:
                raise ValueError("PREVIOUS_BASIS_REQUIRED")
            basis_import_s = self.import_basis(basis)
            self.model.set_param("LPWarmStart", 2)
            evidence_label = EVIDENCE_PERSISTENT_BASIS
        optimize_start = time.perf_counter()
        self.model.optimize()
        optimize_wall = time.perf_counter() - optimize_start
        status = int(self.model.get_int_attr("Status"))
        if status != 2:
            raise RuntimeError("PERSISTENT_SOLVE_NOT_OPTIMAL")
        objective, demand_violation, capacity_violation, extract_s, check_s = self._solution_and_feasibility(instance)
        next_basis = None
        basis_export_s = 0.0
        if not no_basis:
            next_basis, basis_export_s = self.export_basis()
        log_lines = self._new_log_lines()
        native_runtime = float(self.model.get_double_attr("Runtime"))
        total_wall_observed = update_s + basis_import_s + optimize_wall + extract_s + check_s + basis_export_s
        result = {
            "status": "MEASURED", "evidence_label": evidence_label,
            "fixed_universe_label": EVIDENCE_FIXED,
            "record_index": int(instance.snapshot_id.split("-")[-2]), **instance.hashes(),
            "objective": objective, "max_demand_violation": demand_violation,
            "max_capacity_violation": capacity_violation,
            "T_reuse_update": update_s, "T_basis_import": basis_import_s,
            "T_reuse_optimize_wall": optimize_wall,
            # Native Runtime 是 Gurobi 直接统计，不是 Python wall 相减。
            "T_reuse_optimize": native_runtime, "T_gurobi_optimize": native_runtime,
            "T_reuse_extract": extract_s, "T_feasibility_check": check_s,
            "T_basis_export": basis_export_s,
            "T_reuse_total": update_s + basis_import_s + native_runtime + extract_s + check_s + basis_export_s,
            "T_reuse_total_wall_observed": total_wall_observed,
            "simplex_iterations": int(self.model.get_double_attr("IterCount")),
            "barrier_iterations": int(self.model.get_int_attr("BarIterCount")),
            "node_count": float(self.model.get_double_attr("NodeCount")),
            "warm_state_disabled": no_basis,
            "basis_state_available": next_basis is not None,
            "basis_reuse_verified": (not no_basis and basis is not None),
            "solver_api_basis_evidence": (
                {"VBasis_count": len(basis["VBasis"]), "CBasis_count": len(basis["CBasis"])}
                if basis is not None else "NO_BASIS_BY_DESIGN"
            ),
            "solver_log_evidence": log_lines,
        }
        result.update(parse_presolve("\n".join(log_lines)))
        return result, next_basis

    def seed_basis(self, instance: BenchmarkInstance) -> tuple[dict[str, Any], dict[str, list[int]]]:
        self.update(instance)
        self.model.set_param("LPWarmStart", 0)
        start = time.perf_counter()
        self.model.optimize()
        optimize_wall = time.perf_counter() - start
        basis, export_s = self.export_basis()
        return {
            "status": "MEASURED", "mode": "PERSISTENT_BASIS_SEED",
            "T_initial_build": self.T_initial_build, "T_initial_export": self.T_initial_export,
            "T_reuse_optimize": optimize_wall, "T_gurobi_optimize": float(self.model.get_double_attr("Runtime")),
            "T_basis_export": export_s, "simplex_iterations": int(self.model.get_double_attr("IterCount")),
            "basis_state_available": True, "solver_log_evidence": self._new_log_lines(),
        }, basis


def verify_fixed_parity(cold: Mapping[str, Any], reuse: Mapping[str, Any], tolerance: float = 1e-8) -> dict[str, Any]:
    scale = max(1.0, abs(float(cold["objective"])))
    relative = abs(float(cold["objective"]) - float(reuse["objective"])) / scale
    feasible = max(float(reuse["max_demand_violation"]), float(reuse["max_capacity_violation"])) <= tolerance
    status = "FIXED_UNIVERSE_REUSE_PASS" if relative <= tolerance and feasible else "FIXED_UNIVERSE_REUSE_FAIL"
    return {"status": status, "relative_objective_difference": relative, "feasible": feasible, "tolerance": tolerance}


def direct_gurobipy_license_probe(num_variables: int = 6001) -> dict[str, Any]:
    """只做许可 gate，不尝试绕过 direct gurobipy size restriction。"""

    try:
        import gurobipy as gp
        model = gp.Model()
        model.Params.OutputFlag = 0
        model.addVars(num_variables, lb=0.0, ub=1.0, obj=1.0)
        model.ModelSense = gp.GRB.MAXIMIZE
        model.optimize()
        return {"status": "DIRECT_GUROBIPY_FULL_SCALE_LICENSED", "variables": num_variables}
    except Exception as exc:
        return {"status": DIRECT_LICENSE_BLOCKED, "variables": num_variables, "failure_type": type(exc).__name__}


def persistent_api_probe() -> dict[str, Any]:
    """按 AMPL.to_ampls 的真实 import 依赖探测，不把重复 solve 冒充 persistent model。"""

    try:
        import amplpy_gurobi  # type: ignore
        return {"status": "PERSISTENT_GUROBI_API_IMPORT_AVAILABLE", "module": "amplpy_gurobi",
                "version": getattr(amplpy_gurobi, "__version__", "NOT_AVAILABLE")}
    except Exception as exc:
        return {"status": "PERSISTENT_GUROBI_API_NOT_AVAILABLE", "module": "amplpy_gurobi",
                "failure_type": type(exc).__name__}


def driver_capabilities() -> dict[str, Any]:
    """结合 installed module help 与小 LP suffix 返回值验证 timing 语义。"""

    from amplpy import AMPL

    help_run = subprocess.run(
        [sys.executable, "-m", "amplpy.modules", "run", "gurobi", "-="],
        capture_output=True, text=True, timeout=60, check=False,
    )
    help_text = help_run.stdout + help_run.stderr
    ampl = AMPL()
    capture = CaptureOutput()
    ampl.setOutputHandler(capture)
    try:
        ampl.eval("var x >= 0; maximize total: x; subject to cap: x <= 1;")
        ampl.setOption("gurobi_options", "outlev=1 tech:timing=2 tech:stats=3")
        ampl.solve(solver="gurobi", verbose=False)
        returned = driver_statistics(ampl)
    finally:
        ampl.close()
    timing_help = "tech:timing" in help_text and all(name in help_text for name in ("time_read", "time_conversion", "time_setup", "time_solver", "time_output"))
    stats_help = "tech:stats" in help_text and "simplex_iterations" in help_text
    try:
        gurobi_version = metadata.version("ampl-module-gurobi")
    except metadata.PackageNotFoundError:
        gurobi_version = "NOT_AVAILABLE"
    persistent = persistent_api_probe()
    return {
        "driver": "AMPL_GUROBI", "ampl_module_gurobi_version": gurobi_version,
        "options": {
            "tech:timing=2": {"available": timing_help, "verified": timing_help,
                "returned_timing_fields": ["time_read", "time_conversion", "time_setup", "time_solver", "time_output"],
                "semantic_interpretation": "driver wall times; time_solver is direct solver optimize timing"},
            "tech:stats=3": {"available": stats_help, "verified": stats_help,
                "returned_statistics": ["simplex_iterations", "barrier_iterations", "node_count"]},
            "alg:basis": {"available": "alg:basis" in help_text, "verified": "alg:basis" in help_text},
            "alg:start": {"available": "alg:start" in help_text, "verified": "alg:start" in help_text},
            "lp:warmstart": {"available": "lp:warmstart" in help_text, "verified": "lp:warmstart" in help_text},
        },
        "small_lp_returned": returned,
        "persistent_api": persistent,
        "direct_gurobipy_license": direct_gurobipy_license_probe(),
        "persistent_verdict": (
            "PERSISTENT_GUROBI_API_AVAILABLE_PENDING_FULL_SCALE_PROBE"
            if persistent["status"] == "PERSISTENT_GUROBI_API_IMPORT_AVAILABLE" else PERSISTENT_UNAVAILABLE
        ),
    }
