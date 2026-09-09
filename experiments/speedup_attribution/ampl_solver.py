"""与现有 PathFormulation 等价的 sparse AMPL/Gurobi 后端。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import time
from typing import Any, Mapping

from .benchmark_solver import solve_level
from .schemas import BenchmarkInstance, SafetyLimits, estimate_lp


class SolverBackend(str, Enum):
    """求解器抽象；DIRECT_GUROBIPY 保留作小规模 reference。"""

    DIRECT_GUROBIPY = "DIRECT_GUROBIPY"
    AMPL_GUROBI = "AMPL_GUROBI"


class ThreadMode(str, Enum):
    """显式区分论文主比较与第二轮单线程衔接。"""

    DEFAULT = "GUROBI_DEFAULT"
    THREADS_1 = "GUROBI_THREADS_1"
    THREADS_24 = "GUROBI_THREADS_24"


@dataclass(frozen=True)
class CapacityConfiguration:
    name: str
    network_capacity: float
    access_capacity: float


PAPER_LIKE_CONFIG = CapacityConfiguration("PAPER_LIKE_CONFIG", 200.0, 50.0)
CODE_PUBLIC_CONFIG = CapacityConfiguration("CODE_PUBLIC_CONFIG", 200.0, 800.0)


AMPL_MODEL = r"""
set F ordered;
set E ordered;
set V ordered;
set FV within {F, V};
set EV within {E, V};
param demand {F} >= 0;
param capacity {E} >= 0;
var x {V} >= 0;
maximize total: sum {v in V} x[v];
subject to demand_limit {f in F}:
    sum {(ff,v) in FV: ff = f} x[v] <= demand[f];
subject to capacity_limit {e in E}:
    sum {(ee,v) in EV: ee = e} x[v] <= capacity[e];
"""


def sparse_rows(instance: BenchmarkInstance) -> dict[str, Any]:
    """只创建 active-path incidence tuples，绝不创建 dense traffic matrix。"""

    flows = list(instance.active_pairs)
    edges = list(instance.active_path_edges)
    flow_ids = {pair: index for index, pair in enumerate(flows)}
    edge_ids = {edge: index for index, edge in enumerate(edges)}
    variables: list[tuple[tuple[int, int], int, tuple[int, ...]]] = []
    for pair in flows:
        for path_index, path in enumerate(instance.candidate_paths[pair]):
            variables.append((pair, path_index, path))
    fv = [(flow_ids[pair], variable_id) for variable_id, (pair, _, _) in enumerate(variables)]
    ev = [
        (edge_ids[edge], variable_id)
        for variable_id, (_, _, path) in enumerate(variables)
        for edge in zip(path[:-1], path[1:])
    ]
    return {
        "flows": flows,
        "edges": edges,
        "variables": variables,
        "flow_ids": flow_ids,
        "edge_ids": edge_ids,
        "FV": fv,
        "EV": ev,
        # 与 Gurobi NumNZs 一致：只统计约束矩阵，不把 objective 系数计入 NZ。
        "num_nonzeros": len(fv) + len(ev),
    }


def _set_parameter(parameter: Any, values: Mapping[int, float]) -> None:
    parameter.setValues(dict(values))


def solve_ampl_gurobi(
    instance: BenchmarkInstance, thread_mode: ThreadMode = ThreadMode.DEFAULT,
    solver_name: str = "gurobi",
) -> dict[str, Any]:
    """在内存中定义/载入 sparse AMPL 模型并返回可比求解证据。"""

    thread_mode = ThreadMode(thread_mode)
    try:
        from amplpy import AMPL
    except Exception as exc:
        return {"backend": f"AMPL_{solver_name.upper()}", "status": "FAILED", "failure_type": type(exc).__name__}

    rows = sparse_rows(instance)
    ampl = None
    try:
        start = time.perf_counter()
        ampl = AMPL()
        ampl.eval(AMPL_MODEL)
        definition_s = time.perf_counter() - start

        load_start = time.perf_counter()
        ampl.getSet("F").setValues(range(len(rows["flows"])))
        ampl.getSet("E").setValues(range(len(rows["edges"])))
        ampl.getSet("V").setValues(range(len(rows["variables"])))
        ampl.getSet("FV").setValues(rows["FV"])
        ampl.getSet("EV").setValues(rows["EV"])
        _set_parameter(ampl.getParameter("demand"), {
            rows["flow_ids"][pair]: float(instance.demands[pair]) for pair in rows["flows"]
        })
        _set_parameter(ampl.getParameter("capacity"), {
            rows["edge_ids"][edge]: float(instance.capacities[edge]) for edge in rows["edges"]
        })
        data_load_s = time.perf_counter() - load_start

        options = "outlev=0"
        if thread_mode is ThreadMode.THREADS_1:
            options += " threads=1"
        elif thread_mode is ThreadMode.THREADS_24:
            options += " threads=24"
        # DEFAULT 刻意不传 threads，让 Gurobi 使用安装环境默认策略。
        if solver_name != "gurobi":
            # Cross-check solver 固定单线程，避免把并行差异混入 objective parity。
            options = "outlev=0 threads=1"
        ampl.setOption(f"{solver_name}_options", options)
        solve_start = time.perf_counter()
        ampl.solve(solver=solver_name, verbose=False)
        solve_wall_s = time.perf_counter() - solve_start
        if str(ampl.getValue("solve_result")) != "solved":
            return {"backend": f"AMPL_{solver_name.upper()}", "status": "FAILED", "failure_type": "SOLVE_NOT_OPTIMAL"}

        extract_start = time.perf_counter()
        values = ampl.getVariable("x").getValues().toDict()
        objective = float(ampl.getObjective("total").value())
        solution = [float(values.get(variable_id, 0.0)) for variable_id in range(len(rows["variables"]))]
        solution_extract_s = time.perf_counter() - extract_start
        flow_loads = {pair: 0.0 for pair in rows["flows"]}
        edge_loads = {edge: 0.0 for edge in rows["edges"]}
        for value, (pair, _, path) in zip(solution, rows["variables"]):
            flow_loads[pair] += value
            for edge in zip(path[:-1], path[1:]):
                edge_loads[edge] += value
        max_demand_violation = max((flow_loads[p] - instance.demands[p] for p in flow_loads), default=0.0)
        max_capacity_violation = max((edge_loads[e] - instance.capacities[e] for e in edge_loads), default=0.0)
        binding = sum(abs(edge_loads[e] - instance.capacities[e]) <= 1e-7 * max(1.0, instance.capacities[e]) for e in edge_loads)
        total_demand = sum(instance.demands[p] for p in rows["flows"])
        return {
            "backend": f"AMPL_{solver_name.upper()}",
            "status": "MEASURED",
            **instance.hashes(),
            **estimate_lp(instance, "L4"),
            "num_vars": len(rows["variables"]),
            "num_constraints": len(rows["flows"]) + len(rows["edges"]),
            "num_nonzeros": rows["num_nonzeros"],
            "objective": objective,
            "max_demand_violation": max(0.0, max_demand_violation),
            "max_capacity_violation": max(0.0, max_capacity_violation),
            "unsatisfied_demand": total_demand - objective,
            "binding_constraints": binding,
            "binding_constraint_ratio": binding / len(edge_loads) if edge_loads else 0.0,
            "T_ampl_model_definition": definition_s,
            "T_ampl_data_load": data_load_s,
            "T_ampl_to_solver_wall": solve_wall_s,
            "T_solution_extract": solution_extract_s,
            "T_solver_pipeline": definition_s + data_load_s + solve_wall_s + solution_extract_s,
            "T_gurobi_reported_runtime": "GUROBI_INTERNAL_RUNTIME_NOT_AVAILABLE",
            "presolve_post_size": "PRESOLVE_POST_SIZE_NOT_AVAILABLE",
            "iter_count": "NOT_AVAILABLE",
            "thread_mode": thread_mode.value,
            "threads": {ThreadMode.DEFAULT: "DEFAULT", ThreadMode.THREADS_1: 1, ThreadMode.THREADS_24: 24}[thread_mode],
        }
    except Exception as exc:
        # 只保留异常类型，AMPL 原始文本可能包含许可证内容。
        return {"backend": f"AMPL_{solver_name.upper()}", "status": "FAILED", "failure_type": type(exc).__name__}
    finally:
        if ampl is not None:
            try:
                ampl.close()
            except Exception:
                pass


def solve(instance: BenchmarkInstance, backend: SolverBackend) -> dict[str, Any]:
    """统一入口，确保 direct 后端仍使用既有实现。"""

    if backend is SolverBackend.DIRECT_GUROBIPY:
        result = solve_level(instance, "L4", SafetyLimits(20_000, 20_000, 2.0))
        result["backend"] = backend.value
        return result
    if backend is SolverBackend.AMPL_GUROBI:
        return solve_ampl_gurobi(instance)
    raise ValueError(f"unsupported backend: {backend}")


def parity_check(direct: Mapping[str, Any], ampl: Mapping[str, Any], tolerance: float = 1e-8) -> dict[str, Any]:
    """比较 formulation/objective/feasibility；失败时禁止进入 full scale。"""

    required = ("objective", "num_vars", "num_constraints", "num_nonzeros")
    if direct.get("status") != "MEASURED" or ampl.get("status") != "MEASURED":
        return {"status": "AMPL_FORMULATION_PARITY_FAIL", "reason": "BACKEND_NOT_MEASURED"}
    if any(key not in direct or key not in ampl for key in required):
        return {"status": "AMPL_FORMULATION_PARITY_FAIL", "reason": "MISSING_METRIC"}
    scale = max(1.0, abs(float(direct["objective"])))
    relative_objective_difference = abs(float(direct["objective"]) - float(ampl["objective"])) / scale
    sizes_equal = all(direct[key] == ampl[key] for key in ("num_vars", "num_constraints", "num_nonzeros"))
    feasible = all(
        max(0.0, float(result.get(metric, float("inf")))) <= tolerance
        for result in (direct, ampl)
        for metric in ("max_capacity_violation", "max_demand_violation")
    )
    passed = relative_objective_difference <= tolerance and sizes_equal and feasible
    return {
        "status": "AMPL_FORMULATION_PARITY_PASS" if passed else "AMPL_FORMULATION_PARITY_FAIL",
        "relative_objective_difference": relative_objective_difference,
        "sizes_equal": sizes_equal,
        "feasible": feasible,
        "tolerance": tolerance,
    }
