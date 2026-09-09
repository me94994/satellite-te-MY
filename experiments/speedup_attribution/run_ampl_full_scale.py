"""按硬门槛顺序运行第三轮 AMPL official-workload 验证。"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from .ampl_environment import (
    OUTPUT_ROOT, activate_from_environment, collect_environment, license_env_present,
    smoke_test_sizes, write_environment,
)
from .ampl_solver import SolverBackend, parity_check, solve
from .benchmark_large_scale import constellation_instance
from .official_workload import (
    build_inventory, default_search_roots, distribution, scan_flowsets, write_json,
)


def append_status(stage: int, status: str, detail: str) -> None:
    """每个 stage 增量持久化，崩溃时仍保留已完成证据。"""

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_ROOT / "stage_status.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"stage": stage, "status": status, "detail": detail}, sort_keys=True) + "\n")


def run_stage0(run_smoke: bool) -> bool:
    append_status(0, "STARTED", "AMPL install/activate/smoke")
    activation = activate_from_environment()
    environment = collect_environment(run_smoke=False)
    environment["activation"] = activation
    if not license_env_present():
        environment["size_smoke_tests"] = []
        environment["AMPL_Gurobi_solve_success"] = "NOT_EVALUATED"
        write_environment(environment)
        append_status(0, "FAILED", "BLOCKED_AMPL_LICENSE_ENV_NOT_SET")
        return False
    if activation["status"] != "ACTIVATED":
        write_environment(environment)
        append_status(0, "FAILED", "AMPL_LICENSE_ACTIVATION_FAILED")
        return False
    if run_smoke:
        environment["size_smoke_tests"] = smoke_test_sizes()
        environment["AMPL_Gurobi_solve_success"] = all(row["status"] == "COMPLETED" for row in environment["size_smoke_tests"])
    write_environment(environment)
    if not run_smoke or not environment["AMPL_Gurobi_solve_success"]:
        append_status(0, "FAILED", "AMPL_GUROBI_SIZE_LIMIT_VALIDATION_INCOMPLETE")
        return False
    append_status(0, "COMPLETED", "AMPL_GUROBI_FULL_SCALE_AVAILABLE")
    return True


def run_stage1() -> bool:
    """在第二轮四个 K10 点执行 direct-vs-AMPL parity 硬门槛。"""

    append_status(1, "STARTED", "AMPL vs gurobipy parity")
    rows = []
    for nodes, active_sd in ((16, 16), (66, 66), (128, 128), (192, 180)):
        instance = constellation_instance(nodes, active_sd, 10, 42, 1.0)
        direct = solve(instance, SolverBackend.DIRECT_GUROBIPY)
        ampl = solve(instance, SolverBackend.AMPL_GUROBI)
        rows.append({
            "nodes": nodes, "active_sd": active_sd, "direct": direct, "ampl": ampl,
            "gate": parity_check(direct, ampl),
        })
    passed = all(row["gate"]["status"] == "AMPL_FORMULATION_PARITY_PASS" for row in rows)
    result = {"status": "AMPL_FORMULATION_PARITY_PASS" if passed else "AMPL_FORMULATION_PARITY_FAIL", "rows": rows}
    write_json(result, OUTPUT_ROOT / "ampl_parity.json")
    append_status(1, "COMPLETED" if passed else "FAILED", result["status"])
    return passed


def run_stages2_and_3(repository: Path) -> bool:
    """先 inventory，再仅扫描 FlowSet 并计算分布；本函数绝不生成路径或求解。"""

    append_status(2, "STARTED", "official dataset inventory and FlowSet scan")
    inventory = build_inventory(default_search_roots(repository))
    write_json(inventory, OUTPUT_ROOT / "dataset_inventory.json")
    if inventory["status"] != "BENCHMARK_COMPLETE_FOR_MAIN_STARLINK":
        append_status(2, "FAILED", "BENCHMARK_INCOMPLETE")
        return False
    rows = scan_flowsets(inventory)
    csv_path = OUTPUT_ROOT / "official_workload_statistics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    append_status(2, "COMPLETED", f"records={len(rows)}")

    append_status(3, "STARTED", "official workload distributions")
    summaries = {}
    for intensity in (25, 50, 75, 100):
        selected = [row for row in rows if row["intensity"] == intensity]
        summaries[str(intensity)] = {
            metric: distribution([float(row[metric]) for row in selected])
            for metric in ("raw_flow_count", "active_sd_pairs", "total_demand")
        }
        summaries[str(intensity)]["nominal_k10_variables"] = {
            key: 10.0 * value for key, value in summaries[str(intensity)]["active_sd_pairs"].items()
        }
    write_json(summaries, OUTPUT_ROOT / "official_workload_distributions.json")
    append_status(3, "COMPLETED", "official active-SD/K10 distributions")
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-smoke", action="store_true")
    parser.add_argument("--repository", type=Path, default=Path("."))
    args = parser.parse_args()
    if not run_stage0(args.run_smoke):
        raise SystemExit(2)
    if not run_stage1():
        raise SystemExit(3)
    if not run_stages2_and_3(args.repository):
        raise SystemExit(4)
    # Stage 4 需要 official path generation/cache；没有通过 Stage 2/3 前不会启动。
    append_status(4, "FAILED", "NOT_IMPLEMENTED_AFTER_CURRENT_DATA_GATE")
    raise SystemExit(5)


if __name__ == "__main__":
    main()
