"""按硬门槛顺序运行第三轮 AMPL official-workload 验证。"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import pickle
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

from .ampl_environment import (
    OUTPUT_ROOT, activate_from_environment, collect_environment, license_env_present,
    smoke_test_sizes, write_environment,
)
from .ampl_solver import (
    CODE_PUBLIC_CONFIG, PAPER_LIKE_CONFIG, SolverBackend, ThreadMode,
    parity_check, solve, solve_ampl_gurobi,
)
from .benchmark_large_scale import constellation_instance
from .official_workload import (
    SnapshotProvenance, build_inventory, build_official_instance, default_search_roots,
    distribution, inspect_download_archive, load_snapshot, scan_flowsets,
    select_snapshots, write_json,
)


CACHE_ROOT = OUTPUT_ROOT / "cache"
SOLVE_TIMEOUT_S = 600
ROUND2 = {"active_sd": 180, "num_vars": 1800, "num_nonzeros": 18793,
          "gurobi_s": 0.013284, "sate_forward_s": 0.018238, "sate_e2e_s": 0.109980}


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(row), sort_keys=True) + "\n")


def append_status(stage: int, status: str, detail: str) -> None:
    """每个 stage 增量持久化，崩溃时仍保留已完成证据。"""

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_ROOT / "stage_status.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"stage": stage, "status": status, "detail": detail}, sort_keys=True) + "\n")


def run_stage0(run_smoke: bool) -> bool:
    append_status(0, "STARTED", "AMPL install/activate/smoke")
    activation = activate_from_environment() if license_env_present() else {
        "status": "PREACTIVATED_PROBE", "return_code": None,
    }
    environment = collect_environment(run_smoke=False)
    environment["AMPL_LICENSE_ENV_PRESENT"] = license_env_present()
    environment["activation_success"] = activation["status"] == "ACTIVATED"
    environment["activation"] = activation
    if run_smoke:
        environment["size_smoke_tests"] = smoke_test_sizes()
        environment["AMPL_Gurobi_solve_success"] = all(row["status"] == "COMPLETED" for row in environment["size_smoke_tests"])
    write_environment(environment)
    if not run_smoke or not environment.get("AMPL_Gurobi_solve_success"):
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
        ampl = solve_ampl_gurobi(instance, ThreadMode.THREADS_1)
        rows.append({
            "nodes": nodes, "active_sd": active_sd, "direct": direct, "ampl": ampl,
            "gate": parity_check(direct, ampl),
        })
    passed = all(row["gate"]["status"] == "AMPL_FORMULATION_PARITY_PASS" for row in rows)
    result = {"status": "AMPL_FORMULATION_PARITY_PASS" if passed else "AMPL_FORMULATION_PARITY_FAIL", "rows": rows}
    write_json(result, OUTPUT_ROOT / "ampl_parity.json")
    append_status(1, "COMPLETED" if passed else "FAILED", result["status"])
    return passed


def run_stages2_and_3(repository: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """先 inventory，再仅扫描 FlowSet 并计算分布；本函数绝不生成路径或求解。"""

    append_status(2, "STARTED", "official dataset inventory and FlowSet scan")
    inventory = build_inventory(default_search_roots(repository))
    write_json(inventory, OUTPUT_ROOT / "dataset_inventory.json")
    if not inventory["BENCHMARK_PRIMARY_READY"]:
        append_status(2, "FAILED", "BENCHMARK_INCOMPLETE")
        return inventory, []
    rows = scan_flowsets(inventory)
    csv_path = OUTPUT_ROOT / "official_workload_statistics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    append_status(2, "COMPLETED", f"records={len(rows)}")

    append_status(3, "STARTED", "official workload distributions")
    summaries = {}
    for intensity in sorted({int(row["intensity"]) for row in rows}):
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
    return inventory, rows


def build_child(selection_path: Path, result_path: Path) -> int:
    """Stage 5 worker：在带官方 Starlink 依赖的解释器中构造一个 snapshot。"""

    row = json.loads(selection_path.read_text(encoding="utf-8"))
    read_start = time.perf_counter()
    record = load_snapshot(Path(row["source_path"]), int(row["record_index"]))
    read_s = time.perf_counter() - read_start
    provenance = SnapshotProvenance(
        row["dataset"], row["volume"], row["source_path"], int(row["record_index"]),
        int(row["raw_flow_count"]), int(row["active_sd_pairs"]), float(row["total_demand"]),
    )
    build_start = time.perf_counter()
    official, audit = build_official_instance(record, provenance, PAPER_LIKE_CONFIG, cache_dir=CACHE_ROOT)
    payload = {"selection": row, "instance": official, "audit": audit,
               "T_pickle_read": read_s,
               "T_instance_build": time.perf_counter() - build_start}
    result_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = result_path.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(result_path)
    return 0


def solve_child(instance_path: Path, result_path: Path, thread_mode: ThreadMode) -> int:
    """Stage 6/7 worker：一次进程只求解一个 full-scale LP。"""

    with instance_path.open("rb") as handle:
        payload = pickle.load(handle)
    official = payload["instance"]
    start = time.perf_counter()
    result = solve_ampl_gurobi(official.benchmark, thread_mode)
    result.update({
        "selection": payload["selection"],
        "capacity_configuration": "CODE_PUBLIC_CONFIG" if "CODE_PUBLIC" in instance_path.stem else "PAPER_LIKE_CONFIG",
        "actual_unique_k10_vars": sum(len(paths) for paths in official.benchmark.candidate_paths.values()),
        "unique_path_completion_rate": official.unique_path_completion_rate,
        "active_edges": len(official.benchmark.physical_edges),
        "access_edges": sum(edge[0] >= official.user_node_floor or edge[1] >= official.user_node_floor for edge in official.benchmark.physical_edges),
        "network_edges": sum(edge[0] < official.user_node_floor and edge[1] < official.user_node_floor for edge in official.benchmark.physical_edges),
        "mean_path_hops": official.mean_path_hops, "p95_path_hops": official.p95_path_hops,
        "T_full_pipeline": time.perf_counter() - start + payload["T_pickle_read"] + payload["T_instance_build"],
    })
    write_json(result, result_path)
    return 0 if result.get("status") == "MEASURED" else 1


def run_solve_process(instance_path: Path, result_path: Path, thread_mode: ThreadMode,
                      timeout_s: int = SOLVE_TIMEOUT_S) -> dict[str, Any]:
    """父进程执行 600 秒 timeout 并增量保留 STARTED/终态。"""

    started = {"instance": str(instance_path), "thread_mode": thread_mode.value, "status": "STARTED"}
    append_jsonl(OUTPUT_ROOT / "solve_status.jsonl", started)
    try:
        completed = subprocess.run(
            [sys.executable, "-m", "experiments.speedup_attribution.run_ampl_full_scale",
             "--child-solve", str(instance_path), str(result_path), "--thread-mode", thread_mode.value],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=timeout_s, check=False,
        )
        result = (json.loads(result_path.read_text(encoding="utf-8")) if result_path.is_file()
                  else {**started, "status": "FAILED", "failure_type": f"CHILD_EXIT_{completed.returncode}"})
    except subprocess.TimeoutExpired:
        result = {**started, "status": "TIMEOUT", "timeout_s": timeout_s}
    append_jsonl(OUTPUT_ROOT / "solve_status.jsonl", result)
    return result


def run_stage4(rows: Sequence[Mapping[str, Any]], resume: bool) -> list[dict[str, Any]]:
    append_status(4, "STARTED", "deterministic snapshot selection")
    path = OUTPUT_ROOT / "selected_snapshots.json"
    selected = json.loads(path.read_text(encoding="utf-8")) if resume and path.is_file() else select_snapshots(rows)
    write_json(selected, path)
    append_status(4, "COMPLETED", f"snapshots={len(selected)}")
    return selected


def run_stage5(selected: Sequence[Mapping[str, Any]], resume: bool, sate_python: Path) -> list[Path]:
    append_status(5, "STARTED", "K10 paths/cache")
    order = ("median", "P75", "P90", "P95", "P99", "maximum", "P10", "minimum")
    priority = lambda row: order.index(row["selection_label"]) if row["selection_label"] in order else len(order)
    results = []
    for row in sorted(selected, key=priority):
        key = f"{row['volume']}-{row['record_index']}-{row['selection_label']}"
        result_path = OUTPUT_ROOT / "instances" / f"{key}.pkl"
        results.append(result_path)
        if resume and result_path.is_file():
            continue
        job = OUTPUT_ROOT / "jobs" / f"{key}.json"
        write_json(dict(row), job)
        append_status(5, "STARTED", key)
        try:
            completed = subprocess.run(
                [str(sate_python), "-m", "experiments.speedup_attribution.run_ampl_full_scale",
                 "--child-build", str(job), str(result_path)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=SOLVE_TIMEOUT_S, check=False,
            )
            status = "COMPLETED" if completed.returncode == 0 and result_path.is_file() else "FAILED"
        except subprocess.TimeoutExpired:
            status = "TIMEOUT"
        append_status(5, status, key)
    append_status(5, "COMPLETED", f"instances={sum(path.is_file() for path in results)}")
    return [path for path in results if path.is_file()]


def run_stage6(instances: Sequence[Path], resume: bool) -> list[dict[str, Any]]:
    append_status(6, "STARTED", "full-scale Gurobi default")
    rows = []
    for instance in instances:
        result_path = OUTPUT_ROOT / "solves" / f"{instance.stem}-{ThreadMode.DEFAULT.value}.json"
        rows.append(json.loads(result_path.read_text(encoding="utf-8")) if resume and result_path.is_file()
                    else run_solve_process(instance, result_path, ThreadMode.DEFAULT))
    write_json(rows, OUTPUT_ROOT / "full_scale_gurobi.json")
    append_status(6, "COMPLETED", f"measured={sum(row.get('status') == 'MEASURED' for row in rows)}")
    return rows


def run_stage7(instances: Sequence[Path], resume: bool) -> list[dict[str, Any]]:
    append_status(7, "STARTED", "capacity/thread sensitivity")
    rows = []
    # Capacity/thread sensitivity 使用预注册 median；P90/P95 保留给主 default 曲线。
    for instance in [path for path in instances if "median.pkl" in path.name]:
        with instance.open("rb") as handle:
            payload = pickle.load(handle)
        paper = payload["instance"]
        code = paper.with_capacity(CODE_PUBLIC_CONFIG)
        if code.hashes() != paper.hashes():
            raise AssertionError("capacity sensitivity changed topology/demand/path")
        code_path = instance.with_name(instance.stem + "-CODE_PUBLIC.pkl")
        if not code_path.is_file():
            with code_path.open("wb") as handle:
                pickle.dump({**payload, "instance": code}, handle, protocol=pickle.HIGHEST_PROTOCOL)
        for job_path, mode in ((instance, ThreadMode.THREADS_1), (code_path, ThreadMode.DEFAULT)):
            result_path = OUTPUT_ROOT / "solves" / f"{job_path.stem}-{mode.value}.json"
            rows.append(json.loads(result_path.read_text(encoding="utf-8")) if resume and result_path.is_file()
                        else run_solve_process(job_path, result_path, mode))
    write_json(rows, OUTPUT_ROOT / "sensitivity.json")
    append_status(7, "COMPLETED", f"measured={sum(row.get('status') == 'MEASURED' for row in rows)}")
    return rows


def sate_child(instance_path: Path, result_path: Path, checkpoint: Path | None) -> int:
    from .benchmark_k10 import benchmark_instance

    with instance_path.open("rb") as handle:
        payload = pickle.load(handle)
    official = payload["instance"]
    result = benchmark_instance(
        official.benchmark, 4236, 10, checkpoint, "PUBLIC_CODE_K10_DIAGNOSTIC_MODEL", 20, 30,
        network_capacity=200.0, access_capacity=50.0,
    )
    result.update(selection=payload["selection"], official_instance_hashes=official.benchmark.hashes())
    write_json(result, result_path)
    return 0 if result.get("status") == "MEASURED" else 1


def run_stage8(instances: Sequence[Path], resume: bool, sate_python: Path,
               checkpoint: Path | None) -> list[dict[str, Any]]:
    append_status(8, "STARTED", "official SaTE timing")
    rows = []
    for instance in [path for path in instances if any(x in path.stem for x in ("median", "P90", "P95"))]:
        result_path = OUTPUT_ROOT / "sate" / f"{instance.stem}.json"
        if not (resume and result_path.is_file()):
            result_path.parent.mkdir(parents=True, exist_ok=True)
            command = [str(sate_python), "-m", "experiments.speedup_attribution.run_ampl_full_scale",
                       "--child-sate", str(instance), str(result_path)]
            if checkpoint:
                command.extend(("--checkpoint", str(checkpoint)))
            subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                           timeout=SOLVE_TIMEOUT_S, check=False)
        rows.append(json.loads(result_path.read_text(encoding="utf-8")) if result_path.is_file()
                    else {"status": "FAILED", "instance": str(instance)})
    write_json(rows, OUTPUT_ROOT / "official_sate.json")
    append_status(8, "COMPLETED", f"measured={sum(row.get('status') == 'MEASURED' for row in rows)}")
    return rows


def classify_runtime(seconds: float | None) -> str:
    if seconds is None:
        return "NOT_EVALUATED"
    if seconds < 0.1:
        return "OFFICIAL_WORKLOAD_STILL_SMALL"
    if seconds < 1.0:
        return "SUBSECOND_GROWTH"
    if seconds < 10.0:
        return "OFFICIAL_WORKLOAD_EXPLAINS_SOLVER_GROWTH"
    if seconds < 30.0:
        return "STRONG_PAPER_SCALE_GROWTH"
    if 40.0 <= seconds <= 55.0:
        return "PAPER_GUROBI_46S_APPROX_REPRODUCED"
    # 30 秒以上已经进入论文同一几十秒量级；仅 40--55 秒使用更强的近似复现标签。
    return "PAPER_GUROBI_LATENCY_ORDER_REPRODUCED"


def run_stage9(gurobi: Sequence[Mapping[str, Any]], sate: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    append_status(9, "STARTED", "scaling/crossover")
    measured_g = {row.get("selection", {}).get("selection_label"): row for row in gurobi if row.get("status") == "MEASURED"}
    measured_s = {row.get("selection", {}).get("selection_label"): row for row in sate if row.get("status") == "MEASURED"}
    comparisons = []
    for label in sorted(set(measured_g) & set(measured_s)):
        g, s = measured_g[label], measured_s[label]
        solver_s = float(g["T_ampl_to_solver_wall"])
        forward_s = float(s["t_forward"]["median_s"])
        e2e_s = float(s["t_e2e"]["median_s"])
        comparisons.append({"selection_label": label, "forward_speedup_ratio": solver_s / forward_s,
                            "system_e2e_speedup_ratio": solver_s / e2e_s,
                            "forward_crossover": forward_s < solver_s, "e2e_crossover": e2e_s < solver_s})
    median = measured_g.get("median")
    result = {
        "round2_reference": ROUND2,
        "median_runtime_classification": classify_runtime(float(median["T_ampl_to_solver_wall"]) if median else None),
        "comparisons": comparisons,
        "cold_crossover": any(row["e2e_crossover"] for row in comparisons),
        "forward_only_crossover": any(row["forward_crossover"] and not row["e2e_crossover"] for row in comparisons),
    }
    write_json(result, OUTPUT_ROOT / "scaling_crossover.json")
    append_status(9, "COMPLETED", result["median_runtime_classification"])
    return result


def _plot(path: Path, title: str, x: Sequence[float], y: Sequence[float] | None = None,
          xlabel: str = "", ylabel: str = "") -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    if x:
        ax.hist(x, bins=40) if y is None else ax.scatter(x, y, s=20)
    else:
        ax.text(0.5, 0.5, "NOT_EVALUATED", ha="center", va="center", transform=ax.transAxes)
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run_stage10(stats: Sequence[Mapping[str, Any]], gurobi: Sequence[Mapping[str, Any]],
                sensitivity: Sequence[Mapping[str, Any]], sate: Sequence[Mapping[str, Any]]) -> None:
    append_status(10, "STARTED", "reports/plots")
    root = OUTPUT_ROOT / "figures"
    root.mkdir(parents=True, exist_ok=True)
    selected = [row for row in stats if int(row["intensity"]) == 100]
    active, raw = [float(row["active_sd_pairs"]) for row in selected], [float(row["raw_flow_count"]) for row in selected]
    measured = [row for row in gurobi if row.get("status") == "MEASURED"]
    runtime = [float(row["T_ampl_to_solver_wall"]) for row in measured]
    _plot(root / "01_active_sd_distribution.png", "Official active SD distribution", active, xlabel="active SD")
    _plot(root / "02_raw_flows_vs_active_sd.png", "Raw flows vs active SD", raw, active, "raw flows", "active SD")
    _plot(root / "03_k10_vars_distribution.png", "Nominal K10 variables", [10 * x for x in active], xlabel="variables")
    _plot(root / "04_runtime_vs_active_sd.png", "Gurobi runtime vs active SD", [float(r["selection"]["active_sd_pairs"]) for r in measured], runtime, "active SD", "seconds")
    _plot(root / "05_runtime_vs_vars.png", "Gurobi runtime vs variables", [float(r["num_vars"]) for r in measured], runtime, "variables", "seconds")
    _plot(root / "06_runtime_vs_nz.png", "Gurobi runtime vs NZ", [float(r["num_nonzeros"]) for r in measured], runtime, "NZ", "seconds")
    _plot(root / "07_round2_to_round3.png", "Round2 to Round3", [ROUND2["active_sd"]] + [float(r["selection"]["active_sd_pairs"]) for r in measured], [ROUND2["gurobi_s"]] + runtime, "active SD", "seconds")
    sate_rows = [row for row in sate if row.get("status") == "MEASURED"]
    _plot(root / "08_gurobi_vs_sate.png", "Gurobi vs SaTE", list(range(len(sate_rows))), [float(r["t_e2e"]["median_s"]) for r in sate_rows], "snapshot", "seconds")
    cap_rows = [row for row in sensitivity if row.get("status") == "MEASURED"]
    _plot(root / "09_capacity_50_vs_800.png", "Capacity 50 vs 800", list(range(len(cap_rows))), [float(r["objective"]) for r in cap_rows], "solve", "objective")
    _plot(root / "10_paper_reference.png", "Paper 46-47s / 17ms references", [0.017, 46.0, 47.0], xlabel="seconds")
    append_status(10, "COMPLETED", "figures=10")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-smoke", action="store_true")
    parser.add_argument("--repository", type=Path, default=Path("."))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--sate-python", type=Path, default=Path("/home/me94994/anaconda3/envs/satellite-te/bin/python"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--child-build", nargs=2)
    parser.add_argument("--child-solve", nargs=2)
    parser.add_argument("--child-sate", nargs=2)
    parser.add_argument("--thread-mode", choices=[item.value for item in ThreadMode], default=ThreadMode.DEFAULT.value)
    args = parser.parse_args()
    if args.child_build:
        raise SystemExit(build_child(Path(args.child_build[0]), Path(args.child_build[1])))
    if args.child_solve:
        raise SystemExit(solve_child(Path(args.child_solve[0]), Path(args.child_solve[1]), ThreadMode(args.thread_mode)))
    if args.child_sate:
        raise SystemExit(sate_child(Path(args.child_sate[0]), Path(args.child_sate[1]), args.checkpoint))
    archive = args.repository / "input/raw/starlink/_official_download/DataSetForSaTE100.zip"
    if archive.is_file():
        write_json(inspect_download_archive(archive), OUTPUT_ROOT / "download_inventory.json")
    if not run_stage0(args.run_smoke):
        raise SystemExit(2)
    if not run_stage1():
        raise SystemExit(3)
    inventory, rows = run_stages2_and_3(args.repository)
    if not inventory.get("BENCHMARK_PRIMARY_READY"):
        raise SystemExit(4)
    selected = run_stage4(rows, args.resume)
    instances = run_stage5(selected, args.resume, args.sate_python)
    gurobi = run_stage6(instances, args.resume)
    sensitivity = run_stage7(instances, args.resume)
    sate = run_stage8(instances, args.resume, args.sate_python, args.checkpoint)
    run_stage9(gurobi, sate)
    run_stage10(rows, gurobi, sensitivity, sate)


if __name__ == "__main__":
    main()
