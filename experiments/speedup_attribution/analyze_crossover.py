"""Analyze measured crossover data while preserving evidence-class boundaries."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import random
import statistics
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


def load_jsonl(path: Path) -> List[Dict[str, object]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _median_rows(rows: Iterable[Mapping[str, object]], value: str, keys: Sequence[str]) -> List[Dict[str, object]]:
    groups = {}
    for row in rows:
        if row.get("status") != "MEASURED" or not isinstance(row.get(value), (int, float)) or row[value] <= 0:
            continue
        key = tuple(row.get(name) for name in keys)
        groups.setdefault(key, []).append(float(row[value]))
    return [{**dict(zip(keys, key)), value: statistics.median(values)} for key, values in groups.items()]


def scaling_fit(rows: Sequence[Mapping[str, object]], x_key: str, y_key: str, bootstrap_samples: int = 1000) -> Dict[str, object]:
    """Fit measured log-log points and bootstrap the slope without adding fake points."""
    points = [(float(row[x_key]), float(row[y_key])) for row in rows if isinstance(row.get(x_key), (int, float)) and isinstance(row.get(y_key), (int, float)) and row[x_key] > 0 and row[y_key] > 0]
    if len(points) < 3 or len({x for x, _ in points}) < 3:
        return {"status": "NOT_EVALUATED_INSUFFICIENT_MEASURED_POINTS", "samples": len(points)}
    x = np.log([point[0] for point in points])
    y = np.log([point[1] for point in points])
    slope, intercept = np.polyfit(x, y, 1)
    rng = random.Random(20260908)
    boot = []
    for _ in range(bootstrap_samples):
        sample = rng.choices(points, k=len(points))
        if len({item[0] for item in sample}) < 2:
            continue
        sx = np.log([item[0] for item in sample])
        sy = np.log([item[1] for item in sample])
        boot.append(float(np.polyfit(sx, sy, 1)[0]))
    boot.sort()
    pick = lambda q: boot[min(len(boot) - 1, int(q * (len(boot) - 1)))]
    return {
        "status": "MEASURED_FIT",
        "samples": len(points),
        "slope_b": float(slope),
        "intercept_a": float(intercept),
        "bootstrap_slope_95ci": [pick(0.025), pick(0.975)],
    }


def detect_measured_crossover(solver_rows: Sequence[Mapping[str, object]], sate_rows: Sequence[Mapping[str, object]], solver_time_key: str = "total_s") -> Dict[str, object]:
    """Return only a point jointly measured at the same scale; fits cannot create crossover."""
    solver = _median_rows(solver_rows, solver_time_key, ("topology_nodes", "actual_active_flows", "regime"))
    candidates = []
    for sate in sate_rows:
        if sate.get("status") != "MEASURED" or sate.get("k") != 10:
            continue
        e2e = sate.get("t_e2e", {}).get("median_s")
        if not isinstance(e2e, (int, float)):
            continue
        for gurobi in solver:
            if gurobi["topology_nodes"] == sate.get("topology_nodes") and gurobi["actual_active_flows"] == sate.get("active_flows") and gurobi["regime"] == sate.get("regime"):
                candidates.append({"nodes": sate["topology_nodes"], "active_flows": sate["active_flows"], "regime": sate["regime"], "gurobi_s": gurobi[solver_time_key], "sate_e2e_s": e2e})
    crossing = sorted((row for row in candidates if row["sate_e2e_s"] < row["gurobi_s"]), key=lambda row: (row["nodes"], row["active_flows"]))
    return {"status": f"OBSERVED_AT_N={crossing[0]['nodes']}" if crossing else "NOT_OBSERVED", "first_measured": crossing[0] if crossing else None, "matched_measured_points": candidates}


def _plot_lines(path: Path, title: str, xlabel: str, ylabel: str, series: Mapping[str, Sequence[Tuple[float, float]]], log_y: bool = False) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for label, points in series.items():
        if points:
            ordered = sorted(points)
            ax.plot([x for x, _ in ordered], [y for _, y in ordered], marker="o", label=label)
    ax.set_title(title); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.grid(alpha=0.25)
    if log_y: ax.set_yscale("log")
    if series: ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def make_figures(root: Path, solver_rows: Sequence[Mapping[str, object]], sate_rows: Sequence[Mapping[str, object]], pairs: Sequence[Mapping[str, str]], online_rows: Sequence[Mapping[str, object]]) -> None:
    figures = root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    cold = [row for row in solver_rows if row.get("baseline") == "sparse_presolve_cold" and row.get("status") == "MEASURED"]
    def med_points(filtered, xkey, ykey):
        groups = {}
        for row in filtered:
            groups.setdefault(float(row[xkey]), []).append(float(row[ykey]))
        return [(x, statistics.median(values)) for x, values in groups.items()]
    _plot_lines(figures / "figure1_gurobi_total_vs_nodes.png", "Measured Gurobi total runtime", "topology nodes", "seconds", {f"K={k}": med_points([r for r in cold if r.get("k") == k and r.get("family") == "coupled" and r.get("regime") == "congested"], "topology_nodes", "total_s") for k in (5, 10)}, True)
    _plot_lines(figures / "figure2_gurobi_optimize_vs_flows.png", "Measured Gurobi optimize runtime", "active flows", "seconds", {f"K={k}": med_points([r for r in cold if r.get("k") == k and r.get("family") == "traffic" and r.get("regime") == "congested"], "actual_active_flows", "optimize_wall_s") for k in (5, 10)}, True)
    _plot_lines(figures / "figure3_gurobi_vs_vars.png", "Measured Gurobi runtime vs variables", "variables", "seconds", {"K=10": med_points([r for r in cold if r.get("k") == 10], "num_vars", "total_s")}, True)
    _plot_lines(figures / "figure4_gurobi_vs_nonzeros.png", "Measured Gurobi runtime vs nonzeros", "nonzeros", "seconds", {"K=10": med_points([r for r in cold if r.get("k") == 10], "num_nonzeros", "total_s")}, True)
    _plot_lines(figures / "figure5_presolved_vars.png", "Presolved variables", "topology nodes", "count", {regime: med_points([r for r in cold if r.get("k") == 10 and r.get("family") == "coupled" and r.get("regime") == regime and isinstance(r.get("presolved_vars"), (int, float))], "topology_nodes", "presolved_vars") for regime in ("uncongested", "congested")})
    _plot_lines(figures / "figure6_regime_scaling.png", "Uncongested vs controlled stress", "topology nodes", "seconds", {regime: med_points([r for r in cold if r.get("k") == 10 and r.get("family") == "coupled" and r.get("regime") == regime], "topology_nodes", "total_s") for regime in ("uncongested", "congested")}, True)
    solver_best = [r for r in solver_rows if r.get("baseline") == "best_cold_commercial" and r.get("status") == "MEASURED" and r.get("family") == "coupled" and r.get("regime") == "congested"]
    sate_k10 = [r for r in sate_rows if r.get("status") == "MEASURED" and r.get("k") == 10 and r.get("regime") == "congested"]
    _plot_lines(figures / "figure7_cold_vs_sate.png", "Measured cold solver and SaTE latency", "topology nodes", "seconds", {"Gurobi cold": med_points(solver_best, "topology_nodes", "total_s"), "SaTE forward": [(r["topology_nodes"], r["t_forward"]["median_s"]) for r in sate_k10], "SaTE e2e": [(r["topology_nodes"], r["t_e2e"]["median_s"]) for r in sate_k10]}, True)
    online_points = []
    for row in online_rows:
        if row.get("status") == "MEASURED_CONTROLLED_SEQUENTIAL_DIAGNOSTIC":
            online_points.append((row["topology_nodes"], row["methods"]["previous_basis_reuse"]["median_total_s"]))
    _plot_lines(figures / "figure8_online_vs_sate.png", "Measured online solver and SaTE e2e", "topology nodes", "seconds", {"Gurobi online": online_points, "SaTE e2e": [(r["topology_nodes"], r["t_e2e"]["median_s"]) for r in sate_k10]}, True)
    _plot_lines(figures / "figure9_k10_slowdown.png", "Paired K=10 / K=5 slowdown", "active flows", "ratio", {"total": [(float(r["active_flows"]), float(r["total_slowdown"])) for r in pairs if r["regime"] == "congested" and r["family"] == "traffic"]})
    quality = [r for r in sate_k10 if r.get("quality", {}).get("status") == "MEASURED_DIAGNOSTIC"]
    _plot_lines(figures / "figure10_latency_quality.png", "Measured diagnostic latency-quality tradeoff", "latency seconds", "optimality gap", {"SaTE K=10": [(r["t_e2e"]["median_s"], r["quality"]["optimality_gap"]) for r in quality]})
    _plot_lines(figures / "figure11_measured_crossover.png", "Measured crossover only", "topology nodes", "seconds", {"Gurobi cold MEASURED": med_points(solver_best, "topology_nodes", "total_s"), "SaTE e2e MEASURED": [(r["topology_nodes"], r["t_e2e"]["median_s"]) for r in sate_k10]}, True)
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.scatter([4236], [46.5], facecolors="none", edgecolors="tab:red", marker="o", label="Gurobi PAPER_REPORTED_REFERENCE")
    ax.scatter([4236], [0.017], facecolors="none", edgecolors="tab:blue", marker="s", label="SaTE PAPER_REPORTED_REFERENCE")
    ax.set_yscale("log"); ax.set_xlabel("satellites"); ax.set_ylabel("seconds"); ax.set_title("Paper reference points (not measured here)"); ax.grid(alpha=0.25); ax.legend(); fig.tight_layout(); fig.savefig(figures / "figure12_paper_reference.png", dpi=180); plt.close(fig)


def analyze(root: Path) -> Dict[str, object]:
    solver = load_jsonl(root / "solver_scaling.jsonl")
    sate = load_jsonl(root / "sate_scaling.jsonl")
    online = load_jsonl(root / "solver_online.jsonl")
    pairs = list(csv.DictReader((root / "k5_k10_pairs.csv").open(encoding="utf-8"))) if (root / "k5_k10_pairs.csv").is_file() else []
    cold_k10 = [r for r in solver if r.get("status") == "MEASURED" and r.get("k") == 10 and r.get("baseline") == "best_cold_commercial" and r.get("family") == "coupled" and r.get("regime") == "congested"]
    cold_medians = _median_rows(cold_k10, "total_s", ("topology_nodes", "actual_active_flows", "num_vars", "num_nonzeros"))
    optimize_medians = _median_rows(cold_k10, "optimize_wall_s", ("topology_nodes", "actual_active_flows", "num_vars", "num_nonzeros"))
    sate_k10 = [r for r in sate if r.get("status") == "MEASURED" and r.get("k") == 10 and r.get("regime") == "congested"]
    sate_fit_rows = [{"num_vars": r["active_flows"] * 10, "num_nonzeros": r["graph"]["num_hetero_edges"], "forward_s": r["t_forward"]["median_s"], "e2e_s": r["t_e2e"]["median_s"]} for r in sate_k10]
    fits = {
        "gurobi_optimize_vs_vars": scaling_fit(optimize_medians, "num_vars", "optimize_wall_s"),
        "gurobi_total_vs_vars": scaling_fit(cold_medians, "num_vars", "total_s"),
        "gurobi_optimize_vs_nonzeros": scaling_fit(optimize_medians, "num_nonzeros", "optimize_wall_s"),
        "gurobi_total_vs_nonzeros": scaling_fit(cold_medians, "num_nonzeros", "total_s"),
        "sate_forward_vs_path_nodes": scaling_fit(sate_fit_rows, "num_vars", "forward_s"),
        "sate_forward_vs_hetero_edges": scaling_fit(sate_fit_rows, "num_nonzeros", "forward_s"),
        "sate_e2e_vs_hetero_edges": scaling_fit(sate_fit_rows, "num_nonzeros", "e2e_s"),
    }
    cold_crossover = detect_measured_crossover(cold_k10, sate_k10)
    online_flat = [
        {
            "status": "MEASURED",
            "topology_nodes": row["topology_nodes"],
            "actual_active_flows": row["active_flows"],
            "regime": "congested",
            "total_s": row["methods"]["previous_basis_reuse"]["median_total_s"],
        }
        for row in online
        if row.get("status") == "MEASURED_CONTROLLED_SEQUENTIAL_DIAGNOSTIC"
    ]
    online_crossover = detect_measured_crossover(online_flat, sate_k10)
    slowdown = [float(row["total_slowdown"]) for row in pairs]
    measured_solver = [r for r in solver if r.get("status") == "MEASURED" and r.get("k") == 10]
    maximum = max(measured_solver, key=lambda r: (r.get("topology_nodes", 0), r.get("actual_active_flows", 0), r.get("num_vars", 0))) if measured_solver else None
    result = {
        "evidence_boundary": {"measured": "solid markers", "extrapolated": "dashed lines", "paper_reported_reference": "hollow markers"},
        "maximum_measured_solver_problem": None if maximum is None else {key: maximum.get(key) for key in ("topology_nodes", "actual_active_flows", "num_vars", "num_constraints", "num_nonzeros", "total_s", "optimize_wall_s")},
        "k5_k10_total_slowdown_median": statistics.median(slowdown) if slowdown else None,
        "cold_crossover": cold_crossover,
        "online_crossover": online_crossover if online_flat else {"status": "NOT_EVALUATED", "first_measured": None, "matched_measured_points": []},
        "scaling_fits": fits,
        "paper_2738x": "NOT_IDENTIFIABLE",
        "paper_scale_reproduction": "NOT_REPRODUCED",
    }
    (root / "scaling_fit.json").write_text(json.dumps(fits, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (root / "crossover_summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (root / "quality_scaling.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = ["topology_nodes", "active_flows", "k", "regime", "evidence_label", "quality_status", "optimality_gap", "raw_violation", "post_repair_violation", "e2e_median_s"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in sate:
            quality = row.get("quality", {})
            writer.writerow({
                "topology_nodes": row.get("topology_nodes"), "active_flows": row.get("active_flows"), "k": row.get("k"), "regime": row.get("regime"), "evidence_label": row.get("evidence_label"),
                "quality_status": quality.get("status"), "optimality_gap": quality.get("optimality_gap"), "raw_violation": quality.get("max_capacity_violation_before_repair"), "post_repair_violation": quality.get("max_capacity_violation_after_repair"), "e2e_median_s": row.get("t_e2e", {}).get("median_s"),
            })
    make_figures(root, solver, sate, pairs, online)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("output/speedup_attribution/large_scale"))
    args = parser.parse_args()
    result = analyze(args.root)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
