"""Generate the nine preregistered attribution figures from measured artifacts."""

from __future__ import annotations

import json
from pathlib import Path
import statistics

import matplotlib.pyplot as plt


def _rows(path):
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def _save(root, name, title, xlabel, ylabel, x, series, log=False):
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for label, values in series.items():
        ax.plot(x, values, marker="o", label=label)
    ax.set_title(title); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    if log: ax.set_yscale("log")
    if len(series) > 1: ax.legend()
    ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(root / name, dpi=180); plt.close(fig)


def main():
    root = Path("output/speedup_attribution")
    summary = json.loads((root / "solver_summary.json").read_text())
    sate = json.loads((root / "sate_results.json").read_text())
    commercial = json.loads((root / "commercial_results.json").read_text())
    levels = ["L0", "L1", "L2", "L3", "L4"]
    _save(root, "figure1_lp_size.png", "LP size across exact simplification stages", "stage", "count", levels,
          {"vars":[summary[x]["num_vars"] for x in levels], "constraints":[summary[x]["num_constraints"] for x in levels], "nonzeros":[summary[x]["num_nonzeros"] for x in levels]}, True)
    _save(root, "figure2_gurobi_timing.png", "Gurobi timing decomposition", "stage", "seconds", levels,
          {"build":[summary[x]["median_build_s"] for x in levels], "optimize":[summary[x]["median_optimize_s"] for x in levels], "total":[summary[x]["median_total_s"] for x in levels]}, True)
    _save(root, "figure3_presolve_size.png", "Dense before/presolved/manual sparse", "model", "count", ["dense", "presolved", "manual sparse"],
          {"vars":[summary["L1"]["num_vars"], summary["L1"]["presolved_vars"], summary["L3"]["num_vars"]], "constraints":[summary["L1"]["num_constraints"], summary["L1"]["presolved_constraints"], summary["L3"]["num_constraints"]]})
    scaling = [x for x in _rows(root / "scaling.jsonl") if x["status"] == "MEASURED" and x["level"] == "L4"]
    _save(root, "figure4_runtime_vs_flows.png", "Solver runtime vs active-flow count", "active flows", "total seconds", [x["num_active_flows"] for x in scaling], {"L4":[x["total_s"] for x in scaling]})
    _save(root, "figure5_runtime_vs_nonzeros.png", "Solver runtime vs LP nonzeros", "nonzeros", "total seconds", [x["num_nonzeros"] for x in scaling], {"L4":[x["total_s"] for x in scaling]})
    _save(root, "figure6_commercial_vs_sate.png", "Shared-instance deployment latency", "method", "seconds", ["cold Gurobi", "online Gurobi", "SaTE forward", "SaTE e2e"],
          {"median":[commercial["best_cold"]["median_total_s"], commercial["best_online"]["median_total_s"], sate["t_model_forward"]["median_s"], sate["t_sate_total"]["median_s"]]}, True)
    _save(root, "figure7_latency_quality.png", "Latency-quality tradeoff", "optimality gap", "seconds", [0.0, sate["quality"]["optimality_gap"]],
          {"shared methods":[commercial["best_cold"]["median_total_s"], sate["t_sate_total"]["median_s"]]})
    per_label = commercial["best_cold"]["median_total_s"]
    _save(root, "figure8_offline_topology_work.png", "Estimated offline label workload", "training labels", "aggregate solver seconds", [512, 8000], {"diagnostic estimate":[512*per_label, 8000*per_label]})
    local = [summary["L0"]["median_total_s"]/summary["L1"]["median_total_s"], summary["L1"]["median_total_s"]/summary["L3"]["median_total_s"], summary["L3"]["median_total_s"]/summary["L4"]["median_total_s"], commercial["best_cold"]["median_total_s"]/commercial["best_online"]["median_total_s"], commercial["best_online"]["median_total_s"]/sate["t_sate_total"]["median_s"]]
    _save(root, "figure9_stage_local_attribution.png", "Stage-local gains (not multiplicative)", "transition", "local ratio", ["presolve", "manual sparse", "subgraph", "reuse", "neural replacement"], {"local ratio":local}, True)
    print(json.dumps({"figures": 9, "output": str(root)}, sort_keys=True))


if __name__ == "__main__":
    main()
