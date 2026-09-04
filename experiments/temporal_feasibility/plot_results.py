"""Generate dependency-light figures from temporal feasibility CSV outputs."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _floats(rows: list[dict[str, str]], field: str) -> list[float]:
    values = []
    for row in rows:
        try:
            value = float(row[field])
        except (KeyError, TypeError, ValueError):
            continue
        if np.isfinite(value):
            values.append(value)
    return values


def plot_adjacent_vs_random(records: list[dict[str, str]], figures: Path) -> None:
    """Create boxplots for the primary optimal-state distances."""

    pair_types = [
        "adjacent", "random_unrestricted", "random_demand_matched",
        "random_topology_matched", "random_demand_topology_matched",
    ]
    metrics = {
        "path_flow_normalized_l1": "path_flow",
        "edge_load_persistent_normalized_l1": "edge_load",
        "utilization_normalized_l1": "utilization",
        "binding_edge_jaccard_distance": "binding_set",
        "dual_persistent_normalized_l1": "dual",
    }
    for field, label in metrics.items():
        grouped = [_floats([row for row in records if row.get("pair_type") == pair_type], field) for pair_type in pair_types]
        if not any(grouped):
            continue
        fig, axis = plt.subplots(figsize=(9, 4.5))
        axis.boxplot(grouped, tick_labels=["adjacent", "random", "demand", "topology", "joint"], showfliers=False)
        axis.set_ylabel("normalized L1 distance")
        axis.set_title(f"Adjacent vs random control: {label}")
        fig.tight_layout()
        fig.savefig(figures / f"adjacent_vs_random_{label}.png", dpi=160)
        plt.close(fig)


def plot_lag(records: list[dict[str, str]], figures: Path) -> None:
    """Plot median drift at pre-registered lags."""

    metrics = {
        "path_flow_normalized_l1": "path_flow",
        "edge_load_persistent_normalized_l1": "edge_load",
        "utilization_normalized_l1": "utilization",
        "dual_persistent_normalized_l1": "dual",
    }
    for field, label in metrics.items():
        grouped: dict[int, list[float]] = defaultdict(list)
        for row in records:
            if row.get("pair_type") == "adjacent" or row.get("pair_type", "").startswith("lag_"):
                try:
                    grouped[int(row["lag"])].append(float(row[field]))
                except (KeyError, TypeError, ValueError):
                    continue
        if not grouped:
            continue
        lags = sorted(grouped)
        medians = [float(np.median(grouped[lag])) for lag in lags]
        fig, axis = plt.subplots(figsize=(6.5, 4.5))
        axis.plot(lags, medians, marker="o")
        axis.set_xlabel("snapshot lag")
        axis.set_ylabel("median normalized L1 distance")
        axis.set_title(f"Drift vs lag: {label}")
        axis.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(figures / f"drift_vs_lag_{label}.png", dpi=160)
        plt.close(fig)


def plot_runtime(records: list[dict[str, str]], figures: Path) -> None:
    """Plot solver optimize time by benchmark mode, excluding first snapshots."""

    modes = ["cold_rebuild", "reuse_model_auto", "explicit_basis", "explicit_basis_presolve", "reset_basis"]
    grouped = []
    for mode in modes:
        rows = [
            row for row in records
            # CSV rows contain strings while the in-memory smoke path retains bool.
            if row.get("solve_mode") == mode
            and str(row.get("is_first_in_window", False)).lower() != "true"
        ]
        grouped.append(_floats(rows, "optimize_wall_time"))
    if not any(grouped):
        return
    fig, axis = plt.subplots(figsize=(7, 4.5))
    axis.boxplot(grouped, tick_labels=modes, showfliers=False)
    axis.set_ylabel("optimize wall time (s)")
    axis.set_title("Cold vs warm LP solve time")
    fig.tight_layout()
    fig.savefig(figures / "runtime_cold_vs_warm_optimize.png", dpi=160)
    plt.close(fig)

    iterations = [
        _floats([
            row for row in records
            if row.get("solve_mode") == mode and str(row.get("is_first_in_window", False)).lower() != "true"
        ], "iter_count")
        for mode in modes
    ]
    fig, axis = plt.subplots(figsize=(9, 4.5))
    axis.boxplot(iterations, tick_labels=modes, showfliers=False)
    axis.set_ylabel("simplex iterations")
    axis.set_title("Cold vs warm LP IterCount")
    fig.tight_layout()
    fig.savefig(figures / "iterations_cold_vs_warm.png", dpi=160)
    plt.close(fig)


def plot_counterfactual(records: list[dict[str, str]], figures: Path) -> None:
    """Plot traffic-only, topology-only, and actual state drift."""

    components = ["traffic_only", "topology_only", "both"]
    grouped = [
        _floats([row for row in records if row.get("component") == component and row.get("state") == "edge_load"], "distance")
        for component in components
    ]
    if not any(grouped):
        return
    fig, axis = plt.subplots(figsize=(7, 4.5))
    axis.boxplot(grouped, tick_labels=components, showfliers=False)
    axis.set_ylabel("edge-load normalized L1")
    axis.set_title("Traffic-only vs topology-only drift")
    fig.tight_layout()
    fig.savefig(figures / "counterfactual_state_drift.png", dpi=160)
    plt.close(fig)


def plot_transport(records: list[dict[str, str]], figures: Path) -> None:
    """Plot best-history candidates against no-history controls for each state."""

    dynamic = [row for row in records if int(float(row.get("new_edge_count", 0))) or int(float(row.get("deleted_edge_count", 0)))]
    labels, values = [], []
    for state in ("edge_load", "utilization", "binding_state", "dual_price"):
        state_rows = [row for row in dynamic if row.get("state") == state]
        for family in ("transport_", "default_"):
            methods = sorted({row.get("initialization", "") for row in state_rows if row.get("initialization", "").startswith(family)})
            if not methods:
                continue
            best = min(methods, key=lambda method: np.median(_floats([row for row in state_rows if row.get("initialization") == method], "normalized_l1")))
            labels.append(f"{state}\n{family[:-1]}")
            values.append(_floats([row for row in state_rows if row.get("initialization") == best], "normalized_l1"))
    if not values:
        return
    fig, axis = plt.subplots(figsize=(10, 4.5))
    axis.boxplot(values, tick_labels=labels, showfliers=False)
    axis.set_ylabel("normalized L1")
    axis.set_title("Dynamic state transport vs no-history baseline")
    fig.tight_layout()
    fig.savefig(figures / "dynamic_state_transport.png", dpi=160)
    plt.close(fig)


def plot_dual_vs_load(records: list[dict[str, str]], figures: Path) -> None:
    """Show explicitly when zero/degenerate dual drift differs from load drift."""

    adjacent = [row for row in records if row.get("pair_type") == "adjacent"]
    load = _floats(adjacent, "edge_load_persistent_normalized_l1")
    dual = _floats(adjacent, "dual_persistent_normalized_l1")
    if not load or len(load) != len(dual):
        return
    fig, axis = plt.subplots(figsize=(6, 4.5))
    axis.scatter(load, dual, alpha=0.65)
    axis.set_xlabel("edge-load normalized L1")
    axis.set_ylabel("dual normalized L1")
    axis.set_title("Dual distance vs edge-load distance")
    axis.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(figures / "dual_distance_vs_edge_load_distance.png", dpi=160)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default="output/temporal_feasibility")
    parser.add_argument("--output-dir", help="Defaults to <input-dir>/figures")
    args = parser.parse_args()
    input_dir = Path(args.input_dir)
    figures = Path(args.output_dir) if args.output_dir else input_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    continuity = _read_csv(input_dir / "continuity_records.csv")
    solves = _read_csv(input_dir / "solve_records.csv")
    counterfactuals = _read_csv(input_dir / "counterfactual_records.csv")
    transports = _read_csv(input_dir / "transport_records.csv")
    plot_adjacent_vs_random(continuity, figures)
    plot_lag(continuity, figures)
    plot_runtime(solves, figures)
    plot_counterfactual(counterfactuals, figures)
    plot_transport(transports, figures)
    plot_dual_vs_load(continuity, figures)
    print(f"FIGURES_DIR={figures}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
