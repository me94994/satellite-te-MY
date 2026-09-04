"""Traffic-only and topology-only fixed-universe counterfactual LP analysis."""

from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from .extract_primal_dual import calibrate_dual_sign
from .semantic_alignment import jaccard, normalized_l1
from .sequence_schema import Snapshot
from .sequential_lp import CapacityPolicy, GurobiSequentialLP, LPUniverse, ScipySequentialLP, SolveState


def _counterfactual(topology: Snapshot, demand: Snapshot, label: str) -> Snapshot:
    """Combine topology/path availability from one snapshot with another real TM."""

    meta = dict(topology.meta)
    meta["counterfactual"] = label
    return replace(topology, demands=demand.demands, meta=meta)


def _distances(left: SolveState, right: SolveState, persistent_edges: list[tuple[int, int]]) -> dict[str, float]:
    path_keys = sorted(set(left.path_flow).union(right.path_flow))
    edge_keys = sorted(set(left.edge_load).union(right.edge_load))
    left_binding = {edge for edge, active in left.binding_capacity.items() if active}
    right_binding = {edge for edge, active in right.binding_capacity.items() if active}
    return {
        "path_flow": normalized_l1(left.path_flow, right.path_flow, path_keys),
        "edge_load": normalized_l1(left.edge_load, right.edge_load, edge_keys),
        "utilization": normalized_l1(left.edge_utilization, right.edge_utilization, edge_keys),
        "binding_set": 1.0 - jaccard(left_binding, right_binding),
        "dual_price": normalized_l1(left.congestion_price, right.congestion_price, persistent_edges),
    }


def analyze_counterfactuals(
    snapshots: list[Snapshot], backend: str, policy: CapacityPolicy
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Solve z00/z01/z10/z11 and report traffic, topology, both, interaction drift."""

    universe = LPUniverse.from_snapshots(snapshots)
    sign = calibrate_dual_sign(backend)["sign_multiplier"]
    solver: Any = (
        GurobiSequentialLP(universe, policy, sign)
        if backend == "gurobi" else ScipySequentialLP(universe, policy, sign)
    )
    records: list[dict[str, Any]] = []
    for index in range(len(snapshots) - 1):
        current, following = snapshots[index], snapshots[index + 1]
        problems = {
            "z00": _counterfactual(current, current, "G_t_D_t"),
            "z01": _counterfactual(current, following, "G_t_D_t_plus_1"),
            "z10": _counterfactual(following, current, "G_t_plus_1_D_t"),
            "z11": _counterfactual(following, following, "G_t_plus_1_D_t_plus_1"),
        }
        states = {name: solver.solve(problem, "cold_rebuild") for name, problem in problems.items()}
        persistent = sorted(set(current.graph_edges).intersection(following.graph_edges))
        components = {
            "traffic_only": _distances(states["z00"], states["z01"], persistent),
            "topology_only": _distances(states["z00"], states["z10"], persistent),
            "both": _distances(states["z00"], states["z11"], persistent),
        }
        for metric in components["both"]:
            for component in ("traffic_only", "topology_only", "both"):
                records.append(
                    {
                        "transition": index,
                        "source_position": current.position,
                        "target_position": following.position,
                        "component": component,
                        "state": metric,
                        "distance": components[component][metric],
                    }
                )
            records.append(
                {
                    "transition": index,
                    "source_position": current.position,
                    "target_position": following.position,
                    "component": "interaction",
                    "state": metric,
                    "distance": components["both"][metric] - components["traffic_only"][metric] - components["topology_only"][metric],
                }
            )
    if backend == "gurobi":
        solver.close()
    summary: dict[str, Any] = {"states": {}}
    for state_name in ("edge_load", "utilization", "binding_set", "path_flow", "dual_price"):
        summary["states"][state_name] = {}
        for component in ("traffic_only", "topology_only", "both", "interaction"):
            values = [
                row["distance"] for row in records
                if row["state"] == state_name and row["component"] == component
            ]
            summary["states"][state_name][component] = {
                "median": float(np.median(values)),
                "mean": float(np.mean(values)),
                "max": float(np.max(values)),
                "nonzero_count": sum(abs(value) > 1e-12 for value in values),
            }
    # Sparse topology events vanish under an all-transition median, so driver
    # classification uses the preregistered primary-state mean and reports both.
    traffic = np.mean([
        summary["states"][state]["traffic_only"]["mean"]
        for state in ("edge_load", "utilization", "binding_set")
    ])
    topology = np.mean([
        summary["states"][state]["topology_only"]["mean"]
        for state in ("edge_load", "utilization", "binding_set")
    ])
    interaction = np.mean([
        abs(summary["states"][state]["interaction"]["mean"])
        for state in ("edge_load", "utilization", "binding_set")
    ])
    scale = max(float(traffic), float(topology), 1e-12)
    if interaction > scale:
        verdict = "STRONG_INTERACTION"
    elif traffic > 1.5 * topology:
        verdict = "TRAFFIC_DOMINANT"
    elif topology > 1.5 * traffic:
        verdict = "TOPOLOGY_DOMINANT"
    else:
        verdict = "COMPARABLE"
    summary.update(
        {
            "topology_vs_traffic_verdict": verdict,
            "primary_state_mean_traffic_only": float(traffic),
            "primary_state_mean_topology_only": float(topology),
            "primary_state_mean_abs_interaction": float(interaction),
        }
    )
    return records, summary


def write_counterfactual_outputs(
    records: list[dict[str, Any]], summary: dict[str, Any], output_dir: Path
) -> None:
    """Write counterfactual evidence in auditable tabular and JSON form."""

    with (output_dir / "counterfactual_records.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    with (output_dir / "counterfactual_summary.json").open("w", encoding="utf-8-sig") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False, allow_nan=False)
