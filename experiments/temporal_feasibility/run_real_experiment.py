"""Run the complete real-data restricted temporal-feasibility validation."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from dataclasses import replace
from pathlib import Path
from typing import Any

from .analyze_continuity import analyze
from .benchmark_warm_start import benchmark
from .counterfactual import analyze_counterfactuals, write_counterfactual_outputs
from .gurobi_probe import probe_gurobi
from .inspect_dataset import inspect_snapshots, write_manifest
from .plot_results import (
    plot_adjacent_vs_random,
    plot_counterfactual,
    plot_dual_vs_load,
    plot_lag,
    plot_runtime,
    plot_transport,
)
from .real_sequence import adapt_restricted_sequence, build_data_provenance
from .sequence_schema import Snapshot, load_dataset
from .sequential_lp import CapacityPolicy, SolveState


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)


def _write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    if not records:
        raise ValueError(f"Refusing to write empty required output: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    temporary.replace(path)


def _git_commit() -> str:
    """Record the exact checked-out commit without invoking a shell."""

    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    ).stdout.strip()


def _frozen_topology_sequence(snapshots: list[Snapshot]) -> list[Snapshot]:
    """Hold G/path availability at G0 while retaining each real traffic matrix."""

    first = snapshots[0]
    return [
        replace(
            snapshot,
            graph_edges=first.graph_edges,
            edges=first.edges,
            capacities=first.capacities,
            paths=first.paths,
            meta={**snapshot.meta, "experiment_mode": "controlled_frozen_topology_real_traffic"},
        )
        for snapshot in snapshots
    ]


def _write_states(states: list[SolveState], output_dir: Path) -> None:
    _write_csv(output_dir / "solve_records.csv", [state.csv_record() for state in states])
    temporary = output_dir / "state_records.jsonl.tmp"
    with temporary.open("w", encoding="utf-8-sig") as handle:
        for state in states:
            handle.write(json.dumps(state.json_record(), ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(output_dir / "state_records.jsonl")


def _future_topology_records(snapshots: list[Snapshot]) -> list[dict[str, Any]]:
    """Create an oracle recorded-sequence diagnostic without claiming ephemeris prediction."""

    records: list[dict[str, Any]] = []
    for index, snapshot in enumerate(snapshots[:-1]):
        for edge in sorted(snapshot.graph_edges):
            future = []
            for lag in (1, 2, 5):
                future.append(
                    index + lag < len(snapshots) and edge in snapshots[index + lag].graph_edges
                )
            disappearance = None
            for target in range(index + 1, len(snapshots)):
                if edge not in snapshots[target].graph_edges:
                    disappearance = target - index
                    break
            records.append(
                {
                    "position": snapshot.position,
                    "edge_u": edge[0],
                    "edge_v": edge[1],
                    "survives_next_1": future[0],
                    "survives_next_2": future[1],
                    "survives_next_5": future[2],
                    "records_to_disappearance": disappearance if disappearance is not None else "",
                    "provenance": "ORACLE_RECORDED_SEQUENCE_NOT_EPHEMERIS_PREDICTION",
                }
            )
    return records


def _decision(
    continuity_gate: str, warm_gate: str, transport_gate: str
) -> tuple[str, str, str]:
    continuity = "STRONG" if continuity_gate == "PASS" else "MODERATE" if continuity_gate == "PARTIAL" else "WEAK"
    classical = "VERY_STRONG" if warm_gate == "STRONG_WARM_START" else "MODERATE" if warm_gate == "MODERATE_WARM_START" else "WEAK"
    if transport_gate in {"PASS", "PARTIAL"} and continuity != "WEAK":
        recommendation = "TOPOLOGY_TRANSITION_STATE_TRANSPORT"
    elif continuity in {"STRONG", "MODERATE"} and classical != "VERY_STRONG":
        recommendation = "TEMPORAL_RESIDUAL_OPTIMIZER"
    elif continuity in {"STRONG", "MODERATE"}:
        recommendation = "FIXED_TOPOLOGY_REOPTIMIZATION_ONLY"
    else:
        recommendation = "STOP_TEMPORAL_TRACKING"
    return continuity, classical, recommendation


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Execute every evidence stage serially and persist each completed stage."""

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    final_summary = output_dir / "summary.json"
    if args.resume and final_summary.is_file():
        with final_summary.open("r", encoding="utf-8-sig") as handle:
            existing = json.load(handle)
        if existing.get("run_signature") == [args.start_index, args.limit, args.max_flows]:
            return existing

    raw_path, adapted_path = Path(args.raw_path), Path(args.adapted_path)
    slice_manifest = adapt_restricted_sequence(
        raw_path, adapted_path, intensity=args.intensity, start_index=args.start_index,
        limit=args.limit, max_flows=args.max_flows,
    )
    _write_json(output_dir / "real_slice_manifest.json", slice_manifest)
    provenance = build_data_provenance(
        raw_path, adapted_path, args.intensity, args.start_index, args.limit
    )
    provenance["adapter_git_commit"] = _git_commit()
    _write_json(output_dir / "data_provenance.json", provenance)

    snapshots = load_dataset(adapted_path)
    manifest_rows, sequence_summary = inspect_snapshots(snapshots)
    sequence_summary.update({"dataset": str(adapted_path), "evidence_source": "real-data restricted slice"})
    write_manifest(manifest_rows, sequence_summary, output_dir / "sequence_manifest.csv")
    _write_json(output_dir / "sequence_summary.json", sequence_summary)

    gurobi = probe_gurobi(check_size_limit=True)
    _write_json(output_dir / "gurobi_probe.json", gurobi)
    if not gurobi.get("available"):
        raise RuntimeError("Real Gurobi evidence is required but the probe failed")
    if slice_manifest["variable_count"] > 2000 or slice_manifest["constraint_count"] > 2000:
        raise RuntimeError("Restricted slice exceeds the confirmed license-safe size")

    policy = CapacityPolicy(args.network_edge_capacity, args.path_only_edge_capacity)
    continuity_records, transport_records, continuity_summary, states = analyze(
        snapshots, "gurobi", policy, seed=42
    )
    _write_csv(output_dir / "continuity_records.csv", continuity_records)
    _write_csv(output_dir / "transport_records.csv", transport_records)
    _write_states(states, output_dir)

    counterfactual_records, counterfactual_summary = analyze_counterfactuals(
        snapshots, "gurobi", policy
    )
    write_counterfactual_outputs(counterfactual_records, counterfactual_summary, output_dir)

    warm_records, warm_summary = benchmark(
        snapshots, "gurobi", policy, window_size=len(snapshots), stride=len(snapshots),
        tolerance=1e-7,
    )
    _write_csv(output_dir / "warm_start_records.csv", warm_records)
    frozen_records, frozen_summary = benchmark(
        _frozen_topology_sequence(snapshots), "gurobi", policy,
        window_size=len(snapshots), stride=len(snapshots), tolerance=1e-7,
    )
    _write_csv(output_dir / "warm_start_frozen_topology_records.csv", frozen_records)
    _write_csv(output_dir / "future_topology_records.csv", _future_topology_records(snapshots))

    continuity_verdict, classical_verdict, recommendation = _decision(
        continuity_summary["continuity"]["gate_b"], warm_summary["gate_c"],
        continuity_summary["transport"]["gate_d"],
    )
    summary = {
        "run_signature": [args.start_index, args.limit, args.max_flows],
        "evidence_label": "real-data restricted slice",
        "data_provenance": provenance,
        "slice": slice_manifest,
        "gurobi": gurobi,
        "sequence": sequence_summary,
        "continuity": continuity_summary,
        "counterfactual": counterfactual_summary,
        "warm_start_dynamic": warm_summary,
        "warm_start_controlled_frozen_topology": frozen_summary,
        "verdicts": {
            "real_sequence": sequence_summary["real_sequence_verdict"],
            "optimal_state_continuity": continuity_verdict,
            "classical_reoptimization": classical_verdict,
            "topology_vs_traffic_driver": counterfactual_summary["topology_vs_traffic_verdict"],
            "dynamic_state_transport": continuity_summary["transport"]["gate_d"],
            "research_recommendation": recommendation,
        },
        "limitations": [
            "No raw timestamp, orbit epoch, explicit interval, or independent ephemeris is present.",
            "The restricted-license slice is not a full-scale Starlink TE problem.",
            "Persistent-flow selection may suppress traffic churn and limits traffic-only inference.",
            "Future-topology features are oracle observations from the recorded sequence only.",
        ],
    }
    _write_json(final_summary, summary)

    figures = output_dir / "figures"
    figures.mkdir(exist_ok=True)
    plot_adjacent_vs_random(continuity_records, figures)
    plot_lag(continuity_records, figures)
    plot_runtime(warm_records, figures)
    plot_counterfactual(counterfactual_records, figures)
    plot_transport(transport_records, figures)
    plot_dual_vs_load(continuity_records, figures)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-path", required=True)
    parser.add_argument("--adapted-path", required=True)
    parser.add_argument("--output-dir", default="output/temporal_feasibility/real")
    parser.add_argument("--intensity", type=int, default=25)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--max-flows", type=int, default=10)
    parser.add_argument("--network-edge-capacity", type=float, default=200.0)
    parser.add_argument("--path-only-edge-capacity", type=float, default=800.0)
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    summary = run(args)
    print(json.dumps(summary["verdicts"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
