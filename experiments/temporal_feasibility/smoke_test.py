"""Run the complete synthetic pipeline in one deterministic serial process."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from .analyze_continuity import analyze
from .benchmark_warm_start import benchmark
from .inspect_dataset import inspect_snapshots, write_manifest
from .plot_results import plot_adjacent_vs_random, plot_lag, plot_runtime
from .sequence_schema import load_dataset
from .sequential_lp import CapacityPolicy
from .synthetic_fixture import write_fixture


def _write_csv(path: Path, records: list[dict[str, object]]) -> None:
    if not records:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def run_synthetic_smoke(output_dir: Path) -> dict[str, object]:
    """Exercise inspection, LP, continuity, transport, benchmark, and plotting."""

    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = write_fixture(output_dir / "synthetic_fixture.pkl", snapshot_count=16)
    snapshots = load_dataset(dataset_path)
    manifest, sequence_summary = inspect_snapshots(snapshots)
    sequence_summary.update({"dataset": str(dataset_path), "evidence_source": "synthetic_fixture"})
    write_manifest(manifest, sequence_summary, output_dir / "sequence_manifest.csv")

    policy = CapacityPolicy(network_edge_capacity=200.0, path_only_edge_capacity=800.0)
    continuity, transport, continuity_summary, states = analyze(snapshots, "scipy", policy)
    continuity_summary.update({"dataset": str(dataset_path), "evidence_source": "synthetic_fixture"})
    _write_csv(output_dir / "continuity_records.csv", continuity)
    _write_csv(output_dir / "transport_records.csv", transport)
    with (output_dir / "state_records.jsonl").open("w", encoding="utf-8-sig") as handle:
        for state in states:
            handle.write(json.dumps(state.json_record(), ensure_ascii=False, allow_nan=False) + "\n")

    solve_records, warm_summary = benchmark(
        snapshots, "scipy", policy, window_size=8, stride=4, tolerance=1e-7
    )
    warm_summary.update({"dataset": str(dataset_path), "evidence_source": "synthetic_fixture"})
    _write_csv(output_dir / "solve_records.csv", solve_records)
    combined = {
        "evidence_source": "synthetic_fixture",
        "sequence": sequence_summary,
        "continuity": continuity_summary,
        "warm_start": warm_summary,
        "formal_claim_limit": "SciPy/HiGHS functional validation only; no Gurobi timing evidence",
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8-sig") as handle:
        json.dump(combined, handle, indent=2, ensure_ascii=False, allow_nan=False)

    figures = output_dir / "figures"
    figures.mkdir(exist_ok=True)
    plot_adjacent_vs_random(continuity, figures)
    plot_lag(continuity, figures)
    plot_runtime(solve_records, figures)
    return combined


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="output/temporal_feasibility/synthetic_smoke")
    args = parser.parse_args()
    summary = run_synthetic_smoke(Path(args.output_dir))
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
