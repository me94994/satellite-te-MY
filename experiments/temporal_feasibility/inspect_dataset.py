"""Inspect whether adapter output supports a defensible temporal interpretation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from .semantic_alignment import jaccard, normalized_l1
from .sequence_schema import Snapshot, discover_datasets, load_dataset, numeric_data_idx


MANIFEST_FIELDS = [
    "list_position",
    "data_idx",
    "edge_count",
    "graph_hash",
    "total_demand",
    "active_flow_count",
    "semantic_path_count",
    "edge_jaccard_to_prev",
    "normalized_tm_l1_delta_to_prev",
    "semantic_path_overlap_to_prev",
    "is_data_idx_monotonic",
    "data_idx_gap",
    "has_timestamp",
]


def inspect_snapshots(snapshots: list[Snapshot]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build per-sample records and a conservative sequence-level conclusion."""

    rows: list[dict[str, Any]] = []
    seen_indices: set[str] = set()
    duplicate_positions: list[int] = []
    gap_positions: list[int] = []
    nonmonotonic_positions: list[int] = []
    for index, snapshot in enumerate(snapshots):
        previous = snapshots[index - 1] if index else None
        current_numeric = numeric_data_idx(snapshot.data_idx)
        previous_numeric = numeric_data_idx(previous.data_idx) if previous else None
        gap = current_numeric - previous_numeric if current_numeric is not None and previous_numeric is not None else None
        monotonic = None if previous is None else gap is not None and gap > 0
        if previous is not None and monotonic is not True:
            nonmonotonic_positions.append(index)
        if gap is not None and abs(gap - 1.0) > 1e-12:
            gap_positions.append(index)
        canonical_idx = repr(snapshot.data_idx)
        if canonical_idx in seen_indices:
            duplicate_positions.append(index)
        seen_indices.add(canonical_idx)
        rows.append(
            {
                "list_position": snapshot.position,
                "data_idx": snapshot.data_idx,
                "edge_count": len(snapshot.graph_edges),
                "graph_hash": snapshot.graph_hash,
                "total_demand": snapshot.total_demand,
                "active_flow_count": snapshot.active_flow_count,
                "semantic_path_count": len(snapshot.paths),
                "edge_jaccard_to_prev": "" if previous is None else jaccard(previous.graph_edges, snapshot.graph_edges),
                "normalized_tm_l1_delta_to_prev": "" if previous is None else normalized_l1(previous.demands, snapshot.demands),
                "semantic_path_overlap_to_prev": "" if previous is None else jaccard(previous.paths, snapshot.paths),
                "is_data_idx_monotonic": "" if monotonic is None else monotonic,
                "data_idx_gap": "" if gap is None else gap,
                "has_timestamp": snapshot.timestamp is not None,
            }
        )

    graph_hashes = {snapshot.graph_hash for snapshot in snapshots}
    has_all_timestamps = all(snapshot.timestamp is not None for snapshot in snapshots)
    strictly_monotonic = not nonmonotonic_positions
    summary = {
        "snapshot_count": len(snapshots),
        "data_idx_strictly_monotonic": strictly_monotonic,
        "duplicate_data_idx_positions": duplicate_positions,
        "non_unit_gap_positions": gap_positions,
        "nonmonotonic_positions": nonmonotonic_positions,
        "unique_graph_hash_count": len(graph_hashes),
        "fixed_topology_observed": len(graph_hashes) == 1,
        "timestamp_field_present_for_all": has_all_timestamps,
        "source_keys": sorted({key for snapshot in snapshots for key in snapshot.source_keys}),
        "sequence_claim": (
            "PHYSICAL_TIME_METADATA_PRESENT_ORDER_STILL_REQUIRES_SOURCE_VALIDATION"
            if has_all_timestamps and strictly_monotonic
            else "ORDERED_SAMPLE_CORRELATION_ONLY_NOT_EPHEMERIS_AWARE_TEMPORAL_TRACKING"
        ),
        "intensity_or_volume_boundary_status": (
            "NOT_IDENTIFIABLE_FROM_SAMPLE_SCHEMA"
            if not any(key in {"intensity", "volume"} for key in snapshots[0].source_keys)
            else "METADATA_PRESENT_REQUIRES_VALUE_LEVEL_REVIEW"
        ),
    }
    return rows, summary


def write_manifest(rows: list[dict[str, Any]], summary: dict[str, Any], output: Path) -> None:
    """Write CSV plus a sidecar JSON summary without overwriting the input."""

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    summary_path = output.with_suffix(".summary.json")
    with summary_path.open("w", encoding="utf-8-sig") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False, allow_nan=False)


def resolve_dataset(explicit: str | None, search_root: str) -> Path:
    """Resolve an explicit dataset, otherwise use the documented priority order."""

    if explicit:
        return Path(explicit)
    candidates = discover_datasets(search_root)
    if not candidates:
        raise FileNotFoundError(
            f"No Starlink adapter dataset found under {search_root!r}; provide --dataset explicitly"
        )
    return candidates[0]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", help="Adapter-generated list-of-dict pickle")
    parser.add_argument("--search-root", default="input", help="Root used only when --dataset is omitted")
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--output", required=True, help="Manifest CSV path")
    parser.add_argument("--list-datasets", action="store_true", help="Print discovered datasets and exit")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.list_datasets:
        for candidate in discover_datasets(args.search_root):
            print(candidate)
        return 0
    dataset_path = resolve_dataset(args.dataset, args.search_root)
    snapshots = load_dataset(dataset_path, limit=args.limit)
    rows, summary = inspect_snapshots(snapshots)
    summary["dataset"] = str(dataset_path)
    summary["evidence_source"] = "real_adapter_pickle"
    write_manifest(rows, summary, Path(args.output))
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
