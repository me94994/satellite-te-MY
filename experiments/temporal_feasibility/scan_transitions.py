"""Lightweight full-stream topology scanner for raw SaTE pickle records.

The scanner intentionally does not build candidate paths, solver models, SaTE
objects, or DGL graphs.  Sequence boundaries are explicit so the final record
of one source can never be compared with the first record of another source.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence


SHELL_OFFSETS = (0, 1584, 3168, 3516, 4236)
GROUND_OFFSET = 4236
RawEdge = tuple[str, int, int]


@dataclass(frozen=True)
class RawSource:
    """One independently ordered raw pickle stream."""

    sequence_id: str
    path: Path


def iter_pickle_stream(path: Path) -> Iterator[tuple[int, Mapping[str, Any]]]:
    """Yield every dictionary in a concatenated pickle stream exactly once."""

    with path.open("rb") as handle:
        index = 0
        while True:
            try:
                record = pickle.load(handle)
            except EOFError:
                return
            if not isinstance(record, Mapping):
                raise TypeError(f"Raw source record {index} in {path} is not a mapping")
            yield index, record
            index += 1


def canonical_topology_edges(record: Mapping[str, Any]) -> tuple[frozenset[RawEdge], frozenset[RawEdge]]:
    """Return full raw topology edges and the ISL routing edges used by ISM.ISL.

    Links are canonicalized as undirected physical links.  The full signature
    includes relay attachment because it is part of the raw topology record;
    the routing signature includes only links actually used by the current ISL
    TE formulation.
    """

    routing: set[RawEdge] = set()
    for relation, mapping in enumerate(record["InterShell_ISL"]):
        low_offset = SHELL_OFFSETS[relation]
        high_offset = SHELL_OFFSETS[relation + 1]
        for high_local, low_local in enumerate(mapping):
            if int(low_local) < 0:
                continue
            u = high_offset + high_local
            v = low_offset + int(low_local)
            routing.add(("isl", min(u, v), max(u, v)))

    relay: set[RawEdge] = set()
    for satellite, ground in enumerate(record["InterShell_GrdRelay"]):
        if int(ground) >= 0:
            node = GROUND_OFFSET + int(ground)
            relay.add(("relay", min(satellite, node), max(satellite, node)))
    return frozenset(routing | relay), frozenset(routing)


def _hash_rows(rows: Iterable[Sequence[Any]]) -> str:
    """Hash canonical scalar rows without depending on Python hash randomization."""

    digest = hashlib.sha256()
    for row in sorted(tuple(value for value in item) for item in rows):
        digest.update(json.dumps(row, separators=(",", ":"), ensure_ascii=True).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def flow_summary(record: Mapping[str, Any]) -> tuple[int, float, str, frozenset[tuple[int, int]]]:
    """Return input-only traffic summary and a stable FlowSet content hash."""

    rows = [(int(flow[0]), int(flow[1]), float(flow[2])) for flow in record["FlowSet"]]
    active = frozenset((src, dst) for src, dst, _ in rows)
    return len(active), float(sum(demand for _, _, demand in rows)), _hash_rows(rows), active


def topology_severity(previous: frozenset[RawEdge], current: frozenset[RawEdge]) -> dict[str, Any]:
    """Compute raw, non-collapsed topology transition severity metrics."""

    added = current - previous
    deleted = previous - current
    union = previous | current
    degree_delta: Counter[int] = Counter()
    for _, u, v in added:
        degree_delta[u] += 1
        degree_delta[v] += 1
    for _, u, v in deleted:
        degree_delta[u] -= 1
        degree_delta[v] -= 1
    return {
        "added_edges": len(added),
        "deleted_edges": len(deleted),
        "edge_change_count": len(added) + len(deleted),
        "edge_jaccard": len(previous & current) / len(union) if union else 1.0,
        "degree_change_l1": int(sum(abs(value) for value in degree_delta.values())),
    }


def scan_sources(sources: Iterable[RawSource]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Scan independent streams and return record manifest plus event statistics."""

    rows: list[dict[str, Any]] = []
    sequence_statistics: list[dict[str, Any]] = []
    all_regime_lengths: list[int] = []
    for source in sources:
        previous_full: frozenset[RawEdge] | None = None
        previous_routing: frozenset[RawEdge] | None = None
        previous_hash = ""
        previous_routing_hash = ""
        previous_flows: frozenset[tuple[int, int]] = frozenset()
        previous_demand: float | None = None
        transition_count = 0
        full_transition_count = 0
        regime_length = 0
        regime_lengths: list[int] = []
        record_count = 0
        for index, record in iter_pickle_stream(source.path):
            full_edges, routing_edges = canonical_topology_edges(record)
            graph_hash = _hash_rows(full_edges)
            routing_hash = _hash_rows(routing_edges)
            active_count, demand, flow_hash, active_flows = flow_summary(record)
            full_severity = topology_severity(previous_full, full_edges) if previous_full is not None else {}
            routing_severity = topology_severity(previous_routing, routing_edges) if previous_routing is not None else {}
            full_transition = previous_full is not None and graph_hash != previous_hash
            transition = previous_routing is not None and routing_hash != previous_routing_hash
            if full_transition:
                full_transition_count += 1
            if transition:
                transition_count += 1
                if regime_length:
                    regime_lengths.append(regime_length)
                regime_length = 1
            else:
                regime_length += 1
            flow_union = previous_flows | active_flows
            flow_churn = (
                len(previous_flows ^ active_flows) / len(flow_union)
                if previous_full is not None and flow_union else 0.0
            )
            rows.append(
                {
                    "sequence_id": source.sequence_id,
                    "source_file": source.path.name,
                    "source_record_index": index,
                    "graph_hash": graph_hash,
                    "routing_graph_hash": routing_hash,
                    "previous_graph_hash": previous_hash,
                    "previous_routing_graph_hash": previous_routing_hash,
                    "intershell_edge_count": len(routing_edges),
                    "relay_edge_count": len(full_edges) - len(routing_edges),
                    "full_transition": full_transition,
                    "transition": transition,
                    "added_edges": routing_severity.get("added_edges", 0),
                    "deleted_edges": routing_severity.get("deleted_edges", 0),
                    "edge_change_count": routing_severity.get("edge_change_count", 0),
                    "edge_jaccard": routing_severity.get("edge_jaccard", 1.0),
                    "degree_change_l1": routing_severity.get("degree_change_l1", 0),
                    "full_edge_change_count": full_severity.get("edge_change_count", 0),
                    "active_flow_count": active_count,
                    "active_flow_churn": flow_churn,
                    "total_demand": demand,
                    "total_demand_delta": 0.0 if previous_demand is None else demand - previous_demand,
                    "flowset_hash": flow_hash,
                    "evidence_label": "OFFICIAL_REAL_WORKLOAD",
                }
            )
            previous_full, previous_routing = full_edges, routing_edges
            previous_hash, previous_routing_hash = graph_hash, routing_hash
            previous_flows, previous_demand = active_flows, demand
            record_count += 1
        if regime_length:
            regime_lengths.append(regime_length)
        all_regime_lengths.extend(regime_lengths)
        sequence_statistics.append(
            {
                "sequence_id": source.sequence_id,
                "source_file": source.path.name,
                "raw_record_count": record_count,
                "routing_transition_count": transition_count,
                "full_raw_transition_count": full_transition_count,
                "transition_frequency_per_record": transition_count / record_count if record_count else 0.0,
                "regime_lengths": regime_lengths,
            }
        )

    sorted_lengths = sorted(all_regime_lengths)
    summary = {
        "sequences": sequence_statistics,
        "raw_record_count": len(rows),
        "transition_count": sum(item["routing_transition_count"] for item in sequence_statistics),
        "full_raw_transition_count": sum(item["full_raw_transition_count"] for item in sequence_statistics),
        "sequence_boundary_comparisons": 0,
        "routing_transition_definition": "ISM.ISL inter-shell edge set changed within one sequence_id",
        "full_signature_definition": "InterShell_ISL plus InterShell_GrdRelay attachment edges",
        "regime_statistics": _distribution(sorted_lengths),
        "gate_t0": (
            "STRONG" if sum(item["routing_transition_count"] for item in sequence_statistics) >= 50
            else "PASS" if sum(item["routing_transition_count"] for item in sequence_statistics) >= 30
            else "BLOCKED"
        ),
    }
    return rows, summary


def _distribution(values: Sequence[int]) -> dict[str, Any]:
    """Return deterministic nearest-rank regime-length summaries."""

    if not values:
        return {"count": 0, "minimum": None, "p10": None, "p50": None, "p90": None, "maximum": None}

    def percentile(q: float) -> int:
        index = round((len(values) - 1) * q)
        return int(values[index])

    return {
        "count": len(values), "minimum": int(values[0]), "p10": percentile(0.10),
        "p50": percentile(0.50), "p90": percentile(0.90), "maximum": int(values[-1]),
    }


def write_scan_outputs(rows: list[dict[str, Any]], summary: Mapping[str, Any], output_dir: Path) -> None:
    """Atomically persist scanner outputs using UTF-8 BOM for Windows tooling."""

    if not rows:
        raise ValueError("Refusing to write an empty transition manifest")
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "raw_transition_manifest.csv"
    temporary = csv_path.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(csv_path)
    json_path = output_dir / "transition_summary.json"
    temporary_json = json_path.with_suffix(".json.tmp")
    with temporary_json.open("w", encoding="utf-8-sig") as handle:
        json.dump(dict(summary), handle, ensure_ascii=False, indent=2, allow_nan=False)
    temporary_json.replace(json_path)


def _parse_source(value: str) -> RawSource:
    """Parse SEQUENCE_ID=PATH while keeping the boundary user-visible."""

    if "=" not in value:
        raise argparse.ArgumentTypeError("--source must use SEQUENCE_ID=PATH")
    sequence_id, raw_path = value.split("=", 1)
    if not sequence_id or not raw_path:
        raise argparse.ArgumentTypeError("--source requires non-empty sequence ID and path")
    path = Path(raw_path)
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"Raw source does not exist: {path}")
    return RawSource(sequence_id, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", required=True, type=_parse_source)
    parser.add_argument("--output-dir", default="output/temporal_feasibility/transitions")
    args = parser.parse_args()
    rows, summary = scan_sources(args.source)
    write_scan_outputs(rows, summary, Path(args.output_dir))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
