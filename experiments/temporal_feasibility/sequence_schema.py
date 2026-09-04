"""Canonical, order-preserving schema for adapter-generated SaTE snapshots.

This module deliberately loads the adapter's list-of-dict pickle directly.  It
does not pass through ``SaTEEnv``, whose train/test split destroys sequence
adjacency.  Input objects are validated and never modified in place.
"""

from __future__ import annotations

import hashlib
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

FlowKey = tuple[int, int]
EdgeKey = tuple[int, int]
PathKey = tuple[int, int, tuple[int, ...]]


class DatasetFormatError(ValueError):
    """Raised when an input pickle is not an adapter list-of-dict dataset."""


def parse_flow_key(value: object) -> FlowKey:
    """Normalize string or pair flow keys without relying on dict order."""

    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",")]
    elif isinstance(value, (tuple, list)):
        parts = list(value)
    else:
        raise DatasetFormatError(f"Unsupported flow key type: {type(value).__name__}")
    if len(parts) != 2:
        raise DatasetFormatError(f"Flow key must contain src,dst: {value!r}")
    try:
        return int(parts[0]), int(parts[1])
    except (TypeError, ValueError) as exc:
        raise DatasetFormatError(f"Non-integer flow key: {value!r}") from exc


def _edge_from_item(item: object) -> tuple[EdgeKey, float | None]:
    """Read one edge from a tuple/list or a NetworkX-style edge record."""

    if not isinstance(item, (tuple, list)) or len(item) < 2:
        raise DatasetFormatError(f"Invalid graph edge: {item!r}")
    edge = (int(item[0]), int(item[1]))
    capacity: float | None = None
    if len(item) >= 3:
        third = item[2]
        if isinstance(third, Mapping):
            if "capacity" in third:
                capacity = float(third["capacity"])
        elif third is not None:
            capacity = float(third)
    return edge, capacity


@dataclass(frozen=True)
class Snapshot:
    """Immutable semantic view of one adapter sample."""

    position: int
    data_idx: int | float | str
    graph_edges: frozenset[EdgeKey]
    edges: frozenset[EdgeKey]
    capacities: Mapping[EdgeKey, float]
    demands: Mapping[FlowKey, float]
    paths: frozenset[PathKey]
    timestamp: str | None
    source_keys: tuple[str, ...]
    meta: Mapping[str, Any]

    @property
    def total_demand(self) -> float:
        return float(sum(self.demands.values()))

    @property
    def active_flow_count(self) -> int:
        return sum(value > 0.0 for value in self.demands.values())

    @property
    def graph_hash(self) -> str:
        payload = ";".join(f"{u},{v}" for u, v in sorted(self.graph_edges))
        return hashlib.sha256(payload.encode("ascii")).hexdigest()


def snapshot_from_mapping(sample: Mapping[str, Any], position: int) -> Snapshot:
    """Validate one adapter sample and canonicalize semantic keys."""

    missing = {"graph", "tm", "path", "data_idx"}.difference(sample)
    if missing:
        raise DatasetFormatError(f"Sample {position} misses keys: {sorted(missing)}")
    if not isinstance(sample["tm"], Mapping) or not isinstance(sample["path"], Mapping):
        raise DatasetFormatError(f"Sample {position} tm/path must be mappings")

    edges: set[EdgeKey] = set()
    capacities: dict[EdgeKey, float] = {}
    graph_obj = sample["graph"]
    if hasattr(graph_obj, "edges"):
        graph_items = list(graph_obj.edges(data=True))
    elif isinstance(graph_obj, Iterable):
        graph_items = list(graph_obj)
    else:
        raise DatasetFormatError(f"Sample {position} graph is not iterable")
    for item in graph_items:
        edge, capacity = _edge_from_item(item)
        edges.add(edge)
        if capacity is not None:
            if capacity < 0 or not _is_finite(capacity):
                raise DatasetFormatError(f"Invalid capacity for {edge}: {capacity}")
            capacities[edge] = capacity

    demands: dict[FlowKey, float] = {}
    for raw_key, raw_value in sample["tm"].items():
        key = parse_flow_key(raw_key)
        value = float(raw_value)
        if value < 0 or not _is_finite(value):
            raise DatasetFormatError(f"Invalid demand for {key}: {value}")
        if key[0] != key[1]:
            demands[key] = demands.get(key, 0.0) + value

    paths: set[PathKey] = set()
    for raw_flow, raw_paths in sample["path"].items():
        flow = parse_flow_key(raw_flow)
        if not isinstance(raw_paths, Sequence):
            raise DatasetFormatError(f"Paths for {flow} must be a sequence")
        for raw_path in raw_paths:
            nodes = tuple(int(node) for node in raw_path)
            if len(nodes) < 2 or nodes[0] != flow[0] or nodes[-1] != flow[1]:
                raise DatasetFormatError(
                    f"Path endpoints do not match {flow} in sample {position}: {nodes}"
                )
            if any(u == v for u, v in zip(nodes[:-1], nodes[1:])):
                raise DatasetFormatError(f"Self-loop in semantic path: {nodes}")
            paths.add((flow[0], flow[1], nodes))

    # Paths may contain adapter-added access links absent from sample['graph'].
    graph_edges = frozenset(edges)
    path_edges = {edge for key in paths for edge in path_edges_for_key(key)}
    edges.update(path_edges)

    raw_meta = sample.get("meta", {})
    if raw_meta is None:
        raw_meta = {}
    if not isinstance(raw_meta, Mapping):
        raise DatasetFormatError(f"Sample {position} meta must be a mapping when present")
    meta = {str(key): value for key, value in raw_meta.items()}
    timestamp_keys = ("source_timestamp", "timestamp", "time", "datetime", "epoch", "ephemeris_time")
    timestamp = next(
        (
            str(container[key]) for container in (meta, sample) for key in timestamp_keys
            if key in container and container[key] is not None and container[key] != ""
        ),
        None,
    )
    return Snapshot(
        position=position,
        data_idx=sample["data_idx"],
        graph_edges=graph_edges,
        edges=frozenset(edges),
        capacities=capacities,
        demands=demands,
        paths=frozenset(paths),
        timestamp=timestamp,
        source_keys=tuple(sorted(str(key) for key in sample.keys())),
        meta=meta,
    )


def _is_finite(value: float) -> bool:
    return value == value and value not in (float("inf"), float("-inf"))


def path_edges_for_key(path_key: PathKey) -> tuple[EdgeKey, ...]:
    """Return directed edges traversed by a semantic path key."""

    nodes = path_key[2]
    return tuple(zip(nodes[:-1], nodes[1:]))


def load_dataset(
    path: str | Path, limit: int | None = None, start_index: int = 0
) -> list[Snapshot]:
    """Load and validate an adapter-generated pickle while preserving list order."""

    dataset_path = Path(path)
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset does not exist: {dataset_path}")
    with dataset_path.open("rb") as handle:
        raw = pickle.load(handle)
    if not isinstance(raw, list):
        raise DatasetFormatError(
            f"Expected a list-of-dict pickle, got {type(raw).__name__}: {dataset_path}"
        )
    if start_index < 0:
        raise ValueError("--start-index must be non-negative")
    if limit is not None:
        if limit <= 0:
            raise ValueError("--limit must be positive")
        raw = raw[start_index:start_index + limit]
    else:
        raw = raw[start_index:]
    snapshots = []
    for position, sample in enumerate(raw, start=start_index):
        if not isinstance(sample, Mapping):
            raise DatasetFormatError(f"Sample {position} is not a mapping")
        snapshots.append(snapshot_from_mapping(sample, position))
    if not snapshots:
        raise DatasetFormatError(f"Dataset is empty: {dataset_path}")
    return snapshots


def discover_datasets(root: str | Path = "input") -> list[Path]:
    """Find small/reduced Starlink adapter outputs without downloading anything."""

    base = Path(root)
    if not base.exists():
        return []
    candidates = [path for path in base.rglob("*.pkl") if "StarLink_DataSetForAgent" in path.name]

    def priority(path: Path) -> tuple[int, int, str]:
        text = str(path).lower()
        rank = 0 if "fixed_topo" in text and "size500" in text else 1 if "176" in text else 2 if "500" in text else 3
        return rank, path.stat().st_size, str(path)

    return sorted(candidates, key=priority)


def numeric_data_idx(value: int | float | str) -> float | None:
    """Return a finite numeric data_idx when available."""

    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if _is_finite(numeric) else None
