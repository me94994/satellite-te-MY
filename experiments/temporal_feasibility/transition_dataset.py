"""Event-centered real SaTE slices for topology-transition reoptimization."""

from __future__ import annotations

import csv
import hashlib
import json
import pickle
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from lib.data.starlink import SPOnGrid as SPG
from lib.data.starlink.ism import InterShellMode
from lib.data.starlink.user_node import generate_sat2user

from .real_sequence import GROUND_STATION_COUNT, ORBIT_SHELLS
from .sequence_schema import FlowKey, PathKey, Snapshot, path_edges_for_key
from .sequential_lp import CapacityPolicy, LPUniverse


RawPathMap = dict[FlowKey, tuple[tuple[int, ...], ...]]
SLICE_TYPES = ("PERSISTENCE_SLICE", "HIGH_PRESSURE_SLICE", "DEMAND_MATCHED_RANDOM_SLICE")


@dataclass(frozen=True)
class TransitionEvent:
    """One topology transition and its bounded source-record window."""

    event_id: int
    sequence_id: str
    source_file: str
    transition_index: int
    window_indices: tuple[int, ...]


def read_transition_events(manifest_path: Path, max_events: int, radius: int = 2) -> list[TransitionEvent]:
    """Select the earliest real routing transitions without outcome-based filtering."""

    if max_events <= 0 or radius < 1:
        raise ValueError("max_events must be positive and radius must be at least one")
    events: list[TransitionEvent] = []
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["transition"].lower() != "true":
                continue
            index = int(row["source_record_index"])
            indices = tuple(range(max(0, index - radius), index + radius + 1))
            events.append(
                TransitionEvent(
                    event_id=len(events), sequence_id=row["sequence_id"],
                    source_file=row["source_file"], transition_index=index,
                    window_indices=indices,
                )
            )
            if len(events) == max_events:
                break
    return events


def load_raw_indices(path: Path, indices: Iterable[int]) -> dict[int, Mapping[str, Any]]:
    """Read only requested records in one sequential pass over the raw stream."""

    required = sorted(set(indices))
    if not required:
        return {}
    found: dict[int, Mapping[str, Any]] = {}
    with path.open("rb") as handle:
        for index in range(required[-1] + 1):
            try:
                record = pickle.load(handle)
            except EOFError as exc:
                raise ValueError(f"Raw stream ended before requested record {required[-1]}") from exc
            if index in required:
                if not isinstance(record, Mapping):
                    raise TypeError(f"Raw source record {index} is not a mapping")
                found[index] = record
    return found


def _flow_demands(record: Mapping[str, Any]) -> dict[FlowKey, float]:
    """Aggregate duplicate raw FlowSet rows by semantic source/destination."""

    demands: defaultdict[FlowKey, float] = defaultdict(float)
    for flow in record["FlowSet"]:
        src, dst = int(flow[0]), int(flow[1])
        if src != dst:
            demands[(src, dst)] += float(flow[2])
    return dict(demands)


def candidate_paths(record: Mapping[str, Any], flows: Iterable[FlowKey]) -> RawPathMap:
    """Build only official candidate paths needed for deterministic slice selection."""

    available = _flow_demands(record)
    result: RawPathMap = {}
    for flow in sorted(set(flows).intersection(available)):
        paths = SPG.SPOnGrid(
            flow[0], flow[1], record["InterShell_GrdRelay"], record["InterShell_ISL"],
            InterShellMode.ISL, 5,
        )
        if not paths:
            raise RuntimeError(f"No official-adapter candidate path for flow {flow}")
        # The official adapter repeats path 0 to five entries; semantic LP paths
        # are intentionally deduplicated because repeated variables are artifacts.
        result[flow] = tuple(sorted({tuple(int(node) for node in path) for path in paths[:5]}))
    return result


def _all_candidate_flows(records: Sequence[Mapping[str, Any]]) -> tuple[list[FlowKey], Counter[FlowKey]]:
    appearances: Counter[FlowKey] = Counter()
    for record in records:
        appearances.update(_flow_demands(record))
    return sorted(appearances), appearances


def equal_split_pressure(
    records: Sequence[Mapping[str, Any]], path_maps: Sequence[RawPathMap], policy: CapacityPolicy,
    hotspot_count: int = 20,
) -> dict[FlowKey, float]:
    """Score flows using only demands, candidate paths, and fixed capacities."""

    edge_proxy: defaultdict[tuple[int, int], float] = defaultdict(float)
    per_flow_rows: defaultdict[FlowKey, list[tuple[tuple[int, ...], float, int]]] = defaultdict(list)
    for record, paths_by_flow in zip(records, path_maps):
        demands = _flow_demands(record)
        for flow, paths in paths_by_flow.items():
            demand = demands[flow]
            split = demand / len(paths)
            for path in paths:
                per_flow_rows[flow].append((path, demand, len(paths)))
                for edge in zip(path[:-1], path[1:]):
                    edge_proxy[edge] += split
    proxy_util = {
        edge: load / (policy.path_only_edge_capacity if edge[0] >= 4236 or edge[1] >= 4236 else policy.network_edge_capacity)
        for edge, load in edge_proxy.items()
    }
    hotspots = set(
        edge for edge, _ in sorted(proxy_util.items(), key=lambda item: (-item[1], item[0]))[:hotspot_count]
    )
    scores: dict[FlowKey, float] = {}
    for flow, rows in per_flow_rows.items():
        score = 0.0
        for path, demand, path_count in rows:
            score += sum(proxy_util[edge] * demand / path_count for edge in zip(path[:-1], path[1:]) if edge in hotspots)
        scores[flow] = score
    return scores


def _snapshot_from_raw(
    record: Mapping[str, Any], paths_by_flow: RawPathMap, selected: Iterable[FlowKey],
    *, position: int, source_file: str, sequence_id: str, intensity: int,
    demand_scale: float,
) -> Snapshot:
    """Create a minimal semantic Snapshot without constructing the full graph."""

    sat2user = generate_sat2user(sum(a * b for a, b in ORBIT_SHELLS), GROUND_STATION_COUNT, InterShellMode.ISL)
    raw_demands = _flow_demands(record)
    demands: dict[FlowKey, float] = {}
    semantic_paths: set[PathKey] = set()
    network_edges: set[tuple[int, int]] = set()
    for raw_flow in sorted(selected):
        if raw_flow not in raw_demands or raw_flow not in paths_by_flow:
            continue
        flow = (sat2user(raw_flow[0]), sat2user(raw_flow[1]))
        demands[flow] = raw_demands[raw_flow] * demand_scale
        for raw_path in paths_by_flow[raw_flow]:
            nodes = (flow[0],) + raw_path + (flow[1],)
            semantic_paths.add((flow[0], flow[1], nodes))
            network_edges.update(zip(raw_path[:-1], raw_path[1:]))
    path_edges = {edge for path in semantic_paths for edge in path_edges_for_key(path)}
    return Snapshot(
        position=position, data_idx=position, graph_edges=frozenset(network_edges),
        edges=frozenset(network_edges | path_edges), capacities={}, demands=demands,
        paths=frozenset(semantic_paths), timestamp=None,
        source_keys=("FlowSet", "InterShell_GrdRelay", "InterShell_ISL"),
        meta={
            "source_file": source_file, "source_record_index": position,
            "sequence_id": sequence_id, "intensity": intensity,
            "evidence_label": "OFFICIAL_REAL_WORKLOAD" if demand_scale == 1.0 else "REAL_TOPOLOGY_REAL_TRAFFIC_SCALED_STRESS",
            "demand_scale": demand_scale, "adapter_mode": "event_centered_ism_isl_v1",
        },
    )


def _size_for_selection(
    records: Sequence[Mapping[str, Any]], path_maps: Sequence[RawPathMap], selected: Sequence[FlowKey],
    *, positions: Sequence[int], source_file: str, sequence_id: str, intensity: int,
) -> tuple[int, int, int, int]:
    snapshots = [
        _snapshot_from_raw(record, paths, selected, position=position, source_file=source_file,
                           sequence_id=sequence_id, intensity=intensity, demand_scale=1.0)
        for record, paths, position in zip(records, path_maps, positions)
    ]
    universe = LPUniverse.from_snapshots(snapshots)
    return len(universe.flows), len(universe.paths), len(universe.edges), len(universe.edges) + len(universe.flows) + len(universe.paths)


def greedy_license_safe_selection(
    ordered_flows: Sequence[FlowKey], records: Sequence[Mapping[str, Any]], path_maps: Sequence[RawPathMap],
    *, positions: Sequence[int], source_file: str, sequence_id: str, intensity: int,
    constraint_budget: int = 1900, target_count: int | None = None,
) -> tuple[list[FlowKey], tuple[int, int, int, int]]:
    """Greedily maximize ordered flows while keeping the actual LP under budget."""

    satellite_count = sum(a * b for a, b in ORBIT_SHELLS)
    sat2user = generate_sat2user(satellite_count, GROUND_STATION_COUNT, InterShellMode.ISL)
    flow_paths: dict[FlowKey, set[tuple[int, ...]]] = {}
    flow_edges: dict[FlowKey, set[tuple[int, int]]] = {}
    for flow in ordered_flows:
        paths = {
            path for path_map in path_maps for path in path_map.get(flow, ())
        }
        flow_paths[flow] = paths
        edges = {edge for path in paths for edge in zip(path[:-1], path[1:])}
        if paths:
            edges.update({(sat2user(flow[0]), flow[0]), (flow[1], sat2user(flow[1]))})
        flow_edges[flow] = edges
    selected: list[FlowKey] = []
    selected_paths: set[tuple[int, ...]] = set()
    selected_edges: set[tuple[int, int]] = set()
    for flow in ordered_flows:
        if target_count is not None and len(selected) >= target_count:
            break
        next_paths = selected_paths | flow_paths[flow]
        next_edges = selected_edges | flow_edges[flow]
        next_flow_count = len(selected) + 1
        constraint_count = next_flow_count + len(next_paths) + len(next_edges)
        if constraint_count <= constraint_budget:
            selected.append(flow)
            selected_paths, selected_edges = next_paths, next_edges
    if not selected or (target_count is not None and len(selected) != target_count):
        raise RuntimeError(f"Could not construct requested license-safe slice of {target_count} flows")
    # Final construction is the source-of-truth check for incremental counting.
    final_size = _size_for_selection(
        records, path_maps, selected, positions=positions, source_file=source_file,
        sequence_id=sequence_id, intensity=intensity,
    )
    if final_size[3] > constraint_budget:
        raise AssertionError("Incremental restricted-license accounting disagrees with LPUniverse")
    return selected, final_size


def _demand_signature(flow: FlowKey, records: Sequence[Mapping[str, Any]]) -> tuple[float, ...]:
    """Match controls on the complete event-window demand/presence trajectory."""

    return tuple(float(_flow_demands(record).get(flow, 0.0)) for record in records)


def _fast_universe_size(selected: Sequence[FlowKey], path_maps: Sequence[RawPathMap]) -> tuple[int, int, int, int]:
    """Count a semantic universe from cached paths without rebuilding Snapshots."""

    sat2user = generate_sat2user(
        sum(a * b for a, b in ORBIT_SHELLS), GROUND_STATION_COUNT, InterShellMode.ISL
    )
    semantic_paths: set[tuple[FlowKey, tuple[int, ...]]] = set()
    edges: set[tuple[int, int]] = set()
    active_flows: set[FlowKey] = set()
    for flow in selected:
        paths = {path for path_map in path_maps for path in path_map.get(flow, ())}
        if not paths:
            continue
        active_flows.add(flow)
        semantic_paths.update((flow, path) for path in paths)
        edges.update(edge for path in paths for edge in zip(path[:-1], path[1:]))
        edges.update({(sat2user(flow[0]), flow[0]), (flow[1], sat2user(flow[1]))})
    size = (len(active_flows), len(semantic_paths), len(edges), 0)
    return size[:3] + (sum(size[:3]),)


def _demand_matched_random_selection(
    high_pressure: Sequence[FlowKey], candidates: Sequence[FlowKey],
    records: Sequence[Mapping[str, Any]], path_maps: Sequence[RawPathMap],
    *, positions: Sequence[int], source_file: str, sequence_id: str, intensity: int,
    constraint_budget: int, random_seed: int,
) -> tuple[list[FlowKey], tuple[int, int, int, int]] | None:
    """Randomly match demand signatures while enforcing size and path-count parity."""

    demand_maps = [_flow_demands(record) for record in records]
    signatures = {
        flow: tuple(float(demands.get(flow, 0.0)) for demands in demand_maps)
        for flow in candidates
    }
    required = Counter(signatures[flow] for flow in high_pressure)
    pools: defaultdict[tuple[float, ...], list[FlowKey]] = defaultdict(list)
    high_set = set(high_pressure)
    for flow in candidates:
        pools[signatures[flow]].append(flow)
    high_size = _fast_universe_size(list(high_pressure), path_maps)
    for attempt in range(100):
        rng = random.Random(random_seed + 1009 * attempt)
        selected: list[FlowKey] = []
        possible = True
        for signature in sorted(required):
            # Prefer independent flows, but retain the selected flow itself as
            # a deterministic fallback when a demand signature is unique.
            independent = [flow for flow in pools[signature] if flow not in high_set]
            overlap = [flow for flow in pools[signature] if flow in high_set]
            rng.shuffle(independent)
            rng.shuffle(overlap)
            choices = independent + overlap
            count = required[signature]
            if len(choices) < count:
                possible = False
                break
            selected.extend(choices[:count])
        if not possible:
            continue
        size = _fast_universe_size(selected, path_maps)
        path_ratio = size[1] / high_size[1] if high_size[1] else 1.0
        if size[3] <= constraint_budget and 0.8 <= path_ratio <= 1.25:
            actual_size = _size_for_selection(
                records, path_maps, selected, positions=positions, source_file=source_file,
                sequence_id=sequence_id, intensity=intensity,
            )
            if actual_size != size:
                raise AssertionError("Fast demand-matched size accounting disagrees with LPUniverse")
            return selected, actual_size
    return None


def build_event_slices(
    event: TransitionEvent, raw_records: Mapping[int, Mapping[str, Any]], *, intensity: int,
    policy: CapacityPolicy, constraint_budget: int = 1900, random_seed: int = 42,
    demand_scale: float = 1.0,
) -> tuple[dict[str, list[Snapshot]], dict[str, dict[str, Any]], dict[str, Any]]:
    """Build persistence, pressure, and matched-random slices for one event."""

    window_records = [raw_records[index] for index in event.window_indices]
    pair_positions = (event.transition_index - 1, event.transition_index)
    pair_records = [raw_records[index] for index in pair_positions]
    candidates, appearances = _all_candidate_flows(window_records)
    pair_path_maps = [candidate_paths(record, candidates) for record in pair_records]
    pair_candidates = sorted(set().union(*(paths.keys() for paths in pair_path_maps)))
    # Size selection against the whole event window so either flanking stable
    # pair is guaranteed to remain below the restricted-license budget.
    pair_paths_by_position = dict(zip(pair_positions, pair_path_maps))
    window_path_maps = [
        pair_paths_by_position.get(position) or candidate_paths(record, pair_candidates)
        for position, record in zip(event.window_indices, window_records)
    ]
    scores = equal_split_pressure(pair_records, pair_path_maps, policy)
    pressure_order = sorted(
        pair_candidates, key=lambda flow: (-scores.get(flow, 0.0), -appearances[flow], flow[0], flow[1])
    )
    persistence_order = sorted(
        pair_candidates, key=lambda flow: (-appearances[flow], flow[0], flow[1])
    )
    common = dict(
        records=window_records, path_maps=window_path_maps, positions=event.window_indices,
        source_file=event.source_file, sequence_id=event.sequence_id, intensity=intensity,
        constraint_budget=constraint_budget,
    )
    high_max, _ = greedy_license_safe_selection(pressure_order, **common)
    persistence_max, _ = greedy_license_safe_selection(persistence_order, **common)
    initial_count = min(len(high_max), len(persistence_max))
    matched = None
    for common_flow_count in range(initial_count, 0, -1):
        high, high_size = greedy_license_safe_selection(
            pressure_order, **common, target_count=common_flow_count
        )
        persistence, persistence_size = greedy_license_safe_selection(
            persistence_order, **common, target_count=common_flow_count
        )
        matched = _demand_matched_random_selection(
            high, pair_candidates, window_records, window_path_maps,
            positions=event.window_indices, source_file=event.source_file,
            sequence_id=event.sequence_id, intensity=intensity,
            constraint_budget=constraint_budget,
            random_seed=random_seed + event.event_id,
        )
        if matched is not None:
            break
    if matched is None:
        raise RuntimeError("Could not construct a demand- and license-matched random control")
    random_selected, random_size = matched
    selections = {
        "PERSISTENCE_SLICE": (persistence, persistence_size),
        "HIGH_PRESSURE_SLICE": (high, high_size),
        "DEMAND_MATCHED_RANDOM_SLICE": (random_selected, random_size),
    }
    snapshots: dict[str, list[Snapshot]] = {}
    manifests: dict[str, dict[str, Any]] = {}
    for slice_type, (selected, size) in selections.items():
        snapshots[slice_type] = [
            _snapshot_from_raw(
                record, paths, selected, position=position, source_file=event.source_file,
                sequence_id=event.sequence_id, intensity=intensity, demand_scale=demand_scale,
            )
            for record, paths, position in zip(window_records, window_path_maps, event.window_indices)
        ]
        total_demand = [snapshot.total_demand for snapshot in snapshots[slice_type]]
        manifests[slice_type] = {
            "event_id": event.event_id, "transition_index": event.transition_index,
            "slice_type": slice_type, "selection_uses_solver_output": False,
            "selection_inputs": ["FlowSet demand", "candidate paths", "fixed capacity policy"],
            "selected_flows": [[src, dst] for src, dst in selected],
            "flow_count": size[0], "path_count": size[1], "edge_count": size[2],
            "variable_count": size[1], "constraint_count": size[3],
            "constraint_budget": constraint_budget, "total_demand": total_demand,
            "demand_scale": demand_scale,
            "evidence_label": "OFFICIAL_REAL_WORKLOAD" if demand_scale == 1.0 else "REAL_TOPOLOGY_REAL_TRAFFIC_SCALED_STRESS",
        }
    random_manifest = manifests["DEMAND_MATCHED_RANDOM_SLICE"]
    high_manifest = manifests["HIGH_PRESSURE_SLICE"]
    random_manifest["demand_match_exact_to_high_pressure"] = (
        random_manifest["total_demand"] == high_manifest["total_demand"]
    )
    random_manifest["path_count_ratio_to_high_pressure"] = (
        random_manifest["path_count"] / high_manifest["path_count"]
        if high_manifest["path_count"] else 1.0
    )
    high_pair = [
        snapshot for snapshot in snapshots["HIGH_PRESSURE_SLICE"]
        if snapshot.position in pair_positions
    ]
    severity = event_path_severity(high_pair)
    return snapshots, manifests, severity


def event_path_severity(pair: Sequence[Snapshot]) -> dict[str, Any]:
    """Measure route churn on the pair universe without using solver output."""

    if len(pair) != 2:
        raise ValueError("A transition pair must contain exactly two snapshots")
    previous, current = pair
    union = previous.paths | current.paths
    invalidated = previous.paths - current.paths
    born = current.paths - previous.paths
    flows = set(previous.demands) | set(current.demands)
    affected = {
        flow for flow in flows
        if {path for path in previous.paths if path[:2] == flow}
        != {path for path in current.paths if path[:2] == flow}
    }
    return {
        "candidate_path_survival_ratio": len(previous.paths & current.paths) / len(union) if union else 1.0,
        "candidate_path_invalidated_fraction": len(invalidated) / len(previous.paths) if previous.paths else 0.0,
        "candidate_path_birth_fraction": len(born) / len(current.paths) if current.paths else 0.0,
        "number_of_flows_affected_by_transition": len(affected),
        "flow_affected_ratio": len(affected) / len(flows) if flows else 0.0,
        "active_flow_churn": len(set(previous.demands) ^ set(current.demands)) / len(flows) if flows else 0.0,
        "total_demand_delta": current.total_demand - previous.total_demand,
    }


def slice_manifest_digest(manifest: Mapping[str, Any]) -> str:
    """Provide a stable resume identity for a deterministic slice manifest."""

    payload = json.dumps(dict(manifest), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("ascii")).hexdigest()
