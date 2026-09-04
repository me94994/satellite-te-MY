"""Semantic alignment and distance metrics across changing snapshots."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Hashable, Mapping, Sequence, TypeVar

import numpy as np
from scipy.stats import rankdata

from .sequence_schema import EdgeKey, PathKey, Snapshot

KeyT = TypeVar("KeyT", bound=Hashable)


def jaccard(left: set[KeyT] | frozenset[KeyT], right: set[KeyT] | frozenset[KeyT]) -> float:
    """Jaccard similarity, with two empty sets treated as identical."""

    union = set(left).union(right)
    return 1.0 if not union else len(set(left).intersection(right)) / len(union)


def normalized_l1(left: Mapping[KeyT, float], right: Mapping[KeyT, float], keys: Sequence[KeyT] | None = None) -> float:
    """Symmetric L1 drift normalized by the mean vector mass."""

    aligned = list(keys) if keys is not None else sorted(set(left).union(right), key=repr)
    numerator = sum(abs(float(left.get(key, 0.0)) - float(right.get(key, 0.0))) for key in aligned)
    left_mass = sum(abs(float(left.get(key, 0.0))) for key in aligned)
    right_mass = sum(abs(float(right.get(key, 0.0))) for key in aligned)
    denominator = max(0.5 * (left_mass + right_mass), 1e-12)
    return numerator / denominator


def normalized_l2(left: Mapping[KeyT, float], right: Mapping[KeyT, float], keys: Sequence[KeyT] | None = None) -> float:
    """Symmetric Euclidean drift normalized by the mean vector norm."""

    aligned = list(keys) if keys is not None else sorted(set(left).union(right), key=repr)
    lv = np.asarray([float(left.get(key, 0.0)) for key in aligned], dtype=float)
    rv = np.asarray([float(right.get(key, 0.0)) for key in aligned], dtype=float)
    denominator = max(0.5 * (float(np.linalg.norm(lv)) + float(np.linalg.norm(rv))), 1e-12)
    return float(np.linalg.norm(lv - rv) / denominator)


def correlations(left: Mapping[KeyT, float], right: Mapping[KeyT, float], keys: Sequence[KeyT]) -> tuple[float, float]:
    """Return Pearson and Spearman correlations, or NaN when undefined."""

    if len(keys) < 2:
        return math.nan, math.nan
    lv = np.asarray([float(left.get(key, 0.0)) for key in keys], dtype=float)
    rv = np.asarray([float(right.get(key, 0.0)) for key in keys], dtype=float)
    if np.ptp(lv) <= 1e-12 or np.ptp(rv) <= 1e-12:
        return math.nan, math.nan
    pearson = float(np.corrcoef(lv, rv)[0, 1])
    spearman = float(np.corrcoef(rankdata(lv), rankdata(rv))[0, 1])
    return pearson, spearman


def path_birth_death_mass(
    previous: Mapping[PathKey, float], current: Mapping[PathKey, float]
) -> tuple[float, float]:
    """Report mass on paths absent from the other snapshot."""

    previous_keys, current_keys = set(previous), set(current)
    birth = sum(float(current[key]) for key in current_keys - previous_keys)
    death = sum(float(previous[key]) for key in previous_keys - current_keys)
    return birth, death


def topology_delta(previous: Snapshot, current: Snapshot) -> tuple[set[EdgeKey], set[EdgeKey], set[EdgeKey]]:
    """Return persistent, new, and deleted directed edge sets."""

    persistent = set(previous.graph_edges).intersection(current.graph_edges)
    return persistent, set(current.graph_edges) - set(previous.graph_edges), set(previous.graph_edges) - set(current.graph_edges)


def transport_edge_prices(
    previous_prices: Mapping[EdgeKey, float],
    previous_edges: set[EdgeKey] | frozenset[EdgeKey],
    current_edges: set[EdgeKey] | frozenset[EdgeKey],
    new_edge_policy: str,
) -> dict[EdgeKey, float]:
    """Deterministically transport prices across persistent/new/deleted edges."""

    if new_edge_policy not in {"zero", "global_median", "neighbor_median"}:
        raise ValueError(f"Unsupported new-edge policy: {new_edge_policy}")
    persistent = set(previous_edges).intersection(current_edges)
    transported = {edge: float(previous_prices.get(edge, 0.0)) for edge in persistent}
    old_values = [float(previous_prices.get(edge, 0.0)) for edge in previous_edges]
    global_median = float(np.median(old_values)) if old_values else 0.0
    new_edges = set(current_edges) - set(previous_edges)
    for edge in sorted(new_edges):
        if new_edge_policy == "zero":
            transported[edge] = 0.0
        elif new_edge_policy == "global_median":
            transported[edge] = global_median
        else:
            u, v = edge
            neighbors = [
                value
                for candidate, value in previous_prices.items()
                if u in candidate or v in candidate
            ]
            transported[edge] = float(np.median(neighbors)) if neighbors else global_median
    return transported


def initialization_baselines(
    current_edges: set[EdgeKey] | frozenset[EdgeKey],
    reference_prices: Mapping[EdgeKey, float],
    seed: int,
) -> dict[str, dict[EdgeKey, float]]:
    """Create deterministic zero/one/random controls for transport analysis."""

    rng = random.Random(seed)
    scale_values = [abs(float(value)) for value in reference_prices.values()]
    scale = float(np.median(scale_values)) if scale_values else 1.0
    scale = max(scale, 1e-12)
    edges = sorted(current_edges)
    return {
        "zero": {edge: 0.0 for edge in edges},
        "one": {edge: 1.0 for edge in edges},
        "random": {edge: rng.random() * 2.0 * scale for edge in edges},
    }


def classify_change(previous: Snapshot, current: Snapshot, tolerance: float = 1e-12) -> str:
    """Separate topology-only, traffic-only, both, and unchanged transitions."""

    topology_changed = previous.graph_edges != current.graph_edges
    traffic_changed = normalized_l1(previous.demands, current.demands) > tolerance
    if topology_changed and traffic_changed:
        return "both_change"
    if topology_changed:
        return "topology_only"
    if traffic_changed:
        return "traffic_only"
    return "unchanged"


def binding_edges(loads: Mapping[EdgeKey, float], capacities: Mapping[EdgeKey, float], tolerance: float = 1e-7) -> set[EdgeKey]:
    """Identify positive-capacity edges whose residual slack is within tolerance."""

    result = set()
    for edge, capacity in capacities.items():
        if capacity > 0 and capacity - float(loads.get(edge, 0.0)) <= tolerance * max(1.0, capacity):
            result.add(edge)
    return result
