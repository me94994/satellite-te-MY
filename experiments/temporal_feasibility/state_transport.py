"""Deterministic optimizer-state transport and feasibility repair."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Any, Mapping

import numpy as np

from .sequence_schema import EdgeKey, FlowKey, PathKey, Snapshot
from .sequential_lp import AdvancedStart, LPUniverse, SolveState, capacities_for_snapshot, CapacityPolicy


def direct_previous_path_copy(previous: SolveState, current: Snapshot, universe: LPUniverse) -> dict[PathKey, float]:
    """Copy persistent semantic paths, drop deleted paths, and zero new paths."""

    return {
        path: float(previous.path_flow.get(path, 0.0)) if path in current.paths else 0.0
        for path in universe.paths
    }


def repair_path_flow(
    initial: Mapping[PathKey, float], current: Snapshot, universe: LPUniverse,
    policy: CapacityPolicy,
) -> tuple[dict[PathKey, float], dict[str, float]]:
    """Apply demand, capacity, then availability repair using no future optimum."""

    values = {path: max(0.0, float(initial.get(path, 0.0))) for path in universe.paths}
    paths_by_flow: defaultdict[FlowKey, list[PathKey]] = defaultdict(list)
    paths_by_edge: defaultdict[EdgeKey, list[PathKey]] = defaultdict(list)
    for path in universe.paths:
        paths_by_flow[(path[0], path[1])].append(path)
        for edge in universe.path_to_edges[path]:
            paths_by_edge[edge].append(path)

    # Step 1: proportionally enforce each flow's current demand upper bound.
    for flow, paths in paths_by_flow.items():
        load = sum(values[path] for path in paths)
        demand = float(current.demands.get(flow, 0.0))
        if load > demand and load > 0.0:
            factor = demand / load
            for path in paths:
                values[path] *= factor

    # Step 2: only reductions are applied, so a single deterministic edge pass
    # cannot re-violate a previously repaired capacity constraint.
    capacities = capacities_for_snapshot(current, universe, policy)
    for edge in universe.edges:
        paths = paths_by_edge[edge]
        load = sum(values[path] for path in paths)
        capacity = capacities[edge]
        if load > capacity and load > 0.0:
            factor = capacity / load
            for path in paths:
                values[path] *= factor

    # Step 3: future/deleted paths are made unavailable exactly as in the LP.
    for path in universe.paths:
        if path not in current.paths:
            values[path] = 0.0
    audit = path_flow_feasibility(values, current, universe, policy)
    if max(audit.values(), default=0.0) > 1e-7:
        raise AssertionError(f"Deterministic path-flow repair is infeasible: {audit}")
    return values, audit


def path_flow_feasibility(
    values: Mapping[PathKey, float], current: Snapshot, universe: LPUniverse,
    policy: CapacityPolicy,
) -> dict[str, float]:
    """Audit all primal constraints without invoking a solver."""

    flow_load: defaultdict[FlowKey, float] = defaultdict(float)
    edge_load: defaultdict[EdgeKey, float] = defaultdict(float)
    availability_violation = 0.0
    nonnegative_violation = 0.0
    for path, raw_value in values.items():
        value = float(raw_value)
        nonnegative_violation = max(nonnegative_violation, -value)
        flow_load[(path[0], path[1])] += value
        for edge in universe.path_to_edges[path]:
            edge_load[edge] += value
        if path not in current.paths:
            availability_violation = max(availability_violation, value)
    capacities = capacities_for_snapshot(current, universe, policy)
    return {
        "max_nonnegative_violation": max(0.0, nonnegative_violation),
        "max_demand_violation": max(
            [0.0] + [flow_load[flow] - current.demands.get(flow, 0.0) for flow in universe.flows]
        ),
        "max_capacity_violation": max(
            [0.0] + [edge_load[edge] - capacities[edge] for edge in universe.edges]
        ),
        "max_path_availability_violation": max(0.0, availability_violation),
    }


def transport_edge_state(
    previous_values: Mapping[EdgeKey, float], previous_edges: frozenset[EdgeKey],
    current_edges: frozenset[EdgeKey], new_edge_policy: str,
) -> dict[EdgeKey, float]:
    """Copy persistent edge state and initialize new edges deterministically."""

    if new_edge_policy not in {"zero", "neighbor_mean", "global_mean", "global_median", "neighbor_median"}:
        raise ValueError(f"Unsupported new-edge policy: {new_edge_policy}")
    old = [float(previous_values.get(edge, 0.0)) for edge in sorted(previous_edges)]
    global_value = (
        float(np.median(old)) if new_edge_policy.endswith("median") else float(np.mean(old))
    ) if old else 0.0
    result = {
        edge: float(previous_values.get(edge, 0.0))
        for edge in sorted(previous_edges & current_edges)
    }
    for edge in sorted(current_edges - previous_edges):
        neighbors = [
            float(value) for candidate, value in previous_values.items()
            if edge[0] in candidate or edge[1] in candidate
        ]
        if new_edge_policy == "zero":
            result[edge] = 0.0
        elif new_edge_policy.startswith("neighbor") and neighbors:
            result[edge] = (
                float(np.median(neighbors)) if new_edge_policy.endswith("median")
                else float(np.mean(neighbors))
            )
        else:
            result[edge] = global_value
    return result


def transported_advanced_start(
    previous: SolveState, current: Snapshot, universe: LPUniverse, policy: CapacityPolicy,
) -> tuple[AdvancedStart, dict[str, Any]]:
    """Construct repaired primal and semantic dual starts for the next LP."""

    direct = direct_previous_path_copy(previous, current, universe)
    repaired, feasibility = repair_path_flow(direct, current, universe, policy)
    persistent_edges = frozenset(
        edge for edge in universe.edges
        if previous.edge_capacity.get(edge, 0.0) > 0.0 and edge in current.edges
    )
    capacity_dual = {
        edge: float(previous.raw_capacity_pi.get(edge, 0.0)) if edge in persistent_edges else 0.0
        for edge in universe.edges
    }
    demand_dual = {
        flow: float(previous.raw_demand_pi.get(flow, 0.0))
        if flow in current.demands and flow in previous.raw_demand_pi else 0.0
        for flow in universe.flows
    }
    availability_dual = {
        path: float(previous.raw_availability_pi.get(path, 0.0))
        if path in current.paths and path in previous.raw_availability_pi else 0.0
        for path in universe.paths
    }
    return AdvancedStart(repaired, capacity_dual, demand_dual, availability_dual), {
        "feasibility": feasibility,
        "persistent_primal_path_count": sum(path in current.paths and previous.path_flow.get(path, 0.0) != 0.0 for path in universe.paths),
        "persistent_dual_edge_count": len(persistent_edges),
        "dual_uninformative": not any(abs(value) > 1e-12 for value in previous.congestion_price.values()),
    }


def zero_advanced_start(universe: LPUniverse) -> AdvancedStart:
    """Build the complete all-zero PStart/DStart control using the same API."""

    return AdvancedStart(
        path_flow={path: 0.0 for path in universe.paths},
        capacity_dual={edge: 0.0 for edge in universe.edges},
        demand_dual={flow: 0.0 for flow in universe.flows},
        availability_dual={path: 0.0 for path in universe.paths},
    )


def canonical_secondary_costs(universe: LPUniverse) -> dict[PathKey, float]:
    """Return time-independent hop costs with a stable semantic tie-break.

    There is deliberately no previous-solution argument: the diagnostic cannot
    manufacture temporal continuity by minimizing distance to the prior state.
    """

    result: dict[PathKey, float] = {}
    for path in universe.paths:
        digest = hashlib.sha256(repr(path).encode("ascii")).digest()
        tie_break = int.from_bytes(digest[:8], "big") / 2**64
        result[path] = float(len(path[2]) - 1) + 1e-9 * tie_break
    return result
