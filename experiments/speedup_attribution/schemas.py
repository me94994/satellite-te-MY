"""Shared, immutable schemas for fair SaTE/Gurobi comparisons.

The canonical instance is the single source of truth for every formulation.
This prevents a parser or method-specific preprocessing step from silently
changing demands, capacities, paths, or topology.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import pickle
from typing import Dict, Iterable, Mapping, Sequence, Tuple

Edge = Tuple[int, int]
Pair = Tuple[int, int]
Path = Tuple[int, ...]


def _digest(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class BenchmarkInstance:
    """Canonical TE snapshot shared by dense, sparse, and neural consumers."""

    snapshot_id: str
    nodes: Tuple[int, ...]
    physical_edges: Tuple[Edge, ...]
    capacities: Mapping[Edge, float]
    demands: Mapping[Pair, float]
    candidate_paths: Mapping[Pair, Tuple[Path, ...]]
    evidence_label: str = "DIAGNOSTIC_CODE_NATIVE"

    def __post_init__(self) -> None:
        node_set = set(self.nodes)
        edge_set = set(self.physical_edges)
        if set(self.capacities) != edge_set:
            raise ValueError("capacities must cover exactly the physical edge set")
        if any(not math.isfinite(v) or v < 0 for v in self.capacities.values()):
            raise ValueError("capacities must be finite and non-negative")
        for pair, demand in self.demands.items():
            if pair[0] == pair[1] or not set(pair) <= node_set:
                raise ValueError(f"invalid demand pair: {pair}")
            if not math.isfinite(demand) or demand < 0:
                raise ValueError(f"invalid demand for {pair}")
        for pair, paths in self.candidate_paths.items():
            if pair[0] == pair[1] or not set(pair) <= node_set:
                raise ValueError(f"invalid path pair: {pair}")
            if len(paths) != len(set(paths)):
                raise ValueError(f"duplicate candidate path for {pair}")
            for path in paths:
                if len(path) < 2 or (path[0], path[-1]) != pair:
                    raise ValueError(f"path endpoints do not match {pair}: {path}")
                if any(edge not in edge_set for edge in zip(path[:-1], path[1:])):
                    raise ValueError(f"path uses a non-physical edge: {path}")

    @property
    def all_pairs(self) -> Tuple[Pair, ...]:
        return tuple(sorted(self.candidate_paths))

    @property
    def active_pairs(self) -> Tuple[Pair, ...]:
        return tuple(pair for pair in self.all_pairs if self.demands.get(pair, 0.0) > 0)

    @property
    def active_path_edges(self) -> Tuple[Edge, ...]:
        used = {
            edge
            for pair in self.active_pairs
            for path in self.candidate_paths[pair]
            for edge in zip(path[:-1], path[1:])
        }
        return tuple(sorted(used))

    def hashes(self) -> Dict[str, str]:
        """Return independent hashes used as the pre-comparison equality gate."""
        return {
            "demand_hash": _digest([[*p, self.demands.get(p, 0.0)] for p in self.all_pairs]),
            "path_hash": _digest([[*p, [list(x) for x in self.candidate_paths[p]]] for p in self.all_pairs]),
            "capacity_hash": _digest([[*e, self.capacities[e]] for e in sorted(self.physical_edges)]),
            "topology_hash": _digest([list(e) for e in sorted(self.physical_edges)]),
        }

    def paths_for_k(self, k: int) -> "BenchmarkInstance":
        """Select only existing paths; never manufacture or duplicate paths."""
        if k <= 0:
            raise ValueError("k must be positive")
        missing = [p for p, paths in self.candidate_paths.items() if len(paths) < k]
        if missing:
            raise ValueError(f"K={k} unavailable without fake paths for {len(missing)} pairs")
        return BenchmarkInstance(
            self.snapshot_id, self.nodes, self.physical_edges, dict(self.capacities),
            dict(self.demands), {p: paths[:k] for p, paths in self.candidate_paths.items()},
            self.evidence_label,
        )

    def representation_bytes(self) -> Dict[str, int]:
        """Measure serialized data representations, not solver runtime."""
        dense = {
            "demands": {p: self.demands.get(p, 0.0) for p in self.all_pairs},
            "paths": dict(self.candidate_paths),
        }
        sparse = {
            "demands": {p: self.demands[p] for p in self.active_pairs},
            "paths": {p: self.candidate_paths[p] for p in self.active_pairs},
        }
        return {
            "dense_pickle_bytes": len(pickle.dumps(dense, protocol=pickle.HIGHEST_PROTOCOL)),
            "sparse_pickle_bytes": len(pickle.dumps(sparse, protocol=pickle.HIGHEST_PROTOCOL)),
        }


@dataclass(frozen=True)
class SafetyLimits:
    """Fail-closed analytical guard evaluated before model construction."""

    max_vars: int = 2_000
    max_constraints: int = 2_000
    max_estimated_memory_gb: float = 0.25


def estimate_lp(instance: BenchmarkInstance, level: str) -> Dict[str, float]:
    dense = level in {"L0", "L1"}
    pairs = instance.all_pairs if dense else instance.active_pairs
    edges = instance.active_path_edges if level == "L4" else instance.physical_edges
    variables = sum(len(instance.candidate_paths[p]) for p in pairs)
    constraints = len(pairs) + len(edges)
    incidence_nz = sum(
        len(path) - 1 for p in pairs for path in instance.candidate_paths[p]
    )
    nonzeros = variables * 2 + incidence_nz
    # Conservative diagnostic estimate: sparse matrix coefficients plus Python/model overhead.
    estimated_bytes = 256 * (variables + constraints) + 32 * nonzeros
    return {
        "estimated_variables": variables,
        "estimated_constraints": constraints,
        "estimated_nonzeros": nonzeros,
        "estimated_memory_gb": estimated_bytes / 1e9,
    }


def enforce_safety(estimate: Mapping[str, float], limits: SafetyLimits) -> None:
    if (
        estimate["estimated_variables"] > limits.max_vars
        or estimate["estimated_constraints"] > limits.max_constraints
        or estimate["estimated_memory_gb"] > limits.max_estimated_memory_gb
    ):
        raise ResourceWarning("SKIPPED_RESOURCE_BOUND")


def assert_same_problem(*instances: BenchmarkInstance) -> None:
    """Comparison gate required before reporting any speed ratio."""
    if not instances:
        raise ValueError("at least one instance is required")
    expected = instances[0].hashes()
    if any(instance.hashes() != expected for instance in instances[1:]):
        raise AssertionError("SAME PROBLEM INSTANCE gate failed")


def topology_pruning_accounting(full_count: int, representative_count: int, per_label_s: float) -> Dict[str, float]:
    """Account topology pruning only as estimated aggregate offline work."""
    if full_count <= 0 or not 0 < representative_count <= full_count or per_label_s < 0:
        raise ValueError("invalid topology-pruning inputs")
    return {
        "full_training_samples": full_count,
        "representative_samples": representative_count,
        "sample_reduction_x": full_count / representative_count,
        "estimated_full_label_solver_work_s": full_count * per_label_s,
        "estimated_representative_label_solver_work_s": representative_count * per_label_s,
        "online_speedup_x": 1.0,
    }


def compatible_speedup(numerator_s: float, denominator_s: float, same_stage: bool) -> float:
    """Reject multiplication or division of incompatible stage metrics."""
    if not same_stage:
        raise ValueError("incompatible stage-local metrics cannot form a speedup")
    if numerator_s < 0 or denominator_s <= 0:
        raise ValueError("invalid timing")
    return numerator_s / denominator_s
