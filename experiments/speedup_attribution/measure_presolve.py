"""Convenience entry point for dense presolve attribution."""

from .benchmark_solver import diagnostic_instance, solve_level
from .schemas import SafetyLimits


def measure(nodes=16, active_flows=20, k=5):
    instance = diagnostic_instance(nodes, active_flows, k)
    limits = SafetyLimits()
    return [solve_level(instance, level, limits) for level in ("L0", "L1", "L3")]
