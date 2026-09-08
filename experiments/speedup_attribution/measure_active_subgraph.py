"""Convenience entry point for active-subgraph attribution."""

from .benchmark_solver import solve_level
from .schemas import SafetyLimits


def measure(instance):
    return [solve_level(instance, level, SafetyLimits()) for level in ("L3", "L4")]
