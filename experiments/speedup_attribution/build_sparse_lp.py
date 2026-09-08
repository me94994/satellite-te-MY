"""Exact active-flow and active-subgraph LP entry points."""

from .benchmark_solver import build_model
from .schemas import SafetyLimits, enforce_safety, estimate_lp


def build_sparse(instance, active_subgraph: bool, presolve: bool = True, limits: SafetyLimits = SafetyLimits()):
    """Build L2/L3/L4 without changing demand, capacity, paths, or objective."""
    level = "L4" if active_subgraph else ("L3" if presolve else "L2")
    enforce_safety(estimate_lp(instance, level), limits)
    return build_model(instance, level)
