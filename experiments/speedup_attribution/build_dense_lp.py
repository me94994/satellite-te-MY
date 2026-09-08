"""Dense LP entry points; construction always passes the analytical guard."""

from .benchmark_solver import build_model
from .schemas import SafetyLimits, enforce_safety, estimate_lp


def build_dense(instance, presolve: bool, limits: SafetyLimits = SafetyLimits()):
    """Build L0/L1 only after fail-closed resource estimation."""
    level = "L1" if presolve else "L0"
    enforce_safety(estimate_lp(instance, level), limits)
    return build_model(instance, level)
