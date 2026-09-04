"""Finite-difference validation of capacity-dual sign conventions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linprog


def _solve_scipy(capacity: float, demand: float) -> tuple[float, float]:
    result = linprog(
        c=np.asarray([-1.0]),
        A_ub=np.asarray([[1.0], [1.0]]),
        b_ub=np.asarray([capacity, demand]),
        bounds=(0.0, None),
        method="highs-ds",
    )
    if not result.success:
        raise RuntimeError(f"SciPy calibration LP failed: {result.message}")
    return float(-result.fun), float(result.ineqlin.marginals[0])


def _solve_gurobi(capacity: float, demand: float) -> tuple[float, float]:
    try:
        import gurobipy as gp
    except ImportError as exc:
        raise RuntimeError("gurobipy is not installed") from exc
    try:
        model = gp.Model("dual_sign_calibration")
        model.Params.OutputFlag = 0
        model.Params.Threads = 1
        model.Params.Method = 1
        model.Params.Seed = 42
        x = model.addVar(lb=0.0, name="x")
        model.setObjective(x, gp.GRB.MAXIMIZE)
        cap = model.addConstr(x <= capacity, name="cap[0,1]")
        model.addConstr(x <= demand, name="dem[0,1]")
        model.optimize()
    except gp.GurobiError as exc:
        raise RuntimeError(f"Gurobi calibration/license failure: {exc}") from exc
    if model.Status != gp.GRB.OPTIMAL:
        raise RuntimeError(f"Gurobi calibration status is not OPTIMAL: {model.Status}")
    return float(model.ObjVal), float(cap.Pi)


def calibrate_dual_sign(backend: str, epsilon: float = 1e-4) -> dict[str, Any]:
    """Validate zero-slack and unique-bottleneck prices by finite difference."""

    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if backend == "scipy":
        solve = _solve_scipy
    elif backend == "gurobi":
        solve = _solve_gurobi
    else:
        raise ValueError(f"Unsupported backend: {backend}")

    loose_obj, loose_raw_pi = solve(capacity=10.0, demand=1.0)
    bottleneck_obj, bottleneck_raw_pi = solve(capacity=3.0, demand=10.0)
    perturbed_obj, _ = solve(capacity=3.0 + epsilon, demand=10.0)
    finite_difference = (perturbed_obj - bottleneck_obj) / epsilon
    if abs(loose_raw_pi) > 1e-7:
        raise AssertionError(f"Loose capacity should have zero shadow price, got {loose_raw_pi}")
    if abs(finite_difference - 1.0) > 1e-6:
        raise AssertionError(f"Unexpected bottleneck finite difference: {finite_difference}")
    if abs(bottleneck_raw_pi) <= 1e-12:
        raise AssertionError("Bottleneck raw Pi is zero; sign cannot be calibrated")
    sign = 1.0 if bottleneck_raw_pi * finite_difference > 0 else -1.0
    price = sign * bottleneck_raw_pi
    if abs(price - finite_difference) > 1e-6:
        raise AssertionError(f"Dual/finite-difference mismatch: {price} vs {finite_difference}")
    return {
        "backend": backend,
        "epsilon": epsilon,
        "loose_capacity_objective": loose_obj,
        "loose_capacity_raw_pi": loose_raw_pi,
        "bottleneck_objective": bottleneck_obj,
        "bottleneck_raw_pi": bottleneck_raw_pi,
        "perturbed_objective": perturbed_obj,
        "finite_difference_price": finite_difference,
        "sign_multiplier": sign,
        "congestion_price": price,
        "status": "PASS",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("scipy", "gurobi"), default="scipy")
    parser.add_argument("--epsilon", type=float, default=1e-4)
    parser.add_argument("--output")
    args = parser.parse_args()
    result = calibrate_dual_sign(args.backend, args.epsilon)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8-sig") as handle:
            json.dump(result, handle, indent=2, ensure_ascii=False, allow_nan=False)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
