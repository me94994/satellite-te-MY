"""Safely probe Gurobi availability without exposing license credentials."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any


SECRET_ENV_NAMES = ("WLSACCESSID", "WLSSECRET", "LICENSEID")


def probe_gurobi(check_size_limit: bool = True) -> dict[str, Any]:
    """Solve a tiny LP and conservatively detect a package-size restriction."""

    result: dict[str, Any] = {
        "host": platform.node(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "grb_license_file_set": bool(os.environ.get("GRB_LICENSE_FILE")),
        "gurobi_home_set": bool(os.environ.get("GUROBI_HOME")),
        "wls_environment_present": any(bool(os.environ.get(name)) for name in SECRET_ENV_NAMES),
        "secret_values_recorded": False,
    }
    try:
        import gurobipy as gp
    except ImportError as exc:
        result.update({"available": False, "error_type": type(exc).__name__, "error": str(exc)})
        return result

    result["gurobipy_version"] = ".".join(map(str, gp.gurobi.version()))
    model = None
    try:
        model = gp.Model("temporal_two_variable_probe")
        model.Params.OutputFlag = 0
        model.Params.Threads = 1
        model.Params.Method = 1
        model.Params.Seed = 42
        x = model.addVar(lb=0.0, name="x")
        y = model.addVar(lb=0.0, name="y")
        model.setObjective(x + 2.0 * y, gp.GRB.MAXIMIZE)
        model.addConstr(x + y <= 1.0, name="capacity")
        model.optimize()
        result.update(
            {
                "available": model.Status == gp.GRB.OPTIMAL,
                "small_lp_status": int(model.Status),
                "small_lp_objective": float(model.ObjVal) if model.SolCount else None,
                "lp_warm_start_param_info": [str(value) for value in model.getParamInfo("LPWarmStart")],
            }
        )
    except gp.GurobiError as exc:
        result.update({"available": False, "error_type": type(exc).__name__, "error": str(exc)})
        return result
    finally:
        if model is not None:
            model.dispose()

    result["license_size_restriction"] = "NOT_PROBED"
    if check_size_limit and result["available"]:
        size_model = None
        try:
            size_model = gp.Model("temporal_size_limit_probe")
            size_model.Params.OutputFlag = 0
            size_model.Params.Threads = 1
            variables = size_model.addVars(2001, lb=0.0, ub=1.0)
            size_model.setObjective(gp.quicksum(variables.values()), gp.GRB.MAXIMIZE)
            size_model.optimize()
            result["license_size_restriction"] = "UNRESTRICTED_FOR_2001_VARIABLE_LINEAR_PROBE"
        except gp.GurobiError as exc:
            # Error text contains no credential; code 10010 is the documented size-limit failure.
            result["license_size_restriction"] = (
                "RESTRICTED_SIZE_LIMIT_CONFIRMED" if exc.errno == 10010
                else f"SIZE_PROBE_ERROR_{exc.errno}"
            )
        finally:
            if size_model is not None:
                size_model.dispose()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output")
    parser.add_argument("--skip-size-limit-probe", action="store_true")
    args = parser.parse_args()
    result = probe_gurobi(not args.skip_size_limit_probe)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8-sig") as handle:
            json.dump(result, handle, indent=2, ensure_ascii=False, allow_nan=False)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result.get("available") else 2


if __name__ == "__main__":
    raise SystemExit(main())
