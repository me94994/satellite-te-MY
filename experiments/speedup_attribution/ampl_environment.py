"""AMPL 环境审计与许可证安全门槛。

许可证标识符只允许从环境变量传给子进程；本模块从不返回、记录或格式化其值。
"""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


OUTPUT_ROOT = Path("output/speedup_attribution/ampl_full_scale")


def license_env_present() -> bool:
    """只报告许可证变量是否存在，不暴露其内容。"""

    return bool(os.environ.get("AMPL_LICENSE_UUID"))


def activate_from_environment() -> dict[str, Any]:
    """静默激活 AMPL；丢弃可能包含敏感信息的全部子进程输出。"""

    token = os.environ.get("AMPL_LICENSE_UUID")
    if not token:
        return {"status": "BLOCKED_AMPL_LICENSE_ENV_NOT_SET", "return_code": None}
    completed = subprocess.run(
        [sys.executable, "-m", "amplpy.modules", "activate", token],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return {
        "status": "ACTIVATED" if completed.returncode == 0 else "ACTIVATION_FAILED",
        "return_code": completed.returncode,
    }


def installed_modules() -> list[str]:
    """返回已安装模块名；失败时返回空列表，避免泄漏底层诊断文本。"""

    try:
        from amplpy import modules

        return sorted(modules.installed())
    except Exception:
        return []


def _hardware() -> dict[str, Any]:
    result: dict[str, Any] = {
        "os": platform.platform(),
        "cpu": platform.processor() or "NOT_AVAILABLE",
        "logical_cores": os.cpu_count(),
        "physical_cores": "NOT_AVAILABLE",
        "ram_bytes": "NOT_AVAILABLE",
        "gpu": "NOT_AVAILABLE",
        "pytorch": "NOT_AVAILABLE",
        "cuda": "NOT_AVAILABLE",
    }
    try:
        import psutil

        result["physical_cores"] = psutil.cpu_count(logical=False)
        result["ram_bytes"] = psutil.virtual_memory().total
    except Exception:
        pass
    try:
        import torch

        result["pytorch"] = torch.__version__
        result["cuda"] = torch.version.cuda
        if torch.cuda.is_available():
            result["gpu"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    return result


def safe_ampl_probe() -> dict[str, Any]:
    """实例化 AMPL，但仅返回布尔值，不传播可能含许可证正文的异常。"""

    try:
        from amplpy import AMPL

        ampl = AMPL()
        ampl.close()
        return {"AMPL_instantiation_success": True, "failure_type": None}
    except Exception as exc:
        return {"AMPL_instantiation_success": False, "failure_type": type(exc).__name__}


def smoke_test_sizes(sizes: tuple[int, ...] = (5000, 10000)) -> list[dict[str, Any]]:
    """用单线程 Gurobi 求解大于 restricted 限制的 LP；逐点保留受控状态。"""

    rows: list[dict[str, Any]] = []
    try:
        from amplpy import AMPL
    except Exception as exc:
        return [{"variables": size, "status": "FAILED", "failure_type": type(exc).__name__} for size in sizes]
    for size in sizes:
        row: dict[str, Any] = {"variables": size, "status": "STARTED"}
        rows.append(row)
        ampl = None
        try:
            ampl = AMPL()
            ampl.eval(
                f"set I := 1..{size}; var x {{I}} >= 0; "
                "maximize total: sum {i in I} x[i]; "
                "subject to upper {i in I}: x[i] <= 1;"
            )
            # 单线程保证与第二轮边界一致，也避免并行计时噪声。
            ampl.setOption("gurobi_options", "outlev=0 threads=1")
            start = time.perf_counter()
            ampl.solve(solver="gurobi", verbose=False)
            row.update(
                status="COMPLETED" if str(ampl.getValue("solve_result")) == "solved" else "FAILED",
                objective=float(ampl.getObjective("total").value()),
                wall_s=time.perf_counter() - start,
            )
        except Exception as exc:
            row.update(status="FAILED", failure_type=type(exc).__name__)
        finally:
            if ampl is not None:
                try:
                    ampl.close()
                except Exception:
                    pass
    return rows


def collect_environment(run_smoke: bool = False) -> dict[str, Any]:
    """构造不含许可证标识符的环境记录。"""

    try:
        amplpy_version = importlib.metadata.version("amplpy")
    except importlib.metadata.PackageNotFoundError:
        amplpy_version = "NOT_INSTALLED"
    modules = installed_modules()
    probe = safe_ampl_probe()
    result = {
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "amplpy_version": amplpy_version,
        "AMPL_available": amplpy_version != "NOT_INSTALLED" and "base" in modules,
        "installed_modules": modules,
        "gurobi_solver_module_available": "gurobi" in modules,
        "license_env_present": license_env_present(),
        **probe,
        **_hardware(),
    }
    if run_smoke:
        result["size_smoke_tests"] = smoke_test_sizes()
        result["AMPL_Gurobi_solve_success"] = all(
            row["status"] == "COMPLETED" for row in result["size_smoke_tests"]
        )
    else:
        result["AMPL_Gurobi_solve_success"] = "NOT_EVALUATED"
    return result


def write_environment(result: dict[str, Any], path: Path = OUTPUT_ROOT / "environment.json") -> None:
    """原子写入环境证据，且防御性拒绝任何 UUID 字段。"""

    serialized = json.dumps(result, indent=2, sort_keys=True)
    if "AMPL_LICENSE_UUID" in serialized:
        raise ValueError("LICENSE_IDENTIFIER_PERSISTENCE_REJECTED")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(serialized + "\n", encoding="utf-8")
    temporary.replace(path)
