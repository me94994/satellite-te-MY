"""Run the attribution workflow sequentially with no parallel workers."""

from __future__ import annotations

import subprocess
import sys


MODULES = (
    "experiments.speedup_attribution.audit_configuration",
    "experiments.speedup_attribution.measure_representation",
    "experiments.speedup_attribution.benchmark_solver",
    "experiments.speedup_attribution.analyze_results",
    "experiments.speedup_attribution.benchmark_commercial",
    "experiments.speedup_attribution.benchmark_sate",
    "experiments.speedup_attribution.benchmark_scaling",
    "experiments.speedup_attribution.plot_results",
)


def main():
    for module in MODULES:
        # Sequential subprocesses isolate Gurobi/CUDA peak state without concurrency.
        subprocess.run([sys.executable, "-m", module], check=True)


if __name__ == "__main__":
    main()
