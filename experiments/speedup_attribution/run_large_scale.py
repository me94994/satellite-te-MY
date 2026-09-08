"""Run the K=10 diagnosis sequentially with one process at a time."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def _run(module: str, arguments=(), check: bool = True) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    # The experiment and solver use one process/thread for reproducible timing.
    env.update({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"})
    return subprocess.run([sys.executable, "-m", module, *arguments], check=check, env=env)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Use a small smoke scale while retaining all evidence gates.")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    root = Path("output/speedup_attribution/large_scale")
    _run("experiments.speedup_attribution.audit_large_scale")
    solver_args = ["--repeats", "1"] if args.quick else ["--repeats", "3"]
    if args.quick:
        solver_args.extend(["--families", "coupled", "--nodes", "16", "32"])
    if args.resume:
        solver_args.append("--resume")
    _run("experiments.speedup_attribution.benchmark_large_scale", solver_args)
    online_args = ["--nodes", "16", "32", "--repeats", "2"] if args.quick else ["--nodes", "66", "128", "192", "--repeats", "10"]
    _run("experiments.speedup_attribution.benchmark_crossover", online_args)
    training_args = ["--samples", "10", "--epochs", "1", "--batch-size", "2"] if args.quick else ["--samples", "40", "--epochs", "2", "--batch-size", "4"]
    trained = _run("experiments.speedup_attribution.train_k10_diagnostic", training_args, check=False)
    training_path = root / "k10_training.json"
    checkpoint_args = []
    if trained.returncode == 0 and training_path.is_file():
        training = json.loads(training_path.read_text(encoding="utf-8"))
        if training.get("status") == "DIAGNOSTIC_K10_TRAINED":
            checkpoint_args = ["--checkpoint", training["checkpoint"], "--checkpoint-label", "DIAGNOSTIC_K10_TRAINED"]
    sate_args = checkpoint_args
    if args.quick:
        sate_args.extend(["--nodes", "16", "32", "--active-flows", "32", "--warmup", "2", "--repeats-small", "3", "--repeats-large", "3"])
    if args.resume:
        sate_args.append("--resume")
    _run("experiments.speedup_attribution.benchmark_k10", sate_args)
    _run("experiments.speedup_attribution.analyze_crossover")


if __name__ == "__main__":
    main()
