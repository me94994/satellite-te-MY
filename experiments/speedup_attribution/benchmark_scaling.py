"""Reduced diagnostic scaling runner with incremental output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .benchmark_solver import diagnostic_instance, solve_level
from .schemas import SafetyLimits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes", type=int, nargs="+", default=[8, 12, 16, 20])
    parser.add_argument("--output", type=Path, default=Path("output/speedup_attribution/scaling.jsonl"))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for nodes in args.nodes:
        # Vary active-flow count with scale so runtime-vs-flow plots are identifiable.
        instance = diagnostic_instance(nodes=nodes, active_flows=nodes, k=5)
        for level in ("L3", "L4"):
            row = solve_level(instance, level, SafetyLimits())
            with args.output.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
            print(json.dumps({"nodes": nodes, "level": level, "status": row["status"]}))


if __name__ == "__main__":
    main()
