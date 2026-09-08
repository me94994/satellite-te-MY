"""Reduced diagnostic scaling runner with incremental output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .benchmark_large_scale import controlled_stress_instance, constellation_instance
from .benchmark_solver import solve_level
from .schemas import SafetyLimits, assert_k5_prefix_of_k10


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes", type=int, nargs="+", default=[8, 12, 16, 20])
    parser.add_argument("--active-flows", type=int, nargs="+", default=None)
    parser.add_argument("--k", type=int, nargs="+", choices=(5, 10), default=[5, 10])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--regime", choices=("uncongested", "congested"), default="uncongested")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("output/speedup_attribution/scaling.jsonl"))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() and not args.resume:
        args.output.unlink()
    completed = set()
    if args.resume and args.output.is_file():
        completed = {
            (row["topology_nodes"], row["active_flows"], row["k"], row["level"], row["repeat"])
            for row in (json.loads(line) for line in args.output.read_text(encoding="utf-8").splitlines() if line.strip())
        }
    flow_counts = args.active_flows or args.nodes
    if len(flow_counts) not in (1, len(args.nodes)):
        parser.error("--active-flows must contain one value or match --nodes")
    for index, nodes in enumerate(args.nodes):
        active_flows = flow_counts[0] if len(flow_counts) == 1 else flow_counts[index]
        if args.regime == "congested":
            k10, proxy = controlled_stress_instance(nodes, active_flows)
        else:
            k10, proxy = constellation_instance(nodes, active_flows, 10), {}
        k5 = k10.paths_for_k(5)
        assert_k5_prefix_of_k10(k5, k10)
        for k in args.k:
            instance = k5 if k == 5 else k10
            for repeat in range(args.repeats):
                for level in ("L3", "L4"):
                    key = (nodes, active_flows, k, level, repeat)
                    if key in completed:
                        continue
                    row = {**solve_level(instance, level, SafetyLimits()), **proxy, "topology_nodes": nodes, "active_flows": active_flows, "k": k, "regime": args.regime, "repeat": repeat}
                    with args.output.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(row, sort_keys=True) + "\n")
                    print(json.dumps({"nodes": nodes, "active_flows": active_flows, "k": k, "level": level, "status": row["status"]}))


if __name__ == "__main__":
    main()
