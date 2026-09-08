"""Fail-closed paper/code configuration audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re


PAPER_CONFIG = {
    "candidate_paths_per_sd_pair": 10,
    "traffic_representation": "4236x4236 dense matrix before pruning; nonzero active flows after pruning",
    "objective": "maximize total throughput",
    "capacities": "200 Mbps inter-satellite/cross-shell; paper also states 50 Mbps access connections",
    "topology": "Starlink 4236; Iridium 66; mid-size 396 and 1584",
    "dataset": "10000 snapshots, 4:1 train:test; 8000 training reference vs up to 512 representatives",
    "solver_settings": "NOT_IDENTIFIABLE_FROM_CURRENT_ARTIFACT",
}


def audit(repo: Path) -> dict:
    common = (repo / "run/_common.py").read_text(encoding="utf-8-sig")
    baseline_common = (repo / "baselines/run/_common.py").read_text(encoding="utf-8-sig")
    baseline_lp = (repo / "baselines/run/lp.py").read_text(encoding="utf-8-sig")
    main_default = int(re.search(r"ARG_PATH_NUM\s*=\s*(\d+)", common).group(1))
    baseline_default = int(re.search(r"ARG_PATH_NUM\s*=\s*(\d+)", baseline_common).group(1))
    hardcoded = [int(x) for x in re.findall(r"num_paths\s*=\s*(\d+)", baseline_lp)]
    code = {
        "candidate_paths_sate": main_default,
        "candidate_paths_gurobi_default": baseline_default,
        "candidate_paths_gurobi_hardcoded": sorted(set(hardcoded)),
        "traffic_representation": "adapter list-of-dict with active tm entries and paths keyed by represented pairs",
        "objective": "Objective total_flow by default",
        "capacities": "adapter/mode dependent; Starlink OrbitParams use 200/800/800 in run/spaceTE.py",
        "topology": "dataset-selected",
        "dataset": "external input not present in current worktree",
        "solver_settings": "lp_solver Method.CONCURRENT default; thread count optional",
    }
    mismatch = PAPER_CONFIG["candidate_paths_per_sd_pair"] != main_default or main_default != baseline_default
    return {
        "status": "PAPER_CONFIG_CODE_CONFIG_MISMATCH" if mismatch else "PAPER_CONFIG_CODE_CONFIG_MATCH",
        "paper_config": PAPER_CONFIG,
        "code_config": code,
        "formal_comparison_config": "CODE_NATIVE_K5_ONLY" if mismatch else "PAPER_AND_CODE_NATIVE",
        "paper_10_path_reproducibility": "NOT_IDENTIFIABLE_FROM_CURRENT_ARTIFACT" if mismatch else "AVAILABLE",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=Path("output/speedup_attribution/configuration_audit.json"))
    args = parser.parse_args()
    result = audit(args.repo)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
