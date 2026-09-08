"""Summarize compatible stage-local measurements."""

from __future__ import annotations

import json
from pathlib import Path
import statistics


def load_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def summarize(rows):
    measured = [row for row in rows if row.get("status") == "MEASURED"]
    output = {}
    for level in sorted({row["level"] for row in measured}):
        group = [row for row in measured if row["level"] == level]
        output[level] = {
            "samples": len(group),
            "median_build_s": statistics.median(row["build_s"] for row in group),
            "median_optimize_s": statistics.median(row["optimize_wall_s"] for row in group),
            "median_total_s": statistics.median(row["total_s"] for row in group),
            "num_vars": group[0]["num_vars"],
            "num_constraints": group[0]["num_constraints"],
            "num_nonzeros": group[0]["num_nonzeros"],
            "presolved_vars": group[0]["presolved_vars"],
            "presolved_constraints": group[0]["presolved_constraints"],
            "presolved_nonzeros": group[0]["presolved_nonzeros"],
            "objective_min": min(row["objective"] for row in group),
            "objective_max": max(row["objective"] for row in group),
        }
    return output


def main():
    root = Path("output/speedup_attribution")
    result = summarize(load_jsonl(root / "solver_results.jsonl"))
    (root / "solver_summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
