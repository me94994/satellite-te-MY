"""Create a tiny deterministic fixture; synthetic evidence is never labeled real."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any


BASE_EDGES = [
    [0, 1, 5.0],
    [1, 3, 5.0],
    [0, 2, 8.0],
    [2, 3, 8.0],
]
NEW_EDGE = [1, 2, 3.0]


def make_fixture(snapshot_count: int = 16, disorder_data_idx: bool = False) -> list[dict[str, Any]]:
    """Build traffic-only, topology-only, and joint-change transitions."""

    if snapshot_count < 12:
        raise ValueError("snapshot_count must be at least 12 to exercise lag=10")
    dataset: list[dict[str, Any]] = []
    for index in range(snapshot_count):
        has_new_edge = 4 <= index < 10
        graph = [edge[:] for edge in BASE_EDGES]
        if has_new_edge:
            graph.append(NEW_EDGE[:])

        # index 4 is topology-only; index 10 changes topology and traffic.
        demand = 4.0 + float(index if index < 4 else index - 1)
        if index == 4:
            demand = 7.0
        paths = [[0, 1, 3], [0, 2, 3]]
        if has_new_edge:
            paths.append([0, 1, 2, 3])
        if index == 2:
            # Same semantic paths in a different list order.
            paths = list(reversed(paths))
        data_idx = index
        if disorder_data_idx and index in (5, 6):
            data_idx = 11 - index  # Produces a duplicate/nonmonotonic region.
        dataset.append(
            {
                "graph": graph,
                "tm": {"0, 3": demand},
                "path": {"0, 3": paths},
                "data_idx": data_idx,
                "fixture_kind": "synthetic_temporal_feasibility",
            }
        )
    return dataset


def write_fixture(path: str | Path, snapshot_count: int = 16, disorder_data_idx: bool = False) -> Path:
    """Write the small pickle to a caller-controlled path."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as handle:
        pickle.dump(make_fixture(snapshot_count, disorder_data_idx), handle, protocol=4)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--snapshots", type=int, default=16)
    parser.add_argument("--disorder-data-idx", action="store_true")
    args = parser.parse_args()
    output = write_fixture(args.output, args.snapshots, args.disorder_data_idx)
    print(f"SYNTHETIC_FIXTURE={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
