"""Measure representation volume without claiming solver acceleration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import statistics
import time
import tracemalloc

from .benchmark_solver import diagnostic_instance


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes", type=int, default=16)
    parser.add_argument("--active-flows", type=int, default=20)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("output/speedup_attribution/representation.json"))
    args = parser.parse_args()
    start = time.perf_counter()
    instance = diagnostic_instance(args.nodes, args.active_flows, args.k)
    parse_s = time.perf_counter() - start
    dense = {
        "demands": {pair: instance.demands.get(pair, 0.0) for pair in instance.all_pairs},
        "paths": dict(instance.candidate_paths),
    }
    sparse = {
        "demands": {pair: instance.demands[pair] for pair in instance.active_pairs},
        "paths": {pair: instance.candidate_paths[pair] for pair in instance.active_pairs},
    }
    serialization, deserialization, peaks = {}, {}, {}
    for name, value in (("dense", dense), ("sparse", sparse)):
        serialized = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
        serialization[name] = statistics.median(
            (lambda begin=time.perf_counter(): (pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL), time.perf_counter() - begin)[1])()
            for _ in range(100)
        )
        deserialization[name] = statistics.median(
            (lambda begin=time.perf_counter(): (pickle.loads(serialized), time.perf_counter() - begin)[1])()
            for _ in range(100)
        )
        tracemalloc.start()
        pickle.loads(serialized)
        _, peaks[name] = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    result = {
        **instance.representation_bytes(),
        "num_represented_sd_pairs_dense": len(instance.all_pairs),
        "num_represented_sd_pairs_sparse": len(instance.active_pairs),
        "num_represented_paths_dense": sum(map(len, instance.candidate_paths.values())),
        "num_represented_paths_sparse": sum(len(instance.candidate_paths[p]) for p in instance.active_pairs),
        "parse_s": parse_s,
        "serialization_median_s": serialization,
        "deserialization_median_s": deserialization,
        "deserialization_python_peak_bytes": peaks,
        "evidence_label": instance.evidence_label,
        "solver_speedup_claim": "NOT_APPLICABLE_REPRESENTATION_ONLY",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
