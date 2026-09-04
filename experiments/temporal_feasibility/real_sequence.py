"""Build a deterministic license-safe TE slice from official raw SaTE records."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Any, Iterable, Mapping

from tqdm import tqdm

from lib.data.starlink import MultiShellGraph as MSG
from lib.data.starlink import SPOnGrid as SPG
from lib.data.starlink.ism import InterShellMode
from lib.data.starlink.user_node import generate_sat2user

from .sequence_schema import FlowKey, load_dataset
from .sequential_lp import LPUniverse

ADAPTER_VERSION = "real_restricted_slice_v1"
SOURCE_URL = "https://drive.google.com/drive/folders/1h6kbOj4HpqofPNd7lkIJDTut4XF4ipAF"
ORBIT_SHELLS = ((72, 22), (72, 22), (58, 6), (36, 20))
GROUND_STATION_COUNT = 222


def _read_raw_records(path: Path, start_index: int, limit: int) -> list[dict[str, Any]]:
    """Read only the requested record window from a concatenated pickle stream."""

    if start_index < 0 or limit <= 0:
        raise ValueError("start_index must be non-negative and limit must be positive")
    records: list[dict[str, Any]] = []
    with path.open("rb") as handle:
        for index in range(start_index + limit):
            try:
                value = pickle.load(handle)
            except EOFError as exc:
                raise ValueError(f"Raw pickle ended before source record {index}") from exc
            if index >= start_index:
                if not isinstance(value, dict):
                    raise TypeError(f"Raw source record {index} is not a dictionary")
                records.append(value)
    return records


def select_persistent_flows(records: Iterable[Mapping[str, Any]], max_flows: int) -> list[FlowKey]:
    """Select flows by the preregistered (-appearance_count, src, dst) rule."""

    if max_flows <= 0:
        raise ValueError("max_flows must be positive")
    appearances: Counter[FlowKey] = Counter()
    for record in records:
        present = {(int(flow[0]), int(flow[1])) for flow in record["FlowSet"]}
        appearances.update(present)
    return [key for key, _ in sorted(appearances.items(), key=lambda item: (-item[1], item[0][0], item[0][1]))[:max_flows]]


def _static_edges() -> tuple[list[list[int]], int]:
    """Reproduce the official adapter's four fixed intra-shell grids."""

    edges: list[list[int]] = []
    offset = 0
    for orbit_count, satellites_per_orbit in ORBIT_SHELLS:
        latitude = [0] * (orbit_count * satellites_per_orbit)
        _, _, shell_edges = MSG.Inter_Shell_Graph(
            orbit_count, satellites_per_orbit, latitude, offset, 90
        )
        edges.extend(shell_edges)
        offset += orbit_count * satellites_per_orbit
    return edges, offset


def _dynamic_isl_edges(record: Mapping[str, Any]) -> list[list[int]]:
    """Reproduce the official adapter's directed inter-shell ISL construction."""

    offsets = (0, 1584, 3168, 3516, 4236)
    isl = record["InterShell_ISL"]
    edges: list[list[int]] = []
    for relation_index, mapping in enumerate(isl):
        higher_offset = offsets[relation_index + 1]
        lower_offset = offsets[relation_index]
        for local_index, neighbor in enumerate(mapping):
            if neighbor >= 0:
                high = int(local_index + higher_offset)
                low = int(neighbor + lower_offset)
                edges.extend(([high, low], [low, high]))
    return edges


def adapt_restricted_sequence(
    raw_path: str | Path,
    output_path: str | Path,
    *,
    intensity: int,
    start_index: int,
    limit: int,
    max_flows: int,
) -> dict[str, Any]:
    """Adapt selected real flows while retaining only their actually used network edges."""

    source = Path(raw_path)
    records = _read_raw_records(source, start_index, limit)
    selected = select_persistent_flows(records, max_flows)
    selected_set = set(selected)
    _, satellite_count = _static_edges()
    sat2user = generate_sat2user(satellite_count, GROUND_STATION_COUNT, InterShellMode.ISL)
    adapted: list[dict[str, Any]] = []

    for offset, record in enumerate(tqdm(records, desc="Adapting real restricted sequence")):
        source_index = start_index + offset
        demands_by_raw_flow: defaultdict[FlowKey, float] = defaultdict(float)
        for flow in record["FlowSet"]:
            key = (int(flow[0]), int(flow[1]))
            if key in selected_set:
                demands_by_raw_flow[key] += float(flow[2])

        tm: dict[str, float] = {}
        paths_by_flow: dict[str, list[list[int]]] = {}
        restricted_graph_edges: set[tuple[int, int]] = set()
        for src, dst in selected:
            if (src, dst) not in demands_by_raw_flow:
                continue
            raw_paths = SPG.SPOnGrid(
                src, dst, record["InterShell_GrdRelay"], record["InterShell_ISL"],
                InterShellMode.ISL, 5,
            )
            if not raw_paths:
                raise RuntimeError(f"No official-adapter path for flow {(src, dst)} at record {source_index}")
            while len(raw_paths) < 5:
                # Match the official adapter exactly when fewer than five distinct paths exist.
                raw_paths.append(list(raw_paths[0]))
            raw_paths = raw_paths[:5]
            user_src, user_dst = sat2user(src), sat2user(dst)
            flow_key = f"{user_src}, {user_dst}"
            paths_by_flow[flow_key] = [[user_src] + list(path) + [user_dst] for path in raw_paths]
            tm[flow_key] = demands_by_raw_flow[(src, dst)]
            for path in raw_paths:
                restricted_graph_edges.update(zip(path[:-1], path[1:]))

        adapted.append(
            {
                "graph": [list(edge) for edge in sorted(restricted_graph_edges)],
                "tm": tm,
                "path": paths_by_flow,
                "data_idx": source_index,
                "meta": {
                    "source_file": source.name,
                    "source_record_index": source_index,
                    "source_data_idx": None,
                    "intensity": intensity,
                    "adapter_mode": ADAPTER_VERSION,
                    "nominal_interval_s": None,
                    "source_timestamp": None,
                },
            }
        )

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as handle:
        pickle.dump(adapted, handle, protocol=4)

    snapshots = load_dataset(destination)
    universe = LPUniverse.from_snapshots(snapshots)
    appearance_counts = Counter()
    for record in records:
        appearance_counts.update({(int(flow[0]), int(flow[1])) for flow in record["FlowSet"]})
    manifest = {
        "selected_flow_keys": [[src, dst] for src, dst in selected],
        "selected_flow_appearance_counts": [appearance_counts[key] for key in selected],
        "selection_rule": "(-appearance_count, src, dst)",
        "snapshot_range": [start_index, start_index + limit - 1],
        "snapshot_count": limit,
        "path_count": len(universe.paths),
        "edge_count": len(universe.edges),
        "variable_count": len(universe.paths),
        "constraint_count": len(universe.edges) + len(universe.flows) + len(universe.paths),
        "license_limited": True,
        "evidence_label": "real-data restricted slice",
    }
    return manifest


def sha256_file(path: Path) -> str:
    """Hash a source artifact in bounded chunks."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_data_provenance(
    raw_path: Path, adapted_path: Path, intensity: int, start_index: int, limit: int
) -> dict[str, Any]:
    """Record source identity and conservative timing provenance."""

    return {
        "source_url": SOURCE_URL,
        "local_path": str(raw_path),
        "file_size": raw_path.stat().st_size,
        "sha256": sha256_file(raw_path),
        "dataset_intensity": intensity,
        "source_filename": raw_path.name,
        "source_volume": "A" if raw_path.stem.endswith("_A") else "UNKNOWN",
        "first_source_record_index": start_index,
        "last_source_record_index": start_index + limit - 1,
        "adapter_version": ADAPTER_VERSION,
        "adapter_git_commit": "RECORDED_BY_RUNNER",
        "acquisition_date": date.today().isoformat(),
        "adapted_path": str(adapted_path),
        "raw_timestamp_present": False,
        "orbit_epoch_present": False,
        "explicit_time_interval_present": False,
        "ordering_evidence": "CONCATENATED_PICKLE_STREAM_ORDER_ONLY",
        "physical_time_claim": "ORDERED_ONLY",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--provenance", required=True)
    parser.add_argument("--intensity", type=int, required=True)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--max-flows", type=int, default=10)
    args = parser.parse_args()
    raw_path, output_path = Path(args.raw_path), Path(args.output_path)
    manifest = adapt_restricted_sequence(
        raw_path, output_path, intensity=args.intensity, start_index=args.start_index,
        limit=args.limit, max_flows=args.max_flows,
    )
    provenance = build_data_provenance(
        raw_path, output_path, args.intensity, args.start_index, args.limit
    )
    for path, payload in ((Path(args.manifest), manifest), (Path(args.provenance), provenance)):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8-sig") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
    print(json.dumps({"manifest": manifest, "provenance": provenance}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
