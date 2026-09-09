"""Official DataSetForSaTE inventory、FlowSet 统计与 K=10 instance 构造。"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import pickle
import statistics
import time
from typing import Any, Iterable, Iterator, Mapping, Sequence

from .ampl_solver import CapacityConfiguration
from .schemas import BenchmarkInstance, Pair, Path as TEPath


SATELLITE_COUNT = 4236
INTENSITIES = (25, 50, 75, 100)
VOLUMES = ("A", "B")
REQUIRED_KEYS = frozenset(("FlowSet", "InterShell_ISL", "InterShell_GrdRelay"))
# 路径生成器延迟到 Stage 4 加载，避免 Stage 0/2 inventory 被 astropy/GNN 依赖阻塞。
SPG: Any = None


def _path_generator() -> Any:
    global SPG
    if SPG is None:
        from lib.data.starlink import SPOnGrid

        SPG = SPOnGrid
    return SPG


@dataclass(frozen=True)
class SnapshotProvenance:
    dataset: str
    volume: str
    source_path: str
    record_index: int
    raw_flow_count: int
    active_sd_pairs: int
    total_demand: float


@dataclass(frozen=True)
class OfficialBenchmarkInstance:
    """同一 official snapshot 向 AMPL/Gurobi 与 SaTE 提供唯一输入。"""

    provenance: SnapshotProvenance
    benchmark: BenchmarkInstance
    requested_k: int
    topology_hash: str
    demand_hash: str
    path_hash: str
    unique_path_completion_rate: float
    mean_path_hops: float
    p95_path_hops: float

    def hashes(self) -> dict[str, str]:
        return {
            "topology_hash": self.topology_hash,
            "demand_hash": self.demand_hash,
            "path_hash": self.path_hash,
        }


def default_search_roots(repository: Path = Path(".")) -> tuple[Path, ...]:
    """只扫描协议指定位置，避免无界遍历其他磁盘。"""

    return (
        repository / "input",
        repository / "input" / "raw",
        repository / "input" / "raw" / "starlink",
        repository / "raw_data",
    )


def discover_official_files(roots: Iterable[Path]) -> list[Path]:
    found: set[Path] = set()
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("StarLink_DataSetForAgent*_5000_[AB].pkl"):
            if any(f"DataSetForSaTE{i}" in str(path) or f"Agent{i}_" in path.name for i in INTENSITIES):
                found.add(path.resolve())
    return sorted(found)


def _objects_from_pickle(path: Path) -> Iterator[Any]:
    """同时支持单个 list pickle 与连续 pickle stream。"""

    with path.open("rb") as handle:
        while True:
            try:
                value = pickle.load(handle)
            except EOFError:
                return
            if isinstance(value, list) and (not value or isinstance(value[0], Mapping)):
                yield from value
            else:
                yield value


def _dataset_identity(path: Path) -> tuple[int | None, str | None]:
    text = str(path)
    intensity = next((item for item in INTENSITIES if f"DataSetForSaTE{item}" in text or f"Agent{item}_" in path.name), None)
    volume = next((item for item in VOLUMES if path.name.endswith(f"_{item}.pkl")), None)
    return intensity, volume


def inspect_pickle(path: Path) -> dict[str, Any]:
    """检查完整 EOF 与 schema；错误只记录类型，不记录任意原始对象。"""

    intensity, volume = _dataset_identity(path)
    count = 0
    first_keys: list[str] = []
    last_index: int | None = None
    eof_clean = True
    failure_type: str | None = None
    benchmark_usable = True
    try:
        for index, record in enumerate(_objects_from_pickle(path)):
            if not isinstance(record, Mapping):
                benchmark_usable = False
            else:
                if index == 0:
                    first_keys = sorted(str(key) for key in record.keys())
                benchmark_usable = benchmark_usable and REQUIRED_KEYS.issubset(record)
            count += 1
            last_index = index
    except Exception as exc:
        eof_clean = False
        benchmark_usable = False
        failure_type = type(exc).__name__
    return {
        "dataset_intensity": intensity,
        "volume": volume,
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "readable_pickle_records": count,
        "first_record_keys": first_keys,
        "last_readable_record_index": last_index,
        "eof_clean": eof_clean,
        "truncated": not eof_clean,
        "failure_type": failure_type,
        "benchmark_status": "BENCHMARK_USABLE" if benchmark_usable and count > 0 and eof_clean else "BENCHMARK_UNUSABLE",
    }


def build_inventory(roots: Iterable[Path]) -> dict[str, Any]:
    files = [inspect_pickle(path) for path in discover_official_files(roots)]
    present = {(row["dataset_intensity"], row["volume"]) for row in files if row["benchmark_status"] == "BENCHMARK_USABLE"}
    required = {(intensity, volume) for intensity in INTENSITIES for volume in VOLUMES}
    missing = [f"DataSetForSaTE{intensity}/volume_{volume}" for intensity, volume in sorted(required - present)]
    return {
        "status": "BENCHMARK_COMPLETE_FOR_MAIN_STARLINK" if not missing else "BENCHMARK_INCOMPLETE",
        "files": files,
        "missing_files": missing,
    }


def aggregate_flowset(flowset: Iterable[Sequence[Any]]) -> tuple[dict[Pair, float], dict[str, float | int]]:
    """按 active satellite SD 聚合重复 flow，不分配 4236x4236 矩阵。"""

    demands: defaultdict[Pair, float] = defaultdict(float)
    raw_demands: list[float] = []
    raw_count = 0
    for flow in flowset:
        if len(flow) < 3:
            raise ValueError("FLOW_ROW_TOO_SHORT")
        src, dst, demand = int(flow[0]), int(flow[1]), float(flow[2])
        if src == dst or not math.isfinite(demand) or demand < 0:
            raise ValueError("INVALID_FLOW_ROW")
        demands[(src, dst)] += demand
        raw_demands.append(demand)
        raw_count += 1
    active = len(demands)
    return dict(demands), {
        "raw_flow_count": raw_count,
        "active_sd_pairs": active,
        "total_demand": float(sum(raw_demands)),
        "mean_raw_flow_demand": statistics.fmean(raw_demands) if raw_demands else 0.0,
        "median_raw_flow_demand": statistics.median(raw_demands) if raw_demands else 0.0,
        "max_raw_flow_demand": max(raw_demands, default=0.0),
        "aggregation_ratio": raw_count / active if active else 0.0,
        "traffic_matrix_density": active / (SATELLITE_COUNT * (SATELLITE_COUNT - 1)),
    }


def scan_flowsets(inventory: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Stage 2：扫描所有 raw FlowSet，不构建路径也不调用 solver。"""

    rows: list[dict[str, Any]] = []
    for source in inventory["files"]:
        if source["benchmark_status"] != "BENCHMARK_USABLE":
            continue
        read_start = time.perf_counter()
        for index, record in enumerate(_objects_from_pickle(Path(source["path"]))):
            aggregate_start = time.perf_counter()
            _, stats = aggregate_flowset(record["FlowSet"])
            rows.append({
                "dataset": f"DataSetForSaTE{source['dataset_intensity']}",
                "intensity": source["dataset_intensity"],
                "volume": source["volume"],
                "source_path": source["path"],
                "record_index": index,
                **stats,
                "T_pickle_read": time.perf_counter() - read_start,
                "T_flow_aggregate": time.perf_counter() - aggregate_start,
            })
            read_start = time.perf_counter()
    return rows


def percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("EMPTY_DISTRIBUTION")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def distribution(values: Sequence[float]) -> dict[str, float]:
    return {
        "min": min(values), "P10": percentile(values, 0.10), "P25": percentile(values, 0.25),
        "median": percentile(values, 0.50), "P50": percentile(values, 0.50),
        "P75": percentile(values, 0.75), "P90": percentile(values, 0.90),
        "P95": percentile(values, 0.95), "P99": percentile(values, 0.99),
        "max": max(values), "mean": statistics.fmean(values),
        "std": statistics.pstdev(values),
    }


def topology_hash(record: Mapping[str, Any], mode: str) -> str:
    payload = pickle.dumps((mode, record["InterShell_ISL"], record["InterShell_GrdRelay"]), protocol=4)
    return hashlib.sha256(payload).hexdigest()


def generate_unique_paths(
    record: Mapping[str, Any], pairs: Iterable[Pair], requested_k: int = 10,
    mode: Any = None,
) -> tuple[dict[Pair, tuple[TEPath, ...]], list[dict[str, Any]]]:
    """调用公开 SPOnGrid(K=10)，按原顺序去重且绝不补路径。"""

    result: dict[Pair, tuple[TEPath, ...]] = {}
    audit: list[dict[str, Any]] = []
    if mode is None:
        from lib.data.starlink.ism import InterShellMode

        mode = InterShellMode.ISL
    generator = _path_generator()
    for src, dst in sorted(set(pairs)):
        raw_paths = generator.SPOnGrid(
            src, dst, record["InterShell_GrdRelay"], record["InterShell_ISL"], mode, requested_k,
        ) or []
        unique = tuple(dict.fromkeys(tuple(int(node) for node in path) for path in raw_paths))
        result[(src, dst)] = unique
        audit.append({
            "src": src, "dst": dst, "requested_paths": requested_k,
            "returned_paths": len(raw_paths), "unique_paths": len(unique),
            "status": "COMPLETE" if len(unique) == requested_k else "K10_PATH_SHORTFALL",
        })
    return result, audit


def build_official_instance(
    record: Mapping[str, Any], provenance: SnapshotProvenance,
    capacity: CapacityConfiguration, mode: Any = None,
) -> tuple[OfficialBenchmarkInstance, list[dict[str, Any]]]:
    demands, _ = aggregate_flowset(record["FlowSet"])
    paths, audit = generate_unique_paths(record, demands, 10, mode)
    if any(not value for value in paths.values()):
        raise RuntimeError("K10_PATH_GENERATION_EMPTY")
    physical_edges = tuple(sorted({edge for values in paths.values() for path in values for edge in zip(path[:-1], path[1:])}))
    capacities = {
        edge: capacity.access_capacity if edge[0] >= SATELLITE_COUNT or edge[1] >= SATELLITE_COUNT else capacity.network_capacity
        for edge in physical_edges
    }
    benchmark = BenchmarkInstance(
        snapshot_id=f"{provenance.dataset}-{provenance.volume}-{provenance.record_index}-{capacity.name}",
        nodes=tuple(sorted({node for edge in physical_edges for node in edge})),
        physical_edges=physical_edges,
        capacities=capacities,
        demands=demands,
        candidate_paths=paths,
        evidence_label="OFFICIAL_WORKLOAD",
    )
    hashes = benchmark.hashes()
    hops = [len(path) - 1 for values in paths.values() for path in values]
    complete = sum(item["unique_paths"] == 10 for item in audit)
    official = OfficialBenchmarkInstance(
        provenance=provenance, benchmark=benchmark, requested_k=10,
        topology_hash=hashes["topology_hash"], demand_hash=hashes["demand_hash"],
        path_hash=hashes["path_hash"], unique_path_completion_rate=complete / len(audit) if audit else 0.0,
        mean_path_hops=statistics.fmean(hops), p95_path_hops=percentile(hops, 0.95),
    )
    return official, audit


def write_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
