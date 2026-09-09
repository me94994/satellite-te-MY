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
import zipfile

from .ampl_solver import CapacityConfiguration
from .schemas import BenchmarkInstance, Pair, Path as TEPath


SATELLITE_COUNT = 4236
INTENSITIES = (25, 50, 75, 100)
VOLUMES = ("A", "B")
REQUIRED_KEYS = frozenset(("FlowSet", "InterShell_ISL", "InterShell_GrdRelay"))
# 路径生成器延迟到 Stage 4 加载，避免 Stage 0/2 inventory 被 astropy/GNN 依赖阻塞。
SPG: Any = None
CACHE_VERSION = "OFFICIAL_K10_PATH_CACHE_V1"


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
    user_node_floor: int = SATELLITE_COUNT

    def hashes(self) -> dict[str, str]:
        return {
            "topology_hash": self.topology_hash,
            "demand_hash": self.demand_hash,
            "path_hash": self.path_hash,
        }

    def with_capacity(self, configuration: CapacityConfiguration) -> "OfficialBenchmarkInstance":
        """只替换 edge capacity，保证 sensitivity 的 topology/demand/path 完全相同。"""

        capacities = {
            edge: configuration.access_capacity
            if edge[0] >= self.user_node_floor or edge[1] >= self.user_node_floor
            else configuration.network_capacity
            for edge in self.benchmark.physical_edges
        }
        benchmark = BenchmarkInstance(
            snapshot_id=f"{self.provenance.dataset}-{self.provenance.volume}-{self.provenance.record_index}-{configuration.name}",
            nodes=self.benchmark.nodes, physical_edges=self.benchmark.physical_edges,
            capacities=capacities, demands=self.benchmark.demands,
            candidate_paths=self.benchmark.candidate_paths, evidence_label=self.benchmark.evidence_label,
        )
        hashes = benchmark.hashes()
        return OfficialBenchmarkInstance(
            provenance=self.provenance, benchmark=benchmark, requested_k=self.requested_k,
            topology_hash=hashes["topology_hash"], demand_hash=hashes["demand_hash"],
            path_hash=hashes["path_hash"], unique_path_completion_rate=self.unique_path_completion_rate,
            mean_path_hops=self.mean_path_hops, p95_path_hops=self.p95_path_hops,
            user_node_floor=self.user_node_floor,
        )


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
        "sha256": sha256_file(path),
        "source": "OFFICIAL_SATE_GOOGLE_DRIVE",
        "readable_pickle_records": count,
        "first_record_keys": first_keys,
        "last_readable_record_index": last_index,
        "eof_clean": eof_clean,
        "truncated": not eof_clean,
        "failure_type": failure_type,
        "benchmark_status": "BENCHMARK_USABLE" if benchmark_usable and count > 0 and eof_clean else "BENCHMARK_UNUSABLE",
    }


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    """流式计算大型 official artifact 摘要，避免整文件载入内存。"""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_download_archive(path: Path) -> dict[str, Any]:
    """拒绝 HTML/配额页面，只接受可完整读取的 ZIP central directory。"""

    if not path.is_file() or not zipfile.is_zipfile(path):
        return {"status": "DOWNLOAD_ARCHIVE_INVALID", "path": str(path)}
    with zipfile.ZipFile(path) as archive:
        bad_member = archive.testzip()
        members = [
            {"name": item.filename, "size_bytes": item.file_size}
            for item in archive.infolist() if not item.is_dir()
        ]
    return {
        "status": "DOWNLOAD_ARCHIVE_VALID" if bad_member is None else "DOWNLOAD_ARCHIVE_INVALID",
        "path": str(path), "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path), "members": members,
        "bad_member": bad_member,
    }


def build_inventory(roots: Iterable[Path]) -> dict[str, Any]:
    files = [inspect_pickle(path) for path in discover_official_files(roots)]
    present = {(row["dataset_intensity"], row["volume"]) for row in files if row["benchmark_status"] == "BENCHMARK_USABLE"}
    required = {(intensity, volume) for intensity in INTENSITIES for volume in VOLUMES}
    missing = [f"DataSetForSaTE{intensity}/volume_{volume}" for intensity, volume in sorted(required - present)]
    primary_ready = any(intensity == 100 for intensity, _ in present)
    all_intensities_ready = all(any(item == intensity for item, _ in present) for intensity in INTENSITIES)
    a_b_complete = required <= present
    return {
        # 主实验只依赖 DataSetForSaTE100 的任一真实 volume；A/B 完整度另行报告。
        "status": "BENCHMARK_PRIMARY_READY" if primary_ready else "BENCHMARK_INCOMPLETE",
        "BENCHMARK_PRIMARY_READY": primary_ready,
        "BENCHMARK_ALL_INTENSITIES_READY": all_intensities_ready,
        "BENCHMARK_A_B_COMPLETE": a_b_complete,
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
        # 官方 raw artifact 确实含 src_sat == dst_sat；保留这些 flow，避免静默截断 workload。
        if not math.isfinite(demand) or demand < 0:
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


def select_snapshots(
    rows: Sequence[Mapping[str, Any]], intensity: int = 100, limit: int = 10,
) -> list[dict[str, Any]]:
    """仅按 active-SD 分布确定性选点，绝不查看 solver 或 SaTE 结果。"""

    eligible = [dict(row) for row in rows if int(row["intensity"]) == intensity]
    if not eligible:
        raise ValueError("PRIMARY_DATASET_NOT_AVAILABLE")
    ordered = sorted(eligible, key=lambda row: (int(row["active_sd_pairs"]), row["volume"], int(row["record_index"])))
    targets = (("minimum", 0.0), ("P10", 0.10), ("median", 0.50), ("P75", 0.75),
               ("P90", 0.90), ("P95", 0.95), ("P99", 0.99), ("maximum", 1.0))
    selected: list[dict[str, Any]] = []
    used: set[tuple[str, int]] = set()

    def add(label: str, row: Mapping[str, Any]) -> None:
        key = (str(row["source_path"]), int(row["record_index"]))
        if key not in used and len(selected) < limit:
            used.add(key)
            selected.append({**row, "selection_label": label})

    for label, probability in targets:
        target = percentile([float(row["active_sd_pairs"]) for row in ordered], probability)
        closest = min(ordered, key=lambda row: (
            abs(float(row["active_sd_pairs"]) - target), row["volume"], int(row["record_index"]),
        ))
        add(label, closest)
    median_value = percentile([float(row["active_sd_pairs"]) for row in ordered], 0.50)
    near_median = sorted(ordered, key=lambda row: (
        abs(float(row["active_sd_pairs"]) - median_value), row["volume"], int(row["record_index"]),
    ))
    for rank, row in enumerate(near_median, start=1):
        add(f"near_median_{rank}", row)
        if len(selected) >= limit:
            break
    return selected


def load_snapshot(source_path: Path, record_index: int) -> Mapping[str, Any]:
    """按 pickle 原始顺序读取指定 snapshot，不改变或重排 record。"""

    for index, record in enumerate(_objects_from_pickle(source_path)):
        if index == record_index:
            if not isinstance(record, Mapping) or not REQUIRED_KEYS.issubset(record):
                raise ValueError("OFFICIAL_SNAPSHOT_SCHEMA_INVALID")
            return record
    raise IndexError("OFFICIAL_SNAPSHOT_INDEX_OUT_OF_RANGE")


def topology_hash(record: Mapping[str, Any], mode: str) -> str:
    payload = pickle.dumps((mode, record["InterShell_ISL"], record["InterShell_GrdRelay"]), protocol=4)
    return hashlib.sha256(payload).hexdigest()


def path_cache_key(topology_digest: str, mode: str, src: int, dst: int, requested_k: int) -> str:
    """稳定绑定 topology/mode/SD/K，防止跨 topology 误复用路径。"""

    payload = f"{CACHE_VERSION}|{topology_digest}|{mode}|{src}|{dst}|{requested_k}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _mode_name(mode: Any) -> str:
    return str(getattr(mode, "value", mode or "ISL"))


def _satellite_to_user(mode: Any) -> Any:
    """复用官方 adapter 的 generate_sat2user，尤其保留 GrdStation 的额外偏移。"""

    from lib.data.starlink.user_node import generate_sat2user
    from lib.data.starlink.ism import InterShellMode

    selected = mode if mode is not None else InterShellMode.ISL
    return generate_sat2user(SATELLITE_COUNT, 222, selected)


def generate_unique_paths(
    record: Mapping[str, Any], pairs: Iterable[Pair], requested_k: int = 10,
    mode: Any = None, cache_dir: Path | None = None,
) -> tuple[dict[Pair, tuple[TEPath, ...]], list[dict[str, Any]]]:
    """调用公开 SPOnGrid(K=10)，按原顺序去重且绝不补路径。"""

    result: dict[Pair, tuple[TEPath, ...]] = {}
    audit: list[dict[str, Any]] = []
    if mode is None:
        from lib.data.starlink.ism import InterShellMode

        mode = InterShellMode.ISL
    generator = _path_generator()
    topo_digest = topology_hash(record, _mode_name(mode))
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
    for src, dst in sorted(set(pairs)):
        cache_path = None
        raw_paths = None
        cache_lookup_start = time.perf_counter()
        if cache_dir is not None:
            cache_path = cache_dir / f"{path_cache_key(topo_digest, _mode_name(mode), src, dst, requested_k)}.pkl"
            if cache_path.is_file():
                with cache_path.open("rb") as handle:
                    raw_paths = pickle.load(handle)
        cache_lookup_s = time.perf_counter() - cache_lookup_start
        cache_hit = raw_paths is not None
        generation_start = time.perf_counter()
        if raw_paths is None:
            raw_paths = generator.SPOnGrid(
                src, dst, record["InterShell_GrdRelay"], record["InterShell_ISL"], mode, requested_k,
            ) or []
            if cache_path is not None:
                temporary = cache_path.with_suffix(".tmp")
                with temporary.open("wb") as handle:
                    pickle.dump(raw_paths, handle, protocol=pickle.HIGHEST_PROTOCOL)
                temporary.replace(cache_path)
        unique = tuple(dict.fromkeys(tuple(int(node) for node in path) for path in raw_paths))
        result[(src, dst)] = unique
        audit.append({
            "src": src, "dst": dst, "requested_paths": requested_k,
            "returned_paths": len(raw_paths), "unique_paths": len(unique),
            "status": "COMPLETE" if len(unique) == requested_k else "K10_PATH_SHORTFALL",
            "cache_hit": cache_hit,
            # 分开记录 cached 输入与真正的 path generation，避免输入语义混淆。
            "path_cache_lookup_s": cache_lookup_s,
            "path_generation_uncached_s": 0.0 if cache_hit else time.perf_counter() - generation_start,
        })
    return result, audit


def assemble_official_instance(
    provenance: SnapshotProvenance, capacity: CapacityConfiguration,
    demands: Mapping[Pair, float], satellite_paths: Mapping[Pair, tuple[TEPath, ...]],
    audit: Sequence[Mapping[str, Any]], mode: Any = None,
) -> OfficialBenchmarkInstance:
    """从已聚合 demand 和已生成 path 创建 canonical instance，供分阶段计时。"""

    sat2user = _satellite_to_user(mode)
    user_node_floor = int(sat2user(0))
    # 完整 TE path 必须包含 source-user uplink 与 destination-user downlink。
    paths = {
        (int(sat2user(src)), int(sat2user(dst))): tuple(
            (int(sat2user(src)),) + path + (int(sat2user(dst)),) for path in values
        )
        for (src, dst), values in satellite_paths.items()
    }
    demands = {(int(sat2user(src)), int(sat2user(dst))): value for (src, dst), value in demands.items()}
    if any(not value for value in paths.values()):
        raise RuntimeError("K10_PATH_GENERATION_EMPTY")
    physical_edges = tuple(sorted({edge for values in paths.values() for path in values for edge in zip(path[:-1], path[1:])}))
    capacities = {
        edge: capacity.access_capacity if edge[0] >= user_node_floor or edge[1] >= user_node_floor else capacity.network_capacity
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
    return OfficialBenchmarkInstance(
        provenance=provenance, benchmark=benchmark, requested_k=10,
        topology_hash=hashes["topology_hash"], demand_hash=hashes["demand_hash"],
        path_hash=hashes["path_hash"], unique_path_completion_rate=complete / len(audit) if audit else 0.0,
        mean_path_hops=statistics.fmean(hops), p95_path_hops=percentile(hops, 0.95),
        user_node_floor=user_node_floor,
    )


def build_official_instance(
    record: Mapping[str, Any], provenance: SnapshotProvenance,
    capacity: CapacityConfiguration, mode: Any = None, cache_dir: Path | None = None,
) -> tuple[OfficialBenchmarkInstance, list[dict[str, Any]]]:
    demands, _ = aggregate_flowset(record["FlowSet"])
    satellite_paths, audit = generate_unique_paths(record, demands, 10, mode, cache_dir)
    official = assemble_official_instance(provenance, capacity, demands, satellite_paths, audit, mode)
    return official, audit


def write_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
