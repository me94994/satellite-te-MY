"""K=5/K=10 SaTE architecture latency and quality diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import statistics
import time
from typing import Dict, Mapping, Optional, Sequence

import torch
import torch_scatter

from lib.data.starlink.ism import InterShellMode as ISM
from lib.data.starlink.orbit_params import OrbitParams
from lib.spaceTE.sate_actor import SaTEActor
from lib.spaceTE.sate_env import SaTEEnv

from .benchmark_large_scale import constellation_instance, controlled_stress_instance
from .benchmark_solver import solve_level
from .schemas import BenchmarkInstance, SafetyLimits


SATE_SCALE_NODES = (66, 128, 256, 512, 1024, 2048, 4236)


def checkpoint_state(path: Path) -> Mapping[str, torch.Tensor]:
    """Load only a state dictionary; wrappers are accepted without resizing tensors."""
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, Mapping) and "state_dict" in payload:
        payload = payload["state_dict"]
    if not isinstance(payload, Mapping):
        raise ValueError("checkpoint does not contain a state dictionary")
    return payload


def validate_checkpoint_dimension(path: Path, expected_k: int) -> Dict[str, object]:
    """Reject a mismatched checkpoint before constructing or mutating an actor."""
    if not path.is_file():
        return {"status": "BLOCKED_CHECKPOINT_NOT_AVAILABLE", "checkpoint": str(path), "expected_k": expected_k}
    state = checkpoint_state(path)
    weight = state.get("mean_linear.weight")
    bias = state.get("mean_linear.bias")
    actual = list(weight.shape) if weight is not None else None
    valid = actual == [expected_k, expected_k] and bias is not None and list(bias.shape) == [expected_k]
    return {
        "status": "VALID" if valid else "REJECTED_CHECKPOINT_DIMENSION_MISMATCH",
        "checkpoint": str(path),
        "expected_k": expected_k,
        "mean_linear_weight_shape": actual,
        "mean_linear_bias_shape": list(bias.shape) if bias is not None else None,
    }


def _to_sate_data(instance: BenchmarkInstance, topology_nodes: int, data_idx: int = 0) -> Dict[str, object]:
    """Translate one canonical instance without changing paths, demand, or topology."""
    satellite_edges = [list(edge) for edge in instance.physical_edges if edge[0] < topology_nodes and edge[1] < topology_nodes]
    return {
        "graph": satellite_edges,
        "tm": {f"{s}, {t}": demand for (s, t), demand in instance.demands.items()},
        "path": {f"{s}, {t}": [list(path) for path in instance.candidate_paths[(s, t)]] for s, t in instance.active_pairs},
        "data_idx": data_idx,
    }


def make_sate_env(
    instance: BenchmarkInstance,
    topology_nodes: int,
    k: int,
    dataset: Optional[Sequence[Mapping[str, object]]] = None,
    penalized: bool = False,
    work_dir: str = "output/speedup_attribution/large_scale",
) -> SaTEEnv:
    """Use the production SaTE environment under an explicit CODE_CONFIG orbit shape."""
    data = _to_sate_data(instance, topology_nodes)
    records = list(dataset) if dataset is not None else [data, dict(data)]
    params = OrbitParams(
        GrdStationNum=0,
        Offset5=topology_nodes,
        graph_node_num=topology_nodes * 2,
        isl_cap=200,
        uplink_cap=800,
        downlink_cap=800,
        ism=ISM.ISL,
    )
    return SaTEEnv(
        obj="rounded_total_flow",
        problem_path="diagnostic/input/starlink/DIAGNOSTIC_SCALE/ISL",
        num_path=k,
        dummy_path=False,
        edge_disjoint=False,
        dist_metric="min-hop",
        rho=1.0,
        num_failure=0.0,
        device=torch.device("cuda:0"),
        work_dir=work_dir,
        dataset=records,
        supervised=False,
        penalized=penalized,
        flow_lambda=25,
        loss="kl_div",
        orbit_params=params,
    )


def _summary(values, bootstrap_samples: int = 1000) -> Dict[str, object]:
    ordered = sorted(values)
    pick = lambda q: ordered[min(len(ordered) - 1, int(q * (len(ordered) - 1)))]
    rng = random.Random(20260908)
    medians = []
    for _ in range(bootstrap_samples):
        medians.append(statistics.median(rng.choices(ordered, k=len(ordered))))
    medians.sort()
    return {
        "mean_s": statistics.mean(ordered),
        "median_s": statistics.median(ordered),
        "p90_s": pick(0.90),
        "p95_s": pick(0.95),
        "bootstrap_median_95ci_s": [pick_from(medians, 0.025), pick_from(medians, 0.975)],
    }


def pick_from(values, quantile: float):
    return values[min(len(values) - 1, int(quantile * (len(values) - 1)))]


def _timed_cuda(function):
    # Synchronize both boundaries because DGL/PyTorch kernels are asynchronous.
    torch.cuda.synchronize()
    start = time.perf_counter()
    value = function()
    torch.cuda.synchronize()
    return value, time.perf_counter() - start


def benchmark_instance(
    instance: BenchmarkInstance,
    topology_nodes: int,
    k: int,
    checkpoint: Optional[Path],
    checkpoint_label: str,
    warmup: int,
    repeats: int,
) -> Dict[str, object]:
    """Measure graph construction, both GNNs, repair, e2e, memory, and quality."""
    if not torch.cuda.is_available():
        return {"status": "BLOCKED_CUDA_NOT_AVAILABLE", "topology_nodes": topology_nodes, "k": k}
    env = make_sate_env(instance, topology_nodes, k)
    env.reset("test")
    torch.manual_seed(42)
    actor = SaTEActor(env, "EdgeGAT", 0, "linear", "k10-crossover", torch.device("cuda:0"))
    if checkpoint is not None:
        audit = validate_checkpoint_dimension(checkpoint, k)
        if audit["status"] != "VALID":
            return {**audit, "topology_nodes": topology_nodes, "k": k}
        actor.load_state_dict(checkpoint_state(checkpoint))
    actor.eval()

    def forward_parts():
        obs = env.obs
        topo, topo_s = _timed_cuda(lambda: actor.TopoGNN(obs["topo"], obs["capacity"]))
        obs["problem"].nodes["link"].data["x"] = topo
        def allocation():
            value = actor.AlloGNN(obs["problem"], env.edge_index_values.unsqueeze(1))
            return actor.mean_linear(value.reshape(env.num_path_node // env.num_path, -1))
        raw, allo_s = _timed_cuda(allocation)
        return raw, topo_s, allo_s

    def postprocess(raw):
        raw_flow = env.transform_raw_action(raw)
        edge_flow_raw = torch_scatter.scatter(raw_flow[env.p2e[0]], env.p2e[1], dim_size=env.num_edge_node)
        raw_violation = (edge_flow_raw - env.obs["capacity"]).relu().max().item()
        repaired = env.round_action(raw_flow)
        edge_flow = torch_scatter.scatter(repaired[env.p2e[0]], env.p2e[1], dim_size=env.num_edge_node)
        return repaired, raw_violation, (edge_flow - env.obs["capacity"]).relu().max().item()

    for _ in range(warmup):
        env.obs = env._read_obs()
        postprocess(actor.act(env.obs))
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    input_times, topo_times, allo_times, forward_times, post_times, e2e_times = [], [], [], [], [], []
    objectives, raw_violations, repaired_violations = [], [], []
    for _ in range(repeats):
        _, input_s = _timed_cuda(lambda: setattr(env, "obs", env._read_obs()))
        raw, topo_s, allo_s = forward_parts()
        post_value, post_s = _timed_cuda(lambda: postprocess(raw))
        repaired, raw_violation, repaired_violation = post_value
        input_times.append(input_s)
        topo_times.append(topo_s)
        allo_times.append(allo_s)
        forward_times.append(topo_s + allo_s)
        post_times.append(post_s)
        objectives.append(float(repaired.sum().item()))
        raw_violations.append(raw_violation)
        repaired_violations.append(repaired_violation)
        def full_call():
            env.obs = env._read_obs()
            postprocess(actor.act(env.obs))
        _, total_s = _timed_cuda(full_call)
        e2e_times.append(total_s)

    graph = env.obs["problem"]
    quality_allowed = checkpoint is not None and checkpoint_label in {"DIAGNOSTIC_K10_TRAINED", "OFFICIAL_K10_CHECKPOINT"}
    quality: Dict[str, object] = {"status": "NOT_EVALUATED_UNTRAINED_ARCHITECTURE"}
    if quality_allowed:
        gurobi = solve_level(instance, "L4", SafetyLimits(max_vars=1900, max_constraints=1900), method=1)
        optimal = gurobi.get("objective")
        feasible = statistics.median(objectives)
        quality = {
            "status": "MEASURED_DIAGNOSTIC" if gurobi.get("status") == "MEASURED" else "NOT_EVALUATED_GUROBI_BLOCKED",
            "gurobi_objective": optimal,
            "sate_feasible_objective_median": feasible,
            "optimality_gap": None if optimal in (None, 0) else (optimal - feasible) / optimal,
            "max_capacity_violation_before_repair": max(raw_violations),
            "max_capacity_violation_after_repair": max(repaired_violations),
        }
    return {
        "status": "MEASURED",
        "evidence_label": "UNTRAINED_K10_LATENCY_ONLY" if checkpoint is None and k == 10 else checkpoint_label,
        "topology_nodes": topology_nodes,
        "active_flows": len(instance.active_pairs),
        "k": k,
        "hashes": instance.hashes(),
        "warmup": warmup,
        "repeats": repeats,
        "t_input_graph": _summary(input_times),
        "t_topognn": _summary(topo_times),
        "t_allognn": _summary(allo_times),
        "t_forward": _summary(forward_times),
        "t_postprocess": _summary(post_times),
        "t_e2e": _summary(e2e_times),
        "gpu": {
            "name": torch.cuda.get_device_name(0),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        },
        "graph": {
            "num_flow_nodes": graph.num_nodes("flow"),
            "num_path_nodes": graph.num_nodes("path"),
            "num_link_nodes": graph.num_nodes("link"),
            "num_hetero_edges": sum(graph.num_edges(edge_type) for edge_type in graph.canonical_etypes),
        },
        "quality": quality,
    }


def _append(path: Path, row: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes", type=int, nargs="+", default=list(SATE_SCALE_NODES))
    parser.add_argument("--active-flows", type=int, default=180)
    parser.add_argument("--k", type=int, nargs="+", choices=(5, 10), default=[5, 10])
    parser.add_argument("--regime", nargs="+", choices=("uncongested", "congested"), default=["uncongested", "congested"])
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--checkpoint-label", choices=("DIAGNOSTIC_K10_TRAINED", "OFFICIAL_K10_CHECKPOINT"), default="DIAGNOSTIC_K10_TRAINED")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats-small", type=int, default=100)
    parser.add_argument("--repeats-large", type=int, default=30)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("output/speedup_attribution/large_scale/sate_scaling.jsonl"))
    args = parser.parse_args()
    if args.output.exists() and not args.resume:
        args.output.unlink()
    completed = set()
    if args.resume and args.output.is_file():
        completed = {(row["topology_nodes"], row["active_flows"], row["k"], row["regime"], row["evidence_label"]) for row in (json.loads(line) for line in args.output.read_text(encoding="utf-8").splitlines() if line.strip())}
    for nodes in args.nodes:
        flows = min(nodes, args.active_flows)
        for regime in args.regime:
            k10, _ = controlled_stress_instance(nodes, flows) if regime == "congested" else (constellation_instance(nodes, flows, 10), {})
            for k in args.k:
                instance = k10.paths_for_k(k)
                checkpoint = args.checkpoint if k == 10 else None
                label = args.checkpoint_label if checkpoint is not None else ("UNTRAINED_K10_LATENCY_ONLY" if k == 10 else "UNTRAINED_K5_LATENCY_ONLY")
                key = (nodes, flows, k, regime, label)
                if key in completed:
                    continue
                repeats = args.repeats_small if nodes <= 512 else args.repeats_large
                result = benchmark_instance(instance, nodes, k, checkpoint, label, args.warmup, repeats)
                result["regime"] = regime
                _append(args.output, result)
                print(json.dumps({"nodes": nodes, "k": k, "regime": regime, "status": result["status"]}, sort_keys=True))


if __name__ == "__main__":
    main()
