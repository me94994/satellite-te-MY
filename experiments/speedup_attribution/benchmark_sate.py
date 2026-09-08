"""Current-hardware SaTE latency decomposition with synchronized CUDA timing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time

import torch
import torch_scatter

from lib.spaceTE.sate_actor import SaTEActor
from lib.spaceTE.sate_env import SaTEEnv
from .benchmark_solver import iridium_shared_instance, solve_level
from .schemas import SafetyLimits, assert_same_problem


DEFAULT_CHECKPOINT = Path(
    "output/supervised/new_form_Intensity_15_spaceTE/models/"
    "spaceTE_supervised_ep-10_dummy-path-False_flow-lambda-25_layers-0.pt"
)


def _percentiles(values):
    ordered = sorted(values)
    def pick(q):
        return ordered[min(len(ordered) - 1, int(q * (len(ordered) - 1)))]
    return {"median_s": statistics.median(ordered), "p90_s": pick(0.90), "p95_s": pick(0.95)}


def _timed_cuda(callable_):
    # CUDA is asynchronous, so both synchronization points are part of valid timing.
    torch.cuda.synchronize()
    start = time.perf_counter()
    value = callable_()
    torch.cuda.synchronize()
    return value, time.perf_counter() - start


def _to_sate_data(instance):
    satellite_edges = [list(edge) for edge in instance.physical_edges if edge[0] < 66 and edge[1] < 66]
    return {
        "graph": satellite_edges,
        "tm": {f"{s}, {t}": demand for (s, t), demand in instance.demands.items()},
        "path": {f"{s}, {t}": [list(path) for path in instance.candidate_paths[(s, t)]] for s, t in instance.active_pairs},
        "data_idx": 0,
    }


def benchmark(checkpoint: Path, warmup: int, repeats: int, active_flows: int, seed: int):
    if not torch.cuda.is_available():
        return {"status": "BLOCKED_CUDA_NOT_AVAILABLE"}
    if not checkpoint.is_file():
        return {"status": "BLOCKED_CHECKPOINT_NOT_AVAILABLE", "checkpoint": str(checkpoint)}
    instance = iridium_shared_instance(active_flows=active_flows, seed=seed)
    assert_same_problem(instance, instance)
    data = _to_sate_data(instance)
    device = torch.device("cuda:0")
    env = SaTEEnv(
        # SaTE's loader indexes path.parts[-4], matching its normal four-level input layout.
        obj="rounded_total_flow", problem_path="diagnostic/input/iridium/new_form_Intensity_15",
        num_path=5, dummy_path=False, edge_disjoint=False, dist_metric="min-hop", rho=1.0,
        num_failure=0.0, device=device, work_dir="output/speedup_attribution",
        dataset=[data, data], supervised=False, penalized=False, flow_lambda=25,
        loss="kl_div", orbit_params=None,
    )
    env.reset("test")
    actor = SaTEActor(env, "EdgeGAT", 0, "linear", "speedup-attribution", device)
    actor.load_state_dict(torch.load(checkpoint, map_location=device))
    actor.eval()

    def forward_parts():
        obs = env.obs
        topo, topo_s = _timed_cuda(lambda: actor.TopoGNN(obs["topo"], obs["capacity"]))
        obs["problem"].nodes["link"].data["x"] = topo
        def allocation():
            value = actor.AlloGNN(obs["problem"], env.edge_index_values.unsqueeze(1))
            value = value.reshape(env.num_path_node // env.num_path, -1)
            return actor.mean_linear(value)
        raw, allo_s = _timed_cuda(allocation)
        return raw, topo_s, allo_s

    def post(raw):
        raw_flow = env.transform_raw_action(raw)
        edge_flow_raw = torch_scatter.scatter(raw_flow[env.p2e[0]], env.p2e[1], dim_size=env.num_edge_node)
        raw_violation = (edge_flow_raw - env.obs["capacity"]).relu().max().item()
        repaired = env.round_action(raw_flow)
        edge_flow = torch_scatter.scatter(repaired[env.p2e[0]], env.p2e[1], dim_size=env.num_edge_node)
        violation = (edge_flow - env.obs["capacity"]).relu().max().item()
        return repaired, raw_violation, violation

    for _ in range(warmup):
        # AlloGNN mutates node features in place; rebuild from the canonical input.
        env.obs = env._read_obs()
        raw = actor.act(env.obs)
        post(raw)
    torch.cuda.synchronize()

    topo_times, allo_times, forward_times, post_times, total_times, decomposed_total_times, input_times = [], [], [], [], [], [], []
    objectives, raw_violations, repaired_violations = [], [], []
    for _ in range(repeats):
        start = time.perf_counter()
        env.obs = env._read_obs()
        torch.cuda.synchronize()
        input_times.append(time.perf_counter() - start)
        raw, topo_s, allo_s = forward_parts()
        repaired, post_s = _timed_cuda(lambda: post(raw))
        post_value, raw_v, repaired_v = repaired
        topo_times.append(topo_s)
        allo_times.append(allo_s)
        forward_times.append(topo_s + allo_s)
        post_times.append(post_s)
        decomposed_total_times.append(input_times[-1] + topo_s + allo_s + post_s)
        objectives.append(float(post_value.sum().item()))
        raw_violations.append(raw_v)
        repaired_violations.append(repaired_v)

        # Measure the same end-to-end path separately with only boundary synchronization.
        torch.cuda.synchronize()
        total_start = time.perf_counter()
        env.obs = env._read_obs()
        raw_e2e = actor.act(env.obs)
        post(raw_e2e)
        torch.cuda.synchronize()
        total_times.append(time.perf_counter() - total_start)

    gurobi = solve_level(instance, "L4", SafetyLimits(max_vars=2000, max_constraints=2000), method=1)
    optimal = gurobi.get("objective")
    feasible_obj = statistics.median(objectives)
    gap = None if optimal in (None, 0) else (optimal - feasible_obj) / optimal
    graph = env.obs["problem"]
    return {
        "status": "MEASURED_DIAGNOSTIC_SYNTHETIC_IRIDIUM_SHAPE",
        "evidence_label": instance.evidence_label,
        "checkpoint": str(checkpoint),
        "hashes": instance.hashes(),
        "warmup": warmup,
        "repeats": repeats,
        "hardware": {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "cuda": torch.version.cuda},
        "t_sate_input": _percentiles(input_times),
        "t_topognn": _percentiles(topo_times),
        "t_allognn": _percentiles(allo_times),
        "t_model_forward": _percentiles(forward_times),
        "t_postprocess": _percentiles(post_times),
        "t_sate_total": _percentiles(total_times),
        "t_decomposed_sum_with_component_sync": _percentiles(decomposed_total_times),
        "quality": {
            "gurobi_objective": optimal,
            "sate_feasible_objective_median": feasible_obj,
            "optimality_gap": gap,
            "max_capacity_violation_before_repair": max(raw_violations),
            "max_capacity_violation_after_repair": max(repaired_violations),
        },
        "gurobi_shared_l4": gurobi,
        "heterograph": {
            "node_types": list(graph.ntypes),
            "canonical_edge_types": [list(x) for x in graph.canonical_etypes],
            "node_counts": {x: graph.num_nodes(x) for x in graph.ntypes},
            "edge_counts": {str(x): graph.num_edges(x) for x in graph.canonical_etypes},
            "full_graph_runtime_attribution": "NOT_IDENTIFIABLE_WITHOUT_TRAINED_FULL_GRAPH_BASELINE",
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--active-flows", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("output/speedup_attribution/sate_results.json"))
    args = parser.parse_args()
    result = benchmark(args.checkpoint, args.warmup, args.repeats, args.active_flows, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if "heterograph" in result:
        # Keep the requested structural artifact separate from latency results.
        (args.output.parent / "current_heterograph_statistics.json").write_text(
            json.dumps(result["heterograph"], indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(json.dumps({"status": result["status"], "output": str(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
