"""Reproducible fixed-universe LP wrapper with explicit solver provenance.

The mathematical program matches SaTE ``PathFormulation`` for total carried
flow: non-negative path flows, per-flow demand upper bounds, directed-link
capacity upper bounds, and a maximum sum-flow objective.  This experiment adds
only fixed-universe availability bounds so paths absent in a snapshot cannot be
used accidentally.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping

import numpy as np
from scipy.optimize import linprog

from .sequence_schema import EdgeKey, FlowKey, PathKey, Snapshot, load_dataset, path_edges_for_key

BackendName = Literal["gurobi", "scipy"]
SolveMode = Literal[
    "cold_rebuild", "reuse_model", "reuse_model_auto", "explicit_basis",
    "explicit_basis_presolve", "reset_basis",
]


class SolverUnavailableError(RuntimeError):
    """Raised when the requested formal solver cannot be used."""


@dataclass(frozen=True)
class CapacityPolicy:
    """Explicit capacities needed because adapter pickles omit edge attributes."""

    network_edge_capacity: float = 200.0
    path_only_edge_capacity: float = 800.0

    def __post_init__(self) -> None:
        if self.network_edge_capacity <= 0 or self.path_only_edge_capacity <= 0:
            raise ValueError("Capacity defaults must be strictly positive")


@dataclass(frozen=True)
class LPUniverse:
    """Stable variable and constraint indices collected over a sequence window."""

    flows: tuple[FlowKey, ...]
    paths: tuple[PathKey, ...]
    edges: tuple[EdgeKey, ...]
    path_to_edges: Mapping[PathKey, tuple[EdgeKey, ...]]

    @classmethod
    def from_snapshots(cls, snapshots: Iterable[Snapshot]) -> "LPUniverse":
        items = list(snapshots)
        if not items:
            raise ValueError("Cannot build an LP universe from an empty sequence")
        flows = {flow for snapshot in items for flow in snapshot.demands}
        paths = {path for snapshot in items for path in snapshot.paths}
        flows.update((path[0], path[1]) for path in paths)
        edges = {edge for snapshot in items for edge in snapshot.edges}
        path_to_edges = {path: path_edges_for_key(path) for path in paths}
        edges.update(edge for values in path_to_edges.values() for edge in values)
        for flow in flows:
            if not any((path[0], path[1]) == flow for path in paths):
                raise ValueError(f"Active flow has no candidate semantic path in window: {flow}")
        return cls(
            flows=tuple(sorted(flows)),
            paths=tuple(sorted(paths)),
            edges=tuple(sorted(edges)),
            path_to_edges=path_to_edges,
        )


@dataclass
class SolveState:
    """Auditable primal/dual state for one optimal snapshot."""

    position: int
    data_idx: int | float | str
    backend: str
    backend_version: str
    status: str
    solve_mode: str
    warm_start_api: str
    warm_start_effective: bool
    method: int | None
    presolve: int | None
    lp_warm_start: int | None
    model_build_wall_time: float
    rhs_update_wall_time: float
    optimize_wall_time: float
    total_wall_time: float
    solver_runtime: float | None
    iter_count: float | None
    objective: float
    dual_objective: float
    duality_gap: float
    max_capacity_violation: float
    max_demand_violation: float
    max_path_availability_violation: float
    path_flow: dict[PathKey, float]
    edge_load: dict[EdgeKey, float]
    edge_capacity: dict[EdgeKey, float]
    edge_utilization: dict[EdgeKey, float]
    raw_capacity_pi: dict[EdgeKey, float]
    congestion_price: dict[EdgeKey, float]
    raw_demand_pi: dict[FlowKey, float]
    reduced_cost: dict[PathKey, float]
    reduced_cost_convention: str
    binding_capacity: dict[EdgeKey, bool]
    dual_sign_multiplier: float
    dual_sign_validation: str

    def csv_record(self) -> dict[str, Any]:
        """Return scalar fields suitable for ``solve_records.csv``."""

        excluded = {
            "path_flow", "edge_load", "edge_capacity", "edge_utilization",
            "raw_capacity_pi", "congestion_price", "raw_demand_pi",
            "reduced_cost", "binding_capacity",
        }
        return {key: value for key, value in asdict(self).items() if key not in excluded}

    def json_record(self) -> dict[str, Any]:
        """Serialize tuple-keyed maps without lossy string parsing."""

        record = self.csv_record()
        record.update(
            {
                "path_flow": [[p[0], p[1], list(p[2]), value] for p, value in sorted(self.path_flow.items())],
                "edge_load": [[u, v, value] for (u, v), value in sorted(self.edge_load.items())],
                "edge_capacity": [[u, v, value] for (u, v), value in sorted(self.edge_capacity.items())],
                "edge_utilization": [[u, v, value] for (u, v), value in sorted(self.edge_utilization.items())],
                "raw_capacity_pi": [[u, v, value] for (u, v), value in sorted(self.raw_capacity_pi.items())],
                "congestion_price": [[u, v, value] for (u, v), value in sorted(self.congestion_price.items())],
                "raw_demand_pi": [[s, d, value] for (s, d), value in sorted(self.raw_demand_pi.items())],
                "reduced_cost": [[p[0], p[1], list(p[2]), value] for p, value in sorted(self.reduced_cost.items())],
                "binding_capacity": [[u, v, value] for (u, v), value in sorted(self.binding_capacity.items())],
            }
        )
        return record


def capacities_for_snapshot(
    snapshot: Snapshot, universe: LPUniverse, policy: CapacityPolicy
) -> dict[EdgeKey, float]:
    """Resolve capacities without silently guessing absent dynamic edges."""

    result: dict[EdgeKey, float] = {}
    for edge in universe.edges:
        if edge not in snapshot.edges:
            result[edge] = 0.0
        elif edge in snapshot.capacities:
            result[edge] = float(snapshot.capacities[edge])
        elif edge in snapshot.graph_edges:
            result[edge] = policy.network_edge_capacity
        else:
            # Adapter-added user access links do not occur in sample['graph'].
            result[edge] = policy.path_only_edge_capacity
    return result


def _incidence(universe: LPUniverse) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    edge_index = {edge: index for index, edge in enumerate(universe.edges)}
    flow_index = {flow: index for index, flow in enumerate(universe.flows)}
    cap = np.zeros((len(universe.edges), len(universe.paths)), dtype=float)
    dem = np.zeros((len(universe.flows), len(universe.paths)), dtype=float)
    for column, path in enumerate(universe.paths):
        dem[flow_index[(path[0], path[1])], column] = 1.0
        for edge in universe.path_to_edges[path]:
            cap[edge_index[edge], column] += 1.0
    availability = np.eye(len(universe.paths), dtype=float)
    return cap, dem, availability


class ScipySequentialLP:
    """HiGHS dual-simplex fallback; it does not expose reusable simplex bases."""

    backend = "scipy"

    def __init__(self, universe: LPUniverse, policy: CapacityPolicy, dual_sign: float = -1.0):
        self.universe = universe
        self.policy = policy
        self.dual_sign = dual_sign
        start = time.perf_counter()
        self.cap_matrix, self.dem_matrix, self.avail_matrix = _incidence(universe)
        self.build_time = time.perf_counter() - start

    def solve(self, snapshot: Snapshot, mode: SolveMode = "cold_rebuild") -> SolveState:
        total_start = time.perf_counter()
        build_start = time.perf_counter()
        if mode == "cold_rebuild":
            cap_matrix, dem_matrix, avail_matrix = _incidence(self.universe)
            model_build = time.perf_counter() - build_start
        else:
            cap_matrix, dem_matrix, avail_matrix = self.cap_matrix, self.dem_matrix, self.avail_matrix
            model_build = 0.0
        rhs_start = time.perf_counter()
        capacities = capacities_for_snapshot(snapshot, self.universe, self.policy)
        cap_rhs = np.asarray([capacities[edge] for edge in self.universe.edges], dtype=float)
        dem_rhs = np.asarray([snapshot.demands.get(flow, 0.0) for flow in self.universe.flows], dtype=float)
        avail_rhs = np.asarray(
            [snapshot.demands.get((path[0], path[1]), 0.0) if path in snapshot.paths else 0.0 for path in self.universe.paths],
            dtype=float,
        )
        rhs_update = time.perf_counter() - rhs_start
        a_ub = np.vstack([cap_matrix, dem_matrix, avail_matrix])
        b_ub = np.concatenate([cap_rhs, dem_rhs, avail_rhs])
        optimize_start = time.perf_counter()
        result = linprog(
            c=-np.ones(len(self.universe.paths), dtype=float),
            A_ub=a_ub,
            b_ub=b_ub,
            bounds=(0.0, None),
            method="highs-ds",
            options={"presolve": True},
        )
        optimize_wall = time.perf_counter() - optimize_start
        if not result.success:
            raise RuntimeError(f"SciPy/HiGHS failed at position {snapshot.position}: {result.message}")
        path_values = np.asarray(result.x, dtype=float)
        marginals = np.asarray(result.ineqlin.marginals, dtype=float)
        cap_pi = marginals[: len(self.universe.edges)]
        dem_pi = marginals[len(self.universe.edges): len(self.universe.edges) + len(self.universe.flows)]
        avail_pi = marginals[len(self.universe.edges) + len(self.universe.flows):]
        lower_marginals = np.asarray(result.lower.marginals, dtype=float)
        return _assemble_state(
            snapshot=snapshot,
            universe=self.universe,
            backend="scipy",
            backend_version=_scipy_version(),
            status="OPTIMAL",
            mode=mode,
            warm_start_api="NONE_SCIPY_HIGHS_REBUILDS_INTERNAL_MODEL",
            warm_start_effective=False,
            method=1,
            presolve=1,
            lp_warm_start=None,
            model_build=model_build,
            rhs_update=rhs_update,
            optimize_wall=optimize_wall,
            total_wall=time.perf_counter() - total_start,
            solver_runtime=None,
            iter_count=float(result.nit),
            path_values=path_values,
            capacities=capacities,
            raw_capacity_pi=cap_pi,
            raw_demand_pi=dem_pi,
            raw_availability_pi=avail_pi,
            reduced_cost=lower_marginals,
            reduced_cost_convention="SCIPY_MINIMIZATION_LOWER_BOUND_MARGINAL_FOR_NEGATED_OBJECTIVE",
            dual_sign=self.dual_sign,
            dual_validation="FINITE_DIFFERENCE_VALIDATED_SCIPY_MINIMIZATION_MARGINAL_SIGN",
        )


class GurobiSequentialLP:
    """Gurobi model reuse and explicit VBasis/CBasis restoration benchmark."""

    backend = "gurobi"

    def __init__(self, universe: LPUniverse, policy: CapacityPolicy, dual_sign: float):
        try:
            import gurobipy as gp
        except ImportError as exc:
            raise SolverUnavailableError("gurobipy is not installed") from exc
        self.gp = gp
        self.universe = universe
        self.policy = policy
        self.dual_sign = dual_sign
        self.model = None
        self.variables: list[Any] = []
        self.capacity_constraints: list[Any] = []
        self.demand_constraints: list[Any] = []
        self.availability_constraints: list[Any] = []
        self.saved_basis: tuple[list[int], list[int]] | None = None

    def close(self) -> None:
        """Release the native model promptly after a benchmark mode finishes."""

        if self.model is not None:
            self.model.dispose()
            self.model = None

    def _build(self, snapshot: Snapshot, mode: SolveMode) -> float:
        gp = self.gp
        start = time.perf_counter()
        model = gp.Model("temporal_fixed_path_total_flow")
        model.Params.OutputFlag = 0
        model.Params.Threads = 1
        model.Params.Method = 1  # Deterministic dual simplex.
        model.Params.Seed = 42
        # LPWarmStart=2 explicitly enables presolve mapping for supplied bases.
        if mode == "explicit_basis_presolve":
            model.Params.LPWarmStart = 2
        elif mode == "explicit_basis":
            model.Params.LPWarmStart = 1
        variables = [
            model.addVar(lb=0.0, vtype=gp.GRB.CONTINUOUS, name=f"x[{i}]")
            for i in range(len(self.universe.paths))
        ]
        model.setObjective(gp.quicksum(variables), gp.GRB.MAXIMIZE)
        paths_by_edge: dict[EdgeKey, list[int]] = {edge: [] for edge in self.universe.edges}
        paths_by_flow: dict[FlowKey, list[int]] = {flow: [] for flow in self.universe.flows}
        for index, path in enumerate(self.universe.paths):
            paths_by_flow[(path[0], path[1])].append(index)
            for edge in self.universe.path_to_edges[path]:
                paths_by_edge[edge].append(index)
        capacities = capacities_for_snapshot(snapshot, self.universe, self.policy)
        capacity_constraints = [
            model.addConstr(gp.quicksum(variables[i] for i in paths_by_edge[edge]) <= capacities[edge], name=f"cap[{edge[0]},{edge[1]}]")
            for edge in self.universe.edges
        ]
        demand_constraints = [
            model.addConstr(gp.quicksum(variables[i] for i in paths_by_flow[flow]) <= snapshot.demands.get(flow, 0.0), name=f"dem[{flow[0]},{flow[1]}]")
            for flow in self.universe.flows
        ]
        availability_constraints = [
            model.addConstr(variables[i] <= (snapshot.demands.get((path[0], path[1]), 0.0) if path in snapshot.paths else 0.0), name=f"avail[{i}]")
            for i, path in enumerate(self.universe.paths)
        ]
        model.update()
        self.model = model
        self.variables = variables
        self.capacity_constraints = capacity_constraints
        self.demand_constraints = demand_constraints
        self.availability_constraints = availability_constraints
        return time.perf_counter() - start

    def _update_rhs(self, snapshot: Snapshot) -> tuple[float, dict[EdgeKey, float]]:
        if self.model is None:
            raise RuntimeError("Gurobi model is not built")
        start = time.perf_counter()
        capacities = capacities_for_snapshot(snapshot, self.universe, self.policy)
        self.model.setAttr("RHS", self.capacity_constraints, [capacities[edge] for edge in self.universe.edges])
        self.model.setAttr("RHS", self.demand_constraints, [snapshot.demands.get(flow, 0.0) for flow in self.universe.flows])
        self.model.setAttr(
            "RHS", self.availability_constraints,
            [snapshot.demands.get((path[0], path[1]), 0.0) if path in snapshot.paths else 0.0 for path in self.universe.paths],
        )
        self.model.update()
        return time.perf_counter() - start, capacities

    def solve(self, snapshot: Snapshot, mode: SolveMode) -> SolveState:
        gp = self.gp
        total_start = time.perf_counter()
        had_previous_solution = self.model is not None
        if mode == "cold_rebuild" or self.model is None:
            if mode == "cold_rebuild" and self.model is not None:
                # A cold solve must not retain a previous model or basis.
                self.model.dispose()
                self.model = None
            model_build = self._build(snapshot, mode)
            rhs_update = 0.0
            capacities = capacities_for_snapshot(snapshot, self.universe, self.policy)
            restored = False
        else:
            model_build = 0.0
            previous_basis = self.saved_basis
            rhs_update, capacities = self._update_rhs(snapshot)
            restored = False
            if mode in {"explicit_basis", "explicit_basis_presolve"} and previous_basis is not None:
                vbasis, cbasis = previous_basis
                self.model.setAttr("VBasis", self.variables, vbasis)
                all_constraints = self.capacity_constraints + self.demand_constraints + self.availability_constraints
                self.model.setAttr("CBasis", all_constraints, cbasis)
                self.model.update()
                restored = True
            elif mode == "reset_basis":
                # reset(0) discards solution/basis state while preserving the model.
                self.model.reset(0)
        if self.model is None:
            raise RuntimeError("Gurobi model construction failed")
        optimize_start = time.perf_counter()
        try:
            self.model.optimize()
        except gp.GurobiError as exc:
            raise SolverUnavailableError(f"Gurobi optimize/license failure: {exc}") from exc
        optimize_wall = time.perf_counter() - optimize_start
        if self.model.Status != gp.GRB.OPTIMAL:
            raise RuntimeError(f"Gurobi status is not OPTIMAL: {self.model.Status}")
        all_constraints = self.capacity_constraints + self.demand_constraints + self.availability_constraints
        self.saved_basis = (
            list(self.model.getAttr("VBasis", self.variables)),
            list(self.model.getAttr("CBasis", all_constraints)),
        )
        if mode in {"reuse_model", "reuse_model_auto"}:
            api = "MODEL_REUSE_RHS_UPDATE_GUROBI_INTERNAL_BASIS_RETENTION"
            effective = had_previous_solution
        elif mode in {"explicit_basis", "explicit_basis_presolve"}:
            api = "GETATTR_SETATTR_VBASIS_CBASIS"
            effective = restored
        elif mode == "reset_basis":
            api = "MODEL_RESET_0_AFTER_RHS_UPDATE"
            effective = False
        else:
            api = "NEW_MODEL_EACH_SNAPSHOT"
            effective = False
        return _assemble_state(
            snapshot=snapshot,
            universe=self.universe,
            backend="gurobi",
            backend_version=".".join(map(str, gp.gurobi.version())),
            status="OPTIMAL",
            mode=mode,
            warm_start_api=api,
            warm_start_effective=effective,
            method=int(self.model.Params.Method),
            presolve=int(self.model.Params.Presolve),
            lp_warm_start=int(self.model.Params.LPWarmStart),
            model_build=model_build,
            rhs_update=rhs_update,
            optimize_wall=optimize_wall,
            total_wall=time.perf_counter() - total_start,
            solver_runtime=float(self.model.Runtime),
            iter_count=float(self.model.IterCount),
            path_values=np.asarray(self.model.getAttr("X", self.variables), dtype=float),
            capacities=capacities,
            raw_capacity_pi=np.asarray(self.model.getAttr("Pi", self.capacity_constraints), dtype=float),
            raw_demand_pi=np.asarray(self.model.getAttr("Pi", self.demand_constraints), dtype=float),
            raw_availability_pi=np.asarray(self.model.getAttr("Pi", self.availability_constraints), dtype=float),
            reduced_cost=np.asarray(self.model.getAttr("RC", self.variables), dtype=float),
            reduced_cost_convention="GUROBI_NATIVE_RC_FOR_MAXIMIZATION_MODEL",
            dual_sign=self.dual_sign,
            dual_validation="FINITE_DIFFERENCE_CALIBRATED_FOR_ACTIVE_GUROBI_VERSION",
        )


def _assemble_state(
    *, snapshot: Snapshot, universe: LPUniverse, backend: str, backend_version: str,
    status: str, mode: SolveMode, warm_start_api: str, warm_start_effective: bool,
    method: int | None, presolve: int | None, lp_warm_start: int | None,
    model_build: float, rhs_update: float, optimize_wall: float, total_wall: float,
    solver_runtime: float | None, iter_count: float | None, path_values: np.ndarray,
    capacities: Mapping[EdgeKey, float], raw_capacity_pi: np.ndarray,
    raw_demand_pi: np.ndarray, raw_availability_pi: np.ndarray,
    reduced_cost: np.ndarray, reduced_cost_convention: str, dual_sign: float,
    dual_validation: str,
) -> SolveState:
    path_flow = {path: float(max(0.0, path_values[i])) for i, path in enumerate(universe.paths)}
    edge_load = {edge: 0.0 for edge in universe.edges}
    for path, value in path_flow.items():
        for edge in universe.path_to_edges[path]:
            edge_load[edge] += value
    flow_load = {flow: 0.0 for flow in universe.flows}
    for path, value in path_flow.items():
        flow_load[(path[0], path[1])] += value
    cap_violation = max((edge_load[e] - capacities[e] for e in universe.edges), default=0.0)
    dem_violation = max((flow_load[f] - snapshot.demands.get(f, 0.0) for f in universe.flows), default=0.0)
    availability_violation = max(
        (value for path, value in path_flow.items() if path not in snapshot.paths), default=0.0
    )
    cap_pi_map = {edge: float(raw_capacity_pi[i]) for i, edge in enumerate(universe.edges)}
    price = {edge: max(0.0, dual_sign * cap_pi_map[edge]) for edge in universe.edges}
    zero_capacity_violations = [
        (edge, edge_load[edge]) for edge in universe.edges
        if capacities[edge] == 0.0 and edge_load[edge] > 1e-7
    ]
    if zero_capacity_violations:
        raise AssertionError(f"Positive flow on zero-capacity edges: {zero_capacity_violations[:3]}")
    utilization = {
        edge: edge_load[edge] / capacities[edge] if capacities[edge] > 0 else 0.0
        for edge in universe.edges
    }
    binding = {
        edge: capacities[edge] > 0 and capacities[edge] - edge_load[edge] <= 1e-7 * max(1.0, capacities[edge])
        for edge in universe.edges
    }
    demand_rhs = [snapshot.demands.get(flow, 0.0) for flow in universe.flows]
    availability_rhs = [
        snapshot.demands.get((path[0], path[1]), 0.0) if path in snapshot.paths else 0.0
        for path in universe.paths
    ]
    dual_objective = dual_sign * (
        float(np.dot(np.asarray([capacities[e] for e in universe.edges]), raw_capacity_pi))
        + float(np.dot(np.asarray(demand_rhs), raw_demand_pi))
        + float(np.dot(np.asarray(availability_rhs), raw_availability_pi))
    )
    objective = float(sum(path_values))
    return SolveState(
        position=snapshot.position, data_idx=snapshot.data_idx, backend=backend,
        backend_version=backend_version, status=status, solve_mode=mode,
        warm_start_api=warm_start_api, warm_start_effective=warm_start_effective,
        method=method, presolve=presolve, lp_warm_start=lp_warm_start,
        model_build_wall_time=model_build, rhs_update_wall_time=rhs_update,
        optimize_wall_time=optimize_wall, total_wall_time=total_wall,
        solver_runtime=solver_runtime, iter_count=iter_count,
        objective=objective, dual_objective=dual_objective,
        duality_gap=abs(objective - dual_objective),
        max_capacity_violation=max(0.0, float(cap_violation)),
        max_demand_violation=max(0.0, float(dem_violation)),
        max_path_availability_violation=max(0.0, float(availability_violation)),
        path_flow=path_flow, edge_load=edge_load, edge_capacity=dict(capacities),
        edge_utilization=utilization, raw_capacity_pi=cap_pi_map,
        congestion_price=price,
        raw_demand_pi={flow: float(raw_demand_pi[i]) for i, flow in enumerate(universe.flows)},
        reduced_cost={path: float(reduced_cost[i]) for i, path in enumerate(universe.paths)},
        reduced_cost_convention=reduced_cost_convention,
        binding_capacity=binding, dual_sign_multiplier=dual_sign,
        dual_sign_validation=dual_validation,
    )


def _scipy_version() -> str:
    import scipy
    return scipy.__version__


def gurobi_available() -> tuple[bool, str]:
    """Probe import and license with a bounded one-variable model."""

    try:
        import gurobipy as gp
        model = gp.Model("temporal_license_probe")
        model.Params.OutputFlag = 0
        x = model.addVar(lb=0.0)
        model.setObjective(x, gp.GRB.MAXIMIZE)
        model.addConstr(x <= 1.0)
        model.optimize()
        if model.Status != gp.GRB.OPTIMAL:
            return False, f"license probe returned status {model.Status}"
        return True, ".".join(map(str, gp.gurobi.version()))
    except (ImportError, RuntimeError) as exc:
        return False, f"{type(exc).__name__}: {exc}"
    except Exception as exc:
        # The exception is surfaced in provenance; no solver result is fabricated.
        return False, f"{type(exc).__name__}: {exc}"


def choose_backend(requested: str, allow_scipy_fallback: bool) -> tuple[BackendName, str]:
    """Resolve backend and explain any fallback explicitly."""

    available, detail = gurobi_available()
    if requested == "gurobi":
        if available:
            return "gurobi", f"Gurobi {detail} available"
        if not allow_scipy_fallback:
            raise SolverUnavailableError(f"Gurobi unavailable: {detail}")
        return "scipy", f"FUNCTIONAL_FALLBACK_ONLY; Gurobi unavailable: {detail}"
    if requested == "scipy":
        return "scipy", "SciPy/HiGHS explicitly requested; no Gurobi timing claim"
    if available:
        return "gurobi", f"Auto-selected Gurobi {detail}"
    return "scipy", f"FUNCTIONAL_FALLBACK_ONLY; Gurobi unavailable: {detail}"


def write_solve_outputs(states: list[SolveState], output_dir: Path, provenance: Mapping[str, Any]) -> None:
    """Write scalar CSV, full JSONL state, and solver provenance."""

    output_dir.mkdir(parents=True, exist_ok=True)
    if states:
        with (output_dir / "solve_records.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(states[0].csv_record()))
            writer.writeheader()
            writer.writerows(state.csv_record() for state in states)
        with (output_dir / "state_records.jsonl").open("w", encoding="utf-8-sig") as handle:
            for state in states:
                handle.write(json.dumps(state.json_record(), ensure_ascii=False, allow_nan=False) + "\n")
    with (output_dir / "solver_provenance.json").open("w", encoding="utf-8-sig") as handle:
        json.dump(dict(provenance), handle, indent=2, ensure_ascii=False, allow_nan=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--backend", choices=("auto", "gurobi", "scipy"), default="auto")
    parser.add_argument("--allow-scipy-fallback", action="store_true")
    parser.add_argument(
        "--mode",
        choices=("cold_rebuild", "reuse_model", "reuse_model_auto", "explicit_basis", "explicit_basis_presolve", "reset_basis"),
        default="cold_rebuild",
    )
    parser.add_argument("--network-edge-capacity", type=float, default=200.0)
    parser.add_argument("--path-only-edge-capacity", type=float, default=800.0)
    parser.add_argument("--output-dir", default="output/temporal_feasibility")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    snapshots = load_dataset(args.dataset, limit=args.limit)
    universe = LPUniverse.from_snapshots(snapshots)
    backend, backend_detail = choose_backend(args.backend, args.allow_scipy_fallback)
    policy = CapacityPolicy(args.network_edge_capacity, args.path_only_edge_capacity)
    if backend == "gurobi":
        from .extract_primal_dual import calibrate_dual_sign
        calibration = calibrate_dual_sign("gurobi")
        solver: Any = GurobiSequentialLP(universe, policy, calibration["sign_multiplier"])
    else:
        solver = ScipySequentialLP(universe, policy)
    states = [solver.solve(snapshot, mode=args.mode) for snapshot in snapshots]
    provenance = {
        "dataset": args.dataset,
        "snapshot_count": len(snapshots),
        "requested_backend": args.backend,
        "actual_backend": backend,
        "backend_detail": backend_detail,
        "mode": args.mode,
        "threads": 1,
        "simplex_method": "dual simplex",
        "seed": 42,
        "capacity_policy": asdict(policy),
        "python": platform.python_version(),
    }
    write_solve_outputs(states, Path(args.output_dir), provenance)
    print(json.dumps(provenance, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
