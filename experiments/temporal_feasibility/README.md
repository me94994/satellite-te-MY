# SaTE temporal feasibility experiment

This directory tests a narrow hypothesis: whether ordered adapter samples and their
fixed-path LP optima are sufficiently continuous to justify later optimizer-state
tracking. It does not train or modify SaTE, DeepLaDu, TELGEN, or any temporal GNN.

## Evidence boundary

- Input is the adapter-produced list-of-dict pickle and is read in list order.
  `SaTEEnv` is intentionally bypassed because it performs a random train/test split.
- The adapter stores `data_idx` but no timestamp/ephemeris epoch. Monotonic indices
  therefore establish ordered-sample analysis only, not physical-time adjacency.
- Adapter graph lists omit capacity attributes and access links. For Starlink SaTE
  equivalence, commands must explicitly use the repository values 200 for listed
  network edges and 800 for path-only access edges. A dataset containing explicit
  edge capacities takes precedence over these defaults.
- SciPy/HiGHS is a functional fallback. It validates objective, feasibility, state
  extraction, and dual sign, but cannot establish Gurobi basis reuse or Gate C.
- Outputs are caller-controlled and ignored under `output/temporal_feasibility/`.

## Environment check

Run from the repository root. Nothing is downloaded.

```bash
python - <<'PY'
import scipy, matplotlib
print("scipy", scipy.__version__)
print("matplotlib", matplotlib.__version__)
try:
    import gurobipy as gp
    print("gurobi", gp.gurobi.version())
except ImportError as exc:
    print("gurobi unavailable:", exc)
PY

python -m experiments.temporal_feasibility.inspect_dataset \
  --search-root input --list-datasets \
  --output output/temporal_feasibility/unused.csv
```

## Synthetic smoke test

This is one serial process and labels every result as synthetic:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m experiments.temporal_feasibility.smoke_test \
  --output-dir output/temporal_feasibility/synthetic_smoke
```

Unit tests (Gurobi-specific tests explicitly skip when unavailable):

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m pytest -q experiments/temporal_feasibility/tests
```

## Fixed-topology real-data workflow

Prefer `starlink_500_fixed_topo/.../ISL` and pass the adapter pickle itself, not
the directory. The inspector never invokes `train_test_split`.

```bash
python -m experiments.temporal_feasibility.inspect_dataset \
  --dataset input/starlink/starlink_500_fixed_topo/Intensity_100/ISL/StarLink_DataSetForAgent100_5000_Size500.pkl \
  --limit 100 \
  --output output/temporal_feasibility/fixed/sequence_manifest.csv

python -m experiments.temporal_feasibility.benchmark_warm_start \
  --dataset input/starlink/starlink_500_fixed_topo/Intensity_100/ISL/StarLink_DataSetForAgent100_5000_Size500.pkl \
  --limit 100 --backend gurobi --window-size 40 --stride 20 \
  --network-edge-capacity 200 --path-only-edge-capacity 800 \
  --output-dir output/temporal_feasibility/fixed
```

Do not add `--allow-scipy-fallback` to a formal Gurobi benchmark. Without a
working license, the command fails clearly instead of producing misleading timing.

## Dynamic-topology analysis

Use a reduced 176- or 500-node adapter output before any 4236-node dataset:

```bash
python -m experiments.temporal_feasibility.analyze_continuity \
  --dataset input/starlink/starlink_176/ISL/StarLink_DataSetForAgent100_5000_Size176.pkl \
  --limit 100 --backend gurobi \
  --network-edge-capacity 200 --path-only-edge-capacity 800 \
  --output-dir output/temporal_feasibility/dynamic

python -m experiments.temporal_feasibility.plot_results \
  --input-dir output/temporal_feasibility/dynamic
```

For a functional-only run when Gurobi is unavailable, request it explicitly:

```bash
python -m experiments.temporal_feasibility.analyze_continuity \
  --dataset <adapter.pkl> --limit 100 --backend scipy \
  --output-dir output/temporal_feasibility/dynamic_scipy
```

## Outputs

- `sequence_manifest.csv`: list order, `data_idx`, topology hash, demand/path
  counts, and adjacent structural drift.
- `solve_records.csv`: build/update/optimize/total times, solver runtime,
  iterations, objective, dual objective, and feasibility/parity checks.
- `state_records.jsonl`: semantic path flow, edge load/utilization, raw duals,
  calibrated nonnegative congestion prices, binding flags, and reduced costs.
- `continuity_records.csv`: adjacent, lag 1/2/5/10, unrestricted random,
  demand-matched random, and topology-matched random comparisons.
- `transport_records.csv`: copy-persistent/drop-deleted transport with zero,
  global-median, and neighbor-median initialization for new edges, compared with
  default zero/one/deterministic-random vectors.
- `summary.json` and `warm_start_summary.json`: distribution statistics,
  bootstrap median intervals, provenance, limitations, and Gate-ready values.
- `figures/*.png`: matplotlib-only plots.

Dual values can be non-unique for degenerate LPs. Interpret congestion-price
distance together with edge-load distance, binding-set Jaccard, primal/dual
objective parity, and warm-start behavior. Transport distance alone is not a
claim that a primal optimum can be recovered.
