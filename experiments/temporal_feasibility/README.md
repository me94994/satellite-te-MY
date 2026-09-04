# SaTE temporal feasibility experiment

This directory tests a narrow hypothesis: whether ordered adapter samples and their
fixed-path LP optima are sufficiently continuous to justify later optimizer-state
tracking. It does not train or modify SaTE, DeepLaDu, TELGEN, or any temporal GNN.

The second-round real-data runner is intentionally restricted-license safe. It
streams an official raw SaTE pickle volume, selects flows by the fixed
`(-appearance_count, src, dst)` rule, keeps only edges actually used by those
official candidate paths, and labels every result `real-data restricted slice`.

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

## Real restricted validation

After placing one official raw volume under the ignored `input/raw/starlink/`
tree, run all real stages serially with one command:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
python -m experiments.temporal_feasibility.run_real_experiment \
  --raw-path input/raw/starlink/DataSetForSaTE25/StarLink_DataSetForAgent25_5000_A.pkl \
  --adapted-path input/raw/starlink/real_restricted_30x100.pkl \
  --output-dir output/temporal_feasibility/real \
  --intensity 25 --start-index 0 --limit 100 --max-flows 30
```

Use `--resume` only when the completed summary has the same start/limit/flow
signature. The runner fails closed if Gurobi is unavailable or the selected
universe exceeds the confirmed restricted-license limit.

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
- `continuity_records.csv`: adjacent, lag 1/2/5/10/20, unrestricted random,
  demand-matched, topology-matched, and jointly matched random comparisons.
- `transport_records.csv`: load/utilization/binding/dual transport with edge
  birth/death and no-history controls.
- `counterfactual_records.csv`: z00/z01/z10/z11 traffic/topology decomposition.
- `warm_start_records.csv`: cold, reuse, explicit basis, basis+presolve, and
  reset-basis timing/iteration evidence.
- `summary.json`: distribution statistics,
  bootstrap median intervals, provenance, limitations, and Gate-ready values.
- `figures/*.png`: matplotlib-only plots.

Dual values can be non-unique for degenerate LPs. Interpret congestion-price
distance together with edge-load distance, binding-set Jaccard, primal/dual
objective parity, and warm-start behavior. Transport distance alone is not a
claim that a primal optimum can be recovered.
