# SaTE Speedup Attribution Diagnosis

This directory isolates representation, exact LP, solver-native, neural,
heterograph, and offline topology-selection effects. It does not modify SaTE's
objective, demand, capacity, model, or temporal protocol.

Evidence labels are mandatory:

- `MEASURED`: current-machine runtime on the stated instance.
- `DIAGNOSTIC_CODE_NATIVE`: reduced generated instance using code-native K=5.
- `PAPER_REPORTED_REFERENCE`: paper value, never mixed into measured ratios.
- `NOT_IDENTIFIABLE_FROM_CURRENT_ARTIFACT`: required artifact is absent.
- `SKIPPED_RESOURCE_BOUND`: analytical guard prevented construction.

Run the complete reproducible pipeline with one sequential command (no parallel workers):

```bash
python -m experiments.speedup_attribution.run_all
```

Outputs are incremental under `output/speedup_attribution/`, which is ignored
by Git. Dense construction is guarded before Gurobi objects are allocated.

The K=10 large-scale extension is also sequential and single-process:

```bash
python -m experiments.speedup_attribution.run_large_scale
```

It keeps `MEASURED`, `MEASURED_FIT`, `PAPER_REPORTED_REFERENCE`, and
`EXTRAPOLATED_4236` as separate evidence classes. Its generated topology and
stress regimes are diagnostics, never paper-workload substitutes. Use
`--resume` to retain completed JSONL points or `--quick` for a protocol smoke
run; the formal defaults preserve the preregistered scale and repeat counts.
