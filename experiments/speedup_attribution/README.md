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
