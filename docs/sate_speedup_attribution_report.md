# SaTE Speedup Attribution Diagnosis

## Executive conclusion

The available repository does **not** support a paper-scale numerical
attribution of the reported 2738x speedup. The paper uses ten paths per pair;
the released executable paths use five, and the current worktree lacks the raw
and adapted input dataset. Consequently, the 4236-node Gurobi 46--47 s result
was `NOT_IDENTIFIABLE_FROM_CURRENT_ARTIFACT`, not reproduced.

On a measured, hash-locked reduced K=5 diagnostic, application sparsification
reduced a 1200-variable LP to 100 variables, but Gurobi presolve independently
reduced the dense LP to 0 variables because all demands were satisfiable. The
dominant reduced-scale gain was build/representation work. On a separate
same-instance 66-satellite/66-user diagnostic using a shipped checkpoint,
best-cold Gurobi took 1.458 ms, causal same-model/basis Gurobi 0.114 ms, SaTE
forward 6.283 ms, and SaTE end-to-end 14.921 ms. SaTE had an 18.27% objective
gap after feasibility correction. Thus the only defensible classification for
the **measured reduced diagnostic** is `SOLVER_ENGINEERING_DOMINATES`; the
paper-scale A/B/C/D classification is `NOT_IDENTIFIABLE_FROM_CURRENT_ARTIFACT`.

## 1. Paper/code configuration audit

See `docs/sate_speedup_attribution_audit.md`. Paper K=10 and code K=5 mismatch.
All local ratios use four matching canonical hashes. Synthetic K=10 sensitivity
uses genuine unique paths, but is diagnostic rather than a paper reconstruction.

## 2. Simplification stages

- S1 representation pruning measures bytes and parse/storage only.
- S2 exact sparsification removes zero-demand variables and constraints.
- S3 solver-native presolve and causal model/basis reuse are separate.
- S4 graph reduction affects only graph construction/message passing.
- S5 topology selection is offline-only and contributes 1x online.

## 3. Dataset and hardware

Measured on Intel Core i9-13900K (32 logical CPUs; Gurobi forced to one thread)
and NVIDIA RTX 4090, PyTorch 2.2.1, CUDA 12.1. Gurobi 13.0.3 used a legal
restricted non-production license expiring 2027-11-29. The paper dataset is
absent. LP L0--L4 uses a deterministic 16-node reduced topology; the shared
SaTE/Gurobi diagnostic has the 66-satellite plus 66-user shape expected by the
shipped Iridium checkpoint. Neither is a paper baseline.

## 4. Representation reduction

Measured diagnostic serialization fell from 18,298 B (240 pairs, 1,200 paths)
to 1,637 B (20 pairs, 100 paths), an 11.18x storage reduction. Dense/sparse
serialization medians were 0.0806/0.00589 ms; deserialization medians were
0.0415/0.00380 ms; measured Python deserialization peaks were 47,539/4,611 B.
Path generation plus canonical construction was 15.184 ms. Solver impact is not
inferred from these bytes. The paper-reported 335 GB to 15 MB (22,381x) remains a
`PAPER_REPORTED_REFERENCE` about representation/training feasibility.

## 5--7. Exact LP, presolve, and active subgraph

| Stage | Exact? | Vars | Constr | NZ | Build median | Optimize median | Total median | Local observation |
|---|---|---:|---:|---:|---:|---:|---:|---|
| L0 dense, presolve off | yes | 1200 | 304 | 5056 | 3.146 ms | 6.276 ms | 9.538 ms | reference |
| L1 dense, presolve on | yes | 1200 (0 presolved) | 304 (0) | 5056 (0) | 3.116 ms | 0.312 ms | 3.497 ms | 2.73x vs L0 total |
| L2 active flow, presolve off | yes | 100 | 84 | 454 | 0.534 ms | 0.214 ms | 0.774 ms | exact manual sparse |
| L3 active flow, presolve on | yes | 100 (0) | 84 (0) | 454 (0) | 0.538 ms | 0.101 ms | 0.660 ms | 5.30x vs L1 total |
| L4 active subgraph | yes | 100 | 84 | 454 | 0.538 ms | 0.100 ms | 0.703 ms | all 64 edges active; no reduction |

Every objective was exactly 242.0. Allocation equality was not required. The
0/0/0 presolved model shows this reduced workload is uncongested and therefore
does not establish paper-scale solver behavior.

K sensitivity (L3 total): K=1 0.433 ms, K=3 0.513 ms, K=5 0.647 ms, and
diagnostic K=10 0.980 ms. Dense K=10 was `SKIPPED_RESOURCE_BOUND` before model
construction (2,400 estimated variables exceeded the 2,000 guard).

## 8--9. Best cold and online commercial solver

Calibration was fixed to default, dual simplex, and barrier, five repeats each;
minimum median total selected default. Thirty formal repeats gave 1.458 ms total
(0.399 ms optimize). A causal previous-RHS, same-model dual-simplex reoptimization
with prior basis took 0.114 ms total (0.083 ms optimize), 12.83x faster than cold.
No future pair universe was used.

## 10. SaTE latency decomposition

After 20 warmups and 100 synchronized GPU repeats: input graph 3.053 ms,
TopoGNN 3.664 ms, AlloGNN 2.556 ms, full forward 6.283 ms, and
postprocess/repair 6.083 ms. The separately measured monolithic end-to-end
median was 14.921 ms (P90/P95 18.146/26.544 ms); the component-synchronized
sum was 15.585 ms and is retained only as instrumentation evidence. These are
current-hardware measurements on the diagnostic instance, not the missing
Starlink dataset.

## 11. GNN graph structure reduction

The measured current heterograph has node types flow/link/path with 20/396/100
nodes and two canonical edge types: flow-uses-path (100 edges) and
link-constitutes-path (1,087 edges). Paper R1 inter-satellite processing is
implemented separately in TopoGNN; R2/R3 map to the heterogeneous stages. The
paper removes access and folds link information, but no trained full-graph
checkpoint exists. Runtime contribution is
`NOT_IDENTIFIABLE_WITHOUT_TRAINED_FULL_GRAPH_BASELINE`.

## 12. Topology-pruning offline gain

The confirmed paper comparison is 8,000 training samples versus 512
representatives: 15.625x fewer labels/epoch samples. Using the measured
diagnostic cold-label time only as an estimated aggregate workload gives
11.6615 s versus 0.7463 s. This is not wall-clock training time and not a
Starlink estimate. Online contribution is exactly 1x.

## 13. Path computation

Paper reports fewer than 2% paths updated per second and 56 ms mean incremental
calculation. This is `PAPER_REPORTED_REFERENCE`. Missing input prevents a local
full/incremental invalidation benchmark: `NOT_IDENTIFIABLE_FROM_CURRENT_ARTIFACT`.

## 14. Quality parity

Shared-instance Gurobi objective was 229.3333. SaTE feasible objective was
187.4280 (18.27% gap). Maximum capacity excess fell from 16.6424 before repair
to 5.72e-6 after repair (FP32 residual). LP L0--L4 objective parity passed 1e-8.

## 15. Re-attribution of 2738x

The 335 GB to 15 MB fact is confirmed only as representation reduction. The
L0--L4 diagnostic confirms that zero-demand and unused-edge pruning are exact,
and that Gurobi may reproduce much of it via presolve. Commercial reuse can be
large (12.83x locally). The paper-scale residual attributable specifically to
GNN replacement is `NOT_IDENTIFIABLE_FROM_CURRENT_ARTIFACT`. Dividing paper
46--47 s by any local SaTE or Gurobi time is prohibited.

For the shared reduced diagnostic only:

- Cold neural gain = 1.458 / 14.921 = 0.0977x (SaTE is 10.24x slower).
- Neural replacement gain = 0.114 / 14.921 = 0.00761x (SaTE is 131.3x slower).

## 16. Limitations

No paper input artifact, paper-faithful K=10 adapter, solver parameter record,
or trained full heterogeneous-graph baseline is present. Full-scale Gurobi is
`FULL_SCALE_GUROBI_NOT_MEASURED`; dense full construction is prohibited. Old
repository CSVs use unsynchronized CUDA wall timing and remain historical
artifacts only.

## 17. Final classification

- Measured reduced diagnostic: `SOLVER_ENGINEERING_DOMINATES`.
- Paper-reported 4236-node result: `NOT_IDENTIFIABLE_FROM_CURRENT_ARTIFACT`.

The nine figures under `output/speedup_attribution/` preserve stage-local
ratios; no incompatible speedups are multiplied.
