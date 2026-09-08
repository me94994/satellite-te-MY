# SaTE Speedup Attribution: Paper--Code Configuration Audit

Audit HEAD: `84543f0c3bee9112178ae5d6d54c21571bcba8d1`.

| Source / component | Candidate paths | Traffic representation | Objective | Capacities | Topology | Dataset | Solver settings |
|---|---:|---|---|---|---|---|---|
| Paper (`PAPER_CONFIG`) | 10 shortest paths / SD pair | 4236 x 4236 dense before pruning; only nonzero traffic and associated paths after pruning | Max total throughput | 200 Mbps network links; 50 Mbps access links stated in evaluation | 66, 396, 1584, 4236 nodes | 10,000 snapshots, 4:1; 8,000 training reference, up to 512 representatives | `NOT_IDENTIFIABLE_FROM_CURRENT_ARTIFACT` |
| `run/spaceTE.py` + `run/_common.py` (`CODE_CONFIG`) | default 5 | adapter list-of-dict; represented traffic/path keys | default total flow | Starlink `OrbitParams`: 200/800/800 | selected by dataset path | external adapted pickle | no solver |
| `run/lp.py` | default 5 | dense matrix for generic input | same objective enum | graph attributes | selected by dataset path | external generic input | library defaults |
| `baselines/run/lp.py` | hard-coded 5 in both Starlink branches | adapter list-of-dict converted to matrix | same objective enum | reconstructed graph | dataset mode/size | external adapted pickle | current `lp_solver` defaults to concurrent; threads optional |

## Audit decision

`PAPER_CONFIG != CODE_CONFIG`: the paper explicitly says ten precomputed shortest
paths per satellite pair, while both executable defaults are five and the
Starlink Gurobi baseline contains `num_paths = 5` hard-coding. The current
worktree has no `input/` directory, so a paper-faithful ten-path adapter artifact
cannot be inspected or reconstructed.

Formal local comparisons therefore use `CODE_NATIVE_K5_ONLY` and assert the same
demand, path, capacity, and topology hashes. The paper's 46--47 s and 17 ms
values remain `PAPER_REPORTED_REFERENCE`; they are never divided by local
measurements. Paper-faithful K=10 is
`NOT_IDENTIFIABLE_FROM_CURRENT_ARTIFACT`.
