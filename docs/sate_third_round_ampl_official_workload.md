# SaTE Third-Round Official-Workload AMPL Full-Scale Validation

## 1. Executive conclusion

本轮当前最终分类是 **BENCHMARK_INCOMPLETE**。AMPL Academic + Gurobi 已通过 5000/10000 变量实际求解，AMPL formulation 也通过四个第二轮 K10 诊断实例的 direct-gurobipy parity；但 official DataSetForSaTE25/50/75/100 的 A/B raw pickle 在协议路径内全部缺失。因此没有合法依据报告 official active SD、full-scale LP runtime、SaTE official timing、cold crossover 或论文 2738x 复现。

这不是 `AMPL_FULL_SCALE_BLOCKED`：size gate 已解除。下游是被数据 gate 阻断，而不是求解失败。

## 2. What changed from Round 2

新增 `SolverBackend`，正式后端为 `AMPL_GUROBI`，原 `DIRECT_GUROBIPY` 保留作 small-scale parity/reference。新增 sparse AMPL 内存模型、许可证安全审计、official pickle inventory、FlowSet 聚合、K10 unique-path audit 与增量 stage 状态。没有删除或重写既有 Gurobi LP。

## 3. AMPL/Gurobi license validation

| Item | Result |
|---|---:|
| Python | 3.10.18 (`torch280`) |
| amplpy | 0.18.0 |
| AMPL base module | AVAILABLE |
| AMPL Gurobi module | AVAILABLE |
| AMPL instantiate | SUCCESS |
| `AMPL_LICENSE_ENV_PRESENT` in Codex process | false |
| Activation provenance | PREACTIVATED_CONFIRMED_BY_USER |
| 5000-variable solve | COMPLETED, objective 5000, 0.054868 s wall |
| 10000-variable solve | COMPLETED, objective 10000, 0.059209 s wall |
| Size verdict | AMPL_GUROBI_FULL_SCALE_AVAILABLE |

UUID/identifier、license file 内容、activation stdout/stderr 均不写入 artifact。代码中的 activation 子进程将 stdout/stderr 丢弃，只保存 return code。

## 4. Dataset benchmark completeness

DataSetForSaTE25/50/75/100 的 volume A/B 共 8 个 required artifact 均未找到；usable files = 0。因此为 `BENCHMARK_INCOMPLETE`。详细 inventory 见 `docs/official_dataset_benchmark_inventory.md`。

## 5. Official FlowSet statistics

`NOT_RUN_GATE_BLOCKED`。没有 raw `FlowSet`，不能给出 raw-flow、active-SD 或 demand 分布。

## 6. 3M users -> active SD explanation

数学上，300 万 simulated users 不等于 300 万 LP variables。raw user flows 应先按 `(source satellite, destination satellite)` 聚合为 M 个 active SD pairs；nominal K10 variable count 约为 `10*M`，实际变量数则是各 pair 的 unique path 数之和。由于 official artifact 缺失，本轮不能识别 M 的实际分布。

## 7. Official K10 path audit

`NOT_RUN_GATE_BLOCKED`。实现调用公开 `SPOnGrid(..., 10)`，保持返回顺序去重，绝不复制 K5 或第一条路径补齐 K10；不足 10 条会标记 `K10_PATH_SHORTFALL`。

## 8. Official LP size

| Dataset | Snapshot | Raw flows | Active SD | K | Vars | Constr | NZ | Mean hops | Gurobi solve | Gurobi pipeline | SaTE forward | SaTE e2e |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DataSetForSaTE100 | NOT_AVAILABLE | NOT_IDENTIFIABLE | NOT_IDENTIFIABLE | 10 | NOT_IDENTIFIABLE | NOT_IDENTIFIABLE | NOT_IDENTIFIABLE | NOT_IDENTIFIABLE | NOT_RUN_GATE_BLOCKED | NOT_RUN_GATE_BLOCKED | NOT_RUN_GATE_BLOCKED | NOT_RUN_GATE_BLOCKED |

## 9. AMPL-vs-gurobipy parity

全部实例使用相同 demand/path/capacity/topology hashes。NZ 统一定义为约束矩阵 nonzeros，不含 objective 系数。

| Nodes | Active SD | Vars | Constraints | NZ | Objective (both) | Relative diff | Gate |
|---:|---:|---:|---:|---:|---:|---:|---|
| 16 | 16 | 160 | 103 | 1,060 | 180 | 0 | PASS |
| 66 | 66 | 660 | 410 | 5,291 | 807 | 0 | PASS |
| 128 | 128 | 1,280 | 794 | 11,792 | 1,575 | 0 | PASS |
| 192 | 180 | 1,800 | 1,186 | 18,793 | 2,166 | 0 | PASS |

最大 capacity/demand violation 均在 `1e-8` gate 内。结论：`AMPL_FORMULATION_PARITY_PASS`。

## 10. Gurobi full-scale latency

`NOT_RUN_GATE_BLOCKED`。不能回答 official Gurobi 是毫秒、秒或几十秒。

## 11. PAPER_LIKE vs CODE_PUBLIC capacities

实现中严格分离：`PAPER_LIKE_CONFIG=(network 200, access 50)`；`CODE_PUBLIC_CONFIG=(ISL 200, access 800)`。由于数据 gate，两套 official sensitivity 均未运行。

## 12. Presolve behavior

`NOT_RUN_GATE_BLOCKED`。未关闭 Presolve，也未制造慢 baseline。post-Presolve official size 与 IterCount 均不可用。

## 13. Solver cross-check

`NOT_RUN_GATE_BLOCKED`。HiGHS 模块已安装，但没有替代 Gurobi 主结果。

## 14. Official SaTE latency

`NOT_RUN_GATE_BLOCKED`。第二轮 4236-node、180-flow diagnostic 的 forward 18.238 ms / e2e 109.980 ms 仍只属于 `DIAGNOSTIC_K10_TRAINED`，不能当 official measurement。

## 15. Second-round -> third-round scaling extension

| Round | Workload | Active SD | Vars | NZ | Gurobi | SaTE e2e |
|---|---|---:|---:|---:|---:|---:|
| Round 2 | DIAGNOSTIC_SYNTHETIC, capped | 180 | 1,800 | 18,793 | 13.284 ms cold | 37.224 ms (N=192) |
| Round 3 | OFFICIAL_WORKLOAD, uncapped | NOT_IDENTIFIABLE | NOT_IDENTIFIABLE | NOT_IDENTIFIABLE | NOT_RUN_GATE_BLOCKED | NOT_RUN_GATE_BLOCKED |

因此第二轮曲线尚未被 official 点延伸。

## 16. Cold crossover

`NOT_EVALUATED`。没有 official same-instance AMPL/Gurobi 与 SaTE e2e，故不能报告 crossover active SD/vars/NZ。

## 17. Paper 46--47 s comparison

`NOT_EVALUATED`。没有 official runtime，不能给 `PAPER_GUROBI_LATENCY_ORDER_REPRODUCED` 或更强分类。

## 18. Paper ~17 ms comparison

第二轮 topology-shaped diagnostic forward 18.238 ms 与论文 marker 数值接近，但 official path-side workload 缺失，不能判断真实 uncapped workload 下是否仍接近 17–18 ms。

## 19. Re-attribution of 2738x

2738x 仍为 `NOT_IDENTIFIABLE_FROM_CURRENT_ARTIFACT`。本轮只证明 AMPL commercial backend 能越过 restricted size，并证明 formulation parity；尚未测到决定 ratio 的 official M、NZ、Gurobi wall、SaTE forward/e2e。

## 20. Remaining ambiguities

最大且当前唯一硬 blocker 是 official raw A/B pickle 的位置/可用性。拿到 artifact 后仍需审计 K10 shortfall、paper-like/code-public 容量、solver timing boundary、public diagnostic K10 checkpoint 与论文 checkpoint 的差异。

## 21. Final verdict

**BENCHMARK_INCOMPLETE**

已完成：Stage 0 size validation、Stage 1 formulation parity、数据 inventory。
被 gate 阻断：Stage 2 official FlowSet scan，以及 Stage 3–10 的统计、路径、full-scale solve、SaTE、cross-check 与图表。
当前不能声称 `OFFICIAL_WORKLOAD_EXPLAINS_SOLVER_GROWTH`、`OFFICIAL_COLD_CROSSOVER_OBSERVED`、`PAPER_SCALE_CROSSOVER_REPRODUCED` 或 `PAPER_SPEEDUP_REPRODUCED`。
