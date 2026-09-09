# SaTE Third-Round Official Workload Full-Scale Reproduction

## Executive conclusion

第三轮已从 official Google Drive 的 DataSetForSaTE100/A+B 完成 10,000-record 全量扫描、10 个 uncapped snapshot 的真实 K10 path 生成、AMPL Academic Gurobi full-scale 求解，以及 median/P90/P95 的同实例 GPU SaTE timing。主结论为：

- `OFFICIAL_COLD_CROSSOVER_OBSERVED`
- `PAPER_SCALE_CROSSOVER_REPRODUCED`
- `PAPER_GUROBI_LATENCY_ORDER_REPRODUCED`
- `PAPER_GUROBI_46S_APPROX_REPRODUCED=false`
- `PAPER_SPEEDUP_REPRODUCED=false`

官方 active SD 约为第二轮的 18 倍，LP 从 1,800 vars / 18,793 NZ 扩展到中位点 27,546 actual vars / 733,421 NZ；AMPL→Gurobi wall 从 13.284 ms 增至 78.165 s。SaTE forward 仍约 20 ms，但真实 graph-input 构造使 e2e 达 0.71–0.78 s。

## Dataset provenance and statistics

官方 archive 与两份 raw pickle 的大小、SHA256、5,000+5,000 clean EOF 见 `official_dataset_benchmark_inventory.md`。DataSetForSaTE100 的 10,000 records 分布如下：

| Metric | Median | P95 | Max |
|---|---:|---:|---:|
| Raw flows | 3,232 | 3,331 | 3,432 |
| Active SD | 3,225 | 3,325 | 3,428 |
| Nominal K10 vars | 32,250 | 33,250 | 34,280 |

相对第二轮 180 active SD，中位/P95/最大分别为 17.917x / 18.472x / 19.044x。统计不构造 4236×4236 dense matrix，也不设 active-SD cap。

## Official path and LP results

路径严格调用 `SPOnGrid(..., 10)`、按返回顺序去重，不复制第一条 path。ISL 完整路径为 `[src+4236] + satellite_path + [dst+4236]`。中位点 K10 completion 为 75.070%，804/3,225 pairs shortfall；因此 nominal 32,250 vars 对应 actual 27,546 vars。shortfall 不是被补齐或删除，而是 LP 使用真实 unique paths、SaTE 使用显式 inactive slots。

| Dataset | Vol | Snapshot | Raw flows | Active SD | K10 vars | Constr | NZ | Gurobi default | Gurobi T1 | SaTE forward | SaTE e2e |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DataSetForSaTE100 | A | median, idx 339 | 3,232 | 3,225 | 27,546 | 23,536 | 733,421 | 78.165 s | 86.745 s | 19.810 ms | 709.610 ms |
| DataSetForSaTE100 | A | P75, idx 7 | 3,273 | 3,264 | 27,906 | 23,590 | 746,025 | 82.829 s | — | — | — |
| DataSetForSaTE100 | A | P90, idx 38 | 3,307 | 3,302 | 28,041 | 23,681 | 755,012 | 77.644 s | — | 19.454 ms | 756.677 ms |
| DataSetForSaTE100 | A | P95, idx 103 | 3,331 | 3,325 | 28,499 | 23,761 | 770,389 | 82.016 s | — | 20.144 ms | 776.339 ms |
| DataSetForSaTE100 | A | P99, idx 733 | 3,367 | 3,363 | 28,784 | 23,930 | 768,088 | 81.231 s | — | — | — |
| DataSetForSaTE100 | A | max, idx 511 | 3,432 | 3,428 | 29,491 | 24,000 | 787,244 | 81.225 s | — | — | — |
| DataSetForSaTE100 | A | P10, idx 823 | 3,156 | 3,144 | 27,031 | 23,526 | 723,841 | 79.207 s | — | — | — |
| DataSetForSaTE100 | B | min, idx 750 | 2,998 | 2,992 | 25,482 | 23,196 | 679,474 | 68.086 s | — | — | — |

另有两个 deterministic near-median 点：77.396 s 与 78.334 s。所有 solver 点使用 Gurobi default Presolve；`GUROBI_DEFAULT` 不传 threads，第二轮衔接的中位点 `threads=1` 为 86.745 s。本轮按单线程测试约束未运行 threads=24。AMPL 无法可靠取得 post-Presolve size 与 Gurobi internal Runtime，分别标为 `PRESOLVE_POST_SIZE_NOT_AVAILABLE` 和 `GUROBI_INTERNAL_RUNTIME_NOT_AVAILABLE`；表中是实际 `T_ampl_to_solver_wall`。

中位点有 20,311 active LP edges，其中 2,432 access edges、17,879 network edges；mean/P95 path hops 为 25.625/43。所有 access edge 在 PAPER_LIKE 下为 50，在 CODE_PUBLIC 下为 800。

## Capacity sensitivity

中位点保持完全相同 topology/demand/path hashes：

| Config | Network | Access | Objective | Unsatisfied | Gurobi default |
|---|---:|---:|---:|---:|---:|
| PAPER_LIKE | 200 | 50 | 45,833.592 | 102,031.208 | 78.165 s |
| CODE_PUBLIC | 200 | 800 | 90,626.857 | 57,237.943 | 84.587 s |

access=50/800 均真实进入 capacity constraints。800 将 objective 提高约 97.73%，但未解释论文更慢的 solver runtime。

## SaTE official timing

使用 `PUBLIC_CODE_K10_DIAGNOSTIC_MODEL`，只解释 architecture/runtime，不声称 PAPER_MODEL 或 paper quality。对 shortfall 使用每个 flow 固定 10 个 architectural slots；只有实际 unique path slots 参与 link incidence，4,704 个中位点缺失槽位由 mask 强制为零，无 path duplication。所有 raw active flow（含 same-satellite aggregate flow）均保留。

| Snapshot | Flow nodes | Path slots / actual paths | Link nodes | Hetero edges | Forward | E2E | Peak allocated |
|---|---:|---:|---:|---:|---:|---:|---:|
| median | 3,225 | 32,250 / 27,546 | 26,351 | 738,125 | 19.810 ms | 709.610 ms | 3.825 GB |
| P90 | 3,302 | 33,020 / 28,041 | 26,385 | 759,991 | 19.454 ms | 756.677 ms | 3.936 GB |
| P95 | 3,325 | 33,250 / 28,499 | 26,427 | 775,140 | 20.144 ms | 776.339 ms | 4.009 GB |

RTX 4090、warmup=20、repeats=30，所有 CUDA timing 边界均同步。第二轮 18.238 ms 对 official median 19.810 ms，仅增加约 8.62%，因此 `FORWARD_ORDER_MATCHES_PAPER`；但 109.980 ms e2e 增至 709.610 ms，说明第二轮严重低估了 official graph-input system cost。

## Round 2 to Round 3 and crossover

| Round | Active SD | Vars | NZ | Gurobi | SaTE forward | SaTE e2e |
|---|---:|---:|---:|---:|---:|---:|
| Round 2 synthetic diagnostic | 180 | 1,800 | 18,793 | 13.284 ms | 18.238 ms | 109.980 ms |
| Round 3 official median | 3,225 | 27,546 | 733,421 | 78.165 s | 19.810 ms | 709.610 ms |

中位点 activeSD/vars/NZ/runtime ratios 为 17.917x / 15.303x / 39.026x / 5,884.2x。相同 official instance 上 forward 与 e2e 都小于 Gurobi，首次 cold crossover 点为 active SD=3,225、vars=27,546、constraints=23,536、NZ=733,421。

- forward-style ratio：78.165 / 0.019810 = 3,945.8x
- system-level ratio：78.165 / 0.709610 = 110.2x

官方 workload 解释了 solver 从 ms 到几十秒以及 crossover，也使 forward-style ratio 与 paper 2738x 同量级；但 system ratio 只有约 110x，且 Gurobi 78 s 不在 40–55 s 近似窗口、模型不是 paper checkpoint，因此不能声称 `PAPER_SPEEDUP_REPRODUCED` 或精确复现 46–47 s。

## Solver cross-check and figures

HiGHS 在相同中位点、单线程、600 s 子进程边界内未完成，分类为 `TIMEOUT`，不能给 objective parity，也不改变 Gurobi 主结果。10 张规定图位于 ignored artifact `output/speedup_attribution/ampl_full_scale/figures/`：active-SD、raw-vs-active、K10-vars、runtime-vs-active/vars/NZ、Round2→3、Gurobi-vs-SaTE、capacity sensitivity、paper references。

## Reproducibility boundary

Stage 0–10 均已实现；full-scale solve 每点独立 child process、timeout=600 s、状态增量写入，`--resume` 可跳过已有 instance/result。raw、ZIP、cache 与 output 均 Git ignored。工程验证与科学结论分开：公开 diagnostic K10 checkpoint 的质量为 `NOT_PAPER_QUALITY`，只使用其同步 CUDA latency。
