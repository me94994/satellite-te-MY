# SaTE Round 3.1：Full-Scale Gurobi Timing Decomposition and Reuse

## 1. Executive conclusion

在未改变 official FlowSet、active SD、K=10、candidate paths、capacity 或 Presolve default 的条件下，record 339 的 3 次 cold solve-call wall 中位数为 **80.847 s**，而 Gurobi driver 直接返回的 `time_solver`（`T_gurobi_optimize`）仅为 **9.546 s**。driver 五项只覆盖 solve wall 的 **12.170%**；未覆盖 wall 为 **70.938 s**。AMPL 自身 `_ampl_time` 的直接 CPU 增量为 **74.248 s**，因此 cold 主瓶颈是 AMPL 侧模型生成/序列化路径，而非 Gurobi optimize。

官方 AMPLS bridge 已成功保留一个 full-scale Gurobi model。10-snapshot fixed universe 的 9 次 transition 中，persistent no-basis 总延迟中位数为 **17.342 s**，相对 cold solver pipeline 快 **4.744x**；显式 previous basis 虽经 API、basis 数量和 solver log 三重验证，却将中位迭代数从 45,294 增至 77,869，总延迟恶化到 **49.475 s**。最强公平 classical reuse 仍比 SaTE e2e 709.610 ms 慢 **24.439x**。

最终分类：`MODEL_CONSTRUCTION_DOMINATES_COLD`、`REUSE_REMOVES_MODELING_BOTTLENECK`、`SATE_REMAINS_FASTER_AFTER_CLASSICAL_REUSE`。

## 2. Why 78 s needed decomposition

第三轮的 78.165 s 是 `ampl.solve(solver="gurobi")` 外层 wall，不能等同于 Gurobi optimize。本轮只复用同一 official canonical workload，把 application、AMPL、driver、native optimize、output 和 Python extraction 分开计时；`T_gurobi_optimize` 从 driver/native Gurobi statistics 直接取得，未用 wall 相减推导。

## 3. Timing definitions

- Application/canonical：`T_pickle_read`、`T_flow_aggregate`、`T_path_generation_uncached`、`T_path_cache_lookup`、`T_canonical_instance`。
- AMPL：`T_sparse_rows`、`T_ampl_create`、`T_ampl_model_definition`、`T_ampl_data_load`。
- Driver：`T_driver_read`、`T_driver_conversion`、`T_driver_setup`、`T_gurobi_optimize`、`T_driver_output`。
- Result：`T_solution_extract`、`T_feasibility_check`。
- Totals：`T_solve_call_wall` 是 solve 外层 wall；`T_solver_pipeline` 从 sparse rows 到 feasibility；`T_full_pipeline` 从 raw snapshot 到最终可行解。

输入严格分为 `APPLICATION_INPUT = T_sparse_rows + T_ampl_data_load = 1.232 s` 和 `SOLVER_INPUT = T_driver_read + T_driver_conversion = 0.067 s`；`T_input_total = 1.299 s`。建模严格分为 `AMPL_MODEL_BUILD = T_ampl_create + T_ampl_model_definition = 0.096 s` 与 `GUROBI_SETUP = 0.068 s`；按规定口径 `T_modeling_total = 0.164 s`。这些显式 API 调用不包含 solve 内部的 AMPL generation/serialization。

## 4. AMPL/Gurobi driver timing capability

实际环境为 `ampl-module-gurobi 20260908`、`amplpy-gurobi 0.2.4`。安装 driver help 和小 LP 共同验证：

- `tech:timing=2` 返回 `time_read`、`time_conversion`、`time_setup`、`time_solver`、`time_output`，语义均为 driver wall；`time_solver` 是直接 solver timing。
- `tech:stats=3` 返回 simplex/barrier/node statistics。
- `alg:basis`、`alg:start`、`lp:warmstart` 存在；小 LP 的重复求解日志验证了 basis warm-start。
- direct `gurobipy` 6,001-variable probe 得到 `DIRECT_GUROBIPY_FULL_SCALE_NOT_LICENSED`，没有绕过许可限制。
- 官方 `AMPL.to_ampls("gurobi")` bridge 在 34,511 variables / 25,424 constraints / 920,850 NZ 上验证成功，因此 persistent 结果不是 AMPL repeated solve 的冒充。

## 5. Cold timing decomposition

record 339、default method 的独立 cold solve wall 为 80.847、84.577、79.547 s；对应 native optimize 为 9.789、9.385、9.546 s。下表均取中位数，“%”统一除以 `T_solve_call_wall`；pipeline 外项目的百分比仅用于量级对照。

| Metric | Median | % of solve wall |
|---|---:|---:|
| sparse rows | 0.113 s | 0.139% |
| AMPL create | 0.096 s | 0.118% |
| AMPL model definition | 0.000100 s | 0.0001% |
| AMPL data load | 1.119 s | 1.384% |
| driver read | 0.017 s | 0.021% |
| driver conversion | 0.050 s | 0.062% |
| Gurobi setup | 0.068 s | 0.084% |
| Gurobi optimize | 9.546 s | 11.807% |
| driver output | 0.000070 s | 0.0001% |
| solution extract | 0.013 s | 0.017% |
| solve call wall | 80.847 s | 100.000% |
| full solver pipeline | 82.271 s | 101.761% |

Driver components sum 为 **9.681 s**，`accounted_fraction = 0.121696`，`unaccounted_wall = 70.938 s`，非负 timing gate PASS。`T_solver_pipeline` 还包括 solve 外 sparse/data/extract/check，故可以大于 solve wall。

## 6. Where the 78 s is spent

Gurobi optimize 只占 solve wall 的 11.807%，不能把 78 s 称为 solver optimize。未覆盖 wall 占 87.830%；独立的 AMPL internal CPU 计时为 74.248 s，与该量级一致，但 CPU time 与 wall time 不是可加分项，不能用它精确填平 accounting。证据支持“AMPL generation/serialization 主导”，不支持把 70.938 s 再武断拆成更细字段。

Cold default 的 simplex iterations 为 12,128、barrier iterations 为 29、node count 为 0。Presolve 从 23,536 rows / 27,546 cols / 733,421 NZ 降至 18,005 rows / 27,428 cols / 701,104 NZ，即删除 1,263 rows 和 118 cols。

## 7. Ordered official sequence

序列是 `DataSetForSaTE100/A` 的 ordered raw records 329..349，共 21 个 snapshot；artifact 中没有时间间隔证据，因此不称为 1-second sequence。record 339 仍为 3,232 raw flows、3,225 active SD、27,546 unique path vars、23,536 constraints、733,421 NZ。

在已有 K10 cache 下，record 339 的 application measurements 为：`T_pickle_read=7.063 s`、`T_flow_aggregate=0.001 s`、`T_path_generation_uncached=0`、`T_path_cache_lookup=53.742 s`（path-phase wall 53.858 s）、`T_canonical_instance=0.172 s`。按 observable wall 口径，`T_full_pipeline = 7.063 + 0.001 + 53.858 + 0.172 + 82.271 = 143.365 s`。pickle 是包含 5,000 records 的单一 list，此读取成本没有伪装成单-record I/O。

## 8. Structure reuse opportunities

21 个 snapshot 分别记录 topology、active SD、demand、candidate path、path incidence、capacity structure 和完整 structure hashes。20 个相邻 transition 中：

- `EXACT_SAME_STRUCTURE`: **0**。
- `SAME_SUPERSET_COMPATIBLE`: **20**。
- `STRUCTURE_REBUILD_REQUIRED`: **0**（对本轮 fixed-universe implementation）。

因此没有把 B 类 transition 偷标为 A 类；正式 persistent reuse 使用 fixed-universe exact construction。

## 9. Fixed-universe exact reuse construction

union model 对 inactive flow 设置 demand=0，对 inactive semantic path 设置 UB=0，对 snapshot 中不存在的 edge 设置 capacity=0；active path 恢复其 UB。由此每个 snapshot 的可行变量、flow conservation 和 edge capacity 与其 cold 模型逐项等价。全部测试点相对 objective 误差约 `1e-15`，最大 feasibility violation 小于 `3.4e-11`，满足 `1e-8` parity gate。

| Window | Vars | Constraints | NZ | Vars / median | NZ / median |
|---|---:|---:|---:|---:|---:|
| 5 snapshots | 30,991 | 24,590 | 827,186 | 1.125x | 1.128x |
| 10 snapshots | 34,511 | 25,424 | 920,850 | 1.255x | 1.256x |
| 21 snapshots | 44,228 | 27,325 | 1,180,434 | 1.609x | 1.609x |

正式 reuse 采用居中的 10-snapshot union（records 334..343），测量 9 个有序 transition（335..343）；这是 full-scale 每次仍需数十秒时的受控窗口。

## 10. Same-model reuse without basis

R1 `OFFICIAL_AMPL_OBJECT_REUSE` 保持同一 AMPL object/model definition，但每次重填完整集合和数据。5 次 transition 的 data update 中位数 1.018 s、总时间 89.407 s、Gurobi optimize 18.803 s；每次仍出现约 0.017 s driver read，并重新 setup，故 **不是** persistent Gurobi reuse。

R2 `OFFICIAL_PERSISTENT_REUSE_NO_BASIS` 通过 AMPLS 一次 export 后保持同一个 native Gurobi model，只更新 RHS/UB/capacity。每次显式 `GRBreset` 并设置 `LPWarmStart=0`，禁用 previous basis、primal start 和 dual start。一次性 initial build/export 为 92.593/90.936 s，不计入 steady-state online latency。

## 11. Previous-basis reuse

R3 `OFFICIAL_PERSISTENT_REUSE_PREVIOUS_BASIS` 在同一 sequence 上显式导出/导入 34,511 个 `VBasis` 和 25,424 个 `CBasis`，设置 `LPWarmStart=2`。9/9 transition 的 API state 与 solver log 均确认 basis 被使用，故不是依据迭代变化倒推。

但 previous basis 在该结构变化序列上不合适：迭代中位数由 45,294 上升至 77,869，optimize 由 17.248 s 上升至 49.362 s。Presolve-off 只作为 sanity，未进入主结果；主结果 Presolve 保持 default，并出现 warm-start crush 证据。

## 12. Iteration/presolve evidence

Persistent union 的首次主结果 Presolve 从 25,424 rows / 34,511 cols / 920,850 NZ 删除 7,334 rows / 7,157 cols，得到 18,090 rows / 27,354 cols / 700,444 NZ。LP node count 为 0。reuse no-basis 的 iteration median/P90/P95/max 为 45,294 / 48,993 / 51,167 / 53,341；basis 为 77,869 / 118,226 / 120,397 / 122,567。

## 13. Cold vs reuse comparison

表中 optimize 均为 Gurobi 自身直接统计；reuse total 是 update/import/native optimize/extract/check/export 之和。Cold total 使用完整 solver pipeline；Build/setup 仅列明确定义的 AMPL build + driver setup，不吸收未解析 wall。

| Mode | Update | Build/setup | Optimize | Extract | Total | Iter |
|---|---:|---:|---:|---:|---:|---:|
| Cold | 1.232 s application input | 0.164 s | 9.546 s | 0.013 s | 82.271 s | 12,128 |
| Reuse no basis | 0.048 s | 0 steady-state | 17.248 s | 0.004 s | 17.342 s | 45,294 |
| Reuse + basis | 0.050 s + 0.006 s import | 0 steady-state | 49.362 s | 0.004 s + 0.005 s export | 49.475 s | 77,869 |

Reuse sequence statistics：no-basis total median/P90/P95/max 为 17.342/20.377/20.401/20.425 s；basis 为 49.475/192.325/211.421/230.518 s。`ModelReuseSpeedup=4.744x`，`BasisIncrementalSpeedup=0.351x`，`TotalReuseSpeedup=1.663x`。这里的 TotalReuseSpeedup 按指定公式使用 R3；工程上最快的是 R2，而不是 R3。

## 14. Gurobi vs SaTE after reuse

| Method | Latency | vs SaTE forward | vs SaTE e2e |
|---|---:|---:|---:|
| Gurobi cold | 80.847 s solve wall | 4,081.145x | 113.932x |
| Gurobi reuse no basis | 17.342 s | 875.422x | 24.439x |
| Gurobi reuse+basis | 49.475 s | 2,497.499x | 69.722x |
| SaTE forward | 19.810 ms | 1x | 0.028x |
| SaTE e2e | 709.610 ms | 35.821x | 1x |

## 15. Implication for paper 2738x

论文约 46–47 s / 17 ms 对应约 2,706–2,765x，通常概括为 2,738x。用本轮最强 classical online baseline（persistent no-basis 17.342 s）和同一个 17 ms reference，倍率变为约 **1,020x**：solver engineering 明显削弱了 paper-like speedup，但没有消除数量级差距。用本仓库实测 SaTE e2e 709.610 ms 比较，系统级优势是 **24.439x**，而不是把 forward-only 与 solver total 混为一谈。basis 路线更慢，不应被选择为“最强”baseline。

## 16. Limitations

- driver timing 未覆盖 70.938 s solve wall；AMPL internal CPU 提供归因证据，但不能当作 wall 分项强行闭合。
- ordered records 没有 timestamp interval 证据。
- exact-same-structure pair 为零，因此 R2/R3 是经过 parity 证明的 fixed-universe exact reuse，不是 A 类原生相同结构。
- 10-snapshot union 的 9 个 transition 已满足“仍为数十秒时至少 5 次”；它不是对全部 20 个 transition 的延迟分布估计。
- basis warm-start 已验证，却在本序列恶化；不能外推为所有 workload 都无益。
- application path-cache lookup 仍很昂贵，但它属于 canonical pipeline，不计入 steady-state persistent solver latency。

## 17. Final verdict

78 s 的主要部分不是 Gurobi optimize；真正 optimize 中位数为 9.546 s。AMPLS persistent model 将 online classical latency 降至 17.342 s，但 default-Presolve 下 previous simplex basis 使其恶化到 49.475 s。因此 corrected result 是：模型复用能移除大部分 cold 建模瓶颈，basis 没有移除 optimize 瓶颈，且 SaTE e2e 在最强公平 reuse 后仍快 24.439x。

正式 artifacts 位于 `output/speedup_attribution/gurobi_decomposition/`；八幅图分别覆盖 cold decomposition、total、optimize-only、iterations、ordered sequence、NZ/reuse benefit、Gurobi/SaTE 和 paper reference/corrected baseline。
