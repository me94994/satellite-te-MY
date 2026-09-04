# 真实 Starlink topology-transition optimizer-state transport 最终资格审查

日期：2026-09-05

## 1. 结论先行

本轮在 SaTE 官方 `DataSetForSaTE25` volume A 的完整 5,000-record pickle stream
中扫描到 1,862 次与当前 `ISM.ISL` TE 模型一致的 topology transitions，并对按
原始顺序出现的前 50 次 transition 完成 event-centered、pair-universe、单线程
Gurobi 13.0.3 实验。预注册判决为：

`STOP_TOPOLOGY_TRANSITION_TRACKING`

原因不是 event 不足或 workload 无拥塞。Gate T0 达到 STRONG，input-only
HIGH_PRESSURE_SLICE 在未缩放的官方 intensity-25 demand 下也确实产生拥塞；但
best classical previous-basis reuse 在 stable 与 transition 上的 median IterCount
都为 0，transition optimize-time 中位数只增加 13.20%，未达到 20% degradation
阈值。完整 transported PStart+DStart 反而从 0 墠至 55.5 iterations，并使
optimize/end-to-end latency 显著恶化。距离更近没有转化为 solver-level benefit。

因此本轮不授权 graph-state transport、learned residual、1--3 step refinement，
也不授权任何 GNN/GAT/GRU/LSTM/Transformer/RL/DeepLaDu/TELGEN 开发。

## 2. 起点、审查与根因修复

- 起始 HEAD：`d86a381f2548da0be228a4fc5130fa5b2c24061f`
  (`Validate temporal TE on real Starlink sequences with Gurobi`)。
- 复查了两个既有报告、`sequence_schema.py`、`semantic_alignment.py`、
  `sequential_lp.py`、`analyze_continuity.py`、`benchmark_warm_start.py`、
  `real_sequence.py`、`run_real_experiment.py`、`gurobi_probe.py` 与原测试。
- 既有 semantic path key、fixed-universe availability、capacity=0 future-edge、
  VBasis/CBasis 完整向量和 `LPWarmStart=1/2` 实现没有发现 parity 根因错误。
- 上轮只在 100 records 的 selected-path induced graph 上看到 3 次 transition；
  本轮将 transition detection 移到 raw stream，并明确分离：
  - full raw signature：`InterShell_ISL + InterShell_GrdRelay`；
  - routing signature：仅官方 `ISM.ISL` adapter 真正加入 TE 图的
    `InterShell_ISL` 边。
  Gate T0 与 LP event 只使用 routing signature，避免把 relay attachment 变化
  错当成当前 LP 的图变化。
- 微秒级 benchmark 最初固定先跑 transition、再跑 stable，会给后者系统性
  cache/process warm-up 优势；正式版本按 event 奇偶交替 pair 顺序，消除该
  timing-order 根因。
- demand-matched random 最终按完整 5-record window 的真实 demand/presence
  signature 精确匹配，并限制 path-count ratio 在 `[0.8,1.25]`；不再只匹配
  flow count。

## 3. 数据、provenance 与 lightweight scan

唯一扫描文件：

- sequence_id：`DataSetForSaTE25_A`
- 文件：`StarLink_DataSetForAgent25_5000_A.pkl`
- 官方 workload intensity：25
- raw records：5,000
- 文件大小：278,509,040 bytes
- SHA256：`ba362369d64587365668b08150447c764b67f996beea17bc25d941f73026a9fe`
- 来源：仓库 README 指向的 SaTE 作者公开 Google Drive
- timestamp / orbit epoch / physical cadence：均不存在

扫描器只顺序反序列化 `FlowSet`、`InterShell_GrdRelay`、`InterShell_ISL`，不
生成 candidate paths、不构造 Gurobi/SaTE/DGL graph。不同 source 必须显式给
不同 `sequence_id`，跨文件比较计数固定为 0。

扫描结果：

- routing topology transitions：1,862 / 5,000 records，frequency=0.3724/record；
- full raw signature transitions：3,553；
- routing regimes：1,863；
- regime length minimum/P10/P50/P90/max：`1 / 1 / 2 / 5 / 19` records。

没有物理 cadence provenance，因此不换算为秒，也不使用 Ephemeris-Aware
命名。已记录的 future topology 只能标为 `ORACLE_RECORDED_SEQUENCE`。

## 4. Event-centered 数据与 selection controls

正式实验使用 earliest 50 transitions，选择规则不查看任何 solver outcome。
每个 event 保留 `tau-2 ... tau+2`，transition pair 为 `(tau-1,tau)`，并在 raw
routing hash 相等时保留一个 flanking stable pair。50 events 中 48 个有可用
stable control。

三个 slice 的 event-window universe 范围（minimum / median / maximum）：

| slice | flows | paths/variables | edges | constraints |
|---|---:|---:|---:|---:|
| PERSISTENCE | 42 / 44 / 47 | 198 / 212 / 222 | 1366 / 1496.5 / 1605 | 1606 / 1746.5 / 1869 |
| HIGH_PRESSURE | 42 / 44 / 47 | 178 / 209 / 227 | 1077 / 1316.5 / 1636 | 1302 / 1567 / 1896 |
| DEMAND_MATCHED_RANDOM | 42 / 44 / 47 | 169 / 191 / 212 | 1476 / 1631 / 1670 | 1688 / 1869 / 1900 |

random 与 high-pressure 在 50/50 events 的完整 window total-demand trajectory
完全相同，path-count ratio 范围为 `0.8164--1.0558`。三者使用同一个
`constraints <= 1900` 安全上限。selection inputs 只有 FlowSet demand、官方
candidate paths 与固定 capacity policy；Gurobi objective、Pi、binding、
IterCount、optimal path flow 均不进入选择。

## 5. Pair-universe 与 Gurobi API

每个 pair 都固定 union flows、semantic paths、edges 与矩阵结构。在某一侧
不存在的 path，其 availability RHS=0；不存在的 edge，其 capacity RHS=0。
这是 `KNOWN-NEXT-TOPOLOGY PAIR-UNIVERSE`，即允许 classical baseline 使用
recorded next topology，并明确不是 topology prediction。

Gurobi provenance：

- gurobipy 13.0.3，tiny LP 为 OPTIMAL；
- 2,001-variable probe 命中 size-limit，分类为
  `RESTRICTED_SIZE_LIMIT_CONFIRMED`；
- `Threads=1, Method=1, Seed=42, OutputFlag=0`；
- `GRB_LICENSE_FILE`、`GUROBI_HOME`、WLS 环境均未设置；只记录布尔状态，不
  记录任何 secret。

根据 Gurobi 官方的 [PStart attribute 文档](https://docs.gurobi.com/projects/optimizer/en/current/reference/attributes/variable.html#pstart)、
[DStart attribute 文档](https://docs.gurobi.com/projects/optimizer/en/current/reference/attributes/constraintlinear.html#dstart)
与 [LPWarmStart 参数文档](https://docs.gurobi.com/projects/optimizer/en/current/reference/parameters.html#parameter-LPWarmStart)，
PStart/DStart 必须覆盖完整 variable/constraint vector；`LPWarmStart=2` 用于把
advanced start 映射到 presolved model。由于本轮为 dual
simplex，单独 PStart 不是其优先 start，故仅作为受支持但预期不生效的诊断。
PStart+DStart 提交前先 `reset(0)`，确保 retained basis 不会暗中主导结果。

## 6. 官方 workload congestion

本轮没有缩放 demand；以下全部是 `OFFICIAL_REAL_WORKLOAD` intensity 25，且
只代表 input-only restricted slices，不是 full-scale Starlink workload。

| slice | binding-rate median | nonzero capacity-dual rate median | events with unsatisfied demand | median unsatisfied demand |
|---|---:|---:|---:|---:|
| PERSISTENCE | 0.9412% | 0.0627% | 22% | 0 |
| HIGH_PRESSURE | 1.6030% | 0.3697% | 100% | 815 |
| DEMAND_MATCHED_RANDOM | 0% | 0% | 0% | 0 |

HIGH_PRESSURE 相对 persistence 的 paired binding-rate median difference 为
0.009110，bootstrap 95% CI `[0.005506,0.012644]`，Wilcoxon p=`8.78e-10`；
相对 demand-matched random 为 0.016030，CI `[0.013356,0.018082]`，
p=`8.88e-16`。unsatisfied-demand 相对两类 control 的 median difference 分别
为 810 与 815，CI 均严格大于 0，p=`3.76e-10`。

所以 high-pressure selection 确实构造了有信息的真实-demand congestion regime；
无需运行 1.5/2/4 scaled stress，`stress_scales_run=[]`。这不等于随机官方
workload 普遍拥塞，而是证明 input-only restricted selection 可以暴露真实
traffic 下的竞争资源。

## 7. Transition severity 与 path churn

50 个正式 transitions 的 raw severity：

- edge births median/mean/P90/max：`2 / 2.28 / 4 / 6`；
- edge deaths median/mean/P90/max：`2 / 2.32 / 4 / 6`；
- total edge changes median/mean/P90/P95/max：`4 / 4.6 / 8 / 12 / 12`；
- edge Jaccard median：0.998376；
- degree-change L1 median：4；
- raw active-flow churn median：0.05736。

HIGH_PRESSURE pair path severity：

- path survival ratio median/mean：`0.97608 / 0.97132`；
- path invalidation median/mean/P90/P95/max：
  `0 / 0.01285 / 0.02800 / 0.04785 / 0.07805`；
- new-path rate median/mean/P90/max：`0 / 0.01645 / 0.04958 / 0.07353`；
- affected-flow ratio median/mean/P90/max：
  `0.02273 / 0.02386 / 0.04556 / 0.08889`。

Path invalidation 与 best-classical IterCount 的 Spearman rho=0.3347，p=0.0175；
与 optimize time 的 rho=0.3378，p=0.0164。按 invalidation tertile（ties 导致组
大小不相等），small/medium/large 的 median IterCount 为 `0/1/1`。这说明 hard
tail 与 path churn 有关联，但 classical median transition solve 仍为 0 iteration，
且 transport 没有 operational benefit，因此不能据此授权 learned hard-event
solver。

## 8. Stable vs transition classical reoptimization

Gate primary slice 为 HIGH_PRESSURE；selection controls 分开汇总，不把同一
transition 的三种 slice 当成三个独立事件。

Best classical 为 `DIRECT_PREVIOUS_BASIS`：

| regime | n | IterCount median/mean/P90/P95 | optimize median | end-to-end median |
|---|---:|---:|---:|---:|
| STABLE | 48 | 0 / 2.104 / 6.6 / 10.3 | 0.2257 ms | 1.2764 ms |
| TRANSITION | 50 | 0 / 4.06 / 14.1 / 17.1 | 0.2555 ms | 1.3352 ms |

ratio-of-medians optimize increase 为 13.20%。48 个同 event stable/transition
pair 的 IterCount increase median=0，bootstrap CI `[0,0.5]`，one-sided
Wilcoxon p=0.0996；log optimize-time increase median=0.09515，CI
`[-0.02111,0.19001]`，p=0.0617。按预注册阈值 median IterCount <=1 且
optimize increase <20%，Gate T1 为 `NO_DEGRADATION`。

作为尺度参考，transition cold median 为 85.5 iterations、1.6217 ms optimize、
9.6988 ms total；pair-universe basis reuse 已经极强。

## 9. Deterministic state transport

所有距离为相对当前原 LP optimum 的 normalized L1；path-flow 受 LP degeneracy
影响，因此不把它解释为唯一 primal geometry。可选 canonical-primal 目标已实现
为完全 time-independent 的 hop-count + semantic-key tiny tie-break，并有测试
证明接口不接受 previous solution，但正式主结果未运行该可选诊断，状态为
`NOT_EVALUATED_OPTIONAL_DIAGNOSTIC`。

HIGH_PRESSURE 50-transition median distance：

| state / initialization | median distance |
|---|---:|
| edge load, repaired path transport | 0.17264 |
| edge load, new-edge zero | 0.16968 |
| edge load, new-edge neighbor mean | **0.16828** |
| edge load, new-edge global mean | 0.17432 |
| edge load, no-history equal split | 0.75603 |
| edge load, all zero | 2.0 |
| utilization, repaired path transport | 0.17306 |
| utilization, new-edge neighbor mean | **0.17202** |
| utilization, no-history equal split | 0.76967 |
| path flow, repaired transport | 0.44167 |
| path flow, no-history equal split | 1.56803 |
| binding-set Jaccard distance, repaired transport | 0.26121 |
| dual, best zero/global-median-new transport | 0.5 |
| dual, all zero | 2.0 |

Transport 显著优于 zero、global mean 与 input-only no-history：edge-load 相对
no-history 的 median benefit=0.63615，CI `[0.54118,0.68742]`，p=`8.88e-16`；
utilization 为 0.64719，CI `[0.54836,0.70308]`，p=`8.88e-16`。但是 demand、
capacity、availability repair 在这 50 个 previous solutions 上没有改变 direct
semantic copy：三种 primary state 相对 `direct_copy_no_repair` 的 paired median
benefit 都为 0，CI `[0,0]`，p=1.0。new-edge neighbor/zero/global policy 相对
direct copy 的 paired median benefit 也均为 0，未达到有意义 effect threshold。

因此 Gate T2 为 `FAIL`。最有效的 state-level 信息是 persistent edge load/
utilization 与 semantic path flow；binding/dual 在 high-pressure slice 已非全零，
但没有证据表明它们是优于 direct copy 的 transition state，更不能预设 lambda
是最佳 state。

## 10. Transport 是否降低 solver cost

HIGH_PRESSURE transition 结果：

| method | IterCount median | optimize median | operational total median |
|---|---:|---:|---:|
| best classical direct basis | **0** | **0.2555 ms** | **1.3352 ms** |
| auto model reuse | 0 | 0.2565 ms | 1.1567 ms |
| basis + presolve | 0 | 1.1493 ms | 2.4076 ms |
| reset | 85.5 | 1.5618 ms | 2.5559 ms |
| transported PStart only | 85.5 | 1.5829 ms | 5.8034 ms |
| transported PStart+DStart | 55.5 | 1.4231 ms | 5.8184 ms |
| zero PStart+DStart | 153.5 | 1.7366 ms | 3.0016 ms |

PStart only与 reset 完全一致，符合 dual-simplex API priority 的预期。完整
PStart+DStart 比 zero start 好，但无法接近 retained basis。

相对 best classical，transported PStart+DStart：

- paired IterCount difference `(classical - transport)` median=-54，bootstrap
  95% CI `[-59,-50.5]`，benefit-direction Wilcoxon p≈1，rank-biserial=-1；
- log optimize speedup median=-1.7435，CI `[-1.7861,-1.6462]`，p=1；等价于
  **-471.7% reduction**，即 transport optimize time 约为 classical 的 5.72 倍；
- end-to-end log speedup median=-1.4561，CI `[-1.5168,-1.4079]`，p=1；等价于
  **-328.9% reduction**，即 operational total 约为 classical 的 4.29 倍。

所有 1,200 个 transition method solves（50 events × 3 slices × 8 methods）均
objective parity PASS、feasibility PASS；失败不是由不可行 start 或目标改变造成。
Gate T3 为 `FAIL`，而且方向是显著恶化，不只是“不显著改善”。

## 11. Gate T0--T4 与最终判定

| Gate | 结果 | 证据 |
|---|---|---|
| T0 enough events | **STRONG** | 全流 1,862 transitions；正式评价 50 |
| T1 classical degradation | **NO_DEGRADATION** | stable/transition median IterCount=0/0；optimize +13.20% <20% |
| T2 state transport benefit | **FAIL** | 优于 no-history，但不优于 direct copy/repair baseline |
| T3 operational solver benefit | **FAIL** | 55.5 vs 0 iterations；optimize 与 total 显著恶化 |
| T4 congestion informativeness | **PASS_OFFICIAL_REAL_WORKLOAD** | unscaled intensity-25 high-pressure 有 binding、dual、unsatisfied demand |

T0 PASS/STRONG + T1 NO_DEGRADATION 已直接落入预注册 CASE A；T2/T3 的负面
结果进一步排除 learned solver。最终：

`STOP_TOPOLOGY_TRANSITION_TRACKING`

## 12. 逐项回答

1. 扫描 raw records：5,000。
2. 真实 routing topology transitions：1,862；正式 solver 评价 50。
3. regime length P10/P50/P90/max：1/2/5/19 records。
4. severity：edge changes median/mean/P90/P95=4/4.6/8/12。
5. 官方 workload intensity：25 volume A；因 transitions 足够，未下载 100。
6. 真实 congestion：是，只在 input-only selected restricted slices 范围内陈述。
7. binding rate：high-pressure median 1.6030%；persistence 0.9412%；random 0%。
8. nonzero capacity-dual ratio：high-pressure median 0.3697%；persistence
   0.0627%；random 0%。
9. high-pressure 更拥塞：是；50/50 unsatisfied vs 11/50 persistence、0/50 random，
   paired CI 与 Wilcoxon 均通过。
10. stable classical warm IterCount：median 0，mean 2.104，P90/P95=6.6/10.3。
11. transition classical warm IterCount：median 0，mean 4.06，P90/P95=14.1/17.1。
12. classical transition degradation：NO_DEGRADATION；optimize median +13.20%。
13. path invalidation 与 solver cost：rho=0.3347 iterations、0.3378 time，p<0.05；
    但 median invalidation=0，classical median iterations=0。
14. state transport：edge-load neighbor mean 最小 median 0.16828；utilization
    neighbor mean 0.17202；path 0.44167；binding 0.26121；dual best 0.5。
15. transport 只距离更近，没有减少 solver iterations。
16. PStart/DStart API：支持且完整提交；PStart-only 不生效，PStart+DStart 有效
    改变 crash basis，但远弱于 basis reuse。
17. 相对 best classical：IterCount 增加 median 54；optimize 约慢 5.72 倍，total
    约慢 4.29 倍。
18. 统计：T3 iteration CI `[-59,-50.5]`；optimize log CI
    `[-1.7861,-1.6462]`；benefit-direction p≈1。
19. congestion 来源：官方未缩放 real demand；没有 scaled stress。
20. Gate：T0 STRONG、T1 NO_DEGRADATION、T2 FAIL、T3 FAIL、T4 PASS_OFFICIAL。
21. 最终：`STOP_TOPOLOGY_TRANSITION_TRACKING`。

## 13. Blockers 与 negative results

- restricted license 仍禁止 full-scale LP；本结论严格限定于 license-safe、
  input-only restricted slices。
- 无 timestamp、orbit epoch、physical interval 或 independent ephemeris；只能
  陈述 ordered records 和 oracle recorded next topology。
- high-pressure 是 deterministic selection，不应冒充随机部署 workload；random
  control 无拥塞本身也是负面边界。
- 多数 transition 的 selected candidate paths 不变；raw edge change 不等价于
  TE-relevant path invalidation。
- path-flow optimum 可能退化；state distance 不是 operational value。
- hard-transition tail 存在，但 best basis reuse 的 primary median 已为 0，且完整
  transported start 明显更慢，不能用 tail correlation 绕过 T3。

## 14. 输出与验证

正式输出位于被 Git 忽略的：

`output/temporal_feasibility/topology_transition/`

包含要求的 provenance、raw manifest、events、regime statistics、三类 slice
manifests、transition/stable LP records、warm-start records、state/solver transport
records、congestion/severity statistics、summary、10 张主图；未运行 stress，故无
Figure S1。

最终单命令测试：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
python -m pytest -q experiments/temporal_feasibility/tests
```

结果：`31 passed`。所有新增文本保持 UTF-8 BOM；原有中文报告未被覆盖。
