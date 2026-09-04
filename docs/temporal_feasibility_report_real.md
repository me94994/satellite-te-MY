# 真实 Starlink 时序 LP 第二轮正式验证

日期：2026-09-04

## 1. 结论先行

本轮在 SaTE 作者公开的 `DataSetForSaTE25` volume A 原始数据上完成了真实
Gurobi 验证，但证据只覆盖 **real-data restricted slice**，不是 full-scale
Starlink。最终主结论为：

`STOP_TEMPORAL_TRACKING`

这不是对所有 Starlink temporal TE 的普遍机制否定，而是对当前可验证路线的
fail-closed 决策：jointly matched control 没有显示额外的 adjacent advantage；
LP 完全无拥塞，binding set 与 capacity dual 全零；动态拓扑转移只有 3 次，
transport 虽有很大的 edge-load/utilization 点估计优势，但统计力不足。现有证据
不支持进入 residual GNN + 1--3 step unfolding。

## 2. 代码与环境

- 起始 HEAD：`70c0e0a552ae4c94bc8646cb8bcc51e9444150a0`，提交说明为
  `Add temporal LP feasibility experiments`。
- 系统：WSL2 Linux 6.6.87.2，Python 3.13.5，单进程执行。
- Gurobi：13.0.3；`Threads=1, Method=1, Seed=42, OutputFlag=0`。
- 本报告对应的实现提交为包含本文件的最终提交；原始数据与全部 `output/`
  均被 Git 忽略。
- 用户已有的 `environment.yaml` 修改未被本轮编辑或纳入提交。

代码审查修复了上一轮距离定义的根因错误：normalized L1/L2 原来使用较大
向量范数作为分母，本轮按预注册定义改为两向量范数的平均值；path-flow 改为
完整 fixed universe，而非只比较共同路径。统计同时补齐 utilization、binding
distance、joint demand+topology matching、5,000 次 bootstrap 和配对检验。

## 3. 数据来源与 provenance

数据来自仓库 README 唯一指向的 SaTE 作者公开 Google Drive：

`https://drive.google.com/drive/folders/1h6kbOj4HpqofPNd7lkIJDTut4XF4ipAF`

只下载 `DataSetForSaTE25.zip`（112,349,687 bytes），并只解压 volume A：

- 文件：`StarLink_DataSetForAgent25_5000_A.pkl`
- 大小：278,509,040 bytes
- SHA256：`ba362369d64587365668b08150447c764b67f996beea17bc25d941f73026a9fe`
- 记录范围：0--99，共 100 条连续 pickle records
- 原始 keys：`FlowSet, InterShell_GrdRelay, InterShell_ISL`
- 原始 timestamp：无
- orbit epoch：无
- explicit interval：无

仓库源码只能证明 adapter 按 concatenated pickle stream 的顺序读取并编号。
没有找到“每条记录间隔 1 秒”的 artifact/source 证据，因此结论严格写为
`ordered original SaTE simulation sequence`，不写成 physically timestamped 或
1-second adjacency。完整 provenance 位于被忽略的
`output/temporal_feasibility/real/data_provenance.json`。

## 4. Gurobi license 探测

官方 pip package 能成功优化 2-variable LP，状态为 `OPTIMAL`、目标 2.0。
2,001-variable 探针返回 Gurobi size-limit error，故分类为
`RESTRICTED_SIZE_LIMIT_CONFIRMED`。`GRB_LICENSE_FILE`、`GUROBI_HOME` 和 WLS
凭证环境均未设置；探针只记录布尔存在性，从不记录 secret 内容。

`Model.getParamInfo("LPWarmStart")` 在 13.0.3 中报告范围 `-1..2`，本轮实际
运行了默认模式、显式 basis (`LPWarmStart=1`) 及 basis+presolve
(`LPWarmStart=2`)。

## 5. Restricted slice 与规模

在前 100 records 中按 `(-appearance_count, src, dst)` 固定排序选择 flows。
10、20、40-flow 层级分别先构造；40-flow universe 有 2,541 constraints，超过
restricted limit，未求解。最终采用最接近限制且不越界的 30-flow slice：

- 30 个 flow 均在 100 records 中出现 100 次；
- 每个 raw flow 保留官方 adapter 生成的最多 5 条候选路径；
- 每 snapshot 去重后的语义路径数为 142；全窗口 path universe 为 157；
- 全窗口 edge universe 为 1,766；
- 157 variables，1,953 constraints；
- snapshot edge count 范围 1,549--1,585；
- 100 snapshots 中有 4 个 graph hashes、3 次 topology transition；
- restricted total demand 恒为 1,500。

这类选择满足 license-safe 的预注册规则，但抑制了 flow churn，因而不能提供
有力的 traffic-only 推断。

## 6. Gate A：sequence provenance

`data_idx` 与 `source_record_index` 都为连续 0--99；无重复、无跳号、无逆序，
且全部来自同一个 volume A 文件。因此：

- Gate A：`PARTIAL`
- Real sequence verdict：`ORDERED_ONLY`
- `TIME_SUPPORTED`：否

本轮可研究 ordered-sequence continuity，但没有资格作 ephemeris-aware 强表述。

## 7. LP correctness 与 optimal state

100 个 cold Gurobi solves 全部 `OPTIMAL`：objective 恒为 1,500，primal-dual
gap、capacity/demand/path-availability violation 均为 0。有限差分校准中 Gurobi
capacity raw Pi 为 1.0，capacity 增加 `1e-4` 得到的目标导数为
1.0000000000021103，dual 符号校准通过。

负面结果是 slice 在每个 snapshot 都完全承载需求：binding capacity edge 数
恒为 0，capacity dual price 全为 0。由于 total-throughput objective 下多条路径
仍可任意分流，primal/path/edge state 也可能退化；状态接近不等价于唯一 optimum
接近。

## 8. Adjacent vs random 正式统计

下表为 normalized L1；CI 是 5,000-resample median bootstrap 95% CI。

| state | adjacent median / mean | jointly matched median / mean | paired Wilcoxon p |
|---|---:|---:|---:|
| edge load | 0 / 0.00767465 | 0 / 0.00767465 | 1.0 |
| utilization | 0 / 0.00813037 | 0 / 0.00813037 | 1.0 |
| binding-set distance | 0 / 0 | 0 / 0 | 1.0 |
| path flow | 0 / 0.01481481 | 0 / 0.01481481 | 1.0 |
| dual price | 0 / 0 | 0 / 0 | 1.0 |

上述所有 median CI 都为 `[0,0]`。由于 jointly matched random 的 median 也是
0，adjacent/random median ratio 不可定义，不能把 0/0 解释为通过 0.7 gate。
作为对照，edge-load unrestricted random 与 demand-matched random median 均为
0.07960199，但 topology matching 后优势完全消失；这说明表观相邻优势由稀疏
topology regime 解释，而不是相同 topology 内额外的时间邻近性。

Gate B：`FAIL`；Optimal-state continuity verdict：`WEAK`。

## 9. Lag curve

Edge-load normalized L1 的 lag 1/2/5/10/20：

- median：`0 / 0 / 0 / 0 / 0.07960199`
- mean：`0.00767465 / 0.01550592 / 0.03998896 / 0.07990443 / 0.10668484`

Utilization 的对应 median 为 `0 / 0 / 0 / 0 / 0.08432148`，mean 为
`0.00813037 / 0.01642666 / 0.04236349 / 0.08464794 / 0.11301502`。
lag curve 随 lag 上升，但这与 4 个长 topology regimes 完全相容；joint match
已表明不能据此单独认定 optimizer-state temporal continuity。

## 10. Traffic-only / topology-only / interaction

每个相邻 transition 都实际求解 `z00/z01/z10/z11`，路径可用性来自对应 topology，
不存在的 path 被 fixed-universe availability constraint 固定为 0。

- traffic-only：所有 state 的 mean/median/max 均为 0；
- topology-only edge load：mean 0.00767465，max 0.40650407，3/99 非零；
- topology-only utilization：mean 0.00813037，max 0.43074884，3/99 非零；
- topology-only path flow：mean 0.01481481，max 0.8，3/99 非零；
- binding 与 dual：全零；
- interaction：全零。

按稀疏事件不会被 median 抹除的 primary-state mean 分类，结果为
`TOPOLOGY_DOMINANT`。但这里的 traffic-only 为零是 persistent-flow slice 的直接
结果，不能外推到完整 intensity-25 traffic matrix。

## 11. Gurobi warm-start benchmark

所有方法使用同一 universe/objective/snapshot/capacity/demand/path availability，
全部通过 objective parity 与 feasibility gate。以下为动态真实 slice 的中位数：

| mode | optimize ms | total ms | IterCount | optimize speedup | total speedup |
|---|---:|---:|---:|---:|---:|
| cold rebuild | 0.7869 | 9.2556 | 28 | 1.00x | 1.00x |
| reuse model auto | 0.1931 | 1.0307 | 0 | 4.08x | 8.98x |
| explicit VBasis/CBasis | 0.2064 | 1.2793 | 0 | 3.81x | 7.23x |
| explicit basis + presolve | 0.7404 | 1.8346 | 21 | 1.06x | 5.04x |
| reset basis | 0.7096 | 1.5451 | 28 | 1.11x | 5.99x |

reuse 与 explicit basis 的 iteration reduction median 都是 100%；reset basis 为
0%。这同时分离了 model-build saving：即使 reset basis 的 optimize speedup 很弱，
total time 仍因不重建 Python/Gurobi model 而明显更短。

受控 frozen-topology real-traffic 实验固定 `G=G0` 与 path universe；由于选中
flows 的 demand 恒定，它实质上重复同一个 LP，只用于 API/control，不是原始
fixed-topology artifact。其 optimize speedup 为 reuse 4.73x、explicit basis
4.58x，IterCount 均由 28 降到 0。动态 Gate C 依据 4.08x 最佳 warm speedup
分类为 `MODERATE_WARM_START`；最终 Classical verdict 为 `MODERATE`。

## 12. Dynamic state transport

只在 3 次真实 edge birth/death transitions 上评价。persistent edge copy、deleted
edge drop，并对 new edge 比较 zero/global/neighbor/capacity-normalized/random：

- edge load：best transport 0.27163782，best no-history 1.08892010，p=0.125；
- utilization：best transport 0.26675514，best no-history 1.07815631，p=0.125；
- binding：transport 与 no-history 都为 0；
- dual：transport 与 no-history 都为 0。

edge load/utilization 的点估计优势很大，但 n=3 时单侧 exact Wilcoxon 的最小
可达 p-value 为 0.125，不能称显著。Gate D 因此为 `PARTIAL`，不是 PASS。
结果支持“若继续采集证据，load/utilization 比 raw dual 更值得追踪”，但当前不
足以授权下一阶段模型开发。

## 13. Prediction advantage

输出保存了 recorded sequence 中 edge 的 `survives_next_1/2/5` 与
records-to-disappearance。这是
`ORACLE_RECORDED_SEQUENCE_NOT_EPHEMERIS_PREDICTION` diagnostic。由于 raw
artifact 没有独立 ephemeris，本轮未测试“future topology 超过 current+previous
state”的增量预测价值，不能选择 `EPHEMERIS_AWARE_PREDICTIVE_TE`。

## 14. Gate 与二维科研判定

| Gate | 结果 | 解释 |
|---|---|---|
| A | PARTIAL | pickle 原始顺序可信，无物理时间证据 |
| B | FAIL | adjacent 与 jointly matched control 无差异 |
| C | MODERATE_WARM_START | 动态 optimize 最佳 4.08x，parity PASS |
| D | PARTIAL | load/util 点估计有利，但只有 3 次变化、p=0.125 |

二维位置为 continuity weak + classical warm-start moderate。按预注册 CASE 3，
temporal neural solver 不值得做。Gate D 的 limited-scale signal 尚不足以覆盖
Gate B 与样本统计力缺口。

## 15. 最终规定格式

### 1. Real sequence verdict

`ORDERED_ONLY`

### 2. Optimal-state continuity verdict

`WEAK`

- path flow：WEAK；matched 后无 adjacent advantage，且 primal 退化。
- edge load：WEAK；unmatched 有 lag pattern，jointly matched 后消失。
- utilization：WEAK；同 edge load。
- binding set：WEAK/UNINFORMATIVE；全空集合。
- dual price：WEAK/UNINFORMATIVE；全零。

### 3. Classical reoptimization verdict

`MODERATE`

动态 slice reuse optimize speedup 4.08x、explicit basis 3.81x；两者 median
IterCount 从 28 降至 0。

### 4. Topology vs Traffic driver

`TOPOLOGY_DOMINANT`（只对 persistent-flow restricted slice 成立）。

### 5. Dynamic state transport verdict

`PARTIAL`

### 6. Research recommendation

`A. STOP_TEMPORAL_TRACKING`

## 16. Blocker 与 negative results

- 无 timestamp、orbit epoch、interval 或 independent ephemeris；Gate A 只能 PARTIAL。
- restricted license 阻止 40-flow/2,541-constraint 及 full 4,236-satellite LP。
- 公开目录未提供 `starlink_500_fixed_topo`、176/500 reduced dynamic artifact；
  本轮只有 full raw topology 上的 path-induced restricted subgraph。
- persistent-flow 选择导致 total demand 恒定，traffic-only 问题没有激励变化。
- 所有 LP 无 capacity binding，dual 与 binding state 没有信息。
- 只有 3 次 dynamic transitions，transport 检验统计力不足。
- full-scale adjacent-vs-matched-random、真实 fixed-topology traffic variation、
  ephemeris prediction advantage 均为 `NOT_EVALUATED`。
- 未实现任何 GRU/LSTM/Transformer/GNN/unfolding/RL/MIP/IPM network。

进入 residual GNN + 1--3 step unfolding：**否**。若未来取得 unrestricted
license 或官方 176/500 dynamic/fixed-topology artifact，应先增加 topology
transition 数和非平凡 congestion，再重新资格审查；不得直接沿用本轮 PARTIAL
transport 点估计作为模型开发授权。
