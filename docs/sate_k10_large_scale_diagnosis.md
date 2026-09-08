# SaTE K=10 Large-Scale Crossover Diagnosis

## 1. Executive conclusion

本轮在不改变 LP 数学模型、不关闭 Presolve、不扩展 Gurobi 许可边界的前提下，完成了
paired K=5/K=10、topology/traffic/coupled、uncongested/controlled-stress 和
K=10 SaTE CUDA scaling。结论是：**K=10 solver scaling 为 `STRONG_GROWTH`，但
`NO_CROSSOVER_WITHIN_LICENSE_SAFE_RANGE`**。因此总体判定为
`PARTIAL_PAPER_TREND_REPRODUCED`，不是 paper speedup reproduction。

restricted-safe 最大 solver 点为 192 satellites、180 active flows、K=10，含
1800 variables、1332 constraints、18793 nonzeros。该点 best cold commercial
total 为 13.284 ms；same-model/no-basis 为 3.179 ms，previous-basis online 为
0.678 ms。相同 diagnostic scale 的 K=10 SaTE forward/e2e 分别为
8.157/37.224 ms，cold 与 online crossover 均未出现。

SaTE diagnostic scale 实际测到 4236 satellites：K=10 forward 为 18.238 ms，
e2e 为 109.980 ms。该点不是 paper dataset，且 restricted license 不能给出同点
Gurobi optimum/quality，不能与论文 46--47 s / 17 ms 合成 speedup。

## 2. K=10 artifact/checkpoint availability

本地扫描 786 个既有 `.pt`：`mean_linear.weight` 为 764 个 5x5、14 个 5x35、
8 个 6x6，没有 10x10。README 的公开 Google Drive 未提供可在当前环境中识别并
最小下载的 K=10 checkpoint/data 文件，因此记录
`PAPER_K10_ARTIFACT_NOT_AVAILABLE`，没有下载整个目录，也没有 resize/copy K=5
权重。

本轮另行训练了 `DIAGNOSTIC_K10_TRAINED`：66 satellites、40 flows、40 samples，
32 train/8 held-out、2 epochs、batch 4，训练时间 2.314 s（RTX 4090）。checkpoint
审计为 `mean_linear.weight=[10,10]`、bias `[10]`。训练 seed 1000--1039；正式
benchmark 使用 seed 42，未进入训练集。

## 3. Paper/code configuration

本轮保持 `CODE_CONFIG`：ISL 200、uplink/downlink 800。论文描述的 network
200 Mbps/access 50 Mbps 仍记为 `PAPER_CONFIG`，没有静默修改，也没有将
`PAPER_LIKE_CONFIG` 与现有模型质量混合。论文 K=10 与公开执行代码 K=5 的差异
通过 paired path pool 显式处理。

## 4. Experimental scales

- Family A topology：flows=40，N=16/32/48/66/96/128/160/192。
- Family B traffic：N=192，flows=10/20/40/80/120/160/180。
- Family C coupled：flows=min(N,180)，相同 N 轴。
- Dense K=10：N=6/8/10/12/14；构造前 analytical guard。
- SaTE diagnostic：N=66/128/192/256/396/512/1024/1584/2048/4236；最大 180 flows。
- 每个 solver sparse 点 K5/K10 使用同一 P10 pool，P5 严格为前五条路径；每条
  路径唯一，topology/demand/capacity 完全相同。

所有 topology 是 deterministic constellation-shaped diagnostic，不是 Starlink
paper workload。每点保留 topology/demand/capacity/path hash 和 paired pool hash。

## 5. Gurobi license boundary

环境为 Gurobi 13.0.3 restricted license（日志显示到 2027-11-29），Threads=1，
安全上限预注册为 vars<=1900、constraints<=1900。requested flow 超界时只在构模前
降低实际 flows；本轮最大 1800/1332，未触碰许可限制。所有 414 条 sparse formal
rows 均 `MEASURED`，无越界构造或选择性丢点。

## 6. Dense scaling

Dense K=10 仅运行安全小规模：

| N | vars | constraints | NZ | median total |
|---:|---:|---:|---:|---:|
| 6 | 300 | 66 | 1728 | 1.190 ms |
| 8 | 560 | 104 | 3344 | 2.269 ms |
| 10 | 900 | 150 | 5720 | 8.947 ms |
| 12 | 1320 | 204 | 8616 | 15.855 ms |
| 14 | 1820 | 266 | 11760 | 25.346 ms |

该实验只说明 representation/build 增长，不用于 full-scale crossover。

## 7. Sparse K=10 scaling

Coupled/controlled-stress 的 best cold commercial 实测曲线如下：

| N | flows | selected method | parse | build | presolve probe | optimize | total | IterCount |
|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 16 | 16 | dual simplex | 0.058 ms | 0.665 ms | 0.170 ms | 0.193 ms | 0.916 ms | 0 |
| 32 | 32 | default | 0.111 ms | 1.313 ms | 0.292 ms | 0.340 ms | 1.765 ms | 0 |
| 48 | 48 | dual simplex | 0.167 ms | 1.943 ms | 0.427 ms | 0.566 ms | 2.671 ms | 1 |
| 66 | 66 | dual simplex | 0.295 ms | 3.203 ms | 0.702 ms | 0.897 ms | 4.418 ms | 10 |
| 96 | 96 | dual simplex | 0.398 ms | 4.351 ms | 1.025 ms | 1.381 ms | 6.120 ms | 14 |
| 128 | 128 | dual simplex | 0.555 ms | 5.972 ms | 1.403 ms | 1.882 ms | 8.325 ms | 14 |
| 160 | 160 | default | 0.802 ms | 7.596 ms | 1.759 ms | 2.525 ms | 11.023 ms | 28 |
| 192 | 180 | dual simplex | 1.025 ms | 9.299 ms | 2.168 ms | 2.887 ms | 13.284 ms | 31 |

从首末点看 total 境长 14.51x，optimize 增长 14.99x，故 K10 solver scaling
判为 `STRONG_GROWTH`。`presolve probe` 是独立的 presolved-model audit，不重复计入
`total=parse+build+optimize`。

## 8. Congested scaling

Regime U 使用原始 deterministic demand；Regime C 只根据 solver 前的 equal-split
proxy，在 1.0/1.5/2.0/4.0 中取首个触发点，并标
`CONTROLLED_STRESS_DIAGNOSTIC`。

U 在 N=16 被 Presolve 化为 0 vars；N=32--192 仍保留 142/218/432/742/1051/
1454/1586 vars，但 optimum 满足全部 demand、实际 binding ratio 为 0。C 在全部
N 保留完整 160--1800 vars，不再出现 0-variable core；N>=48 出现非零 binding
edge（约 0.40%--2.98%），N=160 出现 32 units unsatisfied demand。

拥塞没有造成一致的 wall-time penalty：同 N 的 C/U sparse-cold total ratio为
1.108/0.791/0.946/1.026/0.936/0.918/1.005/0.932。也就是说 controlled stress
成功阻止 trivial 0-core，但在本 restricted 范围内并未让 Gurobi 明显更慢。

## 9. K5 vs K10

46 个 paired points 的 K10/K5 中位 slowdown：build 1.346x、optimize 1.607x、
total 1.402x；total 范围 1.000--1.967x。Traffic/C family 在 10→180 flows 的
total slowdown 为 1.081/1.153/1.213/1.344/1.359/1.415/1.400x。K=10 同时增加
build 与 optimize，后者的相对增幅更大；它使 solver burden 更明显，但没有把
measured range 推过 crossover。

## 10. SaTE K10 latency

GPU 为 RTX 4090。每点 warmup=20；N<=512 repeats=100，较大点 repeats=30；每段
使用两端 `torch.cuda.synchronize()`。下表为 Regime C 中位数：

| N | flows | forward | e2e | hetero edges | peak allocated |
|---:|---:|---:|---:|---:|---:|
| 66 | 66 | 7.251 ms | 23.917 ms | 5291 | 40.1 MB |
| 128 | 128 | 6.842 ms | 28.341 ms | 11792 | 76.1 MB |
| 192 | 180 | 8.157 ms | 37.224 ms | 18793 | 113.8 MB |
| 256 | 180 | 7.911 ms | 37.575 ms | 20650 | 123.8 MB |
| 396 | 180 | 8.917 ms | 45.615 ms | 23804 | 144.2 MB |
| 512 | 180 | 9.984 ms | 51.994 ms | 27354 | 164.4 MB |
| 1024 | 180 | 10.178 ms | 52.569 ms | 34966 | 218.4 MB |
| 1584 | 180 | 12.458 ms | 64.734 ms | 40379 | 258.5 MB |
| 2048 | 180 | 14.872 ms | 75.046 ms | 43436 | 286.5 MB |
| 4236 | 180 | 18.238 ms | 109.980 ms | 62842 | 449.8 MB |

4236-node forward 接近 paper-reported 17 ms 只是一项趋势性/量级观察；本轮 graph、
traffic、checkpoint、hardware 和 paper artifact 不匹配，不能称 reproduction。

K10/K5 的 TopoGNN ratio 在代表点约 1.01--1.12；AlloGNN 大规模 ratio 增至
1.83--2.35；postprocess 多数约 0.97--1.24；e2e ratio 约 1.15--1.45。K 增加主要
作用于 path-side AlloGNN，TopoGNN 基本不变。

## 11. SaTE K10 quality

只有 `DIAGNOSTIC_K10_TRAINED` 评价质量。C regime 的 optimality gap：N=66
1.46%、N=128 0.97%、N=192 4.49%、N=256 1.90%。repair 后最大 capacity excess
为 9.16e-5 以下（FP32 数值尺度）；raw/post-repair 值均保留在 artifact。N>=396
因 exact L4 超过 restricted-safe constraint guard，quality 为
`NOT_EVALUATED_GUROBI_BLOCKED`，不是零 gap。

## 12. Cold crossover

同 N/flows/regime 的 measured comparison：

- N=66：Gurobi 4.418 ms vs SaTE e2e 23.917 ms；
- N=128：8.325 ms vs 28.341 ms；
- N=192：13.284 ms vs 37.224 ms。

判定：`NOT_OBSERVED`，并具体记为
`NO_CROSSOVER_WITHIN_LICENSE_SAFE_RANGE`。拟合不创建 measured crossover。

## 13. Online crossover

`CONTROLLED_SEQUENTIAL_DIAGNOSTIC` 使用固定的 previous demand=0.95*current，
不查看 solver outcome。N=66/128/192 的 previous-basis online total 为
0.199/0.461/0.678 ms；same-model/no-basis 为 0.945/1.976/3.179 ms。对应 SaTE
e2e 均更慢。因此 online crossover：`NOT_OBSERVED`，但该 online 结论只属于受控
连续 sequence，不外推独立随机 snapshots 或 paper trace。

## 14. Scaling exponents

对 coupled/C measured medians 做 log-log fit，bootstrap 95% CI：

- Gurobi optimize vs vars：b=1.155，CI [1.101, 1.252]；
- Gurobi total vs vars：b=1.108，CI [1.067, 1.185]；
- Gurobi optimize vs NZ：b=0.980，CI [0.942, 1.068]；
- Gurobi total vs NZ：b=0.940，CI [0.914, 0.985]；
- SaTE K10 forward vs hetero edges：b=0.388，CI [0.199, 0.724]；
- SaTE K10 e2e vs hetero edges：b=0.609，CI [0.456, 0.891]。

这是 `MEASURED_FIT`，不是 paper-scale solver measurement。SaTE forward 对固定
path-node count 的 fit 因 flows 在 180 饱和而可识别性较弱，主要采用 hetero-edge
轴解释。任何 4236 solver 预测只能标 `EXTRAPOLATED_4236`，本报告不把它作为
crossover 证据。

## 15. Paper Trend Tests P1--P6

| Test | Result | Evidence boundary |
|---|---|---|
| P1 Gurobi runtime 随规模增加 | **SUPPORTED** | diagnostic K10 total/optimize 均约 15x 增长 |
| P2 GNN inference scaling 更平缓 | **SUPPORTED (DIAGNOSTIC TREND)** | b约0.609 vs Gurobi total/NZ约0.940 |
| P3 大规模出现 GNN latency advantage | **NOT SUPPORTED** | 共同 measured 范围无 cold/online crossover |
| P4 K=10 增加传统 solver 负担 | **SUPPORTED** | paired total 1.402x，optimize 1.607x |
| P5 latency advantage 时仍有可接受质量 | **NOT EVALUATED** | 没有 measured latency advantage；已有 gap 只属 diagnostic |
| P6 46--47 s vs 17 ms 可复现 | **NOT REPRODUCED** | 缺 paper-faithful data/checkpoint/formulation-scale solver point |

## 16. What part of paper conclusion is reproduced

`PARTIAL_PAPER_TREND_REPRODUCED`：Gurobi 随 vars/NZ 明显增长；K=10 对 exact solver
的相对负担大于 SaTE topology side；SaTE 的 measured graph-scaling exponent 更小；
4236-node diagnostic forward 与 17 ms 同量级。没有重现 crossover、2738x 或
paper quality，因此不能升级为 `CROSSOVER_REPRODUCED`。

## 17. What remains unidentifiable

- 官方 paper-faithful K=10 adapted data 与训练 metadata/checkpoint缺失；
- 本地 `input/` 无数据，Google Drive 目录无法识别最小 K10 artifact；
- restricted license 把 exact paired solver 限制在 1800 vars/1332 constraints；
- 396/1584/4236 的 exact quality 和 Gurobi latency未测；
- paper 200/50 与 code 200/800/800 capacity mismatch 未消除；
- paper hardware、solver formulation、active commodity count 与本 diagnostic 不同。

因此 paper 2738x 判定为 `NOT_IDENTIFIABLE`，不是 `REPRODUCED`。

## 18. Next-step recommendation

优先获取一个带 provenance 的官方 K=10 adapted snapshot 与 `[10,10]` checkpoint；
若当前环境合法获得 unrestricted academic license，再按 396→1584→4236、1→3→10
snapshots 逐级运行 active-flow sparse + Presolve + active-subgraph。不要运行
full-scale dense，不要用拟合替代 measured crossover。若 artifact/license 仍缺失，
当前应停在 `PARTIAL_PAPER_TREND_REPRODUCED`。

## Reproducibility and tests

主要 artifact 位于 gitignored `output/speedup_attribution/large_scale/`：
`environment.json`、`data_provenance.json`、`k10_audit.json`、
`solver_scaling.jsonl`、`k5_k10_pairs.csv`、`sate_scaling.jsonl`、
`quality_scaling.csv`、`crossover_summary.json`、`scaling_fit.json` 和 `figures/`。

测试命令为单进程完整 suite；最终结果：**38 passed in 2.46 s**。本轮起始提交：
`341ab7530f1fc997e6066810425dc20f657f8627`。
