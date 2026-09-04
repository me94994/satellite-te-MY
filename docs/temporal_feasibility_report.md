# SaTE 时序优化可行性报告

日期：2026-09-04

## 1. 范围与代码审查结论

本轮只研究 SaTE 的 fixed-path continuous LP，没有修改 SaTE 主模型、训练
入口或 `lib/lp`。审查得到：

- `StarlinkAdapter` / `StarlinkReducedAdapter` 按输入 pickle 的读取顺序 append
  样本，并生成递增 `data_idx`；输出样本只有 `graph/tm/path/data_idx`，没有
  timestamp、轨道历元或 ephemeris 标识。
- `utils/scripts/env` 在给 `INPUT_DIR` 默认赋值之前先展开
  `RAW_INPUT_DIR=${INPUT_DIR}/raw/starlink`；若调用环境未预设 `INPUT_DIR`，
  默认 raw 路径会变为 `/raw/starlink`。实验不依赖该变量，要求显式
  `--dataset` 或只读探测 `input/`。
- adapter 的 `adapt()` 会在输出目录已存在时递归删除再创建；本轮没有调用
  adapter，也没有覆盖任何原始数据。
- mixed adapter 按 intensity 文件循环拼接并重新编号；样本 schema 不保存
  intensity/volume 边界，因此仅凭输出 pickle 无法可靠恢复这些边界。
- `run/spaceTE.py` 读取整个 list-of-dict；`SaTEEnv` 随后调用固定种子的
  `train_test_split`，所以原训练/测试入口不能用于相邻时序分析。
- SaTE 的路径语义来自 `(src,dst,path_nodes)`；路径列表位置不是稳定语义键。
  adapter 在候选不足时会复制第一条路径，实验因此按语义键去重。
- `PathFormulation` 的 total-flow LP 是每路径非负连续变量、链路容量上界、
  commodity demand 上界，并最大化总路径流量。实验 wrapper 保持该数学问题，
  仅增加 fixed universe 中“当前快照未出现路径必须为零”的显式可用性约束。
- Starlink adapter 的 `graph` 是边列表且不含 capacity；`SaTEEnv` 从
  `OrbitParams` 赋 ISL=200、access=800。实验通过显式 CLI capacity policy
  重建同一含义，不从数据中静默猜测。
- 现有 `LpSolver` 默认为 concurrent method，且 `PathFormulation` 每次新建
  model；现有代码没有顺序 RHS 更新、basis 保存/恢复或 dual 提取证据。
- `TopoGNN` / `AlloGNN` / `SaTEActor` 对每个快照独立前向；没有 previous
  optimizer state 或跨快照 state transport。`TopoGNN` 还将 self-loop
  feature 硬编码到 CUDA，这与本轮 CPU-sized LP 测试隔离。本轮未改变这些模块。
- 现有 `LpSolver.solve_lp()` 捕获部分 Gurobi 异常后不再抛出，且未强制检查
  `OPTIMAL`；实验 wrapper 改为 fail-closed，但没有改变主入口行为。

## 2. 实验设计与可证伪点

实验直接读取 adapter pickle，不进入 `SaTEEnv`。manifest 检查 `data_idx`
单调性、重复/跳变、graph hash、需求漂移、语义路径 overlap 及 timestamp
字段。LP 在窗口内固定 flow/path/edge universe，当前未激活 flow RHS 为 0，
动态消失 edge capacity 为 0。

Gurobi 正式设置固定为 `Threads=1, Method=1 (dual simplex), OutputFlag=0,
Seed=42`。三组分别为新建 model、复用 model 只改 RHS、显式读取并恢复
`VBasis/CBasis`；首个快照标记并从 warm 统计排除。`Var.Start`、`PStart/DStart`
未作为已验证方法，也不会因设置字段就声称 warm start 生效。

capacity dual 的统一非负 `congestion_price` 不硬编码 Pi 符号。每个实际
backend 先运行“容量不紧张”和“唯一瓶颈”两个小 LP，用容量 `+eps` 的目标
有限差分校准符号。

## 3. 本机实际可用证据

- 真实 Starlink adapter 数据：**未发现**。仓库 `input/` 不存在，常见
  OneDrive 搜索范围内也未发现 176/500/fixed-topology 目标 pickle。
- Gurobi：**不可用**。当前 Python 环境导入 `gurobipy` 失败。
- SciPy/HiGHS：可用，仅用于 synthetic 功能级验证；不得作为正式 Gurobi
  runtime 或 basis warm-start 证据。

实际验证环境为 SciPy 1.15.3、matplotlib 3.10.0、pytest 8.3.4。

## 4. Synthetic fixture 结果

所有此节结果均来自 4 节点、16 snapshots 的 synthetic fixture，不代表
Starlink/SaTE 真实数据。

- sequence inspector：`data_idx` 严格单调，无重复或跳号；存在 2 个 graph
  hash；没有 timestamp。因此即便在 synthetic 中也只标记 ordered-sample
  correlation，不标记 ephemeris-aware tracking。
- dual sign：SciPy 原始 bottleneck marginal 为 `-1.0`；容量从 3 增到
  `3+1e-4` 的有限差分为 `1.0000000000021103`；统一 nonnegative price 为
  `1.0`。容量不紧张案例 raw marginal 为 `-0.0`，验证通过。
- adjacent path-flow / persistent-edge-load normalized L1 median 均为
  `0.08333`；demand-matched random 均为 `0.125`，点估计 ratio `0.6667`。
  但 bootstrap 95% ratio interval 为 `[0.0, 1.3]`，越过 0.7 判据，故
  synthetic Gate B 为 `FAIL_OR_INCONCLUSIVE`，不能按点估计写 PASS。
- unrestricted random 的上述 median 为 `0.46154`。lag 1/2/5/10 median
  分别为 `0.08333 / 0.14835 / 0.36364 / 0.5`，fixture 中存在预期衰减；
  这只证明分析代码能识别预设相关性。
- dual distance 的 adjacent 与 demand-matched random median 都为 0，说明
  小 LP 的零价格区间/退化使该 median 不具区分力；binding-set Jaccard 两者
  median 都为 1.0。因此没有用 dual 指标补强连续性结论。
- cold/reuse/explicit_basis 三组 objective、capacity、demand、path
  availability 与 primal-dual parity 全部通过。由于实际 backend 是 SciPy，
  后两组不保存/恢复 solver basis，`warm_start_effective=false`。最后一次
  smoke 中 reuse/explicit 的 ratio-of-medians reduction 分别为 `-2.79%`
  和 `-0.99%`，paired bootstrap 95% 区间分别为
  `[-4.37%, 2.61%]`、`[-8.04%, 2.58%]`；这是微秒级噪声，不是
  warm-start 证据。
- 动态 transport：topology-only transition 上 transport 与 default-zero
  距离同为 0；both-change 上 transport L1 为 1.0，default-one 为 0.5；
  汇总 dynamic median 中最佳 transport 与 default-zero 同为 0.5，未严格
  优于 baseline；paired median difference 95% 区间为 `[0.0, 0.0]`。
  synthetic Gate D 为 `FAIL_OR_INCONCLUSIVE`。

实际执行命令：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
python -m pytest -q experiments/temporal_feasibility/tests

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
python -m experiments.temporal_feasibility.smoke_test \
  --output-dir output/temporal_feasibility/synthetic_smoke
```

最终测试结果：`6 passed, 1 skipped`；skip 是 Gurobi-specific basis test，
原因是 `gurobipy` 不可导入。完整 synthetic 输出位于被 Git 忽略的
`output/temporal_feasibility/synthetic_smoke/`。

## 5. Gate A--D

| Gate | 状态 | 原因 |
|---|---|---|
| A 数据具有可信时间次序 | **BLOCKED** | 无真实数据；静态代码仅证明读取顺序编号，不证明相邻物理时刻 |
| B adjacent optimal state 显著近于 random | **BLOCKED** | 无真实序列的相邻/匹配随机统计 |
| C Gurobi warm start median 至少加速约 20% | **BLOCKED** | 无 `gurobipy`/license，也无真实 fixed-topology 序列 benchmark |
| D 动态 deterministic transport 优于 baseline | **BLOCKED** | 无真实动态序列；synthetic 只能验证实现 |

Gate 为 BLOCKED 时不作方向性科学分类。

## 6. 当前研究建议

**不建议进入下一阶段。** 目前只能确认实验工具链可构造，不能以静态代码
审查或 synthetic fixture 支持“最优状态追踪”假设。最小下一步是提供带
provenance 的 reduced Starlink adapter pickle（优先 fixed-topology 500 ISL，
再 176/500 dynamic）和可用 Gurobi license，按 100 snapshots 预注册流程运行。
即便后续 fixed-topology Gate A--C 通过，也应先限制在 fixed-topology temporal
solver；只有 dynamic Gate D 与物理 timestamp provenance 同时成立，才讨论
ephemeris-aware graph-state transport。Residual GNN + 1--3 step unfolding 不在
当前证据支持范围。

## 7. 未完成项与阻塞项

- 未运行真实 100-snapshot sequence（数据缺失）。
- 未运行 Gurobi cold/reuse/explicit-basis benchmark（包或 license 缺失）。
- 未验证 176/500 动态 Starlink edge transport（数据缺失）。
- 未验证 failure perturbation；adapter 输出没有预注册 failure 标签。
- 未运行任何 formal/large-scale experiment，状态为 `NOT_EVALUATED`。
