# Official DataSetForSaTE benchmark inventory

扫描日期：2026-09-09
扫描范围：仓库 `input/`、`input/raw/`、`input/raw/starlink/`、`raw_data/`，以及 `utils/scripts/env` 所指向的仓库内目录。

## 结论

`BENCHMARK_INCOMPLETE`

当前 checkout 没有发现可读取的 official raw pickle。以下 8 个主实验条目均缺失：

| Dataset | Volume A | Volume B |
|---|---:|---:|
| DataSetForSaTE25 | MISSING | MISSING |
| DataSetForSaTE50 | MISSING | MISSING |
| DataSetForSaTE75 | MISSING | MISSING |
| DataSetForSaTE100 | MISSING | MISSING |

`output/` 中存在以 DataSetForSaTE 命名的历史模型日志，但它们不是包含 `FlowSet`、`InterShell_ISL`、`InterShell_GrdRelay` 的 raw benchmark artifact，未被误认成 official workload。

机器可读 inventory 位于被 Git 忽略的 `output/speedup_attribution/ampl_full_scale/dataset_inventory.json`。原始数据与 adapted 数据均未加入 Git。
