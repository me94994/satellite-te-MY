# Official DataSetForSaTE benchmark inventory

扫描日期：2026-09-09

## Official source

- source：`OFFICIAL_SATE_GOOGLE_DRIVE`
- README folder：`1h6kbOj4HpqofPNd7lkIJDTut4XF4ipAF`
- 本轮单文件：`DataSetForSaTE100.zip`
- archive size：220,852,554 bytes
- archive SHA256：`2970188061e0917f1d9685a4d9d37be8a70cfd34aff6c16ec27a7d41e0f9552b`
- ZIP validation：`DOWNLOAD_ARCHIVE_VALID`，central directory 与所有 member CRC 均通过

## Raw files

| Dataset | Volume | Raw file | Bytes | Records | EOF | SHA256 |
|---|---|---|---:|---:|---|---|
| DataSetForSaTE100 | A | `StarLink_DataSetForAgent100_5000_A.pkl` | 428,121,539 | 5,000 | CLEAN | `bbaca946d6b7d60e85c1c42455f42b197c89d6981eb927cec8f71b0881c4defb` |
| DataSetForSaTE100 | B | `StarLink_DataSetForAgent100_5000_B.pkl` | 426,556,460 | 5,000 | CLEAN | `caa2d918b6c790eec2ce69e8dc1321b3bed36f94ebd962888e81d64c03010562` |

两份文件每条 record 均含 `FlowSet`、`InterShell_ISL`、`InterShell_GrdRelay`。本轮没有下载 25/50/75；历史 25/A 仅作 checksum 参考，不进入本轮主实验。

## Readiness

- `BENCHMARK_PRIMARY_READY=true`
- `BENCHMARK_ALL_INTENSITIES_READY=false`
- `BENCHMARK_A_B_COMPLETE=false`

主实验只要求 DataSetForSaTE100 至少一个可用 volume，因此 A/B 均可用时不会再因 25/50/75 缺失而阻断。archive、raw pickle 与 path cache 均位于 Git ignored 路径。
