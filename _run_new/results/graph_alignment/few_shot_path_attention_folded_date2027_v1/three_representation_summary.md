# Three-Representation Cross-Seed Summary

- Seeds: 42, 205, 316, 327, 330, 501.
- Encoder configurations: 270 (all shots); decoder configurations: 54 (10-shot groups only).
- Common test samples per configuration: 420.
- Paired confidence intervals use seed-level two-sided 95% Student-t intervals.

## Overall

| Representation | Encoder Top-1 | Target-only exact | Task exact | Task valid | Stage exact | GED |
|---|---:|---:|---:|---:|---:|---:|
| Netlist | 59.48% | 98.90% | 76.60% | 96.69% | 95.01% | 0.9418 |
| Vpath Attention | 68.30% | 98.90% | 77.98% | 97.02% | 96.14% | 0.9467 |
| Path Attention | 75.26% | 98.90% | 82.05% | 97.84% | 97.02% | 0.9603 |

## Task Exact By Shot

| Shot | Netlist | Vpath Attention | Path Attention |
|---:|---:|---:|---:|
| 2 | n/a | n/a | n/a |
| 4 | n/a | n/a | n/a |
| 6 | n/a | n/a | n/a |
| 8 | n/a | n/a | n/a |
| 10 | 76.60% | 77.98% | 82.05% |

- Decoder 只在 10-shot 上运行，其余 shot 的 task 指标为 n/a。

## Task Exact By Backbone

| Backbone | Netlist | Vpath Attention | Path Attention |
|---|---:|---:|---:|
| graphsage | 76.31% | 75.83% | 81.11% |
| gat | 66.67% | 70.28% | 78.29% |
| gin | 86.83% | 87.82% | 86.75% |

## Paired Task Exact

| Comparison | Mean delta | Seed-level 95% CI | Win/Tie/Loss |
|---|---:|---:|---:|
| Vpath Attention - Netlist | +1.38pp | [-3.36, +6.11]pp | 10/1/7 |
| Path Attention - Vpath Attention | +4.07pp | [+1.44, +6.71]pp | 14/0/4 |
| Path Attention - Netlist | +5.45pp | [+1.39, +9.51]pp | 15/0/3 |

## Decoder Ceiling

- Canonical graph exact: 250/252.0 (99.21%).
- Canonical valid graph: 100.00%.
- Canonical stage exact: 100.00%.
- Per-seed graph exact: 42=42/42.0, 205=42/42.0, 316=42/42.0, 327=42/42.0, 330=41/42.0, 501=41/42.0.

## Interpretation

- Vpath Attention - Netlist: not separable from zero (mean +1.38pp, 95% CI [-3.36, +6.11]pp, win/tie/loss 10/1/7).
- Path Attention - Vpath Attention: statistically positive (mean +4.07pp, 95% CI [+1.44, +6.71]pp, win/tie/loss 14/0/4).
- Path Attention - Netlist: statistically positive (mean +5.45pp, 95% CI [+1.39, +9.51]pp, win/tie/loss 15/0/3).
- 每个表示都有独立的同结构 decoder；比较在 seed × backbone × 10-shot 的 18 个单元上配对。
- 该比较不构成 vpath 边与 hlink 边各自贡献的单因素消融。
