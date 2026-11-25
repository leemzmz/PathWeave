# Three-Representation Cross-Seed Summary

- Seeds: 42, 205, 316, 327, 330, 501.
- Encoder configurations: 270 (all shots); decoder configurations: 54 (10-shot groups only).
- Common test samples per configuration: 420.
- Paired confidence intervals use seed-level two-sided 95% Student-t intervals.

## Overall

| Representation | Encoder Top-1 | Target-only exact | Task exact | Task valid | Stage exact | GED |
|---|---:|---:|---:|---:|---:|---:|
| Netlist | 62.69% | 98.68% | 79.68% | 97.24% | 95.29% | 0.9524 |
| Vpath Attention | 70.39% | 98.68% | 80.79% | 96.43% | 95.63% | 0.9460 |
| Path Attention | 78.11% | 98.68% | 87.21% | 98.28% | 98.08% | 0.9707 |

## Task Exact By Shot

| Shot | Netlist | Vpath Attention | Path Attention |
|---:|---:|---:|---:|
| 2 | n/a | n/a | n/a |
| 4 | n/a | n/a | n/a |
| 6 | n/a | n/a | n/a |
| 8 | n/a | n/a | n/a |
| 10 | 79.68% | 80.79% | 87.21% |

- Decoder 只在 10-shot 上运行，其余 shot 的 task 指标为 n/a。

## Task Exact By Backbone

| Backbone | Netlist | Vpath Attention | Path Attention |
|---|---:|---:|---:|
| graphsage | 81.15% | 77.86% | 84.60% |
| gat | 72.34% | 74.96% | 85.52% |
| gin | 85.56% | 89.56% | 91.51% |

## Paired Task Exact

| Comparison | Mean delta | Seed-level 95% CI | Win/Tie/Loss |
|---|---:|---:|---:|
| Vpath Attention - Netlist | +1.11pp | [-5.93, +8.16]pp | 5/0/13 |
| Path Attention - Vpath Attention | +6.42pp | [+2.99, +9.84]pp | 17/0/1 |
| Path Attention - Netlist | +7.53pp | [+0.27, +14.78]pp | 16/0/2 |

## Decoder Ceiling

- Canonical graph exact: 248/252.0 (98.41%).
- Canonical valid graph: 99.96%.
- Canonical stage exact: 99.91%.
- Per-seed graph exact: 42=42/42.0, 205=41/42.0, 316=41/42.0, 327=42/42.0, 330=40/42.0, 501=42/42.0.

## Interpretation

- Vpath Attention - Netlist: not separable from zero (mean +1.11pp, 95% CI [-5.93, +8.16]pp, win/tie/loss 5/0/13).
- Path Attention - Vpath Attention: statistically positive (mean +6.42pp, 95% CI [+2.99, +9.84]pp, win/tie/loss 17/0/1).
- Path Attention - Netlist: statistically positive (mean +7.53pp, 95% CI [+0.27, +14.78]pp, win/tie/loss 16/0/2).
- 每个表示都有独立的同结构 decoder；比较在 seed × backbone × 10-shot 的 18 个单元上配对。
- 该比较不构成 vpath 边与 hlink 边各自贡献的单因素消融。
