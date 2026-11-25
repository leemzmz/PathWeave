# DATE 2027 新数据实验结果

- 数据：42 macro × 20 core；folded 255、non-folded 585；六 seed 全部纳入。
- 使用固定模型和训练配方；输入、模型、原始指标及预处理图独立发布。
- 检索、decoder 和 shortcut 表为百分数均值 ± 六 seed 样本标准差；benchmark 与开销单独标注口径。

## 主结果

| Backbone | 表示 | 2-shot | 4-shot | 6-shot | 8-shot | 10-shot |
|---|---|---:|---:|---:|---:|---:|
| GraphSAGE | Netlist | 27.18 ± 1.62 | 45.44 ± 1.78 | 60.16 ± 2.16 | 67.98 ± 1.30 | 70.71 ± 3.25 |
| GraphSAGE | Vpath-only | 43.33 ± 7.40 | 65.00 ± 1.84 | 76.07 ± 1.82 | 80.87 ± 1.51 | 80.71 ± 3.56 |
| GraphSAGE | PathWeave | 55.16 ± 4.40 | 71.87 ± 1.91 | 81.31 ± 3.24 | 86.98 ± 1.50 | 87.10 ± 1.54 |
| GAT | Netlist | 29.72 ± 4.30 | 47.30 ± 5.25 | 61.63 ± 2.35 | 68.85 ± 2.11 | 69.84 ± 3.34 |
| GAT | Vpath-only | 38.69 ± 3.42 | 55.08 ± 3.41 | 65.71 ± 2.84 | 71.87 ± 3.56 | 74.25 ± 3.79 |
| GAT | PathWeave | 45.44 ± 5.28 | 66.75 ± 5.15 | 79.76 ± 4.62 | 83.89 ± 3.73 | 83.25 ± 2.54 |
| GINE | Netlist | 32.62 ± 6.94 | 60.67 ± 4.97 | 76.67 ± 7.11 | 84.64 ± 1.89 | 88.77 ± 2.46 |
| GINE | Vpath-only | 43.73 ± 4.12 | 67.90 ± 4.09 | 83.69 ± 3.52 | 89.33 ± 2.79 | 88.33 ± 2.96 |
| GINE | PathWeave | 46.79 ± 7.52 | 70.40 ± 2.98 | 86.79 ± 3.23 | 91.39 ± 2.40 | 92.02 ± 1.72 |

- PathWeave 严格最优：15/15；优于 Netlist：15/15；优于 Vpath-only：15/15。
- Vpath-only 优于 Netlist：14/15。
- PathWeave 相对 Netlist 平均增益 15.78 pp；相对 Vpath-only 6.96 pp。
- 相对 Netlist：2-shot 增益 19.29 pp；10-shot 增益 11.02 pp。

## Embedding 几何

| 表示 | hard-pair 正 margin (%) | GED Spearman |
|---|---:|---:|
| Netlist | 62.55 | 0.7232 |
| Vpath-only | 70.25 | 0.7701 |
| PathWeave | 76.52 | 0.7769 |

## Decoder（10-shot）

| Backbone | 指标 | Netlist | PathWeave |
|---|---|---:|---:|
| GraphSAGE | Correct | 76.31 ± 2.67 | 81.11 ± 5.15 |
| GraphSAGE | Valid | 95.48 ± 1.30 | 97.38 ± 0.74 |
| GAT | Correct | 66.67 ± 4.74 | 78.29 ± 2.18 |
| GAT | Valid | 96.11 ± 2.41 | 97.18 ± 1.53 |
| GINE | Correct | 86.83 ± 2.48 | 86.75 ± 5.99 |
| GINE | Valid | 98.49 ± 0.51 | 98.97 ± 0.89 |

## Shortcut

| Shot | Size | Type/port | GINE Netlist | GINE PathWeave |
|---|---:|---:|---:|---:|
| 2 | 5.63 ± 1.80 | 20.12 ± 1.98 | 32.62 ± 6.94 | 46.79 ± 7.52 |
| 4 | 5.63 ± 0.68 | 22.50 ± 1.34 | 60.67 ± 4.97 | 70.40 ± 2.98 |
| 6 | 6.47 ± 0.83 | 24.25 ± 1.92 | 76.67 ± 7.11 | 86.79 ± 3.23 |
| 8 | 6.07 ± 1.19 | 25.04 ± 1.07 | 84.64 ± 1.89 | 91.39 ± 2.40 |
| 10 | 6.79 ± 1.65 | 26.59 ± 1.77 | 88.77 ± 2.46 | 92.02 ± 1.72 |

## Benchmark 与开销

- cores_vertices：44.022619。
- cores_edges：59.455952。
- targets_vertices：13.714286。
- targets_edges：15.285714。
- vpaths：6.494048。
- target_ged：0.529522。
- intra_ged：0.617621。
- inter_ged：0.524260。

| 表示 | 顶点 | 有向消息边 | Payload KiB | 构建均值 ms | 构建中位数 ms |
|---|---:|---:|---:|---:|---:|
| netlist | 44.023 | 118.912 | 5.436 | 2.836 | 2.670 |
| vpath | 44.023 | 438.667 | 12.930 | 4.016 | 3.945 |
| pathweave | 44.023 | 468.437 | 23.185 | 4.320 | 4.229 |

## 评测口径与限制

- Signal 特征为 9-D（含 OT），论文文字为 8-D；以发布的张量 schema 为准。
- CM 在当前实现中为无向，论文文字描述为有向；输入图按实现语义构建。
- Hard-pair 指标取 GED 最近的 5 个候选中 embedding 最难负例，不是单一最近 GED 目标。
- Encoder 本次固定 CPU 单线程；decoder 显式 CUDA；环境与源码哈希见 [来源记录](../provenance/run_manifest.json)。
- 新旧数据还涉及当前 CktPath 提取源码版本差异，不能将新旧数值变化全部归因于 folded。
- 新电路完成自动语法、PathIR 与目标对应检查；本轮未开展模拟电路专家审阅或 SPICE 性能验证。
- 跨宏 GED 是 seed 20261124 的 1000 对无重复抽样；宏内 7980 对与 target 861 对全量。
- 共享 LDA 保持原论文五个 macro，不依据新结果筛选；仅拟合 seed 42 训练 support。
- 人工图审：PathWeave 对 DFCFC2/3 分离更清楚，ACBC/ACBC2/DFCFC 仍有重叠；不宣称所有类别完全分离或二维邻近指标全部最优。
- 图与原始坐标：embedding_diagnostic.{eps,pdf,png,svg}、embedding_diagnostic_raw.npz。
- paper_numbers.json、full_matrix.json、paired_encoder_gains.csv 保留原始值与配对差异。
- `paper_tables.tex` 提供对应数值的 LaTeX 表格。
- 完整性及原归档保护检查见 final_audit.json。

## Decoder target-side 诊断

- Canonical target 重构低于 100%：14/54 组；均值 98.90%，最小值 76.19%。
- 各组均保持 3000 步；target-side 仅为诊断，不用作筛选或追加训练的门槛。
- Decoder 结果同时反映编码质量与有限预算解码器误差，按论文定位作为次级证据解读。
