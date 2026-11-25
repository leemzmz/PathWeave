# 数据集

## 组成
| 数据 | Macro | Core | Folded | Non-folded |
|---|---:|---:|---:|---:|
| `_data/` | 42 | 840 | 0 | 840 |
| `_data_new/` | 42 | 840 | 255 | 585 |

- 两套数据对齐相同的 42 个功能宏模型；每 macro 20 个裸 core。
- 42 个 `gm_rc_graph.json` 在两套数据中内容一致。
- 每个 `topo_*/` 包含 `macro_model.json`、`gm_rc_graph.json` 和 `core_00.cir` 至 `core_19.cir`。
- 新数据的 `info.json`、生成清单与验证报告保存 seed、结构选择和文件摘要；控制台日志不随数据分发。

## 新数据结构覆盖
| 输入级结构 | 差分负载深度 1 | 差分负载深度 2 | 合计 |
|---|---:|---:|---:|
| Non-folded | 341 | 244 | 585 |
| Folded | 87 | 168 | 255 |

- 42/42 macro 均覆盖这四种组合。
- 深度 2 表示差分镜像负载包含额外 cascode 对。
- 深度 1 不表示整个电路不存在其他 cascode。
- Folded 由输入 drain 与异型共栅 source 的连接判定，并与生成标记交叉核验。

## 生成与验证
- 生成器：CktPath `14938f288c5dcd94f939b02970c7cf15a5505890`。
- 输入固定为 `_data/` 中的 42 份宏模型，副本保存在 `_data_new/source_models/`。
- Base seed=42、seed step=17、每 macro 20 core、peripheral=0、bias addon=none。
- 每条 vpath 最多 4 MOS，差分镜像负载两侧 stack 对齐。
- `python Run/generate_folded_dataset.py` 生成新集合；输出存在时拒绝覆盖。
- `python Run/generate_folded_dataset.py --verify-only` 校验并更新验证报告。
- 验证语法、PathIR 语义、完整锚点、结构覆盖、目标图与网表哈希。
- 重建同一集合需使用记录的生成器版本、输入与参数；旧集合的历史生成过程不由新生成器替代。

## 多样性与边界
- 两套数据每宏均为 20 个不同的 attribute-aware WL 哈希，无宏内同构对。
- 名称和器件编号差异不作为结构多样性的依据。
- 新旧抽样对比见 `_run_new/results/analysis/dataset_diversity_comparison.json`。
- 抽样距离不等于全量 GED 相似度，也不能作为统计显著性结论。
- 新实验的完整 benchmark 统计见 `_run_new/results/analysis/benchmark_structure.json`、`ged_summary.json`。
- 拓扑与语义检查不等同于 SPICE 性能验证或模拟电路专家审阅。

## 发布模型输入

- `preprocessed/data/` 和 `preprocessed/data_new/` 保存两套数据的三种输入图，均为 840 样本。
- 预处理文件只包含张量、特征 schema 和来源哈希，不包含提取器实现。
- 与原始网表、目标图和实验协议绑定；模型复算不再次调用 CktPath。
