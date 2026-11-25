# 当前状态

## 论文

- 论文：PathWeave: A Path-Aware Structural Representation for Macromodel Topology Recovery of Analog Circuits。
- 已被 DATE 2027 录用。

## 功能
- 晶体管电路 → 图表示 → 128-D embedding → Gm+RC 宏模型检索。
- 三种表示：`graph_netlist`、`graph_vpath_attention`、`graph_path_attention`。
- 三种 backbone：GraphSAGE、GAT、GINE。
- Encoder 支持 train、evaluate、summarize、all 四种阶段。
- 10-shot 的独立 GRU decoder 提供拓扑重构评估。

## 数据与实验

- `_data/` 为论文投稿数据；`_data_new/` 为投稿后重新生成的数据，晶体管电路的结构差异更大。

| 输入 | 实验根 | 协议 |
|---|---|---|
| `_data/` | `_run/` | `few_shot_path_attention_v1` |
| `_data_new/` | `_run_new/` | `few_shot_path_attention_folded_date2027_v1` |

- 每套数据：42 macro、840 core、42 target graph。
- 新数据：folded 255、non-folded 585；每宏覆盖 folded/non-folded × 差分负载深度 1/2。
- 每套完整实验：6 seed、270 encoder 配置、54 decoder 配置、330 checkpoint。
- 新实验另有 GED、shortcut、表示开销和 embedding 图，入口见 [复现指南](reproducibility.md)。

## 使用与复现
- 默认主项目入口读取 `_data/`，写入 `_run/`。
- `PATHWEAVE_DATASET_DIR`、`PATHWEAVE_RUN_DIR` 支持独立实验。
- 两套归档均仅包含 README、模型和结果；当前未保留新实验完整训练源码快照。
- 两套输入与模型可用统一入口只读核验；预处理图入口支持无需 CktPath 的权重重载和指标复算。
- 预处理图实测：新数据 270 encoder + 54 decoder 全指标一致；旧数据主要检索指标及 decoder 一致，14 组 Spearman 有微小差异。
- 原始训练入口仍检查输入与源码哈希、split 和预算，不能通过评估入口绕过训练续跑约束。
- 不用修改 protocol 的方式绕过差异；详见 [设计规则](design_rules.md)。

## 评测边界
- 评测为已覆盖 macro 内的 held-out core，不是 unseen-macro 泛化。
- 电路经过拓扑与语义检查，未进行本轮 SPICE 性能验证。
- Decoder 是次级评测，不能单独作为 encoder 质量结论。
