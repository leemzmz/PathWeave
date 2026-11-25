# 模型说明

## 输入表示
| 表示 | Physical node | Physical edge | Signal edge |
|---|---:|---:|---:|
| Netlist | 10-D type | 4-D Norm/D/G/S | 无 |
| Vpath-only | 10-D type | 4-D Norm/D/G/S | 2-D |
| PathWeave | 10-D type | 4-D Norm/D/G/S | 9-D |

- 三者使用同一 net/device 二分图与 physical 顶点集合。
- Vpath 共现边连接同一路径的成员，记录归一化相对位置差。
- Vpath-only 的两列为 `VPATH_COMEMBER`、`DELTA_POSITION`。
- PathWeave 另含 HLINK、GM/CM/RC/OL/OT 类型和 `HLINK_DIRECTED`。
- 无向 hlink 双向展开，direction 列为 0；有向关系保留其方向。
- 当前 CktPath 中 CM 为无向；所有定义以固定源码和 schema 为准。
- Signal 端点必须属于 physical 顶点；不附加 macro label、target 信息或 path 节点。

## Circuit encoder
- GraphSAGE：edge-gated 消息传播；GAT：PyG GATConv；GINE：PyG GINEConv。
- Physical backbone：2 层、hidden/output 128-D、dropout 0.1。
- 路径表示每层先 physical 传播，再 signal attention；两部分参数独立。
- Signal attention 内部 32-D，包含 edge feature、入边 softmax、residual 和 LayerNorm。
- 多重 signal 边逐条参与 attention。
- Readout：mean pooling + L2 normalization。

## Target encoder 与对齐
- Target 是 electrical node 与 Gm/R/C primitive 构成的二分功能图。
- GraphSAGE target encoder 输出 128-D，按 seed 训练 300 epochs 后冻结。
- 监督来自 42 个 target 的 approximate GED similarity matrix。
- Circuit loss：paired cosine、batch geometry、target retrieval、hard-negative manifold 与 ranking。
- Hard-negative Top-K=5，每 anchor 选择 2 个 negative macro；temperature=0.1。
- Margin scale=0.5、manifold weight=0.5、ranking weight=0.25。

## 训练协议
- Seed：42、205、316、327、330、501。
- 每 macro 20 core；10 train pool、10 test；shot 子集为 2/4/6/8/10。
- Circuit batch size=32、learning rate=1e-3；预算为 200/400/600/800/1000 步。
- 配对方法共享 split、target encoder 与训练预算。

## Decoder
- 2 层 GRU，token dim=64、hidden dim=256、dropout=0.1。
- 每个 backbone/表示组合独立权重，只连接 10-shot encoder。
- 公共 2-shot support 共 84 core；每组使用自己的 encoder 提取 latent。
- 插值：`normalize((1-α)·z_target + α·z_circuit)`，α 为 0.25/0.5/0.75/1.0。
- 每组共 42 canonical + 84×4 calibration 样本。
- 两类样本独立取 batch，损失等权相加；AdamW、3000 步、无 early stopping。
- Teacher forcing 训练，greedy decoding 评估；最大步数由实际 token 长度确定。

## 指标
- 检索：Top-1、Top-3、MRR、stage exact。
- 结构几何：所有 test core pairs 的 Spearman/Pearson，以及 hard-pair margin。
- Hard-pair：从 GED 最近的 5 个候选中选 embedding 最难负例。
- Decoder：valid graph、graph exact、stage exact、GED similarity。
- Graph exact 比较 Gm+RC semantic key；valid graph 检查表示中的连接约束。
- Target-side 仅为 canonical embedding 重构诊断，不用于筛选结果。
- 表格报告六 seed 均值和样本标准差；每组保留原始值和重载验证。
- 通用汇总的 `target_side_graph_exact` 来自对应 decoder group；canonical 计数按所有 group 的实际样本累计。
- 历史归档中的 `target_only_graph_exact` 列是 seed 级汇总，不能用于表示之间的 target-side 比较；逐组诊断以原始 decoder report 为准。

## 预处理图重载

- 图张量的节点/边特征顺序与归档 checkpoint 保持一致，物理图 10-D、signal 2-D/9-D。
- `Run/evaluate_preprocessed.py` 按原始 split 加载权重并复算指标，无需 CktPath。
- 默认严格比较；诊断模式明确记录不匹配项，不改变历史结果。
- Encoder 使用单线程 CPU；decoder 严格对照使用 CUDA，CPU decoder 仅保证可执行，不保证生成序列完全相同。
