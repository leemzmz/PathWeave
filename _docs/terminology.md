# 术语

| 术语 | 定义 |
|---|---|
| macro / topology | 一个 Gm+RC 功能宏模型，对应一份 target graph |
| core | 同一 macro 的一种晶体管级实现 |
| shot | 每个 macro 用于 circuit encoder 训练的 core 数 |
| train pool | 每 macro 固定 10 个候选训练 core，shot 使用嵌套前缀 |
| common test | 每 macro 独立的 10 个 core，配对配置共用 |
| physical graph | Net/device 二分连接图，与 main path 不是同一概念 |
| main path | CktPath 中用于主通路与锚点分析的路径 |
| vpath | 满足器件电流方向约束的 VDD-to-GND 简单路径 |
| hlink / VpathLink | Vpath 之间的 GM/CM/RC/OL/OT 关系，可携带方向 |
| hpath | CktPath 内部用于推导部分 hlink 的路径，不直接进入 tensor schema |
| anchor | Hlink 的物理顶点端点 |
| signal edge | Vpath 共现关系或 hlink 在 physical 顶点上的表示 |
| Netlist | 只使用 physical 图的 backend：`graph_netlist` |
| Vpath-only | 添加 vpath 共现边：`graph_vpath_attention` |
| PathWeave | 添加 vpath 和 hlink：`graph_path_attention` |
| GINE | 使用边特征的 GIN；配置名为 `gin` |
| target encoder | 将 Gm+RC graph 编码到 128-D 的 GraphSAGE |
| circuit encoder | 将晶体管电路编码到同一空间的 GNN |
| decoder group | 独立 decoder 权重的粒度：backbone × representation |
| shared decoder | 共享结构和训练配方，不表示跨 group 共享权重 |
| target-side diagnostic | 对 canonical target embedding 的解码诊断 |
| graph exact | 生成图与目标图的 Gm+RC semantic key 一致 |
| approximate GED | 基于图属性和连接的确定性图编辑距离近似 |
| hard negative | 由目标 GED 邻近关系选择的跨 macro 训练负样本 |
| reload audit | 独立加载 checkpoint 后重算并核对指标 |
| protocol | 数据、源码、schema、split、seed 和预算构成的实验身份 |
| full_matrix | 270 个配置的逐项原始指标，不做跨 seed 聚合 |
