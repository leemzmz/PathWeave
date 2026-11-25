# 架构

## 数据流
```text
core_XX.cir → Netlist / Vpath-only / PathWeave → circuit encoder → 128-D embedding
                                                               ↓
gm_rc_graph.json → target encoder → canonical embedding → 检索与结构相似度评估
                                                               ↓
                                            10-shot GRU decoder → Gm+RC graph
```

## 输入与表示
- CktPath 提供物理图、vpath 和 hlink 提取。
- `GraphSAGE/graph_data.py` 将 net/device 二分图转为 tensor。
- `SignalAttentionData.py` 在相同 physical 顶点上附加 vpath 共现和 hlink 边。
- Netlist 只使用 physical 边；Vpath-only 添加共现边；PathWeave 再加入 hlink。
- 三者均不读取 target graph 或 macro label 作为输入特征。

## 发布图数据
- `PreprocessedData.py` 保存和校验纯张量输入；模型与 tensor 类型的导入不依赖 CktPath。
- `export_preprocessed.py` 由维护者调用提取器导出三种表示。
- `evaluate_preprocessed.py` 直接读取图张量，加载原始权重，按原 split 复算并比对指标。
- CktPath 仅在处理原始网表或提取 hlink 时延迟导入。

## 编码与对齐
- `CircuitEncoder.py` 提供 GraphSAGE、GAT、GINE physical backbone。
- 两个路径表示在每层 physical 传播之后执行独立的 32-D signal attention。
- Mean pooling 和 L2 normalization 得到 128-D circuit embedding。
- `GmRcGraph.py` 将功能图编码到同一空间；target encoder 先训练，随后冻结。
- `GraphAlignment.py` 负责对齐、几何保持、检索和 hard-negative 损失。
- 具体 schema、预算与指标见 [模型说明](model_reference.md)。

## 拆分与协议
- `GraphAlignmentData.py` 发现样本并按 Gm+RC semantic key 建立 target bank。
- 每 macro 10 个 train pool、10 个 test；shot 子集使用 train pool 的嵌套前缀。
- `EncoderFewShotExperiment.py` 保存 split、数据/源码哈希与训练配置。
- Checkpoint 加载检查 schema、seed、目标编码器身份、split 和训练步数。

## 解码
- `GmRcGraphDecoder.py` 实现可逆图 token 表达及两层 GRU。
- 每个 backbone/表示组合拥有独立 decoder 权重。
- 仅使用该组 10-shot encoder，在公共 2-shot train support 上校准。
- Decoder 结果与 target-side 诊断独立报告，不改变 encoder 检索结果。

## 执行边界
- 主项目入口路径由 `_project_paths.py` 解析，支持独立数据和输出根。
- Encoder all 在独立进程中依次执行训练、重载评估、汇总。
- Decoder、全矩阵导出和跨 seed 汇总使用独立入口。
- 两套公开归档只保存模型和结果，不包含训练源码快照；当前代码不等同于历史训练版本。
- `Run/check_archive.py` 依据目录识别协议，只读核验输入与产物，无需导入 CktPath。
- `_data/`、`_data_new/` 与归档模型不作为测试或临时输出目录。
