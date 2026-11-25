# 设计决策

## 共享 physical 顶点
- Netlist、Vpath-only 和 PathWeave 使用同一顶点集合。
- 路径结构以额外边表达，避免改变节点集合带来的额外比较因素。

## 独立 signal attention
- Physical 消息传播与 signal attention 参数独立，按层交替执行。
- 两个路径表示统一使用 32-D attention，使差异集中于 signal 关系。
- 多重关系逐条保留，避免端点合并丢失 hlink 类型和方向。

## 检索与解码分离
- Encoder 检索是主评测；decoder 用于验证 embedding 的拓扑重构能力。
- Decoder 架构和训练配方共享，权重按组独立，避免跨组校准混杂。
- Target-side 重构只作为诊断，防止据其选择 seed 或调整预算。

## 实验身份固定
- 用数据/源码哈希及 split manifest 标识实验，不用文件夹名称替代来源验证。
- 两套发布归档采用相同结构，只保存模型和结果，不附带私有依赖或训练源码快照。
- 发布预处理图而不发布私有提取器，实现无需 CktPath 的权重重载和指标复算。
- 独立复算入口检查历史输入、split、模型契约和权重，不修改历史协议或放宽训练续跑检查。
- 新实验使用独立输出目录，保留历史原始结果。
