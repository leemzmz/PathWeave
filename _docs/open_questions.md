# 开放问题与已知限制

## 泛化与电路验证
- 当前划分评估已覆盖 macro 内的新 core；unseen-macro 泛化尚未验证。
- Folded/non-folded 覆盖不等于训练/测试中的结构分层均衡。
- 尚未系统验证 bias variant 鲁棒性及 SPICE 电气性能。

## 表示与生成
- Vpath 共现边不保留完整 path identity，多路径共享顶点时可能丢失归属信息。
- RC/OT hlink 与独立 path identity 的增益需要额外消融。
- Approximate GED 的结构几何与模拟电路功能相似性并不等价。
- Decoder 的编码误差和有限训练预算误差需要进一步区分。
- 跨字段图语法约束可能改善解码，但尚未纳入当前固定协议。

## 复现边界
- 旧数据 14 组 encoder 的 `target_manifold_spearman` 存在最多约 `2.67e-5` 差异；原因未确认，主要检索指标及 decoder 不受影响。
- 预处理图支持无 CktPath 的归档模型指标复算；新网表处理仍需提取器。
- 完整训练源码快照未保留；从头训练的历史权重复现仍未验证。
- 跨硬件/依赖版本不承诺逐位一致。
- 表示构建时间依赖机器及运行负载；以记录中的测量环境为准。
