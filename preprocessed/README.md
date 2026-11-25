# 公开预处理图

- `data/` 对应 `_data/` 与 `_run/`；`data_new/` 对应 `_data_new/` 与 `_run_new/`。
- 每包保存 840 个电路的 Netlist、Vpath-only、PathWeave 三种张量表示。
- 仅包含模型输入数据、特征定义和来源哈希，不包含 CktPath 源码或 Python 类实现。
- `graphs.pt` 使用 PyTorch 的 `weights_only=True` 读取；`manifest.json` 绑定原始输入与历史实验协议。
- 复算入口、格式与导出命令见 [复现指南](../_docs/reproducibility.md)。

## 验证结果

- [新数据验证](data_new/verification.json)：270 encoder、54 decoder 全指标完全一致。
- [原始数据验证](data/verification.json)：主要检索指标和全部 decoder 一致；14 组 encoder 的目标流形 Spearman 有微小差异，最大约 `2.67e-5`。
- 实测使用单线程 CPU encoder、CUDA decoder；两套均未导入 CktPath。
- CPU decoder 可执行但不承诺生成序列与 CUDA 相同；使用诊断模式查看差异。
