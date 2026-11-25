# 原始数据实验

- 输入：项目根 `_data/`，42 macro × 20 core。
- 协议：`few_shot_path_attention_v1`，seed 为 42、205、316、327、330、501。
- `models/`：270 circuit encoder、54 decoder、6 target encoder。
- `results/`：逐配置结果、split、protocol 及跨 seed 汇总。
- 从项目根运行 `python Run/check_archive.py` 核验输入与产物。
- 导出和汇总入口位于项目根 `Run/`；默认路径对应本目录。
- 本归档对应历史训练源码，当前开发代码不能直接保证重现旧数值。
- 新数据实验位于相邻 `_run_new/`；说明见项目 `_docs/reproducibility.md`。

## 从预处理图复算模型

- 使用项目根的 `Run/evaluate_preprocessed.py` 加载现有权重并复算 encoder/decoder 指标，无需 CktPath。
- 输入图位于项目根 `preprocessed/data/`；命令和限制见 [复现指南](../_docs/reproducibility.md)。
