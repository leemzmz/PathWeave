# Folded/non-folded 数据实验

- 输入：项目根 `_data_new/`，42 macro × 20 core；folded 255、non-folded 585。
- 协议：`few_shot_path_attention_folded_date2027_v1`；seed 为 42、205、316、327、330、501。
- Signal 特征 9-D、CM 无向；encoder 单线程 CPU、decoder CUDA。
- `models/`：270 circuit encoder、54 decoder、6 target encoder。
- `results/graph_alignment/`：逐配置结果、独立重载记录、split、protocol 及跨 seed 汇总。
- `results/analysis/`：论文表格、实验图、GED、shortcut、表示开销及原始分析数据。
- `results/provenance/run_manifest.json`：原训练环境、输入及源码哈希记录。
- `results/provenance/output_hashes.json`：按当前目录记录的模型、结果和分析产物 SHA256。

## 核验

从项目根执行，无需 CktPath，且不改写归档：

```bash
PATHWEAVE_DATASET_DIR=_data_new PATHWEAVE_RUN_DIR=_run_new python Run/check_archive.py
```

- 核验输入、产物哈希、矩阵数量、JSON 和模型容器完整性；不执行模型推理。
- 本目录与 `_run/` 一致，仅保存 README、模型和结果，不包含训练源码或私有 CktPath。
- 已有权重可通过预处理图入口复算指标；原始网表处理和从头训练仍需 CktPath。
- 原始来源记录与历史审计中的路径、哈希保持原样，可能涵盖未分发的日志、源码或论文文件。
- `results/analysis/final_audit.json` 是原完整实验的历史审计记录，不代表当前发布包包含源码。
- 详细说明见项目 [复现指南](../_docs/reproducibility.md)。

## 结果

- [完整报告](results/analysis/paper_results.md)、[表格片段](results/analysis/paper_tables.tex)、[数值](results/analysis/paper_numbers.json)。
- `results/analysis/embedding_diagnostic.*` 与 NPZ 保存实验图及底层数组。
- 论文报告提到的 `run_manifest.json` 位于 `results/provenance/`。

## 从预处理图复算模型

- 使用项目根的 `Run/evaluate_preprocessed.py` 加载现有权重并复算 encoder/decoder 指标，无需 CktPath。
- 输入图位于项目根 `preprocessed/data_new/`；命令和限制见 [复现指南](../_docs/reproducibility.md)。
