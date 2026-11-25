# 目录结构

## 项目入口
- `README.md`、`README.zh-CN.md`：英文与中文安装、模型复算和训练指南；包含独立 CktPath 说明及 TCAD 文献。
- `LICENSE`、`CONTRIBUTING.md`：MIT 许可与贡献要求。
- `requirements.txt`、`requirements-dev.txt`、`requirements-analysis.txt`：运行、测试和分析依赖。
- `_project_paths.py`：默认数据/产物根及环境变量配置。
- `run_dual_pathweave_experiment.sh`：encoder/decoder 实验入口。

## 核心模块
- `GraphSAGE/graph_data.py`：Netlist 物理图及 tensor。
- `GraphSAGE/PathExtraction.py`：同级 CktPath 的导入接口，仅原始网表处理需要。
- `GraphSAGE/PreprocessedData.py`：安全加载与校验发布图张量。
- `GraphSAGE/SignalAttentionData.py`：vpath 共现及 hlink signal 边。
- `GraphSAGE/CircuitEncoder.py`、`graphsage.py`：三种 physical backbone。
- `GraphSAGE/SignalAttentionEncoder.py`：physical 传播与 signal attention。
- `GraphSAGE/GmRcGraph.py`：目标图、GED、target encoder。
- `Experiment/GraphAlignmentData.py`：样本、target bank、负样本。
- `Experiment/GraphAlignment.py`：encoder 训练与评估。
- `Experiment/EncoderFewShotExperiment.py`：协议、split、路径、哈希与重载契约。
- `Experiment/encoder_backends.py`：正式表示注册。
- `Decoder/GmRcGraphDecoder.py`：图 token codec 与 GRU。
- `Experiment/GmRcDecoder.py`：decoder 校准、训练及评估。

## 运行与测试
- `Run/train_encoder_few_shot.py`、`train_shared_task_decoder.py`：训练与评估。
- `Run/export_path_attention_matrix.py`、`summarize_three_representation_results.py`：导出和汇总。
- `Run/check_archive.py`、`regenerate_targets.py`：归档核验与目标图恢复。
- `Run/export_preprocessed.py`、`evaluate_preprocessed.py`：输入图导出及无 CktPath 的模型复算。
- `Run/generate_folded_dataset.py`：新数据生成及验证。
- `Run/benchmark_realization_diversity.py`、`compare_dataset_diversity.py`：结构多样性诊断。
- `test_*.py`：模型、schema、协议、生命周期及路径配置回归。
- `conftest.py`：测试导入配置，排除模型与结果归档。

## 数据与归档
- `_data/`、`_data_new/`：原始与 folded/non-folded 数据。
- `_run/models/`、`_run/results/`：旧数据的模型与原始结果。
- `_run_new/models/`、`results/`：新数据的模型与原始结果；顶层结构与 `_run/` 一致。
- `_run_new/results/analysis/`：论文表格、图、GED、shortcut、开销和历史审计。
- `_run_new/results/provenance/`：原实验来源、环境记录和当前产物内容哈希。
- 训练源码、CktPath、重复输入、论文源文件及运行脚本/日志不放入公开结果目录。
- `preprocessed/data/`、`preprocessed/data_new/`：分别对应两套数据的公开模型输入张量和来源清单。
- `_experiments/`：独立实验与复算报告，Git 默认忽略。
- `_docs/`：使用说明和架构参考，不保存开发流水记录。
