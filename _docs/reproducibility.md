# 复现指南

## 环境
- 使用 Python 3.11.13；发布图张量的 encoder 评估仅需 `requirements.txt` 和 CPU，无需 CktPath。
- CktPath 为私有仓库，克隆前需申请访问权限；该方法还在两篇论文中应用，链接见 [README](../README.md#cktpath-方法应用)。
- 开发测试安装 `requirements-dev.txt`；完整网表测试还需要 CktPath，独立图数据测试可运行 `python -m pytest -q test_preprocessed.py`。
- 论文分析另安装 `requirements-analysis.txt`。
- 新实验的精确依赖记录在 `_run_new/results/provenance/run_manifest.json`；encoder 使用单线程 CPU，decoder 使用 CUDA。
- 固定 seed 和预算；跨硬件/依赖版本不承诺逐位相同。

## 归档对应

| 输入 | 结果根 | 协议 |
|---|---|---|
| `_data/` | `_run/` | `few_shot_path_attention_v1` |
| `_data_new/` | `_run_new/` | `few_shot_path_attention_folded_date2027_v1` |

- 两套归档均仅包含 `README.md`、`models/`、`results/`。
- 模型位于 `models/graph_alignment/<protocol>/`；结果位于对应 `results/` 路径。
- 输入统一由项目根的两套数据目录提供；归档中不重复保存数据。
- 原始 protocol、manifest、checkpoint 和逐 seed 数值保留为复现依据。
- 两套公开归档均不附带训练源码或私有 CktPath；当前开发代码不等同于历史训练版本。

## 核验已有结果

```bash
python Run/check_archive.py
PATHWEAVE_DATASET_DIR=_data_new PATHWEAVE_RUN_DIR=_run_new python Run/check_archive.py
```

- 两条命令均只读运行，使用 Python 标准库，无需 CktPath 或 PyTorch。
- 自动识别旧/新协议；检查输入哈希、矩阵数量、JSON 和模型容器完整性。
- 新归档另核对独立重载记录数量及模型、结果、分析文件的完整哈希清单。
- 该检查确认已保存产物完整，不重新计算推理指标，也不验证外部源码。

## 无 CktPath 的模型重载

先用 CPU 验证一组 encoder：

```bash
python Run/evaluate_preprocessed.py --seeds 42 --shots 2 \
  --backbones graphsage --backends graph_path_attention \
  --output _experiments/reload_example.json
```

完整矩阵（含 CUDA decoder）：

```bash
python Run/evaluate_preprocessed.py --decoder --output _experiments/reload_new.json
python Run/evaluate_preprocessed.py --dataset _data --run _run \
  --graphs preprocessed/data --decoder --allow-differences --output _experiments/reload_original.json
```

- 默认复算全部六 seed、五 shot、三个 backbone、三种表示的 encoder；`--decoder` 增加 10-shot decoder 复算。
- 可用 `--seeds`、`--shots`、`--backbones`、`--backends` 选择子集。
- 入口检查输入和张量包哈希、归档协议哈希、split、目标编码器身份、模型契约、权重哈希及训练预算。
- 默认逐项严格比较 encoder/decoder 指标；不一致立即报错，不输出成功报告。
- `--allow-differences` 将差异与实测值写入诊断报告；`reference_match=false` 不能解释为复现通过。
- 原始数据的历史指标与当前预处理流程存在差异，使用诊断模式报告，不承诺全部严格一致。
- 当前实现与历史源码哈希不同，不据此重写协议；通过独立评估入口验证同一输入和同一权重的输出。
- Encoder 固定单线程 CPU；decoder 默认 CUDA，与原归档一致。
- `--decoder-device cpu --allow-differences` 可在无 GPU 时运行 decoder；自回归输出对数值误差敏感，CPU 不承诺复现 CUDA 的离散指标。
- 仅在指定的全新输出文件写入复算报告。
- 预处理图不包含目标标签作为输入特征；目标图和测试划分继续来自原数据及原归档。

## 已验证结果

| 输入图 | Encoder | Decoder | CktPath 导入 |
|---|---|---|---|
| `preprocessed/data_new/` | 270/270 全指标完全一致 | 54/54 全指标完全一致 | 无 |
| `preprocessed/data/` | 256/270 通过严格比较；主要检索指标全部一致 | 54/54 全指标完全一致 | 无 |

- Encoder 使用单线程 CPU，decoder 使用 CUDA；这是当前环境的实测结果。
- 旧数据其余 14 组仅 `target_manifold_spearman` 超出严格容差，最大差值 `2.674309671069075e-5`；不改写历史结果，也不自动放宽比较门槛。
- 两包的 `verification.json` 保存验证范围、图数据哈希、匹配数量和具体差异。
- 跨机器或改用 CPU decoder 时，应重新检查比较报告。

## 发布图数据

- `preprocessed/data/` 对应 `_data/`；`preprocessed/data_new/` 对应 `_data_new/`。
- `graphs.pt`：三种表示 × 840 样本；只包含张量、列表、字典和字符串，使用 `weights_only=True` 加载。
- Physical 部分包含 `x`、`edge_index`、`edge_attr` 及节点/边 schema。
- 路径表示另含 `signal_index`、`signal_attr` 及 signal schema，分别为 2-D 和 9-D。
- `manifest.json`：格式版本、样本输入哈希、六 seed 协议哈希、提取器/张量构建代码哈希及图数据文件哈希；不包含提取器源码。
- Loader 检查节点与边的类型、形状、有限值、索引范围、schema 和样本完整覆盖。
- 保留原 `_data/` 或 `_data_new/`，供目标图读取及输入身份核验；重载评估不会调用提取器。

维护者重新导出到独立目录（需要 CktPath；目标目录必须尚不存在）：

```bash
python Run/export_preprocessed.py --dataset _data_new --run _run_new --output _experiments/export_data_new
python Run/export_preprocessed.py --dataset _data --run _run --output _experiments/export_data
```

## 复现边界

- 公开模型和预处理图支持复算已发布样本的模型指标，不等于从头复现原电路生成或训练过程。
- 用户自己的新网表仍需 CktPath 预处理；发布内容不包含 CktPath 的实现源码。
- 重新训练的旧入口仍从网表构图，需要 CktPath；当前训练代码不承诺重现历史权重。
- 完整历史训练源码快照未保留；环境/源码哈希与历史审计记录描述原实验，不表示对应快照仍可获取。
- 从头重训使用独立副本与空产物目录，不修改发布归档。

## 当前代码的新实验

```bash
PATHWEAVE_DATASET_DIR=_data_new \
PATHWEAVE_RUN_DIR=_experiments/my_run \
./run_dual_pathweave_experiment.sh --phase all --stage all \
  --baseline all --shot all --gnn all --seed 42 205 316 327 330 501
```

- 相对环境变量路径以项目根为基准；默认值为 `_data/` 与 `_run/`。
- 两项配置控制主训练、评估与结果路径；数据生成器仍固定使用 `_data/`、`_data_new/`。
- Decoder 需要该 seed 的九个 10-shot encoder，使用公共 2-shot support 和固定 3000 步。
- `--refresh` 仅适用于尚无模型或结果的元数据；已有产物时会拒绝刷新，改用独立输出目录。

导出自定义实验：

```bash
PATHWEAVE_RUN_DIR=_experiments/my_run python Run/export_path_attention_matrix.py
PATHWEAVE_RUN_DIR=_experiments/my_run python Run/summarize_three_representation_results.py
```

## 论文结果

- `_run_new/results/analysis/paper_results.md`：完整数值及结论边界。
- 同目录的 `paper_numbers.json`、`paper_tables.tex`：原始统计和四张表格片段。
- `embedding_diagnostic.*`、`embedding_diagnostic_raw.npz`：实验图与底层数据。
- `ged_pairs.jsonl`、`shortcut_results.json`、`representation_overhead.json`：逐项测量。
- `final_audit.json`、`figure_review.json` 保留原实验及人工图审记录。
- 历史来源记录可能包含未分发的生成日志、源码和论文文件哈希；它们不是当前发布包的文件清单。
- 当前归档核验以 `results/provenance/output_hashes.json` 为准；原始模型、指标和输入网表保持不变。
