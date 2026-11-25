# PathWeave

[English](README.md) | **简体中文**

论文 **PathWeave: A Path-Aware Structural Representation for Macromodel Topology Recovery of Analog Circuits** 已被 **DATE 2027** 录用。

PathWeave 将晶体管级电路编码为 128 维 embedding，并与 Gm+RC 功能宏模型对齐；可选的 GRU decoder 用于拓扑重构评估。

- 输入表示：Netlist、Vpath-only、PathWeave。
- GNN backbone：GraphSAGE、GAT、GINE。
- 评测设置：2/4/6/8/10-shot、6 个随机种子。

## 安装

- 已验证：Linux/WSL、Python 3.11.13。

```bash
git clone https://github.com/leemzmz/PathWeave.git app_pathweave
cd app_pathweave
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

- `requirements-dev.txt`：运行依赖及测试工具。
- `requirements-analysis.txt`：论文图、shortcut 等分析所需的额外依赖。

## CktPath

- [CktPath](https://github.com/leemzmz/CktPath) 为私有仓库，可向作者申请访问（available upon request）。
- 处理新网表、电路生成或运行原始训练流程时需要 CktPath。

CktPath 方法也用于以下论文：

1. Mingzhen Li and Guoyong Shi, [**A Signal-Path Recognition Approach to Multistage Op-Amp Pole-Zero Extraction With Applications**](https://ieeexplore.ieee.org/document/11270852), **IEEE TCAD**, 2026. DOI: [10.1109/TCAD.2025.3638265](https://doi.org/10.1109/TCAD.2025.3638265).
2. Yapeng Li, Peng Xu, Mingzhen Li, Tsung-Yi Ho, Bei Yu, and Tinghuan Chen, [**Analog Circuit Representation Learning Informed by Kirchhoff’s Current Law**](https://ieeexplore.ieee.org/document/11614522), **IEEE TCAD**, 2026. DOI: [10.1109/TCAD.2026.3715070](https://doi.org/10.1109/TCAD.2026.3715070).

## 数据与结果

| 输入 | 说明 | 预处理图 | 模型与结果 |
|---|---|---|---|
| `_data/` | 论文投稿时使用的数据 | `preprocessed/data/` | `_run/models/`、`_run/results/` |
| `_data_new/` | 投稿后重新生成的数据，晶体管电路的结构差异更大 | `preprocessed/data_new/` | `_run_new/models/`、`_run_new/results/` |

- 两套数据均为 42 macro × 20 core；新数据含 folded 255 个、non-folded 585 个。
- 每套完整实验包含 270 circuit encoder、54 decoder 和 6 target encoder。
- 新数据的表格、实验图及数值见 [实验报告](_run_new/results/analysis/paper_results.md)。

## 无需 CktPath 复算模型

评估入口加载已有权重和预处理图，重新计算指标并与归档比较。Encoder 使用 CPU；decoder 默认使用原 CUDA 设备。

先验证新数据的一组模型：

```bash
python Run/evaluate_preprocessed.py \
  --seeds 42 --shots 2 --backbones graphsage --backends graph_path_attention \
  --output _experiments/reload_example.json
```

复算新数据的全部 encoder 和 decoder：

```bash
python Run/evaluate_preprocessed.py --decoder --output _experiments/reload_new.json
```

复算原始数据并记录与历史结果的差异：

```bash
python Run/evaluate_preprocessed.py \
  --dataset _data --run _run --graphs preprocessed/data \
  --decoder --allow-differences --output _experiments/reload_original.json
```

- 默认严格比较，存在差异即报错；`--allow-differences` 仅生成诊断报告，不将差异标记为复现通过。
- 输出文件必须是新文件，且位于输入和归档目录之外。
- 在记录的验证环境中，新数据的全部指标与归档完全一致。
- 原始数据的检索及 decoder 指标一致；14 组 encoder 的 `target_manifold_spearman` 差异最大为 `2.67e-5`，因此使用诊断模式。
- CPU decoder 诊断：添加 `--decoder-device cpu --allow-differences`；生成序列可能与 CUDA 结果不同。
- 详细格式、验证结果和复现边界见 [复现指南](_docs/reproducibility.md)。

## 归档核验与训练

只读检查已有归档（无需 CktPath）：

```bash
python Run/check_archive.py
PATHWEAVE_DATASET_DIR=_data_new PATHWEAVE_RUN_DIR=_run_new python Run/check_archive.py
```

使用当前代码启动独立实验：

```bash
PATHWEAVE_DATASET_DIR=_data_new \
PATHWEAVE_RUN_DIR=_experiments/example \
./run_dual_pathweave_experiment.sh \
  --phase all --stage encoder --baseline pathweave \
  --shot 2 --gnn graphsage --seed 42
```

- 默认读取 `_data/`，写入 `_run/`；两项环境变量可指定其他数据与产物根。
- 相对配置路径以项目根为基准；独立实验写入各自的 `models/`、`results/`。
- `--phase all` 依次训练、独立重载评估和汇总；`--stage decoder` 单独运行 10-shot decoder。
- 归档对应各自的训练版本，不能通过刷新协议续跑历史 checkpoint；重载评估通过不保证从头训练得到相同权重。

## 许可证

- 本项目代码采用 [MIT License](LICENSE)。
- 第三方依赖保留各自的许可证和版权声明。
