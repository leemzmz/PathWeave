# PathWeave

[English](README.md) | **简体中文**

论文 **PathWeave: A Path-Aware Structural Representation for Macromodel Topology Recovery of Analog Circuits** 已被 **DATE 2027** 录用。

PathWeave 将晶体管级电路编码为 128 维 embedding，并与 Gm+RC 功能宏模型对齐；可选的 GRU decoder 用于拓扑重构评估。

- 输入表示：Netlist、Vpath-only、PathWeave。
- GNN backbone：GraphSAGE、GAT、GINE。
- 评测设置：42 个 macro、每 macro 20 个电路，2/4/6/8/10-shot、6 个随机种子。
- 支持独立训练、checkpoint 重载验证、逐配置导出和跨 seed 汇总。

## 安装

- 已验证：Linux/WSL、Python 3.11.13。
- 使用发布的预处理图复算 encoder 指标只需 CPU，无需 CktPath；decoder 严格对照默认使用原 CUDA 设备。

```bash
git clone https://github.com/leemzmz/PathWeave.git app_pathweave
cd app_pathweave
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

- `requirements-dev.txt`：运行依赖及测试工具。
- `requirements-analysis.txt`：论文图、shortcut 等分析所需的额外依赖。
- Decoder 可用 `--decoder-device cpu --allow-differences` 在 CPU 上执行诊断；生成序列可能因数值差异改变，不保证与 CUDA 归档完全一致。

## CktPath：可选私有依赖与相关论文

- [CktPath](https://github.com/leemzmz/CktPath) 为私有仓库，可向作者申请访问（available upon request）。
- 使用本仓库已有模型和预处理图复算指标，无需安装 CktPath；公开数据不包含其源码。
- 处理新网表、电路生成或运行原始训练流程时需要 CktPath。
- 获得访问权限后，将其放在项目同级的 `path/`，使用提交 `14938f288c5dcd94f939b02970c7cf15a5505890`。

CktPath 方法还在以下两篇 **IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems（IEEE TCAD）期刊论文**中应用：

1. Mingzhen Li and Guoyong Shi, [**A Signal-Path Recognition Approach to Multistage Op-Amp Pole-Zero Extraction With Applications**](https://ieeexplore.ieee.org/document/11270852), **IEEE TCAD**, 2026. DOI: [10.1109/TCAD.2025.3638265](https://doi.org/10.1109/TCAD.2025.3638265).
2. Yapeng Li, Peng Xu, Mingzhen Li, Tsung-Yi Ho, Bei Yu, and Tinghuan Chen, [**Analog Circuit Representation Learning Informed by Kirchhoff’s Current Law**](https://ieeexplore.ieee.org/document/11614522), **IEEE TCAD**, 2026. DOI: [10.1109/TCAD.2026.3715070](https://doi.org/10.1109/TCAD.2026.3715070).

## 数据与结果

| 输入 | 预处理图 | 模型与结果 |
|---|---|---|
| `_data/` | `preprocessed/data/` | `_run/models/`、`_run/results/` |
| `_data_new/` | `preprocessed/data_new/` | `_run_new/models/`、`_run_new/results/` |

- 两套数据均为 42 macro × 20 core；新数据含 folded 255 个、non-folded 585 个。
- 两套归档均仅包含 `README.md`、`models/`、`results/`；不附带训练源码或 CktPath 快照。
- 每套完整实验包含 270 circuit encoder、54 decoder 和 6 target encoder。
- 新数据的表格、实验图及数值见 [实验报告](_run_new/results/analysis/paper_results.md)。

## 无需 CktPath 复算模型

先验证新数据的一组模型：

```bash
python Run/evaluate_preprocessed.py \
  --seeds 42 --shots 2 --backbones graphsage --backends graph_path_attention \
  --output _experiments/reload_example.json
```

复算新数据的六 seed 完整矩阵（270 encoder、54 decoder；decoder 需要 CUDA）：

```bash
python Run/evaluate_preprocessed.py --decoder --output _experiments/reload_new.json
```

复算原始数据并记录与历史结果的差异：

```bash
python Run/evaluate_preprocessed.py \
  --dataset _data --run _run --graphs preprocessed/data \
  --decoder --allow-differences --output _experiments/reload_original.json
```

- `preprocessed/data/`、`preprocessed/data_new/` 分别保存 840 个电路的三种模型输入图；只含张量、特征定义和哈希，不含 CktPath 源码。
- 入口实际加载权重并重新计算指标，与原始结果逐项比较；不训练，不改写历史协议或归档结果。
- 默认严格比较，存在差异即报错；`--allow-differences` 仅生成诊断报告，不将差异标记为复现通过。
- 原始网表仅用于样本身份和哈希核验；此入口不从网表提取路径。
- 输出文件必须是新文件，且位于输入和归档目录之外。
- 本环境完整验证：新数据 270 encoder + 54 decoder 的全部指标与归档完全一致。
- 原始数据主要检索指标及 54 decoder 全部一致；14 组 encoder 的 `target_manifold_spearman` 有差异，最大约 `2.67e-5`，因此旧数据命令使用诊断模式。
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
- 两套归档均对应各自训练版本；当前代码不能通过刷新协议来续跑历史 checkpoint。
- 预处理图入口用于已有权重的指标复算；原始网表训练入口仍需要 CktPath。
- 重载评估通过不代表从头训练会逐位重现历史权重；不修改既有协议来绕过来源差异。

## 文档与贡献

- [文档索引](_docs/README.md)：架构、模型、数据、术语与复现指南。
- [目录结构](_docs/directory_structure.md)：代码职责和文件位置。
- [贡献指南](CONTRIBUTING.md)：检查要求、实验归档约定与 Git 发布检查。

## 许可证

- 本项目代码采用 [MIT License](LICENSE)。
- 第三方依赖保留各自的许可证和版权声明。
