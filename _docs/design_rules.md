# 设计规则

## 表示
- 目标是完整 Gm+RC 功能图对齐，不能退化为 macro 类别标签预测。
- 三种表示共用相同 physical 顶点；不增加 relation 顶点。
- Physical 特征为 10-D；Vpath-only signal 为 2-D；PathWeave signal 为 9-D。
- Vpath-only 与 PathWeave 共用 32-D signal attention 和相同 readout。
- Hlink 保留类型、有向性及重复关系；无向关系双向展开。
- Circuit 输入不得读取 target graph、macro label 或测试信息。

## 公平对照
- 配对配置共享 seed、train/test、target encoder、loss、batch size 和训练预算。
- 每 macro 固定 10 个训练候选与 10 个测试 core；shot 为 2/4/6/8/10 的嵌套子集。
- Circuit encoder 预算为 `100 × shot` optimizer steps；target encoder 先训练并冻结。
- Hard negative 仅来自当前训练子集中的其他 macro。
- 不将 Netlist 与路径表示的比较解释为单一组件消融。

## Decoder
- 只使用 10-shot encoder；权重按 backbone/表示独立。
- 公共 2-shot train support：每组 84 个 core；插值 α 为 0.25/0.5/0.75/1.0。
- Canonical/calibration 使用独立 batch 和等权损失；固定 3000 步。
- 不读取测试样本校准；不按 target-side 指标筛选 seed、追加训练或中断实验。
- 报告 valid graph、graph exact、stage exact 和 GED similarity；作为次级证据解读。

## 归档
- `_data/` 对应 `_run/`；`_data_new/` 对应 `_run_new/`。
- 两套结果根均仅保留 README、models、results；不包含私有依赖或训练源码快照。
- 新实验通过独立 `PATHWEAVE_RUN_DIR` 隔离输出，不覆盖已有模型与原始结果。
- 数据、源码、schema、split、seed 和预算匹配才允许复用 checkpoint。
- `--refresh` 仅允许更新尚无模型或结果的元数据；已有产物不能刷新协议或补造缺失来源。
- 每个 encoder 配置独立重载；离散指标完全一致，连续指标按既定容差核对。
- 保存逐 seed 原始结果、均值和样本标准差；不选择性报告。
- 代码变更执行相关测试；归档变更额外核验内容哈希。
