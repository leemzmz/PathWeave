# 贡献指南

## 开发环境
- 使用 Python 3.11；安装 `requirements-dev.txt`。
- 原始网表相关测试需要按 README 放置 CktPath；预处理图评估及 `test_preprocessed.py` 不需要私有依赖。
- 先阅读 `_docs/architecture.md`、`_docs/design_rules.md` 和相关模块说明。

## 修改约定
- 保持公开入口、tensor schema、checkpoint 契约和指标含义清晰。
- 新函数采用动词开头的 camelCase；私有函数以 `_` 开头；成员变量以 `m_` 开头。
- 新核心模块使用首字母大写的文件名；脚本使用首字母小写的文件名。
- 保留已有公开模块的导入兼容性。
- 数据、模型与原始结果属于实验归档，不作为临时工作目录；私有源码快照不提交到公开仓库。
- 使用 `PATHWEAVE_RUN_DIR` 为新实验选择独立产物根。
- 源码、schema、split 或训练预算变化后，不能刷新协议冒充原实验结果。
- 文档描述当前行为；设计决策记录原则与理由，不记录操作流水。
- Markdown 摘要文件控制在 250 行以内。

## 提交前检查

无需 CktPath 的公开功能检查：

```bash
python -m pytest -q test_preprocessed.py test_archive.py test_project_paths.py test_result_summary.py
python Run/check_archive.py
PATHWEAVE_DATASET_DIR=_data_new PATHWEAVE_RUN_DIR=_run_new python Run/check_archive.py
```

完整测试还需安装 CktPath：

```bash
python -m pytest -q
bash -n run_dual_pathweave_experiment.sh
```

- 涉及模型行为时，用预处理图入口实际复算指标；不要用只读文件核验替代模型验证。
- PR 描述问题、最终行为和验证结果；同步更新使用文档。
- 不提交本地环境、缓存、凭据、生成日志或独立试验目录。

## Git 发布检查

- 发布内容包括当前代码、文档、MIT 许可证、两套输入数据、预处理图、模型和原始指标。
- 不提交 CktPath 实现、已淘汰的提取代码、生成日志、凭据或本地试验目录。
- 检查待公开提交及其可达历史；只删除工作区文件不能移除历史中的源码。
- 发布前通过上述测试及归档校验，并核对 `preprocessed/*/verification.json` 的已验证范围。
- 保留实验来源记录，不将模型复算通过宣称为从头训练完全复现。
- 推送和仓库可见性变更应与维护者确认；不要未经确认改写共享分支历史。
