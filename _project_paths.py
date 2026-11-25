"""数据和产物路径；相对环境变量路径始终以项目根目录为基准。"""

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


# 解析可选路径配置，使新实验可以使用独立输出而不修改归档。
def _resolveProjectPath(variableName, defaultPath):
    configured = os.environ.get(variableName)
    if not configured:
        return defaultPath
    path = Path(configured).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


DATASET_DIR = _resolveProjectPath("PATHWEAVE_DATASET_DIR", PROJECT_ROOT / "_data")
RUN_ROOT = _resolveProjectPath("PATHWEAVE_RUN_DIR", PROJECT_ROOT / "_run")
MODEL_DIR = RUN_ROOT / "models"
RESULT_DIR = RUN_ROOT / "results"
GRAPH_ALIGNMENT_MODEL_DIR = MODEL_DIR / "graph_alignment"
GRAPH_ALIGNMENT_RESULT_DIR = RESULT_DIR / "graph_alignment"
