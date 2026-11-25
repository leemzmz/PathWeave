import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 模型和结果归档不参与主项目测试发现。
collect_ignore = ["_run", "_run_new"]
