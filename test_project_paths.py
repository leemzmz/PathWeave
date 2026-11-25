import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parent


# 从不同工作目录启动真实子进程，确认实验入口使用同一组数据和产物根。
def _readPaths(cwd, dataset=None, run=None):
    environment = dict(os.environ, PYTHONPATH=str(PROJECT_ROOT))
    for name, value in [("PATHWEAVE_DATASET_DIR", dataset), ("PATHWEAVE_RUN_DIR", run)]:
        environment.pop(name, None)
        if value is not None:
            environment[name] = str(value)
    command = (
        "import json; from _project_paths import DATASET_DIR, RUN_ROOT; "
        "from Experiment.EncoderFewShotExperiment import fewShotConfigurationPaths; "
        "p = fewShotConfigurationPaths(2, 'graphsage', 'graph_netlist', 42); "
        "print(json.dumps({'dataset': str(DATASET_DIR), 'run': str(RUN_ROOT), "
        "'model': str(p['circuit']), 'result': str(p['result'])}))"
    )
    return json.loads(subprocess.check_output(
        [sys.executable, "-c", command], cwd=cwd, env=environment, text=True))


# 默认入口必须继续对应旧数据归档。
def testDefaultArchivePaths(tmp_path):
    paths = _readPaths(tmp_path)
    assert Path(paths["dataset"]) == PROJECT_ROOT / "_data"
    assert Path(paths["run"]) == PROJECT_ROOT / "_run"
    assert Path(paths["model"]).is_relative_to(PROJECT_ROOT / "_run/models")
    assert Path(paths["result"]).is_relative_to(PROJECT_ROOT / "_run/results")


# 显式实验根隔离模型与结果；相对路径不随调用者工作目录漂移。
@pytest.mark.parametrize("relative", [True, False])
def testIndependentRunPaths(tmp_path, relative):
    dataset = Path("_data_new") if relative else tmp_path / "inputs"
    run = Path("_experiments/test") if relative else tmp_path / "outputs"
    expectedDataset = PROJECT_ROOT / dataset if relative else dataset
    expectedRun = PROJECT_ROOT / run if relative else run
    existed = expectedRun.exists()
    paths = _readPaths(tmp_path, dataset, run)
    assert Path(paths["dataset"]) == expectedDataset
    assert Path(paths["run"]) == expectedRun
    assert Path(paths["model"]).is_relative_to(expectedRun / "models")
    assert Path(paths["result"]).is_relative_to(expectedRun / "results")
    assert expectedRun.exists() == existed
