"""用 CktPath 重新生成并校验归档数据中的 Gm+RC 目标图。"""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from _project_paths import DATASET_DIR, GRAPH_ALIGNMENT_RESULT_DIR


CKTPATH_ROOT = PROJECT_ROOT.parent / "path"
PROTOCOL_PATH = (
    GRAPH_ALIGNMENT_RESULT_DIR
    / "few_shot_path_attention_v1/seed_42/protocol.json"
)


# 在复制前校验全部生成文件，避免向归档数据写入不匹配的目标图。
def regenerateTargets():
    generator = CKTPATH_ROOT / "scripts/generate_gm_rc_graph.py"
    if not generator.is_file():
        raise FileNotFoundError(f"CktPath 生成脚本不存在: {generator}")
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    expected = protocol["target_sha256"]
    for savedProtocol in sorted(PROTOCOL_PATH.parent.parent.glob("seed_*/protocol.json")):
        recorded = json.loads(savedProtocol.read_text(encoding="utf-8"))
        if recorded["target_sha256"] != expected:
            raise ValueError(f"历史 seed 的目标图哈希不一致: {savedProtocol}")
    topologyDirs = {
        path.name for path in (DATASET_DIR).iterdir() if path.is_dir()
    }
    if set(expected) != topologyDirs:
        raise ValueError("_data 的 topology 与历史协议不一致")
    with TemporaryDirectory(prefix="pathweave_targets_") as temporary:
        staged = {}
        for topology, digest in sorted(expected.items()):
            source = DATASET_DIR / topology / "macro_model.json"
            if not source.is_file():
                raise FileNotFoundError(f"缺少宏模型: {source}")
            output = Path(temporary) / topology / "gm_rc_graph.json"
            subprocess.run(
                [sys.executable, str(generator), str(source), str(output)],
                check=True,
                capture_output=True,
                text=True,
            )
            actual = hashlib.sha256(output.read_bytes()).hexdigest()
            if actual != digest:
                raise ValueError(f"生成目标图与历史协议不一致: {topology}")
            staged[topology] = output

        for topology in staged:
            target = DATASET_DIR / topology / "gm_rc_graph.json"
            if target.exists():
                if not target.is_file():
                    raise ValueError(f"目标图路径不是文件: {target}")
                actual = hashlib.sha256(target.read_bytes()).hexdigest()
                if actual != expected[topology]:
                    raise ValueError(f"现有目标图与历史协议不一致: {target}")

        for topology, source in staged.items():
            target = DATASET_DIR / topology / "gm_rc_graph.json"
            if not target.exists():
                shutil.copyfile(source, target)
        return len(staged)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    print(f"已校验并恢复 {regenerateTargets()} 份目标图")
