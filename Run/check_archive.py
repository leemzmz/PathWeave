"""只读核对归档输入、正式产物和模型文件的完整性。"""

import hashlib
import json
import sys
from pathlib import Path
from zipfile import ZipFile


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from _project_paths import DATASET_DIR, MODEL_DIR, RESULT_DIR, RUN_ROOT


SUPPORTED_PROTOCOLS = {
    "few_shot_path_attention_v1",
    "few_shot_path_attention_folded_date2027_v1",
}
SEEDS = (42, 205, 316, 327, 330, 501)


# 从模型和结果目录识别归档协议，拒绝缺失、混用和未知协议。
def _resolveProtocolRoot():
    modelNames = {p.name for p in (MODEL_DIR / "graph_alignment").glob("*") if p.is_dir()}
    resultNames = {p.name for p in (RESULT_DIR / "graph_alignment").glob("*") if p.is_dir()}
    if modelNames != resultNames or len(resultNames) != 1:
        raise ValueError("归档必须包含同一协议的模型与结果，且只能包含一个协议")
    protocol = next(iter(resultNames))
    if protocol not in SUPPORTED_PROTOCOLS:
        raise ValueError(f"不支持的归档协议: {protocol}")
    return Path("graph_alignment") / protocol


# 核对新归档发布的模型、结果、分析产物及来源记录内容哈希。
def _checkOutputHashes():
    manifestPath = RESULT_DIR / "provenance" / "output_hashes.json"
    if not manifestPath.exists():
        return 0
    recorded = json.loads(manifestPath.read_text(encoding="utf-8"))
    if not isinstance(recorded, dict) or not recorded:
        raise ValueError("产物哈希清单为空或格式错误")
    root = RUN_ROOT.resolve()
    actualNames = {
        path.relative_to(root).as_posix()
        for folder in (MODEL_DIR, RESULT_DIR)
        for path in folder.rglob("*")
        if path.is_file() and path != manifestPath
    }
    if set(recorded) != actualNames:
        raise ValueError("产物哈希清单与归档文件集合不一致")
    for name, expected in recorded.items():
        relative = Path(name)
        path = (root / relative).resolve()
        if relative.is_absolute() or not path.is_relative_to(root):
            raise ValueError(f"产物哈希路径超出归档: {name}")
        if not path.is_file():
            raise FileNotFoundError(path)
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"产物哈希不匹配: {name}")
    return len(recorded)


# 对历史协议中登记的输入文件逐一核对内容哈希。
def _checkInputHashes(protocol, reference):
    for field, filename in (
        ("core_sha256", None),
        ("target_sha256", "gm_rc_graph.json"),
    ):
        recorded = protocol[field]
        if reference is not None and recorded != reference[field]:
            raise ValueError(f"seed 间 {field} 不一致")
        for name, digest in recorded.items():
            path = (
                DATASET_DIR / name
                if filename is None
                else DATASET_DIR / name / filename
            )
            if not path.is_file():
                raise FileNotFoundError(path)
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != digest:
                raise ValueError(f"历史协议哈希不匹配: {path}")


# 逐 seed 检查正式矩阵所需的文件数量与关键记录。
def _checkSeed(seed, reference, protocolRoot):
    resultRoot = RESULT_DIR / protocolRoot / f"seed_{seed}"
    modelRoot = MODEL_DIR / protocolRoot / f"seed_{seed}"
    protocol = json.loads((resultRoot / "protocol.json").read_text())
    if protocol["protocol"] != protocolRoot.name or protocol["seed"] != seed:
        raise ValueError(f"协议或 seed 与归档路径不符: {resultRoot}")
    _checkInputHashes(protocol, reference)
    expected = (
        (resultRoot, "shot_*/*/*/result.json", 45),
        (resultRoot, "shot_10/*/*/task_decoder_result.json", 9),
        (modelRoot, "shot_*/*/*/circuit_encoder.pt", 45),
        (modelRoot, "shot_10/*/*/shared_task_decoder.pt", 9),
    )
    if protocolRoot.name == "few_shot_path_attention_folded_date2027_v1":
        expected += ((resultRoot, "shot_*/*/*/reload.json", 45),)
    for root, pattern, count in expected:
        actual = len(list(root.glob(pattern)))
        if actual != count:
            raise ValueError(f"文件数量不符: {root}/{pattern}: {actual} != {count}")
    for path in (
        resultRoot / "split_manifest.json",
        resultRoot / "shared_task_decoder_report.json",
        resultRoot / "shared_task_decoder_summary.json",
        modelRoot / "target_encoder.pt",
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    return protocol


# 校验归档 JSON 可解析，PyTorch ZIP 容器的 CRC 完整。
def _checkFileFormats():
    jsonPaths = sorted(
        path
        for root in (DATASET_DIR, RESULT_DIR)
        for path in root.rglob("*.json")
    )
    for path in jsonPaths:
        json.loads(path.read_text(encoding="utf-8"))
    modelPaths = sorted(MODEL_DIR.rglob("*.pt"))
    for path in modelPaths:
        with ZipFile(path) as archive:
            badFile = archive.testzip()
        if badFile is not None:
            raise ValueError(f"模型 ZIP 校验失败: {path}: {badFile}")
    if len(modelPaths) != 330:
        raise ValueError(f"模型文件数量不符: {len(modelPaths)} != 330")
    return len(jsonPaths), len(modelPaths)


# 完成只读归档核对并输出简要计数。
def checkArchive():
    protocolRoot = _resolveProtocolRoot()
    if protocolRoot.name == "few_shot_path_attention_folded_date2027_v1":
        for name in ("run_manifest.json", "output_hashes.json"):
            path = RESULT_DIR / "provenance" / name
            if not path.is_file():
                raise FileNotFoundError(path)
    outputCount = _checkOutputHashes()
    reference = None
    for seed in SEEDS:
        reference = _checkSeed(seed, reference, protocolRoot)
    if len(reference["core_sha256"]) != 840:
        raise ValueError("core 数量不符")
    if len(reference["target_sha256"]) != 42:
        raise ValueError("目标图数量不符")
    jsonCount, modelCount = _checkFileFormats()
    print(
        f"归档核对通过：840 core、42 目标图、270 encoder 结果、"
        f"54 decoder 结果、{modelCount} 模型、{jsonCount} JSON；"
        f"协议 {protocolRoot.name}，{outputCount} 项产物哈希"
    )


if __name__ == "__main__":
    checkArchive()
