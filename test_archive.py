"""归档校验的协议选择、输入保护和产物覆盖回归。"""

import hashlib
import json
from pathlib import Path

import pytest

from Run import check_archive as archive


# 为测试隔离模型、结果和输入目录。
def _configureArchive(tmp_path, monkeypatch):
    monkeypatch.setattr(archive, "RUN_ROOT", tmp_path)
    monkeypatch.setattr(archive, "MODEL_DIR", tmp_path / "models")
    monkeypatch.setattr(archive, "RESULT_DIR", tmp_path / "results")
    monkeypatch.setattr(archive, "DATASET_DIR", tmp_path / "data")


# 两种正式协议均可从相同目录结构中识别。
@pytest.mark.parametrize("protocol", sorted(archive.SUPPORTED_PROTOCOLS))
def test_resolveBothArchiveProtocols(tmp_path, monkeypatch, protocol):
    _configureArchive(tmp_path, monkeypatch)
    for folder in ("models", "results"):
        (tmp_path / folder / "graph_alignment" / protocol).mkdir(parents=True)
    assert archive._resolveProtocolRoot() == Path("graph_alignment") / protocol


# 错配模型、多个协议和未知协议均不应被静默忽略。
@pytest.mark.parametrize("case", ["missing", "mismatch", "multiple", "unknown"])
def test_rejectAmbiguousArchive(tmp_path, monkeypatch, case):
    _configureArchive(tmp_path, monkeypatch)
    old, new = "few_shot_path_attention_v1", "few_shot_path_attention_folded_date2027_v1"
    (tmp_path / "models/graph_alignment" / old).mkdir(parents=True)
    if case != "missing":
        name = new if case == "mismatch" else old
        (tmp_path / "results/graph_alignment" / name).mkdir(parents=True)
    if case == "multiple":
        for folder in ("models", "results"):
            (tmp_path / folder / "graph_alignment" / new).mkdir(parents=True)
    if case == "unknown":
        for folder in ("models", "results"):
            (tmp_path / folder / "graph_alignment" / old).rename(tmp_path / folder / "graph_alignment/unknown")
    with pytest.raises(ValueError):
        archive._resolveProtocolRoot()


# 对迁入 results/analysis 的文件和来源记录进行完整覆盖与哈希检查。
def test_validateRelocatedOutputs(tmp_path, monkeypatch):
    _configureArchive(tmp_path, monkeypatch)
    content = {"models/model.pt": b"model", "results/analysis/table.csv": b"value", "results/provenance/run_manifest.json": b"{}"}
    for name, data in content.items():
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
    manifest = tmp_path / "results/provenance/output_hashes.json"
    hashes = {name: hashlib.sha256(data).hexdigest() for name, data in content.items()}
    manifest.write_text(json.dumps(hashes))
    assert archive._checkOutputHashes() == 3
    artifact = tmp_path / "results/analysis/table.csv"
    artifact.write_bytes(b"changed")
    with pytest.raises(ValueError, match="产物哈希不匹配"):
        archive._checkOutputHashes()
    artifact.write_bytes(content["results/analysis/table.csv"])
    del hashes["results/analysis/table.csv"]
    manifest.write_text(json.dumps(hashes))
    with pytest.raises(ValueError, match="文件集合不一致"):
        archive._checkOutputHashes()


# 公开新归档不能在缺少哈希清单时退化为只检查数量。
def test_requireNewArchiveProvenance(tmp_path, monkeypatch):
    _configureArchive(tmp_path, monkeypatch)
    protocol = "few_shot_path_attention_folded_date2027_v1"
    for folder in ("models", "results"):
        (tmp_path / folder / "graph_alignment" / protocol).mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        archive.checkArchive()


# 核验输入被改动时必须报错，不改写输入或协议。
def test_rejectModifiedInput(tmp_path, monkeypatch):
    _configureArchive(tmp_path, monkeypatch)
    core = tmp_path / "data/topo_example/core_01.cir"
    target = core.parent / "gm_rc_graph.json"
    core.parent.mkdir(parents=True)
    core.write_text("netlist")
    target.write_text("{}")
    protocol = {
        "core_sha256": {"topo_example/core_01.cir": hashlib.sha256(core.read_bytes()).hexdigest()},
        "target_sha256": {"topo_example": hashlib.sha256(target.read_bytes()).hexdigest()},
    }
    archive._checkInputHashes(protocol, None)
    core.write_text("modified")
    with pytest.raises(ValueError, match="历史协议哈希不匹配"):
        archive._checkInputHashes(protocol, None)
    assert core.read_text() == "modified"
