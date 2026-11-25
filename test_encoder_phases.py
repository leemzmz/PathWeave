import json
import sys
from types import SimpleNamespace

import pytest

from Run import train_encoder_few_shot as runner
from Experiment.EncoderFewShotExperiment import compareReloadMetrics


# 每种 CLI phase 都能独立解析，不再依赖 decoder 开关。
@pytest.mark.parametrize("phase", ["train", "evaluate", "summarize", "all"])
def testEncoderPhaseArguments(monkeypatch, phase):
    monkeypatch.setattr(sys, "argv", ["encoder", "--phase", phase])
    assert runner._parseArgs().phase == phase


# all 确实执行三个独立阶段，且不再传递已移除的开关。
def testEncoderAllRunsThreePhases(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["encoder", "--phase", "all"])
    commands = []
    monkeypatch.setattr(runner.subprocess, "run", lambda command, **kwargs: commands.append(command))
    runner.main()
    assert [c[c.index("--phase") + 1] for c in commands] == ["train", "evaluate", "summarize"]
    assert all("--encoder-only" not in c for c in commands)


# 纯 encoder 重载必须检查离散指标，缺失指标组不能静默通过。
def testEncoderReloadRejectsDrift():
    reference = {"encoder_test": {"retrieval_top1": 0.5}}
    assert compareReloadMetrics(reference, reference)["discrete_metrics_exact"]
    with pytest.raises(ValueError, match="differs"):
        compareReloadMetrics(reference, {"encoder_test": {"retrieval_top1": 0.6}})
    with pytest.raises(ValueError, match="groups"):
        compareReloadMetrics(reference, {"decoder_test": {}})


# 用隔离产物贯通 encoder train/evaluate/summarize；模型计算替换成确定性桩。
def testEncoderLifecycleWithoutDecoder(tmp_path, monkeypatch):
    paths = {key: tmp_path / name for key, name in {
        "target": "target.pt", "summary_json": "summary.json",
        "summary_csv": "summary.csv", "summary_md": "summary.md"}.items()}
    config = {key: tmp_path / name for key, name in {
        "circuit": "circuit.pt", "result": "result.json", "reload": "reload.json"}.items()}
    paths["target"].write_bytes(b"target")
    records = [SimpleNamespace(semanticKey="target")]
    cases = [SimpleNamespace(name="core")]
    target = object()
    spec = object()
    metrics = {"retrieval_top1": 0.5, "retrieval_top3": 0.8, "retrieval_mrr": 0.7,
               "stage_exact": 1.0, "structure_spearman": 0.6,
               "target_manifold_spearman": 0.5, "pair_count": 87990}
    monkeypatch.setattr(runner, "_buildSampleCache", lambda *a: {"graph_netlist": (spec, {"core": object()})})
    monkeypatch.setattr(runner, "casesForEncoderFewShot", lambda *a, **kw: (cases, cases * 420))
    monkeypatch.setattr(runner, "encoderFewShotSplitPolicy", lambda *a: {"split": "test"})
    monkeypatch.setattr(runner, "fewShotConfigurationPaths", lambda *a, **kw: config)
    monkeypatch.setattr(runner, "evaluateCircuitAlignment", lambda *a, **kw: dict(metrics))
    monkeypatch.setattr(runner, "loadTargetGraphEncoder", lambda *a, **kw: (
        target, {"training_seed": 42, "target_keys": ["target"]}))
    monkeypatch.setattr(runner, "loadCircuitAlignmentEncoder", lambda *a, **kw: (
        object(), {"completed_optimizer_steps": 200}))

    # 模拟一次训练落盘，后续重载必须读取同一文件摘要。
    def trainCircuit(*args, **kwargs):
        config["circuit"].write_bytes(b"circuit")
        return runner.torch.nn.Identity()

    monkeypatch.setattr(runner, "trainCircuitAlignmentEncoder", trainCircuit)
    targetHash = runner.hashFile(paths["target"])
    runner._trainConfigurations(cases, {}, "manifest", "protocol", records, {},
                                target, targetHash, 42, (2,), ("gin",), ("graph_netlist",))
    original = config["result"].read_bytes()
    runner._evaluateConfigurations(cases, {}, "manifest", "protocol", records, {},
                                   paths, 42, (2,), ("gin",), ("graph_netlist",))
    assert config["result"].read_bytes() == original
    assert "decoder_test" not in json.loads(original)
    runner._summarize(paths, "protocol", 42, (2,), ("gin",), ("graph_netlist",))
    assert json.loads(paths["summary_json"].read_text())["configurations"] == 1
    with pytest.raises(ValueError, match="checksum"):
        runner._evaluateConfigurations(cases, {}, "manifest", "wrong", records, {},
                                       paths, 42, (2,), ("gin",), ("graph_netlist",))


# 构造最小协议目录，回归检查必须观察实际磁盘内容而非仅检查异常。
def _prepareProtocolFixture(tmp_path, monkeypatch, artifact=None):
    resultRoot = tmp_path / "results"
    modelRoot = tmp_path / "models"
    resultRoot.mkdir()
    modelRoot.mkdir()
    paths = {"result_root": resultRoot, "model_root": modelRoot,
             "manifest": resultRoot / "split_manifest.json",
             "protocol": resultRoot / "protocol.json"}
    runner.writeExperimentJson(paths["manifest"], {"seed": 42})
    runner.writeExperimentJson(paths["protocol"], {"version": "old"})
    if artifact:
        target = modelRoot / "target.pt" if artifact == "model" else resultRoot / "result.json"
        target.write_bytes(b"existing experiment")
    monkeypatch.setattr(runner, "fewShotExperimentPaths", lambda seed: paths)
    monkeypatch.setattr(runner, "buildEncoderFewShotManifest", lambda *a, **kw: {"seed": 42})
    monkeypatch.setattr(runner, "validateEncoderFewShotManifest", lambda *a: None)
    monkeypatch.setattr(runner, "_sourcePaths", lambda: [])
    monkeypatch.setattr(runner, "buildExperimentProtocol", lambda *a, **kw: {"version": "new"})
    return paths


# 显式 refresh 也不能改写已存在模型或结果的来源。
@pytest.mark.parametrize("artifact", ["model", "result"])
def testProtocolRefreshPreservesArtifacts(tmp_path, monkeypatch, artifact):
    paths = _prepareProtocolFixture(tmp_path, monkeypatch, artifact)
    before = {f:f.read_bytes() for f in tmp_path.rglob("*") if f.is_file()}
    with pytest.raises(ValueError, match="cannot refresh"):
        runner._prepareProtocol([], 42, refreshProtocol=True)
    assert {f:f.read_bytes() for f in tmp_path.rglob("*") if f.is_file()} == before


# 缺失协议或 split 时不能给已有权重补造新的来源文件。
@pytest.mark.parametrize("missing", ["manifest", "protocol"])
def testProtocolRejectsOrphanArtifacts(tmp_path, monkeypatch, missing):
    paths = _prepareProtocolFixture(tmp_path, monkeypatch, "model")
    paths[missing].unlink()
    before = {f:f.read_bytes() for f in tmp_path.rglob("*") if f.is_file()}
    with pytest.raises(ValueError, match="without provenance"):
        runner._prepareProtocol([], 42, refreshProtocol=True)
    assert {f:f.read_bytes() for f in tmp_path.rglob("*") if f.is_file()} == before


# 尚未训练的协议可以刷新，匹配协议的已有实验仍可正常复用。
def testProtocolRefreshBeforeTrainingAndResume(tmp_path, monkeypatch):
    paths = _prepareProtocolFixture(tmp_path, monkeypatch)
    runner._prepareProtocol([], 42, refreshProtocol=True)
    assert json.loads(paths["protocol"].read_text()) == {"version": "new"}
    (paths["model_root"] / "target.pt").write_bytes(b"trained")
    before = {f:f.read_bytes() for f in tmp_path.rglob("*") if f.is_file()}
    runner._prepareProtocol([], 42)
    assert {f:f.read_bytes() for f in tmp_path.rglob("*") if f.is_file()} == before
