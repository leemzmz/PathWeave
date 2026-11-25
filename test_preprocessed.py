"""无私有依赖的输入恢复、模型导入及损坏检测。"""

import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from GraphSAGE.PreprocessedData import BACKENDS, FORMAT_VERSION, hashFile, loadPreprocessedGraphs, unpackGraph
from GraphSAGE.graph_data import NETLIST_NODE_FEATURE_SCHEMA, EDGE_FEATURE_SCHEMA
from GraphSAGE.SignalAttentionData import SIGNAL_ONLY_EDGE_FEATURE_SCHEMA, VPATH_SIGNAL_EDGE_FEATURE_SCHEMA


# 构造与公开 tensor schema 相同的最小图，不调用提取器。
def _makeGraph(backend):
    value = dict(x=torch.zeros((2,10)), edge_index=torch.tensor([[0,1],[1,0]]),
        edge_attr=torch.zeros((2,4)), node_schema=list(NETLIST_NODE_FEATURE_SCHEMA), edge_schema=list(EDGE_FEATURE_SCHEMA))
    if backend != "graph_netlist":
        schema = VPATH_SIGNAL_EDGE_FEATURE_SCHEMA if backend == "graph_vpath_attention" else SIGNAL_ONLY_EDGE_FEATURE_SCHEMA
        value.update(signal_index=torch.tensor([[0,1],[1,0]]), signal_attr=torch.zeros((2,len(schema))), signal_schema=list(schema))
    return value


# 在独立解释器强制拒绝私有模块导入，公开模型与评估入口仍必须可导入。
def test_importWithoutCktPath():
    root = Path(__file__).resolve().parent
    code = """
import sys, importlib.abc
class DenyPrivate(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'GraphSAGE.PathExtraction' or fullname.split('.')[0] in {'core','extraction'}:
            raise AssertionError('private dependency imported: ' + fullname)
sys.meta_path.insert(0,DenyPrivate())
from Run.evaluate_preprocessed import evaluateArchive
from Experiment.encoder_backends import getGraphBackendSpec
from GraphSAGE.PreprocessedData import unpackGraph
for backend in ('graph_netlist','graph_vpath_attention','graph_path_attention'):
    getGraphBackendSpec(backend)
assert 'GraphSAGE.PathExtraction' not in sys.modules
"""
    subprocess.run([sys.executable, "-B", "-c", code], cwd=root, check=True, capture_output=True, text=True)


# 对每种图表示检查特征宽度、端点和有限值，防止坏输入被模型静默接受。
@pytest.mark.parametrize("backend", BACKENDS)
def test_validateTensorSchemas(backend):
    value = _makeGraph(backend)
    graph = unpackGraph(value,backend)
    assert graph is not None
    value["edge_index"][0,0] = 2
    with pytest.raises(ValueError, match="endpoints"):
        unpackGraph(value,backend)
    value = _makeGraph(backend)
    value["x"][0,0] = float('nan')
    with pytest.raises(ValueError, match="node features"):
        unpackGraph(value,backend)
    value = _makeGraph(backend)
    value["node_schema"] = list(reversed(value["node_schema"]))
    with pytest.raises(ValueError, match="schema mismatch"):
        unpackGraph(value,backend)


# 哈希和输入身份都必须匹配；不接受换图包或换网表。
def test_checkBundleProvenance(tmp_path):
    core = tmp_path / "core.cir"
    target = tmp_path / "target.json"
    core.write_text("netlist")
    target.write_text("{}")
    cases = [SimpleNamespace(name="topo/core.cir", topology="topo", netlistPath=core, targetPath=target)]
    payload = {b: {cases[0].name: _makeGraph(b)} for b in BACKENDS}
    torch.save(payload,tmp_path/"graphs.pt")
    manifest = dict(format=FORMAT_VERSION, graphs_sha256=hashFile(tmp_path/"graphs.pt"),
        core_sha256={cases[0].name:hashFile(core)}, target_sha256={"topo":hashFile(target)})
    (tmp_path/"manifest.json").write_text(json.dumps(manifest))
    _, graphs = loadPreprocessedGraphs(tmp_path,cases)
    assert set(graphs) == set(BACKENDS)
    core.write_text("changed")
    with pytest.raises(ValueError,match="circuit input mismatch"):
        loadPreprocessedGraphs(tmp_path,cases)
    core.write_text("netlist")
    (tmp_path/"graphs.pt").write_bytes(b"damaged")
    with pytest.raises(ValueError,match="checksum mismatch"):
        loadPreprocessedGraphs(tmp_path,cases)


# 诊断选项不能把数值差异伪装为严格匹配。
def test_reportMetricDifferencesExplicitly():
    from Run.evaluate_preprocessed import _compareEvaluation
    reference = {"seed": 42, "encoder_test": {"retrieval_top1": 0.8}}
    actual = {"encoder_test": {"retrieval_top1": 0.7}}
    with pytest.raises(ValueError):
        _compareEvaluation(reference, actual, False)
    report = _compareEvaluation(reference, actual, True)
    assert report["reference_match"] is False
    assert len(report["differences"]) == 1
    assert report["differences"][0]["metric"] == "encoder_test.retrieval_top1"
    assert report["differences"][0]["actual"] == 0.7
    assert _compareEvaluation(reference, reference, False)["reference_match"] is True
