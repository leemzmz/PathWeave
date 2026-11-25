from copy import deepcopy
from pathlib import Path

import pytest
import torch

from Experiment.EncoderFewShotExperiment import compareReloadMetrics
from Experiment.GraphAlignment import (
    DEFAULT_HARD_NEGATIVE_CONFIG, loadCircuitAlignmentEncoder,
    trainCircuitAlignmentEncoder,
)
from Experiment.GraphAlignmentData import (
    buildCircuitAlignmentSamples, buildTargetBank, discoverAlignmentCases,
)
from Experiment.GmRcDecoder import encodeCircuitSamples
from Experiment.encoder_backends import getGraphBackendSpec
from GraphSAGE.SignalAttentionData import (
    VPATH_SIGNAL_EDGE_FEATURE_DIM,
    VPATH_SIGNAL_EDGE_FEATURE_SCHEMA,
)
from GraphSAGE.CircuitEncoder import buildCircuitEncoder
from GraphSAGE.GmRcGraph import GmRcGraphEncoder
from GraphSAGE.graph_data import GraphTensor
from GraphSAGE.graphsage import GraphSAGEEncoder


ROOT = Path(__file__).resolve().parent


# 归档数据缺少功能目标图时，跳过对应训练与重载测试。
def _discoverCasesOrSkip(dataset):
    if not any(Path(dataset).glob("*/gm_rc_graph.json")):
        pytest.skip("_data 中缺少 gm_rc_graph.json")
    return discoverAlignmentCases(dataset)


# Exercise different ports on an unchanged graph with distinguishable neighbors.
def _makeGraph(nodeDim, edgeDim):
    generator = torch.Generator().manual_seed(42)
    x = torch.randn(5, nodeDim, generator=generator)
    edgeIndex = torch.tensor([
        [0, 1, 1, 2, 2, 3, 3, 4, 4, 0, 0, 2],
        [1, 0, 2, 1, 3, 2, 4, 3, 0, 4, 2, 0],
    ])
    edgeAttr = torch.eye(edgeDim)[torch.arange(edgeIndex.shape[1]) % edgeDim]
    return GraphTensor(x, edgeIndex, edgeAttr)


@pytest.mark.parametrize("backbone", ["gat", "gin"])
@pytest.mark.parametrize("backend", ["graph_netlist"])
def testBackboneConsumesPortsAndPreservesGraphBatch(backbone, backend):
    spec = getGraphBackendSpec(backend)
    torch.manual_seed(42)
    model = buildCircuitEncoder(
        backbone, spec.node_feature_dim, edgeFeatureDim=len(spec.edge_feature_schema),
        dropout=0,
    )
    graph = _makeGraph(spec.node_feature_dim, len(spec.edge_feature_schema))
    batch = spec.batch_graphs([graph])
    output = spec.encode_batch(model, batch, "cpu")
    assert output.shape == (1, 128)
    assert torch.allclose(output.norm(dim=1), torch.ones(1))
    swapped = GraphTensor(graph.x, graph.edge_index, graph.edge_attr.roll(1, dims=1))
    changed = spec.encode_batch(model, spec.batch_graphs([swapped]), "cpu")
    assert not torch.allclose(output, changed, atol=1e-6)
    ports = graph.edge_attr.clone().requires_grad_()
    weights = torch.arange(128, dtype=torch.float32)
    loss = (model(graph.x, graph.edge_index, torch.zeros(5, dtype=torch.long), ports) * weights).sum()
    loss.backward()
    assert ports.grad.abs().sum() > 0
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    joined = spec.encode_batch(model, spec.batch_graphs([graph, swapped]), "cpu")
    torch.testing.assert_close(joined, torch.cat([output, changed]), atol=1e-6, rtol=1e-5)
    permutation = torch.tensor([2, 4, 1, 0, 3])
    inverse = torch.argsort(permutation)
    permuted = GraphTensor(graph.x[permutation], inverse[graph.edge_index], graph.edge_attr)
    reordered = spec.encode_batch(model, spec.batch_graphs([permuted]), "cpu")
    torch.testing.assert_close(output, reordered, atol=1e-6, rtol=1e-5)
    with pytest.raises(ValueError, match="edge features"):
        model(graph.x, graph.edge_index, torch.zeros(5, dtype=torch.long))


# 分层交替式必须真正使用 signal 边，且两类边的参数互不共享。
@pytest.mark.parametrize("backbone", ["graphsage", "gat", "gin"])
def testLayeredPathAttentionUsesSignalWithSeparateParameters(backbone):
    spec = getGraphBackendSpec("graph_path_attention")
    path = ROOT / "_data" / "topo_NMCCB" / "core_05.cir"
    graph = spec.build_tensor(path)
    assert graph.m_signalIndex.shape[1] > 0
    assert graph.m_signalIndex.shape[0] == 2
    assert graph.m_graph.edge_feature_schema == ["Norm", "D", "G", "S"]
    assert graph.m_graph.x.shape[1] == 10
    model = spec.build_encoder(
        backbone,
        inputDim=spec.node_feature_dim,
        edgeFeatureDim=len(spec.edge_feature_schema),
        dropout=0,
    )
    model.eval()
    batch = spec.batch_graphs([graph])
    output = spec.encode_batch(model, batch, "cpu")
    joined = spec.encode_batch(model, spec.batch_graphs([graph, graph]), "cpu")
    assert output.shape == (1, 128)
    torch.testing.assert_close(output.norm(dim=1), torch.ones(1))
    torch.testing.assert_close(
        joined, torch.cat([output, output]), atol=1e-6, rtol=1e-5
    )
    blanked = type(graph)(
        m_graph=graph.m_graph,
        m_signalIndex=graph.m_signalIndex,
        m_signalAttr=torch.zeros_like(graph.m_signalAttr),
        m_signalLabels=graph.m_signalLabels,
    )
    changed = spec.encode_batch(model, spec.batch_graphs([blanked]), "cpu")
    assert not torch.allclose(output, changed, atol=1e-6)
    signalParams = {id(p) for p in model.m_signalAttention.parameters()}
    physicalParams = {id(p) for p in model.m_physicalLayers.parameters()}
    assert signalParams
    assert not signalParams & physicalParams


# vpath-only 表示自带独立 2-D signal 特征，schema 中不出现任何 hlink 列。
@pytest.mark.parametrize("backbone", ["graphsage", "gat", "gin"])
def testVpathAttentionUsesDedicatedSchema(backbone):
    spec = getGraphBackendSpec("graph_vpath_attention")
    fullSpec = getGraphBackendSpec("graph_path_attention")
    path = ROOT / "_data" / "topo_NMCCB" / "core_05.cir"
    graph = spec.build_tensor(path)
    fullGraph = fullSpec.build_tensor(path)
    assert graph.m_signalFeatureSchema == VPATH_SIGNAL_EDGE_FEATURE_SCHEMA
    assert graph.m_signalAttr.shape[1] == VPATH_SIGNAL_EDGE_FEATURE_DIM
    assert not any(
        "HLINK" in name for name in graph.m_signalFeatureSchema
    )
    assert graph.m_signalLabels.count("HLINK") == 0
    assert graph.m_signalLabels.count("VPATH_COMEMBER") > 0
    assert graph.m_signalIndex.shape[1] < fullGraph.m_signalIndex.shape[1]
    assert torch.equal(graph.m_graph.x, fullGraph.m_graph.x)
    arguments = {
        "inputDim": spec.node_feature_dim,
        "edgeFeatureDim": len(spec.edge_feature_schema),
        "dropout": 0,
    }
    model = spec.build_encoder(backbone, **arguments)
    fullModel = fullSpec.build_encoder(backbone, **arguments)
    assert model.m_signalDim == VPATH_SIGNAL_EDGE_FEATURE_DIM
    for block in model.m_signalAttention:
        assert block.m_key.in_features == 128 + VPATH_SIGNAL_EDGE_FEATURE_DIM
        assert block.m_query.out_features == 32
        assert block.m_output.out_features == 128
    assert sum(p.numel() for p in model.m_backbone.parameters()) == sum(
        p.numel() for p in fullModel.m_backbone.parameters()
    )
    assert sum(p.numel() for p in model.parameters()) < sum(
        p.numel() for p in fullModel.parameters()
    )
    model.eval()
    output = spec.encode_batch(model, spec.batch_graphs([graph]), "cpu")
    joined = spec.encode_batch(model, spec.batch_graphs([graph, graph]), "cpu")
    assert output.shape == (1, 128)
    torch.testing.assert_close(output.norm(dim=1), torch.ones(1))
    torch.testing.assert_close(
        joined, torch.cat([output, output]), atol=1e-6, rtol=1e-5
    )


# attention 宽度固定 32，且 physical backbone 参数不受影响。
@pytest.mark.parametrize("backbone", ["graphsage", "gat", "gin"])
def testAttentionWidthIsFixedAndPhysicalBackboneUnchanged(backbone):
    spec = getGraphBackendSpec("graph_path_attention")
    graph = spec.build_tensor(ROOT / "_data" / "topo_NMCCB" / "core_05.cir")
    model = spec.build_encoder(
        backbone,
        inputDim=spec.node_feature_dim,
        edgeFeatureDim=len(spec.edge_feature_schema),
        dropout=0,
    )
    plain = buildCircuitEncoder(
        backbone,
        spec.node_feature_dim,
        edgeFeatureDim=len(spec.edge_feature_schema),
        dropout=0,
    )
    assert sum(p.numel() for p in model.m_backbone.parameters()) == sum(
        p.numel() for p in plain.parameters()
    )
    for block in model.m_signalAttention:
        assert block.m_query.out_features == 32
        assert block.m_key.out_features == 32
        assert block.m_value.out_features == 32
        assert block.m_output.in_features == 32
        assert block.m_output.out_features == 128
    assert sum(p.numel() for p in model.m_signalAttention.parameters()) > 0
    model.eval()
    output = spec.encode_batch(model, spec.batch_graphs([graph]), "cpu")
    joined = spec.encode_batch(model, spec.batch_graphs([graph, graph]), "cpu")
    assert output.shape == (1, 128)
    torch.testing.assert_close(output.norm(dim=1), torch.ones(1))
    torch.testing.assert_close(
        joined, torch.cat([output, output]), atol=1e-6, rtol=1e-5
    )


# A default backbone must retain the original class, initialization and state keys.
def testGraphSageFactoryIsUnchanged():
    torch.manual_seed(42)
    original = GraphSAGEEncoder(12, edge_feature_dim=4)
    torch.manual_seed(42)
    current = buildCircuitEncoder("graphsage", 12, edgeFeatureDim=4)
    assert type(current) is GraphSAGEEncoder
    assert original.state_dict().keys() == current.state_dict().keys()
    for key, value in original.state_dict().items():
        assert torch.equal(value, current.state_dict()[key])


@pytest.mark.parametrize("backbone", ["gat", "gin"])
def testBackboneAlignmentAndStrictReload(backbone, tmp_path, monkeypatch):
    torch.set_num_threads(1)
    import Experiment.GraphAlignment as alignment
    monkeypatch.setattr(alignment, "getDevice", lambda: "cpu")
    allCases = _discoverCasesOrSkip(ROOT / "_data")
    topologies = sorted({case.topology for case in allCases})[:6]
    cases = [
        case for topology in topologies
        for case in [c for c in allCases if c.topology == topology][:2]
    ]
    records, indices = buildTargetBank(cases)
    spec = getGraphBackendSpec("graph_netlist")
    samples = buildCircuitAlignmentSamples(cases, indices, spec.build_tensor)
    target = GmRcGraphEncoder(outputDim=128)
    targetState = {key: value.clone() for key, value in target.state_dict().items()}
    calls = []
    originalLoss = alignment._hardNegativeManifoldLosses

    def _recordLoss(*args, **kwargs):
        losses = originalLoss(*args, **kwargs)
        calls.append(float(losses[0].detach()))
        return losses

    monkeypatch.setattr(alignment, "_hardNegativeManifoldLosses", _recordLoss)
    circuitPath = tmp_path / "encoder.pt"
    splitPolicy = {
        "split_version": "test",
        "split_strategy": "test",
        "split_seed": 42,
    }
    model = trainCircuitAlignmentEncoder(
        "graph_netlist", spec, samples, records, target, circuitPath,
        batchSize=12,
        hardNegativeConfig=DEFAULT_HARD_NEGATIVE_CONFIG, circuitBackbone=backbone,
        maxOptimizerSteps=3, sampleSplitPolicy=splitPolicy,
    )
    assert calls and max(calls) > 0
    assert all(torch.equal(value, target.state_dict()[key]) for key, value in targetState.items())
    loaded, checkpoint = loadCircuitAlignmentEncoder(
        circuitPath, "graph_netlist", spec,
        hardNegativeConfig=DEFAULT_HARD_NEGATIVE_CONFIG,
        circuitBackbone=backbone, device="cpu", sampleSplitPolicy=splitPolicy,
    )
    embeddings = encodeCircuitSamples(model, spec, samples, device="cpu")
    torch.testing.assert_close(
        embeddings, encodeCircuitSamples(loaded, spec, samples, device="cpu"), atol=0, rtol=0,
    )
    assert checkpoint["circuit_backbone"] == backbone
    assert checkpoint["completed_optimizer_steps"] == 3
    assert checkpoint["training_policy"] == "fixed_optimizer_steps_all_samples"
    with pytest.raises(ValueError, match="backbone"):
        loadCircuitAlignmentEncoder(
            circuitPath, "graph_netlist", spec, device="cpu",
            hardNegativeConfig=DEFAULT_HARD_NEGATIVE_CONFIG,
            sampleSplitPolicy=splitPolicy,
        )
    checkpoint["backbone_config"]["pooling"] = "sum"
    torch.save(checkpoint, circuitPath)
    with pytest.raises(ValueError, match="configuration"):
        loadCircuitAlignmentEncoder(
            circuitPath, "graph_netlist", spec, device="cpu",
            hardNegativeConfig=DEFAULT_HARD_NEGATIVE_CONFIG,
            circuitBackbone=backbone, sampleSplitPolicy=splitPolicy,
        )


# Replay must not hide a changed exact metric behind a floating-point tolerance.
def testComparisonIsolationAndReloadAudit():
    reference = {
        "encoder_test": {"retrieval_top1": 0.5, "paired_cosine_mean": 0.9},
        "decoder_test": {"graph_exact": 0.5},
    }
    replay = deepcopy(reference)
    replay["encoder_test"]["paired_cosine_mean"] += 1e-7
    assert compareReloadMetrics(reference, replay)["discrete_metrics_exact"]
    replay["decoder_test"]["graph_exact"] += 1e-7
    with pytest.raises(ValueError, match="differs"):
        compareReloadMetrics(reference, replay)
