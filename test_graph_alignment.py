from copy import deepcopy
from itertools import combinations
from pathlib import Path

import torch
import pytest

from Decoder.GmRcGraphDecoder import (
    GM_RC_TOKEN_SCHEMA,
    GmRcGraphDecoder,
    gmRcGraphToTokenIds,
    gmRcTokenIdsToGraphData,
)
from Experiment.GraphAlignment import _hardNegativeManifoldLosses
from Experiment.GraphAlignmentData import (
    CircuitAlignmentSample,
    MacroHardNegativeSampler,
    buildTargetBank,
    discoverAlignmentCases,
)
from GraphSAGE.GmRcGraph import (
    GM_RC_EDGE_FEATURE_DIM,
    GM_RC_GED_VERSION,
    GM_RC_NODE_FEATURE_DIM,
    GM_RC_NODE_FEATURE_SCHEMA,
    GmRcGraphEncoder,
    encodeGmRcBatch,
    gmRcGraphApproximateGed,
    gmRcGraphGedSimilarity,
    gmRcGraphSimilarity,
    gmRcGraphSimilarityVersion,
    gmRcSemanticKey,
    loadGmRcGraph,
)
from GraphSAGE.graph_data import batch_graphs


ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "_data"


# 目标图已从归档数据中移除；仅跳过确实依赖目标图的测试。
def _discoverCasesOrSkip(dataset):
    if not any(Path(dataset).glob("*/gm_rc_graph.json")):
        pytest.skip("_data 中缺少 gm_rc_graph.json")
    return discoverAlignmentCases(dataset)


# 直接加载目标图的用例使用同一缺失数据判定。
def _loadTargetOrSkip(path):
    if not Path(path).is_file():
        pytest.skip(f"缺少目标图: {path}")
    return loadGmRcGraph(path)


def _makeSamplerSample(targetIndex, coreIndex):
    return CircuitAlignmentSample(
        name=f"target_{targetIndex}/core_{coreIndex}.cir",
        topology=f"target_{targetIndex}",
        targetKey=f"key_{targetIndex}",
        targetIndex=targetIndex,
        graph=None,
        feedbackImplementation="none",
    )


def testMacroHardNegativeSamplerUsesNearestOtherMacro():
    samples = [
        _makeSamplerSample(targetIndex, coreIndex)
        for targetIndex in range(3)
        for coreIndex in range(2)
    ]
    similarity = torch.tensor([
        [1.0, 0.9, 0.2],
        [0.9, 1.0, 0.4],
        [0.2, 0.4, 1.0],
    ])
    sampler = MacroHardNegativeSampler(
        samples,
        similarity,
        topK=1,
        negativeCount=1,
        seed=42,
    )
    anchors = [samples[0].name, samples[4].name]
    positive, negative, negativeTargets, negativeScores = (
        sampler.sampleBatch(anchors)
    )
    assert positive[0].targetIndex == 0
    assert positive[0].name != anchors[0]
    assert positive[1].targetIndex == 2
    assert positive[1].name != anchors[1]
    assert negativeTargets.tolist() == [[1], [1]]
    assert [sample.targetIndex for sample in negative] == [1, 1]
    assert torch.allclose(
        negativeScores, torch.tensor([[0.9], [0.4]])
    )
    replay = MacroHardNegativeSampler(
        samples,
        similarity,
        topK=1,
        negativeCount=1,
        seed=42,
    ).sampleBatch(anchors)
    assert [sample.name for sample in replay[0]] == [
        sample.name for sample in positive
    ]
    assert [sample.name for sample in replay[1]] == [
        sample.name for sample in negative
    ]


def testHardNegativeManifoldLossFitsContinuousTargetGeometry():
    anchor = torch.tensor([[1.0, 0.0]])
    positiveCore = torch.tensor([[1.0, 0.0]])
    positiveTarget = torch.tensor([[1.0, 0.0]])
    negativeTarget = torch.tensor([[[0.8, 0.6]]])
    matchingNegativeCore = torch.tensor([[[0.8, 0.6]]])
    manifold, ranking = _hardNegativeManifoldLosses(
        anchor,
        positiveCore,
        matchingNegativeCore,
        positiveTarget,
        negativeTarget,
        marginScale=0.5,
    )
    assert manifold.item() == 0.0
    assert ranking.item() == 0.0

    collapsedNegativeCore = torch.tensor([[[1.0, 0.0]]])
    manifold, _ = _hardNegativeManifoldLosses(
        anchor,
        positiveCore,
        collapsedNegativeCore,
        positiveTarget,
        negativeTarget,
        marginScale=0.5,
    )
    assert manifold.item() > 0.0


def testAlignmentDatasetAndTargetBank():
    cases = _discoverCasesOrSkip(DATASET)
    records, _ = buildTargetBank(cases)
    assert len(cases) == 840
    assert len(records) == 42


def testTopologyCoresShareFunctionalTarget():
    cases = _discoverCasesOrSkip(DATASET)
    targetKeys = {}
    for case in cases:
        targetKeys.setdefault(case.topology, set()).add(
            case.target.semanticKey
        )
    assert all(len(keys) == 1 for keys in targetKeys.values())
    assert sum(
        case.feedbackImplementation == "mos" for case in cases
    ) == 103
    assert sum(
        case.feedbackImplementation == "resistor" for case in cases
    ) == 77


def testMosAndResistorFeedbackShareFunctionalTarget():
    targetPath = DATASET / "topo_ldo1" / "gm_rc_graph.json"
    target = _loadTargetOrSkip(targetPath)
    cases = {
        case.name: case
        for case in _discoverCasesOrSkip(DATASET)
    }
    mos = cases["topo_ldo1/core_00.cir"]
    resistor = cases["topo_ldo1/core_06.cir"]
    assert mos.target.semanticKey == resistor.target.semanticKey
    assert mos.feedbackImplementation == "mos"
    assert resistor.feedbackImplementation == "resistor"
    assert any(
        device["kind"] == "R" for device in target.graphData["devices"]
    )
    assert not any("diff" in name for name in GM_RC_NODE_FEATURE_SCHEMA)


def testTargetEncoderForwardAndSimilarity():
    cases = _discoverCasesOrSkip(DATASET)
    records, _ = buildTargetBank(cases)
    selected = records[:3]
    model = GmRcGraphEncoder(
        inputDim=GM_RC_NODE_FEATURE_DIM,
        hiddenDim=16,
        outputDim=8,
        numLayers=2,
        dropout=0.0,
        edgeFeatureDim=GM_RC_EDGE_FEATURE_DIM,
    )
    embeddings = encodeGmRcBatch(
        model,
        batch_graphs([record.tensor for record in selected]),
        torch.device("cpu"),
    )
    assert embeddings.shape == (3, 8)
    assert torch.allclose(
        torch.linalg.vector_norm(embeddings, dim=1),
        torch.ones(3),
        atol=1e-5,
    )
    assert gmRcGraphSimilarity(
        selected[0].graphData, selected[0].graphData
    ) == 1.0
    assert gmRcGraphSimilarityVersion() == GM_RC_GED_VERSION


def testGmRcGraphTokenRoundTripIsLosslessForAllTargets():
    records, _ = buildTargetBank(_discoverCasesOrSkip(DATASET))
    for record in records:
        tokens = gmRcGraphToTokenIds(record.graphData)
        rebuilt = gmRcTokenIdsToGraphData(tokens)
        assert gmRcSemanticKey(rebuilt) == record.semanticKey


def testGmRcGraphDecoderProducesVariableLengthTokenLogits():
    record = _loadTargetOrSkip(
        DATASET / "topo_NMCCB" / "gm_rc_graph.json"
    )
    tokens = torch.tensor(
        [gmRcGraphToTokenIds(record.graphData)], dtype=torch.long
    )
    decoder = GmRcGraphDecoder(
        embeddingDim=16,
        tokenDim=8,
        hiddenDim=12,
        numLayers=2,
        dropout=0.0,
    )
    logits = decoder(torch.randn(1, 16), tokens[:, :-1])
    assert logits.shape == (
        1,
        tokens.shape[1] - 1,
        len(GM_RC_TOKEN_SCHEMA),
    )


def testPureGraphSimilaritySeparatesDifferentStageGraphs():
    first = _loadTargetOrSkip(
        DATASET / "topo_ACBC" / "gm_rc_graph.json"
    )
    second = _loadTargetOrSkip(
        DATASET / "topo_RAFFC" / "gm_rc_graph.json"
    )
    assert first.stageCounts == (1, 2, 1)
    assert second.stageCounts == (1, 1, 2)
    assert gmRcGraphSimilarity(first.graphData, second.graphData) < 1.0


def testPureGraphSimilarityTracksPrimitiveAndDirection():
    original = _loadTargetOrSkip(
        DATASET / "topo_CB" / "gm_rc_graph.json"
    ).graphData
    changedKind = deepcopy(original)
    compensation = next(
        device for device in changedKind["devices"]
        if device.get("role") == "compensation"
    )
    compensation["kind"] = "R" if compensation["kind"] == "C" else "C"
    assert compensation["kind"] in {"R", "C"}
    assert gmRcGraphSimilarity(original, changedKind) < 1.0

    directional = _loadTargetOrSkip(
        DATASET / "topo_AFFC" / "gm_rc_graph.json"
    ).graphData
    reversedGraph = deepcopy(directional)
    gmDevice = next(
        device for device in reversedGraph["devices"]
        if device.get("role") == "compensation"
        and device["kind"] == "Gm"
    )
    for edge in reversedGraph["edges"]:
        if edge["device"] != gmDevice["id"]:
            continue
        edge["terminal"] = (
            "output" if edge["terminal"] == "input" else "input"
        )
    assert gmRcGraphSimilarity(directional, reversedGraph) < 1.0


def testPureGraphSimilarityMatrixIsSymmetricAndBounded():
    records, _ = buildTargetBank(_discoverCasesOrSkip(DATASET))
    for record in records:
        assert gmRcGraphSimilarity(
            record.graphData, record.graphData
        ) == 1.0
    for first, second in combinations(records, 2):
        forward = gmRcGraphSimilarity(
            first.graphData, second.graphData
        )
        backward = gmRcGraphSimilarity(
            second.graphData, first.graphData
        )
        assert 0.0 <= forward <= 1.0
        assert abs(forward - backward) < 1e-12


def testPureGraphSimilarityIgnoresHiddenMetadata():
    original = _loadTargetOrSkip(
        DATASET / "topo_ldo1" / "gm_rc_graph.json"
    ).graphData
    changed = deepcopy(original)
    for index, device in enumerate(changed["devices"]):
        device["role"] = "hidden"
        device["path"] = 1000 + index
        device["order"] = 1000 - index
        device["unit"] = 2000 + index
        device["stage"] = 3000 + index
        device["differential"] = not device.get("differential", False)
    assert gmRcGraphSimilarity(original, changed) == 1.0
    assert gmRcGraphGedSimilarity(original, changed) == 1.0


def testApproximateGedPrefersSmallerVertexEdgeEdit():
    nmc = _loadTargetOrSkip(
        DATASET / "topo_NMC" / "gm_rc_graph.json"
    ).graphData
    nmcf = _loadTargetOrSkip(
        DATASET / "topo_NMCF" / "gm_rc_graph.json"
    ).graphData
    ciac = _loadTargetOrSkip(
        DATASET / "topo_CIAC" / "gm_rc_graph.json"
    ).graphData
    scafc = _loadTargetOrSkip(
        DATASET / "topo_SCAFC" / "gm_rc_graph.json"
    ).graphData
    assert gmRcGraphApproximateGed(nmc, nmcf) == 3.0
    assert gmRcGraphApproximateGed(ciac, scafc) == 8.0
    assert gmRcGraphGedSimilarity(nmc, nmcf) > gmRcGraphGedSimilarity(
        ciac, scafc
    )
    assert gmRcGraphGedSimilarity(nmc, nmcf) == gmRcGraphGedSimilarity(
        nmcf, nmc
    )
    assert gmRcGraphSimilarity(nmc, nmcf) == (
        gmRcGraphGedSimilarity(nmc, nmcf)
    )
