from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from Experiment.EncoderFewShotExperiment import (
    DECODER_FEW_SHOT,
    DECODER_GROUPS,
    DECODER_INTERPOLATION_ALPHAS,
    DECODER_OPTIMIZER_STEPS,
    ENCODER_FEW_SHOTS,
    FORMAL_DECODER_DEVICE,
    FORMAL_EMBEDDING_DEVICE,
    buildExperimentProtocol,
    buildEncoderFewShotManifest,
    casesForEncoderFewShot,
    decoderConfigurationPaths,
    encoderFewShotSplitPolicy,
    fewShotConfigurationPaths,
    fewShotExperimentPaths,
    optimizerStepsForShot,
)
from Experiment.GraphAlignmentData import buildTargetBank, discoverAlignmentCases
from Experiment.GmRcDecoder import (
    LATENT_CALIBRATED_DECODER_POLICY,
    buildLatentCalibratedDecoderData,
    loadLatentCalibratedGmRcGraphDecoder,
    trainLatentCalibratedGmRcGraphDecoder,
)


ROOT = Path(__file__).resolve().parent


# 归档数据缺少功能目标图时，跳过对应集成测试。
def _discoverCasesOrSkip(dataset):
    if not any(Path(dataset).glob("*/gm_rc_graph.json")):
        pytest.skip("_data 中缺少 gm_rc_graph.json")
    return discoverAlignmentCases(dataset)


def testEncoderFewShotManifestIsNestedAndCoversEveryCore():
    cases = _discoverCasesOrSkip(ROOT / "_data")
    manifest = buildEncoderFewShotManifest(cases, seed=42)
    allNames = {case.name for case in cases}
    trainTen, commonTest = casesForEncoderFewShot(
        manifest, cases, shot=10
    )
    assert len(trainTen) == 420
    assert len(commonTest) == 420
    assert {case.name for case in trainTen}.isdisjoint(
        case.name for case in commonTest
    )
    assert {case.name for case in trainTen} | {
        case.name for case in commonTest
    } == allNames
    previous = set()
    for shot in ENCODER_FEW_SHOTS:
        train, test = casesForEncoderFewShot(manifest, cases, shot=shot)
        names = {case.name for case in train}
        assert len(train) == 42 * shot
        assert len(test) == 420
        assert previous <= names
        assert {case.name for case in test} == {
            case.name for case in commonTest
        }
        previous = names
    policy = encoderFewShotSplitPolicy(manifest, "manifest-hash", 6)
    assert policy["train_cores_per_topology"] == 6
    assert policy["common_test_cores_per_topology"] == 10
    assert policy["manifest_sha256"] == "manifest-hash"


def testEncoderFewShotSeedsCreateIndependentSplits():
    cases = _discoverCasesOrSkip(ROOT / "_data")
    first = buildEncoderFewShotManifest(cases, seed=316)
    second = buildEncoderFewShotManifest(cases, seed=327)
    assert first["seed"] == 316
    assert second["seed"] == 327
    assert first["topologies"] != second["topologies"]
    for manifest in (first, second):
        train, test = casesForEncoderFewShot(manifest, cases, shot=10)
        assert len(train) == 420
        assert len(test) == 420


def testEncoderFewShotProtocolUsesDeterministicFormalEvaluation(tmp_path):
    cases = _discoverCasesOrSkip(ROOT / "_data")
    manifest = buildEncoderFewShotManifest(cases, seed=316)
    protocol = buildExperimentProtocol(
        cases,
        "manifest-hash",
        [ROOT / "Experiment" / "EncoderFewShotExperiment.py"],
        seed=316,
    )
    assert manifest["seed"] == protocol["seed"]
    assert protocol["formal_embedding_device"] == "cpu"
    assert protocol["formal_decoder_device"] == "cuda"
    assert FORMAL_EMBEDDING_DEVICE == "cpu"
    assert FORMAL_DECODER_DEVICE == "cuda"


def testEncoderFewShotPathsAreIsolatedAndDescriptive():
    paths = fewShotConfigurationPaths(
        8, "gin", "graph_path_attention", seed=42
    )
    value = paths["circuit"].as_posix()
    assert "few_shot_path_attention_v1" in value
    assert "/seed_42/shot_8/gin/graph_path_attention/" in value
    other = fewShotConfigurationPaths(
        8, "gin", "graph_path_attention", seed=316
    )["circuit"].as_posix()
    assert "/seed_316/shot_8/gin/graph_path_attention/" in other
    assert other != value


def testDecoderGroupsCoverEveryBackboneAndRepresentation():
    assert len(DECODER_GROUPS) == 9
    assert len(set(DECODER_GROUPS)) == 9
    assert DECODER_FEW_SHOT == 10
    assert DECODER_OPTIMIZER_STEPS == 3000
    assert DECODER_INTERPOLATION_ALPHAS == (0.25, 0.5, 0.75, 1.0)
    for backbone, backend in DECODER_GROUPS:
        paths = decoderConfigurationPaths(backbone, backend, seed=42)
        assert paths["group"] == f"{backbone}__{backend}"
        assert paths["shot"] == DECODER_FEW_SHOT
        assert f"/seed_42/shot_{DECODER_FEW_SHOT}/{backbone}/{backend}/" in (
            paths["decoder"].as_posix()
        )
        assert paths["decoder"].name == "shared_task_decoder.pt"
        assert paths["report"].name == f"{backbone}__{backend}.json"
    other = decoderConfigurationPaths(
        "gin", "graph_netlist", seed=316
    )["decoder"].as_posix()
    assert "/seed_316/shot_10/gin/graph_netlist/" in other
    with pytest.raises(ValueError, match="backbone"):
        decoderConfigurationPaths("unknown", "graph_netlist", seed=42)
    with pytest.raises(ValueError, match="backend"):
        decoderConfigurationPaths("gin", "unknown", seed=42)


def testFewShotScheduleUsesOneHundredStepsPerShot():
    assert {
        shot: optimizerStepsForShot(shot)
        for shot in ENCODER_FEW_SHOTS
    } == {
        2: 200,
        4: 400,
        6: 600,
        8: 800,
        10: 1000,
    }


def testLatentCalibratedDecoderUsesOnlyDeclaredInterpolationPoints():
    cases = _discoverCasesOrSkip(ROOT / "_data")
    records, _ = buildTargetBank(cases)
    records = records[:2]
    references = F.normalize(torch.tensor([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ]), dim=1)
    calibration = F.normalize(torch.tensor([
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 1.0],
    ]), dim=1)
    embeddings, graphData, metadata = buildLatentCalibratedDecoderData(
        references,
        calibration,
        [0, 1],
        records,
        interpolationAlphas=(0.5, 1.0),
    )
    assert embeddings.shape == (6, 3)
    assert len(graphData) == 6
    assert metadata == {
        "reference_samples": 2,
        "calibration_samples": 2,
        "interpolation_alphas": [0.5, 1.0],
        "expanded_train_samples": 6,
    }
    assert torch.allclose(
        torch.linalg.vector_norm(embeddings, dim=1),
        torch.ones(6),
    )
    assert torch.allclose(embeddings[-2:], calibration)


def testLatentCalibratedDecoderIsGroupBoundAndSourceBound(tmp_path):
    cases = _discoverCasesOrSkip(ROOT / "_data")
    records, _ = buildTargetBank(cases)
    records = records[:2]
    references = F.normalize(torch.tensor([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ]), dim=1)
    calibration = F.normalize(torch.tensor([
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 1.0],
    ]), dim=1)
    sources = {
        "graphsage/graph_netlist": "netlist-hash",
        "graphsage/graph_path_attention": "path-attention-hash",
    }
    checkpointPath = tmp_path / "shared_task_decoder.pt"
    _, metrics, _ = trainLatentCalibratedGmRcGraphDecoder(
        references,
        calibration,
        [0, 1],
        records,
        checkpointPath,
        targetCheckpointSha256="target-hash",
        calibrationSources=sources,
        decoderGroup="graphsage__graph_netlist",
        decoderShot=DECODER_FEW_SHOT,
        maxOptimizerSteps=2,
        batchSize=2,
        interpolationAlphas=(1.0,),
        seed=42,
        device="cpu",
    )
    checkpoint = torch.load(
        checkpointPath, map_location="cpu", weights_only=False
    )
    assert metrics["training_policy"] == LATENT_CALIBRATED_DECODER_POLICY
    assert checkpoint["train_samples"] == 4
    assert checkpoint["canonical_samples"] == 2
    assert checkpoint["calibration_train_samples"] == 2
    assert checkpoint["canonical_loss_weight"] == 1.0
    assert checkpoint["calibration_loss_weight"] == 1.0
    assert checkpoint["calibration_sources"] == sources
    assert checkpoint["decoder_group"] == "graphsage__graph_netlist"
    assert checkpoint["decoder_shot"] == DECODER_FEW_SHOT
    assert "backend" not in checkpoint
    assert "circuit_backbone" not in checkpoint
    loadLatentCalibratedGmRcGraphDecoder(
        checkpointPath,
        records,
        "target-hash",
        sources,
        interpolationAlphas=(1.0,),
        calibrationSampleCount=2,
        decoderGroup="graphsage__graph_netlist",
        decoderShot=DECODER_FEW_SHOT,
        device="cpu",
    )
    with pytest.raises(ValueError, match="group mismatch"):
        loadLatentCalibratedGmRcGraphDecoder(
            checkpointPath,
            records,
            "target-hash",
            sources,
            interpolationAlphas=(1.0,),
            decoderGroup="gin__graph_netlist",
            device="cpu",
        )
    with pytest.raises(ValueError, match="shot mismatch"):
        loadLatentCalibratedGmRcGraphDecoder(
            checkpointPath,
            records,
            "target-hash",
            sources,
            interpolationAlphas=(1.0,),
            decoderShot=2,
            device="cpu",
        )
    with pytest.raises(ValueError, match="calibration_sources"):
        loadLatentCalibratedGmRcGraphDecoder(
            checkpointPath,
            records,
            "target-hash",
            {"graphsage/graph_netlist": "changed"},
            interpolationAlphas=(1.0,),
            device="cpu",
        )
    with pytest.raises(ValueError, match="interpolation alpha"):
        loadLatentCalibratedGmRcGraphDecoder(
            checkpointPath,
            records,
            "target-hash",
            sources,
            interpolationAlphas=(0.5, 1.0),
            device="cpu",
        )
    with pytest.raises(ValueError, match="sample count"):
        loadLatentCalibratedGmRcGraphDecoder(
            checkpointPath,
            records,
            "target-hash",
            sources,
            interpolationAlphas=(1.0,),
            calibrationSampleCount=3,
            device="cpu",
        )
