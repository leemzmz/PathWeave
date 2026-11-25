import argparse
import gc
import json
import os
from pathlib import Path
import sys

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from Experiment.EncoderFewShotExperiment import (
    DECODER_FEW_SHOT,
    DECODER_GROUPS,
    DECODER_INTERPOLATION_ALPHAS,
    DECODER_OPTIMIZER_STEPS,
    ENCODER_FEW_SHOT_BACKBONES,
    buildEncoderFewShotManifest,
    casesForEncoderFewShot,
    decoderConfigurationPaths,
    encoderFewShotSplitPolicy,
    fewShotExperimentPaths,
    hashFile,
    jsonFileSha256,
    validateEncoderFewShotManifest,
    writeExperimentJson,
)
from Experiment.GraphAlignment import (
    DEFAULT_HARD_NEGATIVE_CONFIG,
    encodeTargetRecords,
    loadCircuitAlignmentEncoder,
    loadTargetGraphEncoder,
)
from Experiment.GraphAlignmentData import (
    buildCircuitAlignmentSamples,
    buildTargetBank,
    discoverAlignmentCases,
)
from Experiment.GmRcDecoder import (
    LATENT_CALIBRATED_DECODER_POLICY,
    encodeCircuitSamples,
    evaluateGmRcGraphDecoder,
    evaluateTargetSideGmRcGraphDecoder,
    loadLatentCalibratedGmRcGraphDecoder,
    trainLatentCalibratedGmRcGraphDecoder,
)
from Experiment.encoder_backends import getGraphBackendSpec
from _project_paths import DATASET_DIR


CALIBRATION_SHOT = 2
EMBEDDING_DEVICE = torch.device("cpu")
DECODER_DEVICE = torch.device("cuda")
DECODER_METRICS = (
    "valid_graph_rate",
    "graph_exact",
    "token_exact",
    "stage_exact",
    "node_count_exact",
    "device_count_exact",
    "edge_count_exact",
    "primitive_count_exact",
    "ground_presence_acc",
    "ged_similarity_mean",
)


def _parseArgs():
    parser = argparse.ArgumentParser(
        description=(
            "Train one decoder per (backbone, representation) group on the "
            "common 2-shot latent support at 10-shot."
        )
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--steps", type=int, default=DECODER_OPTIMIZER_STEPS
    )
    return parser.parse_args()


def _releaseMemory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _loadManifest(cases, seed, paths):
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    expected = buildEncoderFewShotManifest(cases, seed=seed)
    if manifest != expected:
        raise ValueError("shared task decoder manifest mismatch")
    validateEncoderFewShotManifest(manifest, cases)
    return manifest, jsonFileSha256(paths["manifest"])


def _buildSampleCache(cases, targetIndex):
    cache = {}
    for backend in [item[1] for item in DECODER_GROUPS]:
        if backend in cache:
            continue
        spec = getGraphBackendSpec(backend)
        samples = buildCircuitAlignmentSamples(
            cases, targetIndex, spec.build_tensor
        )
        cache[backend] = (
            spec,
            {sample.name: sample for sample in samples},
        )
    return cache


def _selectSamples(sampleByName, cases):
    return [sampleByName[case.name] for case in cases]


def _loadCircuitModel(
    seed,
    backbone,
    backend,
    spec,
    splitPolicy,
    targetSha256,
):
    checkpointPath = decoderConfigurationPaths(
        backbone, backend, seed
    )["circuit"]
    model, _ = loadCircuitAlignmentEncoder(
        checkpointPath,
        backend,
        spec,
        seed=seed,
        hardNegativeConfig=DEFAULT_HARD_NEGATIVE_CONFIG,
        circuitBackbone=backbone,
        sampleSplitPolicy=splitPolicy,
        targetCheckpointSha256=targetSha256,
        device=EMBEDDING_DEVICE,
    )
    return model, checkpointPath


# 每个 group 只用自己那条 10-shot encoder 在公共 2-shot train cores 上抽校准 latent。
def _collectGroupCalibration(
    seed,
    cases,
    manifest,
    manifestSha256,
    sampleCache,
    targetSha256,
):
    calibrationCases, _ = casesForEncoderFewShot(
        manifest, cases, shot=CALIBRATION_SHOT
    )
    splitPolicy = encoderFewShotSplitPolicy(
        manifest, manifestSha256, DECODER_FEW_SHOT
    )
    calibration = {}
    for backbone, backend in DECODER_GROUPS:
        spec, sampleByName = sampleCache[backend]
        samples = _selectSamples(sampleByName, calibrationCases)
        model, checkpointPath = _loadCircuitModel(
            seed, backbone, backend, spec, splitPolicy, targetSha256
        )
        embeddings = encodeCircuitSamples(
            model,
            spec,
            samples,
            batchSize=64,
            device=EMBEDDING_DEVICE,
        )
        source = f"shot_{DECODER_FEW_SHOT}/{backbone}/{backend}"
        calibration[(backbone, backend)] = {
            "embeddings": embeddings,
            "target_indices": [
                sample.targetIndex for sample in samples
            ],
            "sources": {source: hashFile(checkpointPath)},
        }
        print(
            f"CALIBRATION {source} samples={len(samples)}",
            flush=True,
        )
        del model
        _releaseMemory()
    return calibration


def _decoderNeedsTraining(groupPaths, calibrationSources, steps):
    if not groupPaths["decoder"].exists():
        return True
    stored = torch.load(
        groupPaths["decoder"], map_location=EMBEDDING_DEVICE,
        weights_only=False,
    )
    policy = stored.get("training_policy")
    sourcesMatch = (
        stored.get("calibration_sources")
        == dict(sorted(calibrationSources.items()))
    )
    stepsMatch = stored.get("completed_optimizer_steps") == steps
    del stored
    if policy != LATENT_CALIBRATED_DECODER_POLICY:
        print(
            f"RETRAIN {groupPaths['group']}: decoder policy changed",
            flush=True,
        )
        return True
    if not sourcesMatch:
        print(
            f"RETRAIN {groupPaths['group']}: calibration sources changed",
            flush=True,
        )
        return True
    if not stepsMatch:
        print(
            f"RETRAIN {groupPaths['group']}: optimizer budget changed",
            flush=True,
        )
        return True
    return False


# 单个 group：训练或加载 decoder，写出 group 报告，并只评估 10-shot 配置。
def _runGroup(
    args,
    groupPaths,
    groupCalibration,
    records,
    referenceEmbeddings,
    targetSha256,
    manifest,
    manifestSha256,
    cases,
    sampleCache,
):
    backbone = groupPaths["backbone"]
    backend = groupPaths["backend"]
    spec, sampleByName = sampleCache[backend]
    needsTraining = _decoderNeedsTraining(
        groupPaths, groupCalibration["sources"], args.steps
    )
    if needsTraining:
        _, decoderTrain, _ = trainLatentCalibratedGmRcGraphDecoder(
            referenceEmbeddings,
            groupCalibration["embeddings"],
            groupCalibration["target_indices"],
            records,
            groupPaths["decoder"],
            targetSha256,
            groupCalibration["sources"],
            decoderGroup=groupPaths["group"],
            decoderShot=groupPaths["shot"],
            maxOptimizerSteps=args.steps,
            interpolationAlphas=DECODER_INTERPOLATION_ALPHAS,
            seed=args.seed,
            device=DECODER_DEVICE,
        )
    else:
        print(f"SKIP TRAIN {groupPaths['group']}", flush=True)
        decoderTrain = None
    decoderSha256 = hashFile(groupPaths["decoder"])
    decoder, checkpoint = loadLatentCalibratedGmRcGraphDecoder(
        groupPaths["decoder"],
        records,
        targetSha256,
        groupCalibration["sources"],
        interpolationAlphas=DECODER_INTERPOLATION_ALPHAS,
        calibrationSampleCount=len(groupCalibration["embeddings"]),
        decoderGroup=groupPaths["group"],
        decoderShot=groupPaths["shot"],
        device=DECODER_DEVICE,
    )
    if decoderTrain is None:
        decoderTrain = {
            "training_policy": checkpoint["training_policy"],
            "calibration": checkpoint["calibration"],
            "completed_optimizer_steps": checkpoint[
                "completed_optimizer_steps"
            ],
            "completed_epochs": checkpoint["completed_epochs"],
            **checkpoint["training_metrics"],
        }
    targetMetrics = evaluateTargetSideGmRcGraphDecoder(
        decoder,
        referenceEmbeddings,
        records,
        checkpoint["max_decode_steps"],
        device=DECODER_DEVICE,
    )
    _, testCases = casesForEncoderFewShot(
        manifest, cases, shot=DECODER_FEW_SHOT
    )
    splitPolicy = encoderFewShotSplitPolicy(
        manifest, manifestSha256, DECODER_FEW_SHOT
    )
    samples = _selectSamples(sampleByName, testCases)
    circuitSha256 = hashFile(groupPaths["circuit"])
    if groupPaths["result"].exists():
        existing = json.loads(
            groupPaths["result"].read_text(encoding="utf-8")
        )
        if (
            existing.get("decoder_training_policy")
            == LATENT_CALIBRATED_DECODER_POLICY
            and existing.get("test_samples") == len(samples)
            and existing.get("artifact_sha256")
            == {
                "circuit_encoder": circuitSha256,
                "shared_task_decoder": decoderSha256,
            }
        ):
            print(f"SKIP TASK DECODER {groupPaths['group']}", flush=True)
            metrics = existing["shared_task_decoder"]
            return {
                "group": groupPaths["group"],
                "backbone": backbone,
                "backend": backend,
                "shot": groupPaths["shot"],
                "training": decoderTrain,
                "target_side_test": targetMetrics,
                "metrics": metrics,
                "result": existing,
            }
    circuitModel, _ = _loadCircuitModel(
        args.seed, backbone, backend, spec, splitPolicy, targetSha256
    )
    embeddings = encodeCircuitSamples(
        circuitModel,
        spec,
        samples,
        batchSize=64,
        device=EMBEDDING_DEVICE,
    )
    targetByKey = {
        record.semanticKey: record.graphData for record in records
    }
    metrics = evaluateGmRcGraphDecoder(
        decoder,
        embeddings,
        samples,
        targetByKey,
        checkpoint["max_decode_steps"],
        device=DECODER_DEVICE,
    )
    result = {
        "seed": args.seed,
        "shot": groupPaths["shot"],
        "backbone": backbone,
        "backend": backend,
        "decoder_group": groupPaths["group"],
        "test_samples": len(samples),
        "calibration_shot": CALIBRATION_SHOT,
        "decoder_training_policy": LATENT_CALIBRATED_DECODER_POLICY,
        "artifact_sha256": {
            "circuit_encoder": circuitSha256,
            "shared_task_decoder": decoderSha256,
        },
        "shared_task_decoder": metrics,
    }
    writeExperimentJson(groupPaths["result"], result)
    print(
        f"TASK DECODER {groupPaths['group']} "
        f"exact={metrics['graph_exact']:.4f}",
        flush=True,
    )
    del circuitModel, embeddings, decoder
    _releaseMemory()
    return {
        "group": groupPaths["group"],
        "backbone": backbone,
        "backend": backend,
        "shot": groupPaths["shot"],
        "training": decoderTrain,
        "target_side_test": targetMetrics,
        "metrics": metrics,
        "result": result,
    }


def _resultRow(groupRow):
    taskMetrics = {
        key: float(groupRow["metrics"][key]) for key in DECODER_METRICS
    }
    return {
        "group": groupRow["group"],
        "shot": groupRow["shot"],
        "backbone": groupRow["backbone"],
        "backend": groupRow["backend"],
        "task": taskMetrics,
    }


def _meanMetrics(rows, field):
    selected = [row for row in rows if row[field] is not None]
    if not selected:
        return None
    return {
        key: sum(row[field][key] for row in selected) / len(selected)
        for key in DECODER_METRICS
    }


def _meanNestedMetrics(entries):
    if not entries:
        return None
    keys = sorted(entries[0])
    return {
        key: sum(entry[key] for entry in entries) / len(entries)
        for key in keys
    }


def _aggregateRows(rows):
    def _aggregate(group):
        return {
            "configuration_count": len(group),
            "task": _meanMetrics(group, "task"),
        }

    return {
        "overall": _aggregate(rows),
        "by_backend": {
            backend: _aggregate([
                row for row in rows if row["backend"] == backend
            ])
            for backend in [item[1] for item in DECODER_GROUPS]
        },
        "by_backbone": {
            backbone: _aggregate([
                row for row in rows if row["backbone"] == backbone
            ])
            for backbone in ENCODER_FEW_SHOT_BACKBONES
        },
    }


def main():
    args = _parseArgs()
    if args.steps < 1:
        raise ValueError("shared task decoder steps must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("shared task decoder training requires CUDA")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)

    cases = discoverAlignmentCases(DATASET_DIR)
    paths = fewShotExperimentPaths(args.seed)
    manifest, manifestSha256 = _loadManifest(cases, args.seed, paths)
    records, targetIndex = buildTargetBank(cases)
    targetModel, _ = loadTargetGraphEncoder(
        paths["target"], device=EMBEDDING_DEVICE
    )
    targetSha256 = hashFile(paths["target"])
    referenceEmbeddings = encodeTargetRecords(
        targetModel, records, device=EMBEDDING_DEVICE
    )
    del targetModel
    _releaseMemory()
    sampleCache = _buildSampleCache(cases, targetIndex)
    groupCalibration = _collectGroupCalibration(
        args.seed,
        cases,
        manifest,
        manifestSha256,
        sampleCache,
        targetSha256,
    )

    groupRows = []
    for backbone, backend in DECODER_GROUPS:
        groupPaths = decoderConfigurationPaths(
            backbone, backend, args.seed
        )
        groupRows.append(
            _runGroup(
                args,
                groupPaths,
                groupCalibration[(backbone, backend)],
                records,
                referenceEmbeddings,
                targetSha256,
                manifest,
                manifestSha256,
                cases,
                sampleCache,
            )
        )

    calibrationSources = {}
    for values in groupCalibration.values():
        calibrationSources.update(values["sources"])
    writeExperimentJson(paths["task_decoder_report"], {
        "seed": args.seed,
        "decoder_training_policy": LATENT_CALIBRATED_DECODER_POLICY,
        "decoder_shot": DECODER_FEW_SHOT,
        "group_count": len(groupRows),
        "calibration_shot": CALIBRATION_SHOT,
        "optimizer_steps": args.steps,
        "interpolation_alphas": list(DECODER_INTERPOLATION_ALPHAS),
        "calibration_sources": calibrationSources,
        "artifact_sha256": {
            "target_encoder": targetSha256,
            "shared_task_decoders": {
                row["group"]: hashFile(
                    decoderConfigurationPaths(
                        row["backbone"], row["backend"], args.seed
                    )["decoder"]
                )
                for row in groupRows
            },
        },
        "groups": {
            row["group"]: {
                "backbone": row["backbone"],
                "backend": row["backend"],
                "shot": row["shot"],
                "training": row["training"],
                "target_side_test": row["target_side_test"],
            }
            for row in groupRows
        },
        "training": {
            "completed_optimizer_steps": args.steps,
            "group_count": len(groupRows),
        },
        "target_side_test": _meanNestedMetrics([
            {
                key: value
                for key, value in row["target_side_test"].items()
                if isinstance(value, (int, float))
            }
            for row in groupRows
        ]),
    })

    rows = [_resultRow(row) for row in groupRows]
    writeExperimentJson(paths["task_decoder_summary"], {
        "seed": args.seed,
        "decoder_training_policy": LATENT_CALIBRATED_DECODER_POLICY,
        "decoder_shot": DECODER_FEW_SHOT,
        "configurations": len(rows),
        "calibration_shot": CALIBRATION_SHOT,
        "interpolation_alphas": list(DECODER_INTERPOLATION_ALPHAS),
        "groups": [row["group"] for row in rows],
        "aggregate": _aggregateRows(rows),
        "rows": rows,
    })
    print("TASK DECODER SUMMARY", paths["task_decoder_summary"], flush=True)


if __name__ == "__main__":
    main()
