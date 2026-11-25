import argparse
import csv
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from Experiment.EncoderFewShotExperiment import (
    ALL_ENCODER_FEW_SHOT_BACKENDS,
    ENCODER_FEW_SHOT_BACKBONES,
    ENCODER_FEW_SHOT_BACKENDS,
    ENCODER_FEW_SHOT_PROTOCOL,
    ENCODER_FEW_SHOT_SEED,
    ENCODER_FEW_SHOTS,
    FORMAL_EMBEDDING_DEVICE,
    TARGET_ENCODER_EPOCHS,
    buildEncoderFewShotManifest,
    buildExperimentProtocol,
    casesForEncoderFewShot,
    compareReloadMetrics,
    encoderFewShotSplitPolicy,
    fewShotConfigurationPaths,
    fewShotExperimentPaths,
    hashFile,
    jsonFileSha256,
    optimizerStepsForShot,
    validateEncoderFewShotManifest,
    writeExperimentJson,
)
from Experiment.GraphAlignment import (
    DEFAULT_HARD_NEGATIVE_CONFIG,
    evaluateCircuitAlignment,
    loadCircuitAlignmentEncoder,
    loadTargetGraphEncoder,
    trainCircuitAlignmentEncoder,
    trainTargetGraphEncoder,
)
from Experiment.GraphAlignmentData import (
    buildCircuitAlignmentSamples,
    buildTargetBank,
    discoverAlignmentCases,
)

from Experiment.encoder_backends import getGraphBackendSpec
from GraphSAGE.PathExtraction import CKTPATH_ROOT
from _project_paths import DATASET_DIR


def _parseArgs():
    parser = argparse.ArgumentParser(
        description="Run one seeded encoder-only few-shot experiment."
    )
    parser.add_argument(
        "--phase",
        choices=("all", "train", "evaluate", "summarize"),
        default="all",
    )
    parser.add_argument("--seed", type=int, default=ENCODER_FEW_SHOT_SEED)
    parser.add_argument(
        "--shots",
        nargs="+",
        type=int,
        choices=ENCODER_FEW_SHOTS,
        default=list(ENCODER_FEW_SHOTS),
    )
    parser.add_argument(
        "--backbones",
        nargs="+",
        choices=ENCODER_FEW_SHOT_BACKBONES,
        default=list(ENCODER_FEW_SHOT_BACKBONES),
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=ALL_ENCODER_FEW_SHOT_BACKENDS,
        default=list(ENCODER_FEW_SHOT_BACKENDS),
    )
    parser.add_argument("--refresh-protocol", action="store_true",
                        help="Refresh metadata only when no model or result artifacts exist.")
    return parser.parse_args()


def _releaseMemory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _sourcePaths():
    names = (
        "Experiment/EncoderFewShotExperiment.py",
        "Experiment/GraphAlignment.py",
        "Experiment/GraphAlignmentData.py",
        "Experiment/Metrics.py",
        "Experiment/Runtime.py",
        "Experiment/encoder_backends.py",
        "GraphSAGE/CircuitEncoder.py",
        "GraphSAGE/GmRcGraph.py",
        "GraphSAGE/PathExtraction.py",
        "GraphSAGE/SignalAttentionData.py",
        "GraphSAGE/SignalAttentionEncoder.py",
        "GraphSAGE/graph_data.py",
        "GraphSAGE/graphsage.py",
        "Run/train_encoder_few_shot.py",
        "_project_paths.py",
        "run_dual_pathweave_experiment.sh",
    )
    cktPathSources = sorted(
        path
        for directory in ("core", "extraction")
        for path in (CKTPATH_ROOT / directory).glob("*.py")
    )
    return [PROJECT_ROOT / name for name in names] + cktPathSources


# 仅允许修改尚未产生模型或结果的协议元数据。
def _hasExperimentArtifacts(paths):
    if any(path.is_file() for path in paths["model_root"].rglob("*")):
        return True
    metadata = {paths["manifest"], paths["protocol"]}
    return any(path.is_file() and path not in metadata
               for path in paths["result_root"].rglob("*"))


# 在任何写入前检查归档完整性；已有产物不得刷新或补造来源元数据。
def _prepareProtocol(cases, seed, refreshProtocol=False):
    paths = fewShotExperimentPaths(seed)
    hasArtifacts = _hasExperimentArtifacts(paths)
    if hasArtifacts and not all(paths[key].is_file() for key in ("manifest", "protocol")):
        raise ValueError("experiment artifacts exist without provenance metadata; use a new PATHWEAVE_RUN_DIR")
    expectedManifest = buildEncoderFewShotManifest(cases, seed=seed)
    if paths["manifest"].exists():
        manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
        if manifest != expectedManifest:
            raise ValueError("saved encoder few-shot manifest changed")
    else:
        writeExperimentJson(paths["manifest"], expectedManifest)
        manifest = expectedManifest
    validateEncoderFewShotManifest(manifest, cases)
    manifestSha256 = jsonFileSha256(paths["manifest"])
    expectedProtocol = buildExperimentProtocol(
        cases, manifestSha256, _sourcePaths(), seed=seed
    )
    if paths["protocol"].exists():
        protocol = json.loads(paths["protocol"].read_text(encoding="utf-8"))
        if protocol != expectedProtocol:
            if not refreshProtocol:
                raise ValueError("saved encoder few-shot protocol changed")
            if hasArtifacts:
                raise ValueError("cannot refresh protocol with existing experiment artifacts; use a new PATHWEAVE_RUN_DIR")
            writeExperimentJson(paths["protocol"], expectedProtocol)
            protocol = expectedProtocol
    else:
        writeExperimentJson(paths["protocol"], expectedProtocol)
        protocol = expectedProtocol
    return paths, manifest, manifestSha256, jsonFileSha256(paths["protocol"])


# 加载或训练当前 seed 的目标编码器，并校验目标图身份。
def _prepareTargetEncoder(paths, records, seed):
    if paths["target"].exists():
        targetModel, targetCheckpoint = loadTargetGraphEncoder(paths["target"])
        if targetCheckpoint.get("training_seed") != seed:
            raise ValueError("canonical target encoder seed mismatch")
        if targetCheckpoint.get("target_keys") != [
            record.semanticKey for record in records
        ]:
            raise ValueError("canonical target graph bank mismatch")
    else:
        targetModel, _ = trainTargetGraphEncoder(
            records,
            paths["target"],
            epochs=TARGET_ENCODER_EPOCHS,
            seed=seed,
        )
    return targetModel, hashFile(paths["target"])


def _buildSampleCache(cases, targetIndex, backends):
    cache = {}
    for backend in backends:
        print(f"BUILD GRAPH CACHE {backend}", flush=True)
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


def _trainConfigurations(
    cases,
    manifest,
    manifestSha256,
    protocolSha256,
    records,
    targetIndex,
    targetModel,
    targetSha256,
    seed,
    shots,
    backbones,
    backends,
):
    sampleCache = _buildSampleCache(cases, targetIndex, backends)
    completed = 0
    total = len(shots) * len(backbones) * len(backends)
    startedAll = time.perf_counter()
    for shot in shots:
        optimizerSteps = optimizerStepsForShot(shot)
        trainCases, testCases = casesForEncoderFewShot(
            manifest, cases, shot=shot
        )
        splitPolicy = encoderFewShotSplitPolicy(
            manifest, manifestSha256, shot
        )
        for backbone in backbones:
            for backend in backends:
                configPaths = fewShotConfigurationPaths(
                    shot, backbone, backend, seed=seed
                )
                if configPaths["result"].exists():
                    if not configPaths["circuit"].exists():
                        raise FileNotFoundError("result has no circuit checkpoint")
                    existing = json.loads(
                        configPaths["result"].read_text(encoding="utf-8")
                    )
                    if (
                        existing.get("protocol_sha256") != protocolSha256
                        or existing.get("manifest_sha256") != manifestSha256
                        or existing.get("training", {}).get(
                            "optimizer_steps"
                        ) != optimizerSteps
                    ):
                        raise ValueError(
                            "existing result does not match current protocol"
                        )
                    completed += 1
                    print(
                        f"SKIP COMPLETE {completed}/{total} shot={shot} "
                        f"{backbone} {backend}",
                        flush=True,
                    )
                    continue
                spec, sampleByName = sampleCache[backend]
                trainSamples = _selectSamples(sampleByName, trainCases)
                if configPaths["circuit"].exists():
                    circuitModel, checkpoint = loadCircuitAlignmentEncoder(
                        configPaths["circuit"],
                        backend,
                        spec,
                        seed=seed,
                        hardNegativeConfig=DEFAULT_HARD_NEGATIVE_CONFIG,
                        circuitBackbone=backbone,
                        sampleSplitPolicy=splitPolicy,
                        targetCheckpointSha256=targetSha256,
                    )
                    if checkpoint["completed_optimizer_steps"] != optimizerSteps:
                        raise ValueError("circuit optimizer budget mismatch")
                    trainingSeconds = None
                else:
                    print(
                        f"TRAIN shot={shot} {backbone} {backend}",
                        flush=True,
                    )
                    started = time.perf_counter()
                    circuitModel = trainCircuitAlignmentEncoder(
                        backend,
                        spec,
                        trainSamples,
                        records,
                        targetModel,
                        configPaths["circuit"],
                        batchSize=32,
                        learningRate=1e-3,
                        seed=seed,
                        hardNegativeConfig=DEFAULT_HARD_NEGATIVE_CONFIG,
                        circuitBackbone=backbone,
                        maxOptimizerSteps=optimizerSteps,
                        sampleSplitPolicy=splitPolicy,
                        targetCheckpointSha256=targetSha256,
                    )
                    trainingSeconds = time.perf_counter() - started
                testSamples = _selectSamples(sampleByName, testCases)
                circuitModel = circuitModel.to(FORMAL_EMBEDDING_DEVICE)
                encoderMetrics = evaluateCircuitAlignment(
                    circuitModel,
                    spec,
                    testSamples,
                    records,
                    targetModel,
                    batchSize=64,
                    maxPairs=None,
                    seed=seed,
                    device=FORMAL_EMBEDDING_DEVICE,
                )
                result = {
                    "protocol": ENCODER_FEW_SHOT_PROTOCOL,
                    "seed": seed,
                    "shot": shot,
                    "backbone": backbone,
                    "backend": backend,
                    "split": splitPolicy,
                    "train_samples": len(trainSamples),
                    "test_samples": len(testSamples),
                    "manifest_sha256": manifestSha256,
                    "protocol_sha256": protocolSha256,
                    "training": {
                        "optimizer_steps": optimizerSteps,
                        "batch_size": 32,
                        "learning_rate": 1e-3,
                        "seconds": trainingSeconds,
                    },
                    "artifact_sha256": {
                        "target_encoder": targetSha256,
                        "circuit_encoder": hashFile(
                            configPaths["circuit"]
                        ),
                    },
                    "encoder_test": encoderMetrics,
                }
                writeExperimentJson(configPaths["result"], result)
                completed += 1
                message = (
                    f"DONE {completed}/{total} shot={shot} {backbone} {backend} "
                    f"top1={encoderMetrics['retrieval_top1']:.4f}"
                )
                message += f" elapsed={time.perf_counter() - startedAll:.1f}s"
                print(message, flush=True)
                del circuitModel
                _releaseMemory()


# 独立重载 encoder，严格检查协议与模型摘要，不加载 decoder。
def _evaluateConfigurations(cases, manifest, manifestSha256, protocolSha256,
                            records, targetIndex, paths, seed, shots, backbones, backends):
    torch.use_deterministic_algorithms(True)
    device = torch.device(FORMAL_EMBEDDING_DEVICE)
    targetModel, targetCheckpoint = loadTargetGraphEncoder(paths["target"], device=device)
    if targetCheckpoint.get("training_seed") != seed or targetCheckpoint.get("target_keys") != [r.semanticKey for r in records]:
        raise ValueError("canonical target encoder contract mismatch")
    targetSha256 = hashFile(paths["target"])
    sampleCache = _buildSampleCache(cases, targetIndex, backends)
    for shot in shots:
        _, testCases = casesForEncoderFewShot(manifest, cases, shot=shot)
        splitPolicy = encoderFewShotSplitPolicy(manifest, manifestSha256, shot)
        for backbone in backbones:
            for backend in backends:
                configPaths = fewShotConfigurationPaths(shot, backbone, backend, seed=seed)
                reference = json.loads(configPaths["result"].read_text(encoding="utf-8"))
                expectedHashes = {"target_encoder": targetSha256,
                                  "circuit_encoder": hashFile(configPaths["circuit"])}
                if (reference["protocol_sha256"] != protocolSha256
                        or reference["manifest_sha256"] != manifestSha256
                        or reference["artifact_sha256"] != expectedHashes):
                    raise ValueError("result protocol or artifact checksum mismatch")
                spec, sampleByName = sampleCache[backend]
                testSamples = _selectSamples(sampleByName, testCases)
                circuitModel, checkpoint = loadCircuitAlignmentEncoder(
                    configPaths["circuit"], backend, spec, seed=seed,
                    hardNegativeConfig=DEFAULT_HARD_NEGATIVE_CONFIG,
                    circuitBackbone=backbone, sampleSplitPolicy=splitPolicy,
                    targetCheckpointSha256=targetSha256, device=device)
                if checkpoint["completed_optimizer_steps"] != optimizerStepsForShot(shot):
                    raise ValueError("reloaded circuit optimizer budget mismatch")
                reloaded = {"encoder_test": evaluateCircuitAlignment(
                    circuitModel, spec, testSamples, records, targetModel,
                    batchSize=64, maxPairs=None, seed=seed, device=device)}
                audit = compareReloadMetrics(reference, reloaded)
                writeExperimentJson(configPaths["reload"], {
                    "protocol": ENCODER_FEW_SHOT_PROTOCOL, "seed": seed,
                    "shot": shot, "backbone": backbone, "backend": backend,
                    "manifest_sha256": manifestSha256, "protocol_sha256": protocolSha256,
                    "artifact_sha256": expectedHashes,
                    "evaluation": {"embedding_device": FORMAL_EMBEDDING_DEVICE, "deterministic": True},
                    "reload_audit": audit, **reloaded})
                print(f"RELOAD shot={shot} {backbone} {backend} exact=True", flush=True)
                del circuitModel
                _releaseMemory()


# 提取 encoder 汇总指标；decoder 使用独立的 group 报告。
def _metricRow(result):
    encoder = result["encoder_test"]
    return {
        "shot": result["shot"], "backbone": result["backbone"], "backend": result["backend"],
        "test_samples": result["test_samples"], "optimizer_steps": result["training"]["optimizer_steps"],
        **{key: encoder[key] for key in ("retrieval_top1", "retrieval_top3", "retrieval_mrr",
                                         "stage_exact", "structure_spearman", "target_manifold_spearman")}}


# 校验所有重载记录后输出 encoder JSON/CSV/Markdown 摘要。
def _summarize(paths, protocolSha256, seed, shots, backbones, backends):
    rows = []
    for shot in shots:
        for backbone in backbones:
            for backend in backends:
                configPaths = fewShotConfigurationPaths(shot, backbone, backend, seed=seed)
                result = json.loads(configPaths["result"].read_text(encoding="utf-8"))
                reload = json.loads(configPaths["reload"].read_text(encoding="utf-8"))
                if result["protocol_sha256"] != protocolSha256 or reload["protocol_sha256"] != protocolSha256:
                    raise ValueError("summary protocol checksum mismatch")
                if (result["artifact_sha256"] != reload["artifact_sha256"]
                        or result["manifest_sha256"] != reload["manifest_sha256"]
                        or any(reload[k] != v for k, v in {"seed": seed, "shot": shot, "backbone": backbone, "backend": backend}.items())):
                    raise ValueError("summary reload identity mismatch")
                if not reload["reload_audit"]["discrete_metrics_exact"]:
                    raise ValueError("summary found failed reload")
                compareReloadMetrics(result, reload)
                if result["test_samples"] != 420 or result["encoder_test"]["pair_count"] != 87990:
                    raise ValueError("summary test population mismatch")
                if result["training"]["optimizer_steps"] != optimizerStepsForShot(shot):
                    raise ValueError("summary optimizer budget mismatch")
                rows.append(_metricRow(result))
    if not rows:
        raise ValueError("no configurations selected for summary")
    writeExperimentJson(paths["summary_json"], {
        "protocol": ENCODER_FEW_SHOT_PROTOCOL, "seed": seed,
        "protocol_sha256": protocolSha256, "configurations": len(rows), "rows": rows})
    with paths["summary_csv"].open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# Encoder evaluation", "", f"- Seed: {seed}", "",
             "| Shot | Backbone | Input | Top-1 | Top-3 | MRR |",
             "|---:|---|---|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['shot']} | {row['backbone']} | {row['backend']} | "
                     f"{row['retrieval_top1']:.4f} | {row['retrieval_top3']:.4f} | {row['retrieval_mrr']:.4f} |")
    paths["summary_md"].write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("SUMMARY", paths["summary_md"], flush=True)


# 执行单个 encoder 阶段；decoder 仅由独立入口处理。
def _runPhase(phase, seed, refreshProtocol=False, shots=ENCODER_FEW_SHOTS,
              backbones=ENCODER_FEW_SHOT_BACKBONES, backends=ENCODER_FEW_SHOT_BACKENDS):
    if phase not in ("train", "evaluate", "summarize"):
        raise ValueError("unsupported encoder phase")
    cases = discoverAlignmentCases(DATASET_DIR)
    paths, manifest, manifestSha256, protocolSha256 = _prepareProtocol(cases, seed, refreshProtocol)
    records, targetIndex = buildTargetBank(cases)
    if len(records) != 42:
        raise ValueError(f"expected 42 target graphs, got {len(records)}")
    if phase == "train":
        targetModel, targetSha256 = _prepareTargetEncoder(paths, records, seed)
        _trainConfigurations(cases, manifest, manifestSha256, protocolSha256,
                             records, targetIndex, targetModel, targetSha256,
                             seed, shots, backbones, backends)
    elif phase == "evaluate":
        _evaluateConfigurations(cases, manifest, manifestSha256, protocolSha256,
                                records, targetIndex, paths, seed, shots, backbones, backends)
    else:
        _summarize(paths, protocolSha256, seed, shots, backbones, backends)


# all 依次在独立进程中训练、重载评估和汇总。
def main():
    args = _parseArgs()
    if args.phase != "all":
        _runPhase(args.phase, args.seed, refreshProtocol=args.refresh_protocol,
                  shots=tuple(args.shots), backbones=tuple(args.backbones), backends=tuple(args.backends))
        return
    environment = {**os.environ, "PYTHONHASHSEED": str(args.seed), "PYTHONDONTWRITEBYTECODE": "1"}
    for phase in ("train", "evaluate", "summarize"):
        command = [sys.executable, "-u", str(Path(__file__).resolve()),
                   "--phase", phase, "--seed", str(args.seed)]
        if args.refresh_protocol:
            command.append("--refresh-protocol")
        command.extend(["--shots", *map(str, args.shots), "--backbones", *args.backbones, "--backends", *args.backends])
        subprocess.run(command, cwd=PROJECT_ROOT, env=environment, check=True)


if __name__ == "__main__":
    main()
