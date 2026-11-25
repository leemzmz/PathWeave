"""无需 CktPath，从发布图张量重载归档模型并复算指标。"""

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
from Experiment.GraphAlignmentData import discoverAlignmentCases, buildTargetBank, buildCircuitAlignmentSamples
from Experiment.GraphAlignment import loadTargetGraphEncoder, loadCircuitAlignmentEncoder, evaluateCircuitAlignment
from Experiment.EncoderFewShotExperiment import (
    validateEncoderFewShotManifest, encoderFewShotSplitPolicy, compareReloadMetrics, optimizerStepsForShot,
)
from Experiment.encoder_backends import getGraphBackendSpec
from GraphSAGE.PreprocessedData import BACKENDS, hashFile, loadPreprocessedGraphs


# 默认要求指标一致；显式诊断模式保留全部实测值并标记比较失败。
def _compareEvaluation(reference, actual, allowDifferences):
    try:
        return dict(reference_match=True, **compareReloadMetrics(reference, actual))
    except ValueError as error:
        if not allowDifferences:
            raise
        differences = []
        def _collectDifferences(first, second, path):
            if isinstance(first, dict) and isinstance(second, dict):
                for key in sorted(set(first) | set(second)):
                    _collectDifferences(first.get(key), second.get(key), f"{path}.{key}")
            elif first != second:
                item = dict(metric=path, reference=first, actual=second)
                if isinstance(first, (float,int)) and isinstance(second, (float,int)):
                    item["absolute_difference"] = abs(first-second)
                differences.append(item)
        for group in ("encoder_test", "decoder_test"):
            if group in reference or group in actual:
                _collectDifferences(reference.get(group), actual.get(group), group)
        return dict(reference_match=False, comparison_error=str(error), differences=differences)


# 10-shot decoder 消费重载 encoder 的 embedding；默认保持原 CUDA 评估设备。
def _evaluateDecoder(model, spec, samples, records, directory, referencePath, targetHash, encoderHash, backbone, backend, decoderDevice, allowDifferences):
    from Experiment.GmRcDecoder import loadLatentCalibratedGmRcGraphDecoder, encodeCircuitSamples, evaluateGmRcGraphDecoder

    reference = json.loads(referencePath.read_text())
    checkpointPath = directory / "shared_task_decoder.pt"
    expected = {"circuit_encoder": encoderHash, "shared_task_decoder": hashFile(checkpointPath)}
    if reference["artifact_sha256"] != expected:
        raise ValueError("decoder result/checkpoint hash mismatch")
    decoder, checkpoint = loadLatentCalibratedGmRcGraphDecoder(checkpointPath, records,
        targetHash, {f"shot_10/{backbone}/{backend}": encoderHash},
        calibrationSampleCount=84, decoderGroup=f"{backbone}__{backend}", decoderShot=10,
        device=decoderDevice)
    if checkpoint["completed_optimizer_steps"] != 3000:
        raise ValueError("decoder optimizer budget mismatch")
    embeddings = encodeCircuitSamples(model, spec, samples, batchSize=64, device=torch.device("cpu"))
    metrics = evaluateGmRcGraphDecoder(decoder, embeddings, samples,
        {r.semanticKey: r.graphData for r in records}, checkpoint["max_decode_steps"], device=decoderDevice)
    audit = _compareEvaluation({"decoder_test": reference["shared_task_decoder"]}, {"decoder_test": metrics}, allowDifferences)
    return dict(decoder_test=metrics, decoder_reload_audit=audit, decoder_sha256=expected["shared_task_decoder"])


# 严格按归档输入、split、权重及模型契约重载；不改写历史协议或结果。
def evaluateArchive(dataset, run, bundle, seeds, shots, backbones, backends, decoder=False, decoderDevice="cuda", allowDifferences=False):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    device = torch.device("cpu")
    decoderDevice = torch.device(decoderDevice)
    if decoder and decoderDevice.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("archived decoder used CUDA; use --decoder-device cpu --allow-differences for CPU diagnostics")
    cases = discoverAlignmentCases(dataset)
    records, targetIndex = buildTargetBank(cases)
    metadata, graphs = loadPreprocessedGraphs(bundle, cases)
    roots = [p for p in (run / "results/graph_alignment").iterdir() if p.is_dir()]
    if len(roots) != 1:
        raise ValueError("expected exactly one archived protocol")
    resultRoot = roots[0]
    modelRoot = run / "models/graph_alignment" / resultRoot.name
    cache = {}
    for backend in backends:
        spec = getGraphBackendSpec(backend)
        samples = buildCircuitAlignmentSamples(cases, targetIndex,
            lambda path, b=backend: graphs[b][Path(path).relative_to(dataset).as_posix()])
        cache[backend] = (spec, {s.name: s for s in samples})
    rows = []
    for seed in seeds:
        results = resultRoot / f"seed_{seed}"
        models = modelRoot / f"seed_{seed}"
        protocolHash = hashFile(results / "protocol.json")
        if metadata["protocol_sha256"].get(str(seed)) != protocolHash:
            raise ValueError("tensor package archive protocol mismatch")
        protocol = json.loads((results / "protocol.json").read_text())
        manifestHash = hashFile(results / "split_manifest.json")
        manifest = json.loads((results / "split_manifest.json").read_text())
        if protocol["manifest_sha256"] != manifestHash or protocol["protocol"] != resultRoot.name or protocol["seed"] != seed:
            raise ValueError("archive split/protocol mismatch")
        if metadata["core_sha256"] != protocol["core_sha256"] or metadata["target_sha256"] != protocol["target_sha256"]:
            raise ValueError("archive input mismatch")
        validateEncoderFewShotManifest(manifest, cases, protocol=resultRoot.name)
        names = sorted(name for entry in manifest["topologies"].values() for name in entry["test"])
        targetHash = hashFile(models / "target_encoder.pt")
        targetModel, targetCheckpoint = loadTargetGraphEncoder(models / "target_encoder.pt", device=device)
        if targetCheckpoint["training_seed"] != seed or targetCheckpoint["target_keys"] != [r.semanticKey for r in records]:
            raise ValueError("target encoder identity mismatch")
        for shot in shots:
            policy = encoderFewShotSplitPolicy(manifest, manifestHash, shot, protocol=resultRoot.name)
            for backbone in backbones:
                for backend in backends:
                    relative = Path(f"shot_{shot}") / backbone / backend
                    reference = json.loads((results / relative / "result.json").read_text())
                    encoderPath = models / relative / "circuit_encoder.pt"
                    expectedHashes = {"target_encoder": targetHash, "circuit_encoder": hashFile(encoderPath)}
                    if reference["artifact_sha256"] != expectedHashes or reference["protocol_sha256"] != protocolHash or reference["manifest_sha256"] != manifestHash:
                        raise ValueError("result/model provenance mismatch")
                    spec, byName = cache[backend]
                    model, checkpoint = loadCircuitAlignmentEncoder(encoderPath, backend, spec,
                        seed=seed, device=device, hardNegativeConfig=protocol["hard_negative_config"],
                        circuitBackbone=backbone, sampleSplitPolicy=policy, targetCheckpointSha256=targetHash)
                    if checkpoint["completed_optimizer_steps"] != optimizerStepsForShot(shot):
                        raise ValueError("optimizer budget mismatch")
                    metrics = evaluateCircuitAlignment(model, spec, [byName[n] for n in names], records,
                        targetModel, batchSize=64, maxPairs=None, seed=seed, device=device)
                    audit = _compareEvaluation(reference, {"encoder_test": metrics}, allowDifferences)
                    rows.append(dict(seed=seed, shot=shot, backbone=backbone, backend=backend,
                        artifact_sha256=expectedHashes, encoder_test=metrics, reload_audit=audit))
                    if decoder and shot == 10:
                        rows[-1].update(_evaluateDecoder(model, spec, [byName[n] for n in names], records,
                            models / relative, results / relative / "task_decoder_result.json",
                            targetHash, expectedHashes["circuit_encoder"], backbone, backend, decoderDevice, allowDifferences))
                    matched = rows[-1]["reload_audit"]["reference_match"] and rows[-1].get("decoder_reload_audit", {}).get("reference_match", True)
                    status = "PASS" if matched else "DIFF"
                    print(f"{status} seed={seed} shot={shot} {backbone} {backend}", flush=True)
    if "GraphSAGE.PathExtraction" in sys.modules or "extraction" in sys.modules:
        raise RuntimeError("unexpected CktPath import during tensor-only evaluation")
    return dict(protocol=resultRoot.name, graphs_sha256=metadata["graphs_sha256"],
                cktpath_imported=False, configurations=len(rows), decoder_device=str(decoderDevice) if decoder else None,
                reference_match=all(r["reload_audit"]["reference_match"] and r.get("decoder_reload_audit", {}).get("reference_match",True) for r in rows), results=rows)


# 输出单独复算报告；输出不得落入历史归档或输入数据。
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "_data_new")
    parser.add_argument("--run", type=Path, default=ROOT / "_run_new")
    parser.add_argument("--graphs", type=Path, default=ROOT / "preprocessed/data_new")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42,205,316,327,330,501])
    parser.add_argument("--shots", type=int, nargs="+", choices=[2,4,6,8,10], default=[2,4,6,8,10])
    parser.add_argument("--backbones", nargs="+", choices=["graphsage","gat","gin"], default=["graphsage","gat","gin"])
    parser.add_argument("--backends", nargs="+", choices=BACKENDS, default=list(BACKENDS))
    parser.add_argument("--decoder", action="store_true", help="Also recompute 10-shot decoder metrics")
    parser.add_argument("--decoder-device", choices=["cpu","cuda"], default="cuda")
    parser.add_argument("--allow-differences", action="store_true", help="Diagnostic report only: record metric mismatches instead of raising")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.decoder and 10 not in args.shots:
        raise ValueError("decoder evaluation requires --shots 10")
    output = args.output.resolve()
    protected = [ROOT/n for n in ("_data", "_data_new", "_run", "_run_new")]
    protected += [args.dataset.resolve(), args.run.resolve(), args.graphs.resolve()]
    if any(output.is_relative_to(p) for p in protected) or output.exists():
        raise ValueError("use a new output file outside data, models and published graphs")
    report = evaluateArchive(args.dataset.resolve(), args.run.resolve(), args.graphs.resolve(),
        args.seeds, args.shots, args.backbones, args.backends, args.decoder, args.decoder_device, args.allow_differences)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print("Evaluated configurations:", report["configurations"], "reference_match:", report["reference_match"], flush=True)


if __name__ == "__main__":
    main()
