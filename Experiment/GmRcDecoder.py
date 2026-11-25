from collections import Counter, defaultdict
from pathlib import Path
import random
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from Decoder.GmRcGraphDecoder import (
    GM_RC_DECODER_CONTRACT,
    GM_RC_EOS_ID,
    GM_RC_PAD_ID,
    GM_RC_TOKEN_SCHEMA,
    GmRcGraphDecoder,
    gmRcGraphToTokenIds,
    gmRcTokenIdsToGraphData,
)
from GraphSAGE.GmRcGraph import (
    extractStageCounts,
    gmRcGraphGedSimilarity,
    gmRcSemanticKey,
)
from Experiment.Runtime import getDevice


DEFAULT_DECODER_EPOCHS = 200
DEFAULT_DECODER_BATCH_SIZE = 32
DEFAULT_DECODER_LEARNING_RATE = 1e-3
DEFAULT_DECODER_HIDDEN_DIM = 256
DEFAULT_DECODER_TOKEN_DIM = 64
DEFAULT_DECODER_LAYERS = 2
DEFAULT_DECODER_DROPOUT = 0.1
DEFAULT_DECODER_MAX_STEP_MARGIN = 32
LATENT_CALIBRATED_DECODER_CONTRACT = (
    "shared_functional_decoder_with_circuit_latent_calibration"
)
# 每个 (backbone, 表示) group 独立训练"同结构 + 同配方"的 decoder；只在 10-shot 上运行。
LATENT_CALIBRATED_DECODER_POLICY = (
    "fixed_steps_balanced_canonical_and_common_2shot_interpolation"
    "_per_group_10shot"
)
DECODER_GROUP_FIELD = "decoder_group"
DECODER_SHOT_FIELD = "decoder_shot"
DEFAULT_TASK_DECODER_STEPS = 400
DEFAULT_EVALUATION_BATCH_SIZE = 128
DEFAULT_CALIBRATION_ALPHAS = (0.25, 0.5, 0.75, 1.0)
DEFAULT_CANONICAL_LOSS_WEIGHT = 1.0
DEFAULT_CALIBRATION_LOSS_WEIGHT = 1.0


class GmRcDecoderDataset(Dataset):

    def __init__(self, embeddings, graphData):
        if len(embeddings) != len(graphData):
            raise ValueError("decoder embeddings and targets must align")
        self.m_embeddings = torch.as_tensor(
            embeddings, dtype=torch.float32
        ).cpu()
        self.m_graphData = list(graphData)
        self.m_tokens = [
            gmRcGraphToTokenIds(graph) for graph in self.m_graphData
        ]

    def __len__(self):
        return len(self.m_graphData)

    def __getitem__(self, index):
        return (
            self.m_embeddings[index],
            torch.tensor(self.m_tokens[index], dtype=torch.long),
        )


def _collateDecoder(items):
    embeddings = torch.stack([item[0] for item in items])
    maxLength = max(item[1].shape[0] for item in items)
    tokens = torch.full(
        (len(items), maxLength), GM_RC_PAD_ID, dtype=torch.long
    )
    for index, (_, sequence) in enumerate(items):
        tokens[index, :sequence.shape[0]] = sequence
    return embeddings, tokens


def _setSeed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# 使用 frozen circuit encoder 一次性生成 decoder 输入。
def encodeCircuitSamples(
    circuitModel, spec, samples, batchSize=64, device=None
):
    device = device or getDevice()
    circuitModel.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(samples), batchSize):
            batchSamples = samples[start:start + batchSize]
            graphBatch = spec.batch_graphs([
                sample.graph for sample in batchSamples
            ])
            outputs.append(
                spec.encode_batch(circuitModel, graphBatch, device).cpu()
            )
    return torch.cat(outputs, dim=0)


def _tokenWeights(graphData, device):
    counts = Counter(
        token
        for graph in graphData
        for token in gmRcGraphToTokenIds(graph)[1:]
    )
    total = sum(counts.values())
    weights = torch.ones(len(GM_RC_TOKEN_SCHEMA), dtype=torch.float32)
    for token, count in counts.items():
        weights[token] = (total / max(1, count)) ** 0.5
    active = torch.tensor(
        [token in counts for token in range(len(GM_RC_TOKEN_SCHEMA))]
    )
    weights[active] /= weights[active].mean()
    weights[GM_RC_PAD_ID] = 0.0
    return weights.to(device)


def _decoderLoss(model, embeddings, tokens, tokenWeights):
    logits = model(embeddings, tokens[:, :-1])
    labels = tokens[:, 1:]
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        labels.reshape(-1),
        weight=tokenWeights,
        ignore_index=GM_RC_PAD_ID,
    )


def _evaluateTeacherForced(model, loader, tokenWeights, device):
    model.eval()
    lossTotal = 0.0
    tokenCorrect = 0
    tokenTotal = 0
    with torch.no_grad():
        for embeddings, tokens in loader:
            embeddings = embeddings.to(device)
            tokens = tokens.to(device)
            logits = model(embeddings, tokens[:, :-1])
            labels = tokens[:, 1:]
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                labels.reshape(-1),
                weight=tokenWeights,
                ignore_index=GM_RC_PAD_ID,
            )
            mask = labels != GM_RC_PAD_ID
            tokenCorrect += (
                logits.argmax(dim=-1)[mask] == labels[mask]
            ).sum().item()
            tokenTotal += mask.sum().item()
            lossTotal += loss.item() * embeddings.shape[0]
    return {
        "loss": lossTotal / len(loader.dataset),
        "token_accuracy": tokenCorrect / max(1, tokenTotal),
    }


# 分别计算 canonical 与 calibration loss，避免 anchor 被样本量淹没。
def _fitBalancedGmRcGraphDecoder(
    canonicalEmbeddings,
    canonicalGraphData,
    calibrationEmbeddings,
    calibrationGraphData,
    batchSize=DEFAULT_DECODER_BATCH_SIZE,
    learningRate=DEFAULT_DECODER_LEARNING_RATE,
    canonicalLossWeight=DEFAULT_CANONICAL_LOSS_WEIGHT,
    calibrationLossWeight=DEFAULT_CALIBRATION_LOSS_WEIGHT,
    seed=42,
    device=None,
    maxOptimizerSteps=DEFAULT_TASK_DECODER_STEPS,
):
    if maxOptimizerSteps < 1:
        raise ValueError("decoder optimizer steps must be positive")
    if canonicalLossWeight <= 0.0 or calibrationLossWeight <= 0.0:
        raise ValueError("balanced decoder loss weights must be positive")
    device = device or getDevice()
    _setSeed(seed)
    canonicalDataset = GmRcDecoderDataset(
        canonicalEmbeddings, canonicalGraphData
    )
    calibrationDataset = GmRcDecoderDataset(
        calibrationEmbeddings, calibrationGraphData
    )
    canonicalLoader = DataLoader(
        canonicalDataset,
        batch_size=batchSize,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        collate_fn=_collateDecoder,
    )
    calibrationLoader = DataLoader(
        calibrationDataset,
        batch_size=batchSize,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed + 1),
        collate_fn=_collateDecoder,
    )
    embeddingDim = int(canonicalEmbeddings.shape[1])
    model = GmRcGraphDecoder(
        embeddingDim=embeddingDim,
        tokenDim=DEFAULT_DECODER_TOKEN_DIM,
        hiddenDim=DEFAULT_DECODER_HIDDEN_DIM,
        numLayers=DEFAULT_DECODER_LAYERS,
        dropout=DEFAULT_DECODER_DROPOUT,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learningRate, weight_decay=1e-5
    )
    tokenWeights = _tokenWeights(canonicalGraphData, device)
    canonicalIterator = iter(canonicalLoader)
    calibrationIterator = iter(calibrationLoader)
    calibrationEpochs = 0
    reportEvery = max(1, maxOptimizerSteps // 10)
    for completedOptimizerSteps in range(1, maxOptimizerSteps + 1):
        try:
            canonicalBatch = next(canonicalIterator)
        except StopIteration:
            canonicalIterator = iter(canonicalLoader)
            canonicalBatch = next(canonicalIterator)
        try:
            calibrationBatch = next(calibrationIterator)
        except StopIteration:
            calibrationEpochs += 1
            calibrationIterator = iter(calibrationLoader)
            calibrationBatch = next(calibrationIterator)

        canonicalBatchEmbeddings, canonicalTokens = canonicalBatch
        calibrationBatchEmbeddings, calibrationTokens = calibrationBatch
        canonicalBatchEmbeddings = canonicalBatchEmbeddings.to(device)
        canonicalTokens = canonicalTokens.to(device)
        calibrationBatchEmbeddings = calibrationBatchEmbeddings.to(device)
        calibrationTokens = calibrationTokens.to(device)
        model.train()
        optimizer.zero_grad()
        canonicalLoss = _decoderLoss(
            model,
            canonicalBatchEmbeddings,
            canonicalTokens,
            tokenWeights,
        )
        calibrationLoss = _decoderLoss(
            model,
            calibrationBatchEmbeddings,
            calibrationTokens,
            tokenWeights,
        )
        loss = (
            canonicalLossWeight * canonicalLoss
            + calibrationLossWeight * calibrationLoss
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        if (
            completedOptimizerSteps == 1
            or completedOptimizerSteps % reportEvery == 0
            or completedOptimizerSteps == maxOptimizerSteps
        ):
            print(
                "balanced decoder step "
                f"{completedOptimizerSteps:04d}/{maxOptimizerSteps} "
                f"canonical_loss={canonicalLoss.item():.6f} "
                f"calibration_loss={calibrationLoss.item():.6f}"
            )

    canonicalMetrics = _evaluateTeacherForced(
        model, canonicalLoader, tokenWeights, device
    )
    calibrationMetrics = _evaluateTeacherForced(
        model, calibrationLoader, tokenWeights, device
    )
    maxTargetLength = max(
        len(gmRcGraphToTokenIds(graph))
        for graph in [*canonicalGraphData, *calibrationGraphData]
    )
    # 不再使用固定地板；按实际 token 长度上限 + 少量余量决定解码步数。
    maxDecodeSteps = maxTargetLength + DEFAULT_DECODER_MAX_STEP_MARGIN
    model.eval()
    return model, {
        "completed_optimizer_steps": maxOptimizerSteps,
        "completed_epochs": calibrationEpochs,
        "train_samples": len(canonicalDataset) + len(calibrationDataset),
        "canonical_samples": len(canonicalDataset),
        "calibration_train_samples": len(calibrationDataset),
        "validation_samples": 0,
        "canonical_loss_weight": float(canonicalLossWeight),
        "calibration_loss_weight": float(calibrationLossWeight),
        "loss": (
            canonicalLossWeight * canonicalMetrics["loss"]
            + calibrationLossWeight * calibrationMetrics["loss"]
        ),
        "token_accuracy": (
            canonicalMetrics["token_accuracy"]
            + calibrationMetrics["token_accuracy"]
        ) / 2.0,
        "canonical_loss": canonicalMetrics["loss"],
        "canonical_token_accuracy": canonicalMetrics["token_accuracy"],
        "calibration_loss": calibrationMetrics["loss"],
        "calibration_token_accuracy": calibrationMetrics["token_accuracy"],
    }, maxDecodeSteps


# 用 reference 到真实 circuit embedding 的插值覆盖公共 latent 区域。
def buildLatentCalibratedDecoderData(
    referenceEmbeddings,
    calibrationEmbeddings,
    calibrationTargetIndices,
    targetRecords,
    interpolationAlphas=DEFAULT_CALIBRATION_ALPHAS,
):
    referenceEmbeddings = torch.as_tensor(
        referenceEmbeddings, dtype=torch.float32
    ).cpu()
    calibrationEmbeddings = torch.as_tensor(
        calibrationEmbeddings, dtype=torch.float32
    ).cpu()
    targetIndices = torch.as_tensor(
        calibrationTargetIndices, dtype=torch.long
    ).cpu()
    if referenceEmbeddings.ndim != 2:
        raise ValueError("reference decoder embeddings must be a matrix")
    if calibrationEmbeddings.ndim != 2:
        raise ValueError("calibration decoder embeddings must be a matrix")
    if referenceEmbeddings.shape[1] != calibrationEmbeddings.shape[1]:
        raise ValueError("decoder calibration embedding dimensions differ")
    if len(calibrationEmbeddings) != len(targetIndices):
        raise ValueError("decoder calibration targets do not align")
    if len(referenceEmbeddings) != len(targetRecords):
        raise ValueError("decoder reference targets do not align")
    if targetIndices.numel() and (
        int(targetIndices.min()) < 0
        or int(targetIndices.max()) >= len(targetRecords)
    ):
        raise ValueError("decoder calibration target index is invalid")
    alphas = tuple(float(alpha) for alpha in interpolationAlphas)
    if not alphas or any(alpha <= 0.0 or alpha > 1.0 for alpha in alphas):
        raise ValueError("decoder interpolation alpha must be in (0, 1]")

    embeddings = [referenceEmbeddings]
    graphData = [record.graphData for record in targetRecords]
    pairedReference = referenceEmbeddings[targetIndices]
    pairedGraphs = [
        targetRecords[int(index)].graphData for index in targetIndices
    ]
    for alpha in alphas:
        interpolated = F.normalize(
            (1.0 - alpha) * pairedReference
            + alpha * calibrationEmbeddings,
            p=2,
            dim=1,
        )
        if not torch.isfinite(interpolated).all():
            raise ValueError("decoder calibration produced nonfinite embedding")
        embeddings.append(interpolated)
        graphData.extend(pairedGraphs)
    return torch.cat(embeddings, dim=0), graphData, {
        "reference_samples": len(referenceEmbeddings),
        "calibration_samples": len(calibrationEmbeddings),
        "interpolation_alphas": list(alphas),
        "expanded_train_samples": len(graphData),
    }


# 在公共 2-shot circuit latent 与 reference latent 上训练共享 decoder。
def trainLatentCalibratedGmRcGraphDecoder(
    referenceEmbeddings,
    calibrationEmbeddings,
    calibrationTargetIndices,
    targetRecords,
    checkpointPath,
    targetCheckpointSha256,
    calibrationSources,
    decoderGroup=None,
    decoderShot=None,
    maxOptimizerSteps=DEFAULT_TASK_DECODER_STEPS,
    batchSize=DEFAULT_DECODER_BATCH_SIZE,
    learningRate=DEFAULT_DECODER_LEARNING_RATE,
    interpolationAlphas=DEFAULT_CALIBRATION_ALPHAS,
    seed=42,
    device=None,
):
    if not targetCheckpointSha256:
        raise ValueError("calibrated decoder requires target encoder checksum")
    if not calibrationSources:
        raise ValueError("calibrated decoder requires circuit sources")
    embeddings, graphData, calibration = buildLatentCalibratedDecoderData(
        referenceEmbeddings,
        calibrationEmbeddings,
        calibrationTargetIndices,
        targetRecords,
        interpolationAlphas=interpolationAlphas,
    )
    canonicalCount = len(targetRecords)
    model, trainingMetrics, maxDecodeSteps = (
        _fitBalancedGmRcGraphDecoder(
            embeddings[:canonicalCount],
            graphData[:canonicalCount],
            embeddings[canonicalCount:],
            graphData[canonicalCount:],
            batchSize=batchSize,
            learningRate=learningRate,
            canonicalLossWeight=DEFAULT_CANONICAL_LOSS_WEIGHT,
            calibrationLossWeight=DEFAULT_CALIBRATION_LOSS_WEIGHT,
            seed=seed,
            device=device,
            maxOptimizerSteps=maxOptimizerSteps,
        )
    )
    checkpointPath = Path(checkpointPath)
    checkpointPath.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "model_contract": GM_RC_DECODER_CONTRACT,
        "shared_decoder_contract": LATENT_CALIBRATED_DECODER_CONTRACT,
        "token_schema": GM_RC_TOKEN_SCHEMA,
        "target_checkpoint_sha256": targetCheckpointSha256,
        "target_keys": [record.semanticKey for record in targetRecords],
        "embedding_dim": int(embeddings.shape[1]),
        "token_dim": DEFAULT_DECODER_TOKEN_DIM,
        "hidden_dim": DEFAULT_DECODER_HIDDEN_DIM,
        "num_layers": DEFAULT_DECODER_LAYERS,
        "dropout": DEFAULT_DECODER_DROPOUT,
        "max_decode_steps": maxDecodeSteps,
        "training_seed": int(seed),
        "training_policy": LATENT_CALIBRATED_DECODER_POLICY,
        DECODER_GROUP_FIELD: decoderGroup,
        DECODER_SHOT_FIELD: (
            None if decoderShot is None else int(decoderShot)
        ),
        "calibration_sources": dict(sorted(calibrationSources.items())),
        "calibration": calibration,
        **trainingMetrics,
        "training_metrics": dict(trainingMetrics),
    }
    temporaryPath = checkpointPath.with_suffix(
        checkpointPath.suffix + ".tmp"
    )
    torch.save(payload, temporaryPath)
    temporaryPath.replace(checkpointPath)
    print("latent-calibrated shared decoder saved:", checkpointPath)
    print("latent-calibrated decoder metrics:", trainingMetrics)
    return model, {
        "training_policy": LATENT_CALIBRATED_DECODER_POLICY,
        "calibration": calibration,
        **trainingMetrics,
    }, maxDecodeSteps


# 严格加载公共 latent-calibrated decoder。
def loadLatentCalibratedGmRcGraphDecoder(
    checkpointPath,
    targetRecords,
    targetCheckpointSha256,
    calibrationSources,
    interpolationAlphas=DEFAULT_CALIBRATION_ALPHAS,
    calibrationSampleCount=None,
    decoderGroup=None,
    decoderShot=None,
    device=None,
):
    device = device or getDevice()
    checkpoint = torch.load(
        checkpointPath, map_location=device, weights_only=False
    )
    expected = {
        "model_contract": GM_RC_DECODER_CONTRACT,
        "shared_decoder_contract": LATENT_CALIBRATED_DECODER_CONTRACT,
        "token_schema": GM_RC_TOKEN_SCHEMA,
        "target_checkpoint_sha256": targetCheckpointSha256,
        "target_keys": [record.semanticKey for record in targetRecords],
        "training_policy": LATENT_CALIBRATED_DECODER_POLICY,
        "calibration_sources": dict(sorted(calibrationSources.items())),
        "canonical_loss_weight": DEFAULT_CANONICAL_LOSS_WEIGHT,
        "calibration_loss_weight": DEFAULT_CALIBRATION_LOSS_WEIGHT,
    }
    for key, value in expected.items():
        if checkpoint.get(key) != value:
            raise ValueError(
                f"latent-calibrated decoder checkpoint {key} mismatch"
            )
    if (
        decoderGroup is not None
        and checkpoint.get(DECODER_GROUP_FIELD) != decoderGroup
    ):
        raise ValueError(
            "latent-calibrated decoder checkpoint group mismatch"
        )
    if (
        decoderShot is not None
        and checkpoint.get(DECODER_SHOT_FIELD) != int(decoderShot)
    ):
        raise ValueError(
            "latent-calibrated decoder checkpoint shot mismatch"
        )
    calibration = checkpoint.get("calibration", {})
    expectedAlphas = [float(alpha) for alpha in interpolationAlphas]
    if calibration.get("interpolation_alphas") != expectedAlphas:
        raise ValueError(
            "latent-calibrated decoder interpolation alpha mismatch"
        )
    if (
        calibrationSampleCount is not None
        and calibration.get("calibration_samples")
        != int(calibrationSampleCount)
    ):
        raise ValueError(
            "latent-calibrated decoder calibration sample count mismatch"
        )
    model = GmRcGraphDecoder(
        embeddingDim=checkpoint["embedding_dim"],
        tokenDim=checkpoint["token_dim"],
        hiddenDim=checkpoint["hidden_dim"],
        numLayers=checkpoint["num_layers"],
        dropout=checkpoint["dropout"],
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint


# 在 canonical target embedding 上诊断当前 group decoder 的重构性能。
def evaluateTargetSideGmRcGraphDecoder(
    model,
    embeddings,
    targetRecords,
    maxDecodeSteps,
    batchSize=64,
    device=None,
):
    samples = [
        SimpleNamespace(
            targetKey=record.semanticKey,
            topology=f"target_{index:02d}",
        )
        for index, record in enumerate(targetRecords)
    ]
    return evaluateGmRcGraphDecoder(
        model,
        embeddings,
        samples,
        {
            record.semanticKey: record.graphData
            for record in targetRecords
        },
        maxDecodeSteps,
        batchSize=batchSize,
        device=device,
    )


def _trimGeneratedTokens(tokens):
    values = [int(token) for token in tokens]
    if GM_RC_EOS_ID not in values:
        raise ValueError("decoded sequence has no EOS")
    return values[:values.index(GM_RC_EOS_ID) + 1]


def _primitiveCounts(graphData):
    return Counter(device["kind"] for device in graphData["devices"])


# 在 test circuit embedding 上评估开放式 graph generation。
def evaluateGmRcGraphDecoder(
    model,
    embeddings,
    samples,
    targetByKey,
    maxDecodeSteps,
    batchSize=DEFAULT_EVALUATION_BATCH_SIZE,
    device=None,
):
    device = device or getDevice()
    model.eval()
    generated = []
    with torch.no_grad():
        for start in range(0, len(samples), batchSize):
            batch = embeddings[start:start + batchSize].to(device)
            generated.extend(
                model.generate(batch, maxDecodeSteps).cpu().tolist()
            )

    valid = 0
    graphExact = 0
    tokenExact = 0
    stageExact = 0
    nodeCountExact = 0
    deviceCountExact = 0
    edgeCountExact = 0
    primitiveCountExact = 0
    groundPresenceExact = 0
    gedSimilarityTotal = 0.0
    errors = Counter()
    perTopology = defaultdict(lambda: {
        "sample_count": 0,
        "valid": 0,
        "graph_exact": 0,
        "stage_exact": 0,
        "node_count_exact": 0,
        "device_count_exact": 0,
        "edge_count_exact": 0,
        "primitive_count_exact": 0,
        "ground_presence_exact": 0,
        "ged_similarity_total": 0.0,
    })
    # target 侧量在每个配置内被重复使用，预先算一次避免逐样本重算。
    targetCache = {}
    for key, target in targetByKey.items():
        targetCache[key] = {
            "tokens": gmRcGraphToTokenIds(target),
            "semantic_key": gmRcSemanticKey(target),
            "stage_counts": extractStageCounts(target),
            "has_ground": any(
                node["kind"] == "ground" for node in target["nodes"]
            ),
        }
    for sequence, sample in zip(generated, samples):
        topologyMetrics = perTopology[sample.topology]
        topologyMetrics["sample_count"] += 1
        target = targetByKey[sample.targetKey]
        cached = targetCache[sample.targetKey]
        try:
            trimmed = _trimGeneratedTokens(sequence)
            prediction = gmRcTokenIdsToGraphData(trimmed)
        except (ValueError, IndexError) as error:
            errors[str(error)] += 1
            continue
        valid += 1
        topologyMetrics["valid"] += 1
        tokenMatch = int(trimmed == cached["tokens"])
        graphMatch = int(
            gmRcSemanticKey(prediction) == cached["semantic_key"]
        )
        stageMatch = int(
            extractStageCounts(prediction) == cached["stage_counts"]
        )
        nodeCountMatch = int(
            len(prediction["nodes"]) == len(target["nodes"])
        )
        deviceCountMatch = int(
            len(prediction["devices"]) == len(target["devices"])
        )
        edgeCountMatch = int(
            len(prediction["edges"]) == len(target["edges"])
        )
        primitiveCountMatch = int(
            _primitiveCounts(prediction) == _primitiveCounts(target)
        )
        predictedGround = any(
            node["kind"] == "ground" for node in prediction["nodes"]
        )
        groundMatch = int(predictedGround == cached["has_ground"])
        gedSimilarity = gmRcGraphGedSimilarity(prediction, target)

        tokenExact += tokenMatch
        graphExact += graphMatch
        stageExact += stageMatch
        nodeCountExact += nodeCountMatch
        deviceCountExact += deviceCountMatch
        edgeCountExact += edgeCountMatch
        primitiveCountExact += primitiveCountMatch
        groundPresenceExact += groundMatch
        gedSimilarityTotal += gedSimilarity
        for key, value in (
            ("graph_exact", graphMatch),
            ("stage_exact", stageMatch),
            ("node_count_exact", nodeCountMatch),
            ("device_count_exact", deviceCountMatch),
            ("edge_count_exact", edgeCountMatch),
            ("primitive_count_exact", primitiveCountMatch),
            ("ground_presence_exact", groundMatch),
        ):
            topologyMetrics[key] += value
        topologyMetrics["ged_similarity_total"] += gedSimilarity

    total = len(samples)
    perTopologyMetrics = {}
    for topology, values in sorted(perTopology.items()):
        count = values["sample_count"]
        perTopologyMetrics[topology] = {
            "sample_count": count,
            "valid_graph_rate": values["valid"] / count,
            "graph_exact": values["graph_exact"] / count,
            "stage_exact": values["stage_exact"] / count,
            "node_count_exact": values["node_count_exact"] / count,
            "device_count_exact": values["device_count_exact"] / count,
            "edge_count_exact": values["edge_count_exact"] / count,
            "primitive_count_exact": (
                values["primitive_count_exact"] / count
            ),
            "ground_presence_acc": (
                values["ground_presence_exact"] / count
            ),
            "ged_similarity_mean": (
                values["ged_similarity_total"] / count
            ),
        }
    return {
        "sample_count": total,
        "valid_graph_rate": valid / total if total else 0.0,
        "graph_exact": graphExact / total if total else 0.0,
        "token_exact": tokenExact / total if total else 0.0,
        "stage_exact": stageExact / total if total else 0.0,
        "node_count_exact": nodeCountExact / total if total else 0.0,
        "device_count_exact": deviceCountExact / total if total else 0.0,
        "edge_count_exact": edgeCountExact / total if total else 0.0,
        "primitive_count_exact": (
            primitiveCountExact / total if total else 0.0
        ),
        "ground_presence_acc": (
            groundPresenceExact / total if total else 0.0
        ),
        "ged_similarity_mean": gedSimilarityTotal / total if total else 0.0,
        "invalid_error_counts": dict(errors.most_common(10)),
        "per_topology": perTopologyMetrics,
    }
