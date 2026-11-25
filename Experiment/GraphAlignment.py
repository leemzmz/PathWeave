from itertools import combinations
import json
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from Experiment.GraphAlignmentData import (
    CircuitAlignmentDataset,
    MacroHardNegativeSampler,
    buildHardNegativeCandidates,
    makeAlignmentCollate,
)
from Experiment.Metrics import calculatePearson, calculateSpearman
from Experiment.Runtime import getDevice
from GraphSAGE.GmRcGraph import (
    GM_RC_EDGE_FEATURE_DIM,
    GM_RC_EDGE_FEATURE_SCHEMA,
    GM_RC_ENCODER_CONTRACT,
    GM_RC_GRAPH_CONTRACT,
    GM_RC_NODE_FEATURE_DIM,
    GM_RC_NODE_FEATURE_SCHEMA,
    GmRcGraphEncoder,
    encodeGmRcBatch,
    gmRcGraphSimilarity,
    gmRcGraphSimilarityVersion,
)
from GraphSAGE.graph_data import batch_graphs
from GraphSAGE.CircuitEncoder import buildCircuitEncoder, describeCircuitBackbone


GRAPH_ALIGNMENT_MANIFOLD_SCHEMA = (
    "circuit_to_functional_gm_rc_sampled_manifold_alignment"
)
GRAPH_ALIGNMENT_OUTPUT_DIM = 128
DEFAULT_ALIGNMENT_LOSS_WEIGHTS = {
    "positive": 1.0,
    "geometry": 0.5,
    "retrieval": 0.25,
}
DEFAULT_HARD_NEGATIVE_CONFIG = {
    "mode": "manifold",
    "top_k": 5,
    "negative_count": 2,
    "sampling_temperature": 0.1,
    "margin_scale": 0.5,
    "manifold_loss_weight": 0.5,
    "ranking_loss_weight": 0.25,
}


def _setSeed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


_TARGET_SIMILARITY_CACHE = {}


def _targetSimilarityMatrix(records):
    cacheKey = tuple(record.semanticKey for record in records)
    cached = _TARGET_SIMILARITY_CACHE.get(cacheKey)
    if cached is not None:
        return cached.clone()
    count = len(records)
    matrix = torch.eye(count, dtype=torch.float32)
    for first, second in combinations(range(count), 2):
        score = gmRcGraphSimilarity(
            records[first].graphData, records[second].graphData
        )
        matrix[first, second] = score
        matrix[second, first] = score
    _TARGET_SIMILARITY_CACHE[cacheKey] = matrix
    return matrix.clone()


def _targetBatch(records):
    return batch_graphs([record.tensor for record in records])


def _encodeTargets(model, records, device):
    model.eval()
    with torch.no_grad():
        return encodeGmRcBatch(model, _targetBatch(records), device)


# 使用冻结的 target encoder 生成 canonical target embedding。
def encodeTargetRecords(model, records, device=None):
    device = device or getDevice()
    return _encodeTargets(model.to(device), records, device).cpu()


def _targetGeometryMetrics(embeddings, labels):
    predictions = (embeddings @ embeddings.t()).detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    indices = np.triu_indices(labels.shape[0], k=1)
    expected = labels[indices]
    actual = predictions[indices]
    return {
        "mse": float(np.mean((expected - actual) ** 2)),
        "spearman": calculateSpearman(expected, actual),
        "pearson": calculatePearson(expected, actual),
    }


# 先学习 Gm+RC 目标图自身的连续结构空间。
def trainTargetGraphEncoder(
    records,
    checkpointPath,
    epochs=300,
    learningRate=1e-3,
    hiddenDim=128,
    outputDim=GRAPH_ALIGNMENT_OUTPUT_DIM,
    numLayers=2,
    dropout=0.1,
    seed=42,
):
    if len(records) < 2:
        raise ValueError("target graph encoder requires at least two graphs")
    _setSeed(seed)
    device = getDevice()
    model = GmRcGraphEncoder(
        inputDim=GM_RC_NODE_FEATURE_DIM,
        hiddenDim=hiddenDim,
        outputDim=outputDim,
        numLayers=numLayers,
        dropout=dropout,
        edgeFeatureDim=GM_RC_EDGE_FEATURE_DIM,
    ).to(device)
    graphBatch = _targetBatch(records)
    similarity = _targetSimilarityMatrix(records).to(device)
    offDiagonal = ~torch.eye(
        len(records), dtype=torch.bool, device=device
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)

    for epoch in range(epochs):
        model.train()
        embeddings = encodeGmRcBatch(model, graphBatch, device)
        predictions = embeddings @ embeddings.t()
        loss = F.mse_loss(
            predictions[offDiagonal], similarity[offDiagonal]
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch == 0 or (epoch + 1) % max(1, epochs // 10) == 0:
            print(
                f"Target epoch {epoch + 1:03d}/{epochs} "
                f"loss={loss.item():.6f}"
            )

    embeddings = _encodeTargets(model, records, device)
    metrics = _targetGeometryMetrics(embeddings, similarity.cpu())
    checkpointPath = Path(checkpointPath)
    checkpointPath.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "model_contract": GM_RC_ENCODER_CONTRACT,
        "graph_contract": GM_RC_GRAPH_CONTRACT,
        "similarity_method": "ged",
        "similarity": gmRcGraphSimilarityVersion(),
        "node_feature_schema": GM_RC_NODE_FEATURE_SCHEMA,
        "edge_feature_schema": GM_RC_EDGE_FEATURE_SCHEMA,
        "input_dim": GM_RC_NODE_FEATURE_DIM,
        "edge_feature_dim": GM_RC_EDGE_FEATURE_DIM,
        "hidden_dim": hiddenDim,
        "output_dim": outputDim,
        "num_layers": numLayers,
        "dropout": dropout,
        "training_seed": int(seed),
        "target_keys": [record.semanticKey for record in records],
        "metrics": metrics,
    }, checkpointPath)
    print("target model saved:", checkpointPath)
    print("target metrics:", json.dumps(metrics, sort_keys=True))
    return model, metrics


# 加载并严格校验共享目标图 encoder。
def loadTargetGraphEncoder(checkpointPath, device=None):
    device = device or getDevice()
    checkpoint = torch.load(
        checkpointPath, map_location=device, weights_only=False
    )
    expected = {
        "model_contract": GM_RC_ENCODER_CONTRACT,
        "graph_contract": GM_RC_GRAPH_CONTRACT,
        "similarity_method": "ged",
        "similarity": gmRcGraphSimilarityVersion(),
        "node_feature_schema": GM_RC_NODE_FEATURE_SCHEMA,
        "edge_feature_schema": GM_RC_EDGE_FEATURE_SCHEMA,
    }
    for key, value in expected.items():
        if checkpoint.get(key) != value:
            raise ValueError(
                f"target checkpoint {key} mismatch: "
                f"{checkpoint.get(key)!r} != {value!r}"
            )
    model = GmRcGraphEncoder(
        inputDim=checkpoint["input_dim"],
        hiddenDim=checkpoint["hidden_dim"],
        outputDim=checkpoint["output_dim"],
        numLayers=checkpoint["num_layers"],
        dropout=checkpoint["dropout"],
        edgeFeatureDim=checkpoint["edge_feature_dim"],
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint


def _geometryLoss(circuitEmbedding, targetEmbedding):
    if circuitEmbedding.shape[0] < 2:
        return circuitEmbedding.new_zeros(())
    circuitSimilarity = circuitEmbedding @ circuitEmbedding.t()
    targetSimilarity = targetEmbedding @ targetEmbedding.t()
    mask = ~torch.eye(
        circuitEmbedding.shape[0],
        dtype=torch.bool,
        device=circuitEmbedding.device,
    )
    return F.mse_loss(circuitSimilarity[mask], targetSimilarity[mask])


def _normalizeHardNegativeConfig(config):
    if config is None:
        raise ValueError("hard-negative configuration is required")
    normalized = dict(DEFAULT_HARD_NEGATIVE_CONFIG)
    normalized.update(config)
    normalized["top_k"] = int(normalized["top_k"])
    normalized["negative_count"] = int(normalized["negative_count"])
    for key in (
        "sampling_temperature",
        "margin_scale",
        "manifold_loss_weight",
        "ranking_loss_weight",
    ):
        normalized[key] = float(normalized[key])
    if normalized["mode"] != "manifold":
        raise ValueError("hard-negative mode must be manifold")
    if normalized["margin_scale"] < 0.0:
        raise ValueError("hard-negative margin scale must be nonnegative")
    if normalized["manifold_loss_weight"] < 0.0:
        raise ValueError("hard manifold loss weight must be nonnegative")
    if normalized["ranking_loss_weight"] < 0.0:
        raise ValueError("hard ranking loss weight must be nonnegative")
    return normalized


def _hardNegativeManifoldLosses(
    anchorEmbedding,
    positiveCoreEmbedding,
    negativeCoreEmbedding,
    positiveTargetEmbedding,
    negativeTargetEmbedding,
    marginScale,
):
    positiveCoreCosine = torch.sum(
        anchorEmbedding * positiveCoreEmbedding, dim=1
    )
    negativeCoreCosine = torch.sum(
        anchorEmbedding.unsqueeze(1) * negativeCoreEmbedding,
        dim=2,
    )
    expectedCosine = torch.sum(
        positiveTargetEmbedding.unsqueeze(1)
        * negativeTargetEmbedding,
        dim=2,
    ).detach()
    anchorNegativeTargetCosine = torch.sum(
        anchorEmbedding.unsqueeze(1) * negativeTargetEmbedding,
        dim=2,
    )
    anchorPositiveTargetCosine = torch.sum(
        anchorEmbedding * positiveTargetEmbedding, dim=1
    )
    positiveGeometryLoss = F.smooth_l1_loss(
        positiveCoreCosine,
        torch.ones_like(positiveCoreCosine),
    )
    coreGeometryLoss = F.smooth_l1_loss(
        negativeCoreCosine, expectedCosine
    )
    crossGeometryLoss = F.smooth_l1_loss(
        anchorNegativeTargetCosine, expectedCosine
    )
    manifoldLoss = (
        positiveGeometryLoss + coreGeometryLoss + crossGeometryLoss
    ) / 3.0
    margins = marginScale * (1.0 - expectedCosine)
    rankingLoss = F.relu(
        anchorNegativeTargetCosine
        - anchorPositiveTargetCosine.unsqueeze(1)
        + margins
    ).mean()
    return manifoldLoss, rankingLoss


# 训练一个 baseline 的电路 encoder 对齐冻结的目标图空间。
def trainCircuitAlignmentEncoder(
    backend,
    spec,
    samples,
    targetRecords,
    targetModel,
    checkpointPath,
    batchSize=32,
    learningRate=1e-3,
    hiddenDim=128,
    outputDim=GRAPH_ALIGNMENT_OUTPUT_DIM,
    numLayers=2,
    dropout=0.1,
    temperature=0.1,
    lossWeights=None,
    seed=42,
    hardNegativeConfig=None,
    circuitBackbone="graphsage",
    maxOptimizerSteps=None,
    sampleSplitPolicy=None,
    targetCheckpointSha256=None,
):
    if batchSize < 1:
        raise ValueError("alignment batch size must be positive")
    if maxOptimizerSteps is None or maxOptimizerSteps < 1:
        raise ValueError("alignment optimizer steps are required")
    if not samples:
        raise ValueError("cannot train alignment encoder without samples")
    if sampleSplitPolicy is None:
        raise ValueError("alignment training requires a split policy")
    lossWeights = dict(DEFAULT_ALIGNMENT_LOSS_WEIGHTS if lossWeights is None else lossWeights)
    hardNegativeConfig = _normalizeHardNegativeConfig(
        hardNegativeConfig
    )
    _setSeed(seed)
    device = getDevice()
    targetModel = targetModel.to(device)
    for parameter in targetModel.parameters():
        parameter.requires_grad = False
    targetEmbeddingBank = _encodeTargets(
        targetModel, targetRecords, device
    ).detach()

    backboneConfig = describeCircuitBackbone(circuitBackbone)
    modelBuilder = spec.build_encoder or buildCircuitEncoder
    model = modelBuilder(
        circuitBackbone,
        inputDim=spec.node_feature_dim,
        hiddenDim=hiddenDim,
        outputDim=outputDim,
        numLayers=numLayers,
        dropout=dropout,
        edgeFeatureDim=len(spec.edge_feature_schema),
    ).to(device)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        CircuitAlignmentDataset(samples),
        batch_size=batchSize,
        shuffle=True,
        collate_fn=makeAlignmentCollate(spec.batch_graphs),
        generator=generator,
    )
    targetSimilarity = _targetSimilarityMatrix(targetRecords)
    hardNegativeSampler = MacroHardNegativeSampler(
        samples,
        targetSimilarity,
        topK=hardNegativeConfig["top_k"],
        negativeCount=hardNegativeConfig["negative_count"],
        temperature=hardNegativeConfig["sampling_temperature"],
        seed=seed,
    )
    hardNegativeCandidates = {
        targetRecords[index].semanticKey: [
            targetRecords[other].semanticKey
            for other in candidates
        ]
        for index, candidates in (
            hardNegativeSampler.candidateTargets().items()
        )
    }
    optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)

    totalOptimizerSteps = int(maxOptimizerSteps)
    completedOptimizerSteps = 0
    completedEpochs = 0
    reportEvery = max(1, totalOptimizerSteps // 10)
    while completedOptimizerSteps < totalOptimizerSteps:
        model.train()
        for graphBatch, targetIndices, names in loader:
            targetIndices = targetIndices.to(device)
            circuitEmbedding = spec.encode_batch(model, graphBatch, device)
            targetEmbedding = targetEmbeddingBank[targetIndices]
            positiveLoss = (1.0 - torch.sum(
                circuitEmbedding * targetEmbedding, dim=1
            )).mean()
            geometryLoss = _geometryLoss(
                circuitEmbedding, targetEmbedding
            )
            logits = circuitEmbedding @ targetEmbeddingBank.t() / temperature
            retrievalLoss = F.cross_entropy(logits, targetIndices)
            hardManifoldLoss = circuitEmbedding.new_zeros(())
            hardRankingLoss = circuitEmbedding.new_zeros(())
            (
                positiveSamples,
                negativeSamples,
                negativeTargets,
                _,
            ) = hardNegativeSampler.sampleBatch(names)
            positiveCoreEmbedding = spec.encode_batch(
                model,
                spec.batch_graphs([
                    sample.graph for sample in positiveSamples
                ]),
                device,
            )
            negativeCoreEmbedding = spec.encode_batch(
                model,
                spec.batch_graphs([
                    sample.graph for sample in negativeSamples
                ]),
                device,
            ).reshape(
                len(names),
                hardNegativeConfig["negative_count"],
                -1,
            )
            negativeTargetEmbedding = targetEmbeddingBank[
                negativeTargets.to(device)
            ]
            hardManifoldLoss, hardRankingLoss = _hardNegativeManifoldLosses(
                circuitEmbedding,
                positiveCoreEmbedding,
                negativeCoreEmbedding,
                targetEmbedding,
                negativeTargetEmbedding,
                hardNegativeConfig["margin_scale"],
            )
            loss = (
                lossWeights["positive"] * positiveLoss
                + lossWeights["geometry"] * geometryLoss
                + lossWeights["retrieval"] * retrievalLoss
            )
            loss = (
                loss
                + hardNegativeConfig["manifold_loss_weight"]
                * hardManifoldLoss
                + hardNegativeConfig["ranking_loss_weight"]
                * hardRankingLoss
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            completedOptimizerSteps += 1
            if (
                completedOptimizerSteps == 1
                or completedOptimizerSteps % reportEvery == 0
                or completedOptimizerSteps == totalOptimizerSteps
            ):
                print(
                    f"{backend} step "
                    f"{completedOptimizerSteps:04d}/{totalOptimizerSteps} "
                    f"loss={loss.item():.6f} "
                    f"positive={positiveLoss.item():.6f} "
                    f"geometry={geometryLoss.item():.6f} "
                    f"retrieval={retrievalLoss.item():.6f} "
                    f"hard_manifold={hardManifoldLoss.item():.6f} "
                    f"hard_ranking={hardRankingLoss.item():.6f}"
                )
            if completedOptimizerSteps >= totalOptimizerSteps:
                break
        completedEpochs += 1

    checkpointPath = Path(checkpointPath)
    checkpointPath.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "model_contract": GRAPH_ALIGNMENT_MANIFOLD_SCHEMA,
        "backend": backend,
        "backend_model_contract": spec.model_contract,
        "message_type": backboneConfig["message_type"],
        **({
            "circuit_backbone": circuitBackbone,
            "backbone_config": backboneConfig,
        } if circuitBackbone != "graphsage" else {}),
        "node_feature_schema": spec.node_feature_schema,
        "edge_feature_schema": spec.edge_feature_schema,
        "input_dim": spec.node_feature_dim,
        "edge_feature_dim": len(spec.edge_feature_schema),
        "hidden_dim": hiddenDim,
        "output_dim": outputDim,
        "num_layers": numLayers,
        "dropout": dropout,
        "temperature": temperature,
        "loss_weights": lossWeights,
        "hard_negative_config": hardNegativeConfig,
        "hard_negative_candidates": hardNegativeCandidates,
        "target_model_contract": GM_RC_ENCODER_CONTRACT,
        "target_similarity_method": "ged",
        "target_similarity": gmRcGraphSimilarityVersion(),
        "target_keys": [record.semanticKey for record in targetRecords],
        "sample_split_policy": sampleSplitPolicy,
        "training_seed": int(seed),
        "training_policy": "fixed_optimizer_steps_all_samples",
        "completed_optimizer_steps": completedOptimizerSteps,
        "completed_epochs": completedEpochs,
        **({
            "target_checkpoint_sha256": targetCheckpointSha256,
        } if targetCheckpointSha256 is not None else {}),
    }, checkpointPath)
    print("circuit model saved:", checkpointPath)
    return model


# 加载并校验一个 baseline 的对齐 encoder。
def loadCircuitAlignmentEncoder(
    checkpointPath,
    backend,
    spec,
    seed=42,
    device=None,
    hardNegativeConfig=None,
    circuitBackbone="graphsage",
    sampleSplitPolicy=None,
    targetCheckpointSha256=None,
):
    device = device or getDevice()
    checkpoint = torch.load(
        checkpointPath, map_location=device, weights_only=False
    )
    backboneConfig = describeCircuitBackbone(circuitBackbone)
    if checkpoint.get("circuit_backbone", "graphsage") != circuitBackbone:
        raise ValueError("alignment checkpoint circuit backbone mismatch")
    if circuitBackbone != "graphsage" and checkpoint.get("backbone_config") != backboneConfig:
        raise ValueError("alignment checkpoint backbone configuration mismatch")
    hardNegativeConfig = _normalizeHardNegativeConfig(
        hardNegativeConfig
    )
    expected = {
        "model_contract": GRAPH_ALIGNMENT_MANIFOLD_SCHEMA,
        "backend": backend,
        "backend_model_contract": spec.model_contract,
        "message_type": backboneConfig["message_type"],
        "node_feature_schema": spec.node_feature_schema,
        "edge_feature_schema": spec.edge_feature_schema,
        "target_model_contract": GM_RC_ENCODER_CONTRACT,
        "target_similarity_method": "ged",
        "target_similarity": gmRcGraphSimilarityVersion(),
        "hard_negative_config": hardNegativeConfig,
    }
    for key, value in expected.items():
        if checkpoint.get(key) != value:
            raise ValueError(
                f"alignment checkpoint {key} mismatch: "
                f"{checkpoint.get(key)!r} != {value!r}"
            )
    if sampleSplitPolicy is None:
        raise ValueError("alignment loading requires a split policy")
    if checkpoint.get("sample_split_policy") != sampleSplitPolicy:
        raise ValueError("alignment checkpoint split policy mismatch")
    if targetCheckpointSha256 is not None and checkpoint.get(
        "target_checkpoint_sha256"
    ) != targetCheckpointSha256:
        raise ValueError("alignment checkpoint target checksum mismatch")
    if checkpoint["input_dim"] != spec.node_feature_dim:
        raise ValueError("alignment checkpoint input dimension mismatch")
    modelBuilder = spec.build_encoder or buildCircuitEncoder
    model = modelBuilder(
        circuitBackbone,
        inputDim=checkpoint["input_dim"],
        hiddenDim=checkpoint["hidden_dim"],
        outputDim=checkpoint["output_dim"],
        numLayers=checkpoint["num_layers"],
        dropout=checkpoint["dropout"],
        edgeFeatureDim=len(spec.edge_feature_schema),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint


def _encodeCircuitSamples(model, spec, samples, batchSize, device):
    loader = DataLoader(
        CircuitAlignmentDataset(samples),
        batch_size=batchSize,
        shuffle=False,
        collate_fn=makeAlignmentCollate(spec.batch_graphs),
    )
    embeddings = []
    targetIndices = []
    names = []
    model.eval()
    with torch.no_grad():
        for graphBatch, batchTargets, batchNames in loader:
            embeddings.append(
                spec.encode_batch(model, graphBatch, device).cpu()
            )
            targetIndices.extend(batchTargets.tolist())
            names.extend(batchNames)
    return (
        torch.cat(embeddings, dim=0),
        np.asarray(targetIndices, dtype=np.int64),
        names,
    )


def _samplePairs(count, maxPairs, seed):
    total = count * (count - 1) // 2
    if maxPairs is None or total <= maxPairs:
        return list(combinations(range(count), 2))
    rng = random.Random(seed)
    selected = set()
    while len(selected) < maxPairs:
        first, second = rng.sample(range(count), 2)
        if first > second:
            first, second = second, first
        selected.add((first, second))
    return sorted(selected)


# 在完全独立的 test core 上评估跨图对齐、结构几何与图检索。
def evaluateCircuitAlignment(
    model,
    spec,
    samples,
    targetRecords,
    targetModel,
    batchSize=64,
    maxPairs=None,
    seed=42,
    hardNegativeTopK=5,
    device=None,
):
    device = device or getDevice()
    circuitEmbeddings, targetIndices, _ = _encodeCircuitSamples(
        model, spec, samples, batchSize, device
    )
    targetEmbeddings = _encodeTargets(
        targetModel.to(device), targetRecords, device
    ).cpu()
    pairedTargets = targetEmbeddings[targetIndices]
    pairedCosine = torch.sum(
        circuitEmbeddings * pairedTargets, dim=1
    ).numpy()
    logits = (circuitEmbeddings @ targetEmbeddings.t()).numpy()
    ranking = np.argsort(-logits, axis=1)
    top1 = ranking[:, 0]
    top3 = ranking[:, :min(3, ranking.shape[1])]
    reciprocalRanks = []
    for row, targetIndex in zip(ranking, targetIndices):
        rank = int(np.flatnonzero(row == targetIndex)[0]) + 1
        reciprocalRanks.append(1.0 / rank)

    stageExact = []
    for predicted, expected in zip(top1, targetIndices):
        stageExact.append(
            targetRecords[int(predicted)].stageCounts
            == targetRecords[int(expected)].stageCounts
        )
    exact = top1 == targetIndices
    negativeLogits = logits.copy()
    negativeLogits[np.arange(len(samples)), targetIndices] = -np.inf
    hardestNegative = np.max(negativeLogits, axis=1)
    alignmentMargins = pairedCosine - hardestNegative
    top3Exact = np.asarray([
        targetIndex in row
        for targetIndex, row in zip(targetIndices, top3)
    ], dtype=bool)
    stageExact = np.asarray(stageExact, dtype=bool)
    topologies = np.asarray([sample.topology for sample in samples])
    perTopology = {}
    for topology in sorted(set(topologies.tolist())):
        mask = topologies == topology
        predicted, counts = np.unique(top1[mask], return_counts=True)
        perTopology[topology] = {
            "cases": int(np.sum(mask)),
            "expected_target_index": int(targetIndices[mask][0]),
            "retrieval_top1": float(np.mean(exact[mask])),
            "retrieval_top3": float(np.mean(top3Exact[mask])),
            "stage_exact": float(np.mean(stageExact[mask])),
            "paired_cosine_mean": float(np.mean(pairedCosine[mask])),
            "predicted_target_counts": {
                str(int(index)): int(count)
                for index, count in zip(predicted, counts)
            },
        }

    pairs = _samplePairs(len(samples), maxPairs, seed)
    targetSimilarity = _targetSimilarityMatrix(targetRecords).numpy()
    hardCandidates = buildHardNegativeCandidates(
        targetSimilarity, topK=hardNegativeTopK
    )
    hardCandidateIndices = np.asarray([
        hardCandidates[index] for index in targetIndices
    ], dtype=np.int64)
    hardNegativeLogits = np.take_along_axis(
        logits, hardCandidateIndices, axis=1
    )
    hardestSimilarityNegative = np.max(
        hardNegativeLogits, axis=1
    )
    similarityHardMargins = pairedCosine - hardestSimilarityNegative
    labels = []
    targetManifoldLabels = []
    predictions = []
    for first, second in pairs:
        labels.append(targetSimilarity[
            targetIndices[first], targetIndices[second]
        ])
        targetManifoldLabels.append(float(torch.dot(
            targetEmbeddings[targetIndices[first]],
            targetEmbeddings[targetIndices[second]],
        )))
        predictions.append(float(torch.dot(
            circuitEmbeddings[first], circuitEmbeddings[second]
        )))
    labels = np.asarray(labels, dtype=np.float64)
    targetManifoldLabels = np.asarray(
        targetManifoldLabels, dtype=np.float64
    )
    predictions = np.asarray(predictions, dtype=np.float64)

    sameTargetMask = np.asarray([
        targetIndices[first] == targetIndices[second]
        for first, second in pairs
    ], dtype=bool)
    similarityHardPairMask = np.asarray([
        (
            targetIndices[second]
            in hardCandidates[targetIndices[first]]
            or targetIndices[first]
            in hardCandidates[targetIndices[second]]
        )
        for first, second in pairs
    ], dtype=bool)
    targetToCircuitTop1 = []
    targetToCircuitMrr = []
    for targetIndex in sorted(set(targetIndices.tolist())):
        rankedCircuits = np.argsort(-logits[:, targetIndex])
        positives = targetIndices[rankedCircuits] == targetIndex
        targetToCircuitTop1.append(bool(positives[0]))
        targetToCircuitMrr.append(
            1.0 / (int(np.flatnonzero(positives)[0]) + 1)
        )

    implementations = np.asarray([
        sample.feedbackImplementation for sample in samples
    ])
    mosMask = implementations == "mos"
    resistorMask = implementations == "resistor"
    crossRealizationCosines = []
    for first, second in combinations(range(len(samples)), 2):
        if samples[first].topology != samples[second].topology:
            continue
        if {implementations[first], implementations[second]} != {
            "mos", "resistor"
        }:
            continue
        crossRealizationCosines.append(float(
            torch.dot(circuitEmbeddings[first], circuitEmbeddings[second])
        ))
    crossRealizationCosines = np.asarray(
        crossRealizationCosines, dtype=np.float64
    )

    def _maskedMean(values, mask):
        return float(np.mean(values[mask])) if np.any(mask) else None

    metrics = {
        "cases": len(samples),
        "target_graphs": len(targetRecords),
        "paired_cosine_mean": float(np.mean(pairedCosine)),
        "paired_cosine_min": float(np.min(pairedCosine)),
        "hardest_negative_cosine_mean": float(np.mean(hardestNegative)),
        "alignment_margin_mean": float(np.mean(alignmentMargins)),
        "alignment_margin_min": float(np.min(alignmentMargins)),
        "alignment_margin_positive_rate": float(np.mean(
            alignmentMargins > 0.0
        )),
        "similarity_hard_top_k": int(hardNegativeTopK),
        "similarity_hard_margin_mean": float(np.mean(
            similarityHardMargins
        )),
        "similarity_hard_margin_min": float(np.min(
            similarityHardMargins
        )),
        "similarity_hard_margin_positive_rate": float(np.mean(
            similarityHardMargins > 0.0
        )),
        "retrieval_top1": float(np.mean(exact)),
        "retrieval_top3": float(np.mean(top3Exact)),
        "retrieval_mrr": float(np.mean(reciprocalRanks)),
        "target_to_circuit_top1": float(np.mean(targetToCircuitTop1)),
        "target_to_circuit_mrr": float(np.mean(targetToCircuitMrr)),
        "stage_exact": float(np.mean(stageExact)),
        "per_topology": perTopology,
        "mos_feedback_cases": int(np.sum(mosMask)),
        "resistor_feedback_cases": int(np.sum(resistorMask)),
        "mos_feedback_top1": _maskedMean(exact, mosMask),
        "resistor_feedback_top1": _maskedMean(exact, resistorMask),
        "cross_realization_pair_count": int(len(crossRealizationCosines)),
        "cross_realization_cosine_mean": (
            float(np.mean(crossRealizationCosines))
            if len(crossRealizationCosines) else None
        ),
        "cross_realization_cosine_min": (
            float(np.min(crossRealizationCosines))
            if len(crossRealizationCosines) else None
        ),
        "pair_count": len(pairs),
        "same_target_pair_count": int(np.sum(sameTargetMask)),
        "same_target_cosine_mean": _maskedMean(
            predictions, sameTargetMask
        ),
        "different_target_cosine_mean": _maskedMean(
            predictions, ~sameTargetMask
        ),
        "similarity_hard_pair_count": int(np.sum(
            similarityHardPairMask
        )),
        "similarity_hard_pair_cosine_mean": _maskedMean(
            predictions, similarityHardPairMask
        ),
        "similarity_hard_pair_structure_mse": float(np.mean((
            labels[similarityHardPairMask]
            - predictions[similarityHardPairMask]
        ) ** 2)),
        "similarity_hard_pair_structure_spearman": calculateSpearman(
            labels[similarityHardPairMask],
            predictions[similarityHardPairMask],
        ),
        "similarity_hard_pair_target_manifold_mse": float(np.mean((
            targetManifoldLabels[similarityHardPairMask]
            - predictions[similarityHardPairMask]
        ) ** 2)),
        "similarity_hard_pair_target_manifold_spearman": calculateSpearman(
            targetManifoldLabels[similarityHardPairMask],
            predictions[similarityHardPairMask],
        ),
        "structure_mse": float(np.mean((labels - predictions) ** 2)),
        "structure_spearman": calculateSpearman(labels, predictions),
        "structure_pearson": calculatePearson(labels, predictions),
        "target_manifold_mse": float(np.mean((
            targetManifoldLabels - predictions
        ) ** 2)),
        "target_manifold_spearman": calculateSpearman(
            targetManifoldLabels, predictions
        ),
        "target_manifold_pearson": calculatePearson(
            targetManifoldLabels, predictions
        ),
    }
    return metrics
