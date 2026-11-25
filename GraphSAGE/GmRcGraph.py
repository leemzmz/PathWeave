from collections import Counter, defaultdict
from dataclasses import dataclass
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from GraphSAGE.graph_data import GraphTensor
from GraphSAGE.graphsage import GraphSAGEEncoder


GM_RC_NODE_FEATURE_SCHEMA = [
    "is_boundary",
    "is_internal",
    "is_ground",
    "is_gm",
    "is_r",
    "is_c",
    "is_input_pin",
    "is_main_boundary",
    "is_stage_output",
    "is_output_pin",
    "boundary_position",
    "stage_position",
]
GM_RC_EDGE_FEATURE_SCHEMA = [
    "passive",
    "gm_input",
    "gm_output",
]
GM_RC_NODE_FEATURE_DIM = len(GM_RC_NODE_FEATURE_SCHEMA)
GM_RC_EDGE_FEATURE_DIM = len(GM_RC_EDGE_FEATURE_SCHEMA)
GM_RC_GRAPH_CONTRACT = "functional_gm_rc_graph_v2"
GM_RC_ENCODER_CONTRACT = "gm_rc_graphsage_alignment_target_v4_emb128"
GM_RC_GED_VERSION = "bipartite_ged_vertex_edge_v1"


@dataclass(frozen=True)
class GmRcGraphRecord:
    graphData: dict
    tensor: GraphTensor
    semanticKey: str
    stageCounts: tuple


def _loadJson(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _validateGraphData(graphData):
    nodes = graphData.get("nodes", [])
    devices = graphData.get("devices", [])
    edges = graphData.get("edges", [])
    nodeIds = [node.get("id") for node in nodes]
    deviceIds = [device.get("id") for device in devices]
    if len(nodeIds) != len(set(nodeIds)):
        raise ValueError("duplicate Gm+RC electrical node id")
    if len(deviceIds) != len(set(deviceIds)):
        raise ValueError("duplicate Gm+RC device id")

    knownNodes = set(nodeIds)
    knownDevices = set(deviceIds)
    terminals = defaultdict(list)
    for edge in edges:
        nodeId = edge.get("node")
        deviceId = edge.get("device")
        if nodeId not in knownNodes:
            raise ValueError(f"unknown Gm+RC electrical node: {nodeId}")
        if deviceId not in knownDevices:
            raise ValueError(f"unknown Gm+RC device: {deviceId}")
        terminals[deviceId].append(edge.get("terminal"))

    for device in devices:
        deviceId = device["id"]
        kind = device.get("kind")
        values = terminals[deviceId]
        if len(values) != 2:
            raise ValueError(
                f"Gm+RC device {deviceId} must have two terminals"
            )
        if kind == "Gm" and set(values) != {"input", "output"}:
            raise ValueError(
                f"Gm device {deviceId} requires input/output terminals"
            )
        if kind in {"R", "C"} and set(values) != {"a", "b"}:
            raise ValueError(
                f"passive device {deviceId} requires a/b terminals"
            )
        if kind not in {"Gm", "R", "C"}:
            raise ValueError(f"unsupported Gm+RC device kind: {kind}")


# 加载 dataset 定义的 topology 级功能 Gm+RC 目标图。
def loadGmRcGraph(targetPath):
    graphData = _loadJson(targetPath)
    for device in graphData.get("devices", []):
        device.pop("differential", None)
    _validateGraphData(graphData)
    tensor = gmRcGraphToTensor(graphData)
    return GmRcGraphRecord(
        graphData=graphData,
        tensor=tensor,
        semanticKey=gmRcSemanticKey(graphData),
        stageCounts=extractStageCounts(graphData),
    )


def _setFeature(features, name, value=1.0):
    features[GM_RC_NODE_FEATURE_SCHEMA.index(name)] = float(value)


# 将 Gm+RC 二分图转换为双向消息边；被动端口不编码人为 a/b 方向。
def gmRcGraphToTensor(graphData):
    _validateGraphData(graphData)
    electricalNodes = graphData["nodes"]
    devices = graphData["devices"]
    allItems = [*(('node', item) for item in electricalNodes),
                *(('device', item) for item in devices)]
    itemToIndex = {
        (category, item["id"]): index
        for index, (category, item) in enumerate(allItems)
    }
    maxBoundary = max(
        [node.get("boundary", 0) for node in electricalNodes] or [0]
    )
    maxStage = max(
        [node.get("stage", 0) for node in electricalNodes] or [0]
    )

    nodeFeatures = []
    nodeLabels = []
    for category, item in allItems:
        features = [0.0] * GM_RC_NODE_FEATURE_DIM
        if category == "node":
            kind = item.get("kind")
            _setFeature(features, f"is_{kind}")
            roles = set(item.get("roles", []))
            for role, featureName in (
                ("input-pin", "is_input_pin"),
                ("main-boundary", "is_main_boundary"),
                ("stage-output", "is_stage_output"),
                ("output-pin", "is_output_pin"),
            ):
                if role in roles:
                    _setFeature(features, featureName)
            if kind == "boundary":
                _setFeature(
                    features,
                    "boundary_position",
                    item.get("boundary", 0) / max(1, maxBoundary),
                )
            if item.get("stage") is not None:
                _setFeature(
                    features,
                    "stage_position",
                    item.get("stage", 0) / max(1, maxStage),
                )
        else:
            _setFeature(features, f"is_{item['kind'].lower()}")
        nodeFeatures.append(features)
        nodeLabels.append(f"{category}:{item['id']}")

    deviceById = {device["id"]: device for device in devices}
    messageEdges = []
    edgeFeatures = []
    edgeLabels = []
    for edge in graphData["edges"]:
        nodeIndex = itemToIndex[("node", edge["node"])]
        deviceIndex = itemToIndex[("device", edge["device"])]
        feature = [0.0] * GM_RC_EDGE_FEATURE_DIM
        deviceKind = deviceById[edge["device"]]["kind"]
        if deviceKind == "Gm":
            featureName = f"gm_{edge['terminal']}"
        else:
            featureName = "passive"
        feature[GM_RC_EDGE_FEATURE_SCHEMA.index(featureName)] = 1.0
        for source, target in (
            (nodeIndex, deviceIndex),
            (deviceIndex, nodeIndex),
        ):
            messageEdges.append((source, target))
            edgeFeatures.append(feature)
            edgeLabels.append(featureName)

    return GraphTensor(
        x=torch.tensor(nodeFeatures, dtype=torch.float32),
        edge_index=torch.tensor(
            messageEdges, dtype=torch.long
        ).t().contiguous(),
        edge_attr=torch.tensor(edgeFeatures, dtype=torch.float32),
        node_labels=nodeLabels,
        edge_labels=edgeLabels,
        node_feature_schema=GM_RC_NODE_FEATURE_SCHEMA,
        edge_feature_schema=GM_RC_EDGE_FEATURE_SCHEMA,
    )


def extractStageCounts(graphData):
    counts = Counter(
        int(device["stage"])
        for device in graphData["devices"]
        if device.get("role") == "main"
    )
    return tuple(counts[index] for index in range(1, max(counts, default=0) + 1))


def _nodeSemanticName(node):
    kind = node.get("kind")
    if kind == "boundary":
        return "B" + str(node.get("boundary"))
    if kind == "ground":
        return "GND"
    return "X" + str(node.get("order", 0))


def _pathSemanticData(graphData, pathId, devices):
    nodes = {node["id"]: node for node in graphData["nodes"]}
    edgesByDevice = defaultdict(list)
    for edge in graphData["edges"]:
        edgesByDevice[edge["device"]].append(edge)
    pathItems = []
    for device in sorted(devices, key=lambda item: item.get("order", 0)):
        terminals = []
        for edge in edgesByDevice[device["id"]]:
            terminal = edge["terminal"]
            if device["kind"] in {"R", "C"}:
                terminal = "passive"
            terminals.append((terminal, _nodeSemanticName(nodes[edge["node"]])))
        pathItems.append((device["kind"], tuple(sorted(terminals))))
    return tuple(pathItems)


# 生成与 topology 名称、path 编号和器件 id 无关的目标图语义键。
def gmRcSemanticKey(graphData):
    boundaries = sorted(
        (
            int(node.get("boundary", 0)),
            tuple(sorted(node.get("roles", []))),
            node.get("stage"),
        )
        for node in graphData["nodes"]
        if node.get("kind") == "boundary"
    )
    main = sorted(
        (
            int(device.get("unit", 0)),
            int(device.get("stage", 0)),
        )
        for device in graphData["devices"]
        if device.get("role") == "main"
    )
    paths = defaultdict(list)
    for device in graphData["devices"]:
        if device.get("role") == "compensation":
            paths[device.get("path")].append(device)
    pathData = sorted(
        _pathSemanticData(graphData, pathId, devices)
        for pathId, devices in paths.items()
    )
    semantic = {
        "boundaries": boundaries,
        "main": main,
        "paths": pathData,
        "has_ground": any(
            node.get("kind") == "ground" for node in graphData["nodes"]
        ),
    }
    return json.dumps(semantic, sort_keys=True, separators=(",", ":"))


def _featureLabel(values):
    return tuple(round(float(value), 8) for value in values)


def _featureDistance(first, second):
    difference = sum(
        abs(firstValue - secondValue)
        for firstValue, secondValue in zip(first, second)
    )
    return min(1.0, difference / 2.0)


def _minimumAssignment(costs):
    size = len(costs)
    if size == 0:
        return []
    rowPotential = [0.0] * (size + 1)
    columnPotential = [0.0] * (size + 1)
    matching = [0] * (size + 1)
    predecessor = [0] * (size + 1)
    for row in range(1, size + 1):
        matching[0] = row
        column = 0
        minimum = [float("inf")] * (size + 1)
        used = [False] * (size + 1)
        while True:
            used[column] = True
            matchedRow = matching[column]
            delta = float("inf")
            nextColumn = 0
            for candidate in range(1, size + 1):
                if used[candidate]:
                    continue
                reduced = (
                    costs[matchedRow - 1][candidate - 1]
                    - rowPotential[matchedRow]
                    - columnPotential[candidate]
                )
                if reduced < minimum[candidate]:
                    minimum[candidate] = reduced
                    predecessor[candidate] = column
                if minimum[candidate] < delta:
                    delta = minimum[candidate]
                    nextColumn = candidate
            for candidate in range(size + 1):
                if used[candidate]:
                    rowPotential[matching[candidate]] += delta
                    columnPotential[candidate] -= delta
                else:
                    minimum[candidate] -= delta
            column = nextColumn
            if matching[column] == 0:
                break
        while True:
            previousColumn = predecessor[column]
            matching[column] = matching[previousColumn]
            column = previousColumn
            if column == 0:
                break
    assignment = [0] * size
    for column in range(1, size + 1):
        assignment[matching[column] - 1] = column - 1
    return assignment


def _starDistance(first, second):
    size = len(first) + len(second)
    if size == 0:
        return 0.0
    forbidden = float(size + 1)
    costs = [[0.0] * size for _ in range(size)]
    for firstIndex in range(len(first)):
        for secondIndex in range(len(second)):
            edgeDistance = _featureDistance(
                first[firstIndex][0], second[secondIndex][0]
            )
            neighborDistance = _featureDistance(
                first[firstIndex][1], second[secondIndex][1]
            )
            costs[firstIndex][secondIndex] = 0.5 * (
                edgeDistance + neighborDistance
            )
        for dummyIndex in range(len(first)):
            column = len(second) + dummyIndex
            costs[firstIndex][column] = (
                1.0 if dummyIndex == firstIndex else forbidden
            )
    for secondIndex in range(len(second)):
        row = len(first) + secondIndex
        for candidate in range(len(second)):
            costs[row][candidate] = (
                1.0 if candidate == secondIndex else forbidden
            )
    assignment = _minimumAssignment(costs)
    return sum(costs[row][column] for row, column in enumerate(assignment))


def _attributedGraph(graphData):
    tensor = gmRcGraphToTensor(graphData)
    nodeFeatures = tuple(
        _featureLabel(row) for row in tensor.x.tolist()
    )
    edges = []
    adjacency = [[] for _ in nodeFeatures]
    for edgeIndex in range(0, tensor.edge_index.shape[1], 2):
        source = int(tensor.edge_index[0, edgeIndex])
        target = int(tensor.edge_index[1, edgeIndex])
        edgeFeature = _featureLabel(
            tensor.edge_attr[edgeIndex].tolist()
        )
        edges.append((source, target, edgeFeature))
        adjacency[source].append((edgeFeature, nodeFeatures[target]))
        adjacency[target].append((edgeFeature, nodeFeatures[source]))
    return nodeFeatures, tuple(edges), tuple(
        tuple(items) for items in adjacency
    )


def _initialGedAssignment(first, second):
    firstNodes, _, firstAdjacency = first
    secondNodes, _, secondAdjacency = second
    size = len(firstNodes) + len(secondNodes)
    forbidden = float(size * 4 + 1)
    costs = [[0.0] * size for _ in range(size)]
    for firstIndex, firstFeature in enumerate(firstNodes):
        for secondIndex, secondFeature in enumerate(secondNodes):
            nodeDistance = _featureDistance(
                firstFeature, secondFeature
            )
            starDistance = _starDistance(
                firstAdjacency[firstIndex],
                secondAdjacency[secondIndex],
            )
            costs[firstIndex][secondIndex] = (
                nodeDistance + 0.5 * starDistance
            )
        for dummyIndex in range(len(firstNodes)):
            column = len(secondNodes) + dummyIndex
            costs[firstIndex][column] = (
                1.0 + 0.5 * len(firstAdjacency[firstIndex])
                if dummyIndex == firstIndex else forbidden
            )
    for secondIndex in range(len(secondNodes)):
        row = len(firstNodes) + secondIndex
        for candidate in range(len(secondNodes)):
            costs[row][candidate] = (
                1.0 + 0.5 * len(secondAdjacency[secondIndex])
                if candidate == secondIndex else forbidden
            )
    return _minimumAssignment(costs)


def _mappedEditCost(first, second, assignment):
    firstNodes, firstEdges, _ = first
    secondNodes, secondEdges, _ = second
    firstCount = len(firstNodes)
    secondCount = len(secondNodes)
    nodeMapping = {}
    cost = 0.0
    matchedSecondNodes = set()
    for firstIndex in range(firstCount):
        secondIndex = assignment[firstIndex]
        if secondIndex < secondCount:
            nodeMapping[firstIndex] = secondIndex
            matchedSecondNodes.add(secondIndex)
            cost += _featureDistance(
                firstNodes[firstIndex], secondNodes[secondIndex]
            )
        else:
            cost += 1.0
    cost += secondCount - len(matchedSecondNodes)

    secondEdgesByNodes = {
        tuple(sorted((source, target))): feature
        for source, target, feature in secondEdges
    }
    matchedSecondEdges = set()
    for source, target, feature in firstEdges:
        if source not in nodeMapping or target not in nodeMapping:
            cost += 1.0
            continue
        mappedNodes = tuple(sorted((
            nodeMapping[source], nodeMapping[target]
        )))
        if mappedNodes not in secondEdgesByNodes:
            cost += 1.0
            continue
        cost += _featureDistance(
            feature, secondEdgesByNodes[mappedNodes]
        )
        matchedSecondEdges.add(mappedNodes)
    cost += len(secondEdgesByNodes) - len(matchedSecondEdges)
    return cost


def _improveGedAssignment(first, second, assignment):
    assignment = list(assignment)
    bestCost = _mappedEditCost(first, second, assignment)
    while True:
        bestSwap = None
        swapCost = bestCost
        for firstRow in range(len(assignment)):
            for secondRow in range(firstRow + 1, len(assignment)):
                assignment[firstRow], assignment[secondRow] = (
                    assignment[secondRow], assignment[firstRow]
                )
                candidateCost = _mappedEditCost(
                    first, second, assignment
                )
                assignment[firstRow], assignment[secondRow] = (
                    assignment[secondRow], assignment[firstRow]
                )
                if candidateCost < swapCost - 1e-12:
                    swapCost = candidateCost
                    bestSwap = (firstRow, secondRow)
        if bestSwap is None:
            return bestCost
        firstRow, secondRow = bestSwap
        assignment[firstRow], assignment[secondRow] = (
            assignment[secondRow], assignment[firstRow]
        )
        bestCost = swapCost


def _directionalApproximateGed(first, second):
    assignment = _initialGedAssignment(first, second)
    return _improveGedAssignment(first, second, assignment)


def _approximateGedGraphs(firstGraph, secondGraph):
    return min(
        _directionalApproximateGed(firstGraph, secondGraph),
        _directionalApproximateGed(secondGraph, firstGraph),
    )


# 计算确定性的 attributed bipartite GED 上界近似。
def gmRcGraphApproximateGed(first, second):
    firstGraph = _attributedGraph(first)
    secondGraph = _attributedGraph(second)
    return _approximateGedGraphs(firstGraph, secondGraph)


# 将 approximate GED 按较大图的 V+E 数归一化。
def gmRcGraphGedSimilarity(first, second):
    firstGraph = _attributedGraph(first)
    secondGraph = _attributedGraph(second)
    normalizer = max(
        len(firstGraph[0]) + len(firstGraph[1]),
        len(secondGraph[0]) + len(secondGraph[1]),
        1,
    )
    distance = min(
        1.0,
        _approximateGedGraphs(firstGraph, secondGraph) / normalizer,
    )
    return 1.0 - distance


def gmRcGraphSimilarity(first, second):
    return gmRcGraphGedSimilarity(first, second)


def gmRcGraphSimilarityVersion():
    return GM_RC_GED_VERSION


def gmRcGraphDistance(first, second):
    return 1.0 - gmRcGraphSimilarity(first, second)


class GmRcGraphEncoder(nn.Module):

    def __init__(
        self,
        inputDim=GM_RC_NODE_FEATURE_DIM,
        hiddenDim=128,
        outputDim=128,
        numLayers=2,
        dropout=0.1,
        edgeFeatureDim=GM_RC_EDGE_FEATURE_DIM,
    ):
        super().__init__()
        self.m_nodeEncoder = GraphSAGEEncoder(
            input_dim=inputDim,
            hidden_dim=hiddenDim,
            output_dim=hiddenDim,
            num_layers=numLayers,
            dropout=dropout,
            edge_feature_dim=edgeFeatureDim,
        )
        self.m_projection = nn.Linear(hiddenDim * 2 + 1, outputDim)

    def forward(self, x, edgeIndex, batch, edgeAttr):
        nodeEmbedding = self.m_nodeEncoder.encodeNodes(
            x, edgeIndex, edge_attr=edgeAttr
        )
        graphCount = int(batch.max().item()) + 1 if batch.numel() else 0
        means = nodeEmbedding.new_zeros((graphCount, nodeEmbedding.shape[1]))
        maximums = nodeEmbedding.new_full(
            (graphCount, nodeEmbedding.shape[1]), -torch.inf
        )
        counts = nodeEmbedding.new_zeros(graphCount)
        means.index_add_(0, batch, nodeEmbedding)
        counts.index_add_(0, batch, torch.ones_like(batch, dtype=x.dtype))
        means = means / counts.clamp(min=1.0).unsqueeze(1)
        maximums.scatter_reduce_(
            0,
            batch.unsqueeze(1).expand_as(nodeEmbedding),
            nodeEmbedding,
            reduce="amax",
            include_self=True,
        )
        sizeFeature = torch.log1p(counts).unsqueeze(1)
        embedding = self.m_projection(
            torch.cat([means, maximums, sizeFeature], dim=1)
        )
        return F.normalize(embedding, p=2, dim=1)


# 编码一个 batch 的 Gm+RC 图。
def encodeGmRcBatch(model, graphBatch, device):
    x, edgeIndex, edgeAttr, batch = graphBatch
    return model(
        x.to(device),
        edgeIndex.to(device),
        batch.to(device),
        edgeAttr.to(device),
    )
