from dataclasses import dataclass, field
from pathlib import Path

import torch


from GraphSAGE.graph_data import (
    GraphTensor,
    batch_graphs,
    ckt_graph_to_tensor,
    extractNetlistManager,
)


# 已发布 physical 特征契约，不依赖 CktPath 的运行时类型。
SIGNAL_NODE_FEATURE_SCHEMA = [
    "Bias", "In", "Out", "Net", "VDD", "GND", "Res", "Cap", "PMOS", "NMOS",
]
HLINK_TYPE_NAMES = ("GM", "CM", "RC", "OL", "OT")
# 完整 signal 边：vpath 共现 + hlink 类型。
SIGNAL_ONLY_EDGE_FEATURE_SCHEMA = [
    "VPATH_COMEMBER",
    "HLINK",
    *[f"HLINK_{name}" for name in HLINK_TYPE_NAMES],
    "HLINK_DIRECTED",
    "DELTA_POSITION",
]
# vpath-only signal 边：自带独立 schema，不含任何 hlink 列。
VPATH_SIGNAL_EDGE_FEATURE_SCHEMA = [
    "VPATH_COMEMBER",
    "DELTA_POSITION",
]
SIGNAL_NODE_FEATURE_DIM = len(SIGNAL_NODE_FEATURE_SCHEMA)
SIGNAL_ONLY_EDGE_FEATURE_DIM = len(SIGNAL_ONLY_EDGE_FEATURE_SCHEMA)
VPATH_SIGNAL_EDGE_FEATURE_DIM = len(VPATH_SIGNAL_EDGE_FEATURE_SCHEMA)
# pathweave 注意力：完整 signal 边 + 32-D 分层交替编码。
PATH_ATTENTION_MODEL_CONTRACT = "graph_path_attention_dim32_v1_emb128"
# vpath 注意力：独立 2-D signal 特征 + 32-D 分层交替编码。
VPATH_ATTENTION_MODEL_CONTRACT = "graph_vpath_attention_dim32_v1_emb128"
# 所有 signal 注意力的宽度统一为 32。
SIGNAL_ATTENTION_DIM32 = 32


@dataclass
class SignalAttentionBranchTensor:
    m_graph: GraphTensor
    m_signalIndex: torch.Tensor
    m_signalAttr: torch.Tensor
    m_signalLabels: list
    m_signalFeatureSchema: list = field(default_factory=list)


# 只接受显式声明的 physical 类型，保持与 Netlist 相同的顶点集合。
def _buildPhysicalBase(graph):
    unsupported = sorted(
        f"{vertex.name}:{vertex.vtype.name}"
        for vertex in graph.vertices.values()
        if vertex.vtype.name not in SIGNAL_NODE_FEATURE_SCHEMA
    )
    if unsupported:
        raise ValueError(
            "path attention graph found unsupported physical vertices: "
            + ", ".join(unsupported)
        )
    base = ckt_graph_to_tensor(graph)
    indexes = [
        base.node_feature_schema.index(name)
        for name in SIGNAL_NODE_FEATURE_SCHEMA
    ]
    base.x = base.x[:, indexes]
    base.node_feature_schema = SIGNAL_NODE_FEATURE_SCHEMA
    return base


# 按给定 schema 生成一条 signal 边属性；schema 不含的列自然保持为零。
def _makeSignalAttr(
    reference,
    schema,
    edgeType,
    hlinkType=None,
    isDirected=False,
    deltaPosition=0.0,
):
    attr = reference.new_zeros((1, len(schema)))
    attr[0, schema.index(edgeType)] = 1.0
    if hlinkType is not None:
        attr[0, schema.index(f"HLINK_{hlinkType}")] = 1.0
        attr[0, schema.index("HLINK_DIRECTED")] = 1.0 if isDirected else 0.0
    attr[0, schema.index("DELTA_POSITION")] = float(deltaPosition)
    return attr


# signal 端点必须落在 physical 顶点集合内，否则拒绝构造。
def _resolveIndex(vertexToIndex, vertex, context):
    index = vertexToIndex.get(vertex)
    if index is None:
        raise ValueError(
            f"path attention {context} anchor is not a physical vertex: "
            f"{getattr(vertex, 'name', vertex)}"
        )
    return index


def _vertexIndexes(graph):
    vertices = list(graph.vertices.values())
    return {vertex: index for index, vertex in enumerate(vertices)}


# vpath 共现边：同一电流通路内的成员对，双向展开并带归一化位置差。
def _collectVpathEdges(manager, vertexToIndex, reference, schema):
    edges = []
    edgeAttrs = []
    edgeLabels = []
    for path in manager.getVpaths():
        denominator = max(1, len(path.nodes) - 1)
        indexes = [
            _resolveIndex(vertexToIndex, node, "vpath")
            for node in path.nodes
        ]
        for first in range(len(indexes)):
            for second in range(first + 1, len(indexes)):
                attr = _makeSignalAttr(
                    reference,
                    schema,
                    "VPATH_COMEMBER",
                    deltaPosition=abs(first - second) / denominator,
                )
                edges.extend((
                    (indexes[first], indexes[second]),
                    (indexes[second], indexes[first]),
                ))
                edgeAttrs.extend((attr, attr.clone()))
                edgeLabels.extend(("VPATH_COMEMBER", "VPATH_COMEMBER"))
    return edges, edgeAttrs, edgeLabels


# hlink 类型边：VpathLink 的 anchor 对，有向单向、无向双向。
def _collectHlinkEdges(manager, vertexToIndex, reference, schema):
    from GraphSAGE.PathExtraction import VPATH_LINK_TYPE_LABELS

    edges = []
    edgeAttrs = []
    edgeLabels = []
    for link in manager.getVpathLinks():
        hlinkType = VPATH_LINK_TYPE_LABELS[link.linkType]
        startIndexes = [
            _resolveIndex(vertexToIndex, endpoint.anchor, "hlink")
            for endpoint in link.startEndpoints
        ]
        endIndexes = [
            _resolveIndex(vertexToIndex, endpoint.anchor, "hlink")
            for endpoint in link.endEndpoints
        ]
        attr = _makeSignalAttr(
            reference, schema, "HLINK", hlinkType, link.isDirected
        )
        for start in startIndexes:
            for end in endIndexes:
                edges.append((start, end))
                edgeAttrs.append(attr.clone())
                edgeLabels.append("HLINK")
                if not link.isDirected:
                    edges.append((end, start))
                    edgeAttrs.append(attr.clone())
                    edgeLabels.append("HLINK")
    return edges, edgeAttrs, edgeLabels


def _mergeSignalParts(parts):
    edges = []
    edgeAttrs = []
    edgeLabels = []
    for partEdges, partAttrs, partLabels in parts:
        edges.extend(partEdges)
        edgeAttrs.extend(partAttrs)
        edgeLabels.extend(partLabels)
    return edges, edgeAttrs, edgeLabels


def _branchTensor(base, edges, edgeAttrs, edgeLabels, schema):
    if edges:
        signalIndex = torch.tensor(
            edges, dtype=torch.long
        ).t().contiguous()
        signalAttr = torch.cat(edgeAttrs, dim=0)
    else:
        signalIndex = base.x.new_zeros((2, 0), dtype=torch.long)
        signalAttr = base.x.new_zeros((0, len(schema)))
    return SignalAttentionBranchTensor(
        m_graph=base,
        m_signalIndex=signalIndex,
        m_signalAttr=signalAttr,
        m_signalLabels=edgeLabels,
        m_signalFeatureSchema=list(schema),
    )


# pathweave 注意力：physical 图与完整 signal 边分开保存。
def signalAttentionBranchesToTensor(manager):
    graph = manager.getGraph()
    base = _buildPhysicalBase(graph)
    vertexToIndex = _vertexIndexes(graph)
    edges, edgeAttrs, edgeLabels = _mergeSignalParts((
        _collectVpathEdges(
            manager, vertexToIndex, base.edge_attr,
            SIGNAL_ONLY_EDGE_FEATURE_SCHEMA,
        ),
        _collectHlinkEdges(
            manager, vertexToIndex, base.edge_attr,
            SIGNAL_ONLY_EDGE_FEATURE_SCHEMA,
        ),
    ))
    return _branchTensor(
        base, edges, edgeAttrs, edgeLabels,
        SIGNAL_ONLY_EDGE_FEATURE_SCHEMA,
    )


# vpath 注意力：同一 physical 图，只附加 vpath 共现边。
def vpathAttentionBranchesToTensor(manager):
    graph = manager.getGraph()
    base = _buildPhysicalBase(graph)
    vertexToIndex = _vertexIndexes(graph)
    edges, edgeAttrs, edgeLabels = _collectVpathEdges(
        manager, vertexToIndex, base.edge_attr,
        VPATH_SIGNAL_EDGE_FEATURE_SCHEMA,
    )
    return _branchTensor(
        base, edges, edgeAttrs, edgeLabels, VPATH_SIGNAL_EDGE_FEATURE_SCHEMA
    )


def netlistToSignalAttentionBranches(path):
    return signalAttentionBranchesToTensor(extractNetlistManager(Path(path)))


def netlistToVpathAttentionBranches(path):
    return vpathAttentionBranchesToTensor(extractNetlistManager(Path(path)))


# signal 索引需要按图逐个偏移，physical 侧沿用通用 batch。
def batchSignalAttentionBranches(graphs):
    physicalBatch = batch_graphs([graph.m_graph for graph in graphs])
    indexParts = []
    attrParts = []
    offset = 0
    for graph in graphs:
        indexParts.append(graph.m_signalIndex + offset)
        attrParts.append(graph.m_signalAttr)
        offset += graph.m_graph.x.shape[0]
    if indexParts and indexParts[0].numel():
        signalIndex = torch.cat(indexParts, dim=1)
    else:
        signalIndex = torch.zeros((2, 0), dtype=torch.long)
    if attrParts and attrParts[0].numel():
        signalAttr = torch.cat(attrParts, dim=0)
    else:
        width = graphs[0].m_signalAttr.shape[1] if graphs else 0
        signalAttr = torch.zeros((0, width))
    return (physicalBatch, signalIndex, signalAttr)


# 分层交替编码的批处理入口，物理图与 signal 边分别送入模型。
def encodeSignalAttentionBranches(model, graphBatch, device):
    (x, edge_index, edge_attr, batch), signalIndex, signalAttr = graphBatch
    return model(
        x.to(device),
        edge_index.to(device),
        batch.to(device),
        edge_attr.to(device),
        signalIndex.to(device),
        signalAttr.to(device),
    )
