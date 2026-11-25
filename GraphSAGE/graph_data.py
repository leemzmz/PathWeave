from dataclasses import dataclass
from pathlib import Path

import torch

# 模型输入契约独立于网表提取器；顺序必须与既有 checkpoint 一致。
NETLIST_NODE_FEATURE_SCHEMA = [
    "Bias", "In", "Out", "Net", "VDD", "GND", "Res", "Cap", "PMOS", "NMOS",
]
NODE_FEATURE_SCHEMA = [*NETLIST_NODE_FEATURE_SCHEMA, "Block", "is_key_node"]
NODE_FEATURE_DIM = len(NODE_FEATURE_SCHEMA)
NETLIST_NODE_FEATURE_DIM = len(NETLIST_NODE_FEATURE_SCHEMA)
EDGE_FEATURE_SCHEMA = ["Norm", "D", "G", "S"]
EDGE_FEATURE_DIM = len(EDGE_FEATURE_SCHEMA)
GRAPH_NETLIST_MODEL_CONTRACT = "graph_netlist_physical10_gated_edge_emb128"


@dataclass
class GraphTensor:
    x: torch.Tensor
    edge_index: torch.Tensor
    edge_attr: torch.Tensor
    node_labels: list = None
    edge_labels: list = None
    node_feature_schema: list = None
    edge_feature_schema: list = None


# 移除 MOS body；当前模型不支持 BJT。
def _edgePortLabel(edge):
    if edge.dev_port.name == "B":
        return None
    return edge.dev_port.name


def ckt_graph_to_tensor(graph, key_vertices=None, included_vertices=None):
    unsupported = [
        vertex.name
        for vertex in graph.vertices.values()
        if vertex.vtype.name in {"NPN", "PNP"}
    ]
    if unsupported:
        raise ValueError(
            "graph_netlist does not support BJT devices: "
            + ", ".join(unsupported)
        )

    key_vertices = set(key_vertices or [])
    included_vertices = (
        None if included_vertices is None else set(included_vertices)
    )
    key_vertex_keys = {
        (vertex.name, vertex.vtype)
        for vertex in key_vertices
    }
    vertices = [
        vertex
        for vertex in graph.vertices.values()
        if included_vertices is None or vertex in included_vertices
    ]
    vertex_to_idx = {v: i for i, v in enumerate(vertices)}

    features = []
    node_labels = []
    for vertex in vertices:
        feat = [0.0] * NODE_FEATURE_DIM
        if vertex.vtype.name in NODE_FEATURE_SCHEMA[:-1]:
            feat[NODE_FEATURE_SCHEMA.index(vertex.vtype.name)] = 1.0
        feat[NODE_FEATURE_SCHEMA.index("is_key_node")] = 1.0 if (
            vertex in key_vertices or (vertex.name, vertex.vtype) in key_vertex_keys
        ) else 0.0
        features.append(feat)
        node_labels.append(f"{vertex.name}:{vertex.vtype.name}")

    edges = []
    edge_attrs = []
    edge_labels = []
    for edge in graph.edges:
        if edge.net not in vertex_to_idx or edge.dev not in vertex_to_idx:
            continue
        port_label = _edgePortLabel(edge)
        if port_label is None:
            continue
        src = vertex_to_idx[edge.net]
        dst = vertex_to_idx[edge.dev]
        attr = [0.0] * EDGE_FEATURE_DIM
        attr[EDGE_FEATURE_SCHEMA.index(port_label)] = 1.0

        edges.append((src, dst))
        edge_attrs.append(attr)
        edge_labels.append(port_label)
        edges.append((dst, src))
        edge_attrs.append(attr)
        edge_labels.append(port_label)

    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float32)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, EDGE_FEATURE_DIM), dtype=torch.float32)

    return GraphTensor(
        x=torch.tensor(features, dtype=torch.float32),
        edge_index=edge_index,
        edge_attr=edge_attr,
        node_labels=node_labels,
        edge_labels=edge_labels,
        node_feature_schema=NODE_FEATURE_SCHEMA,
        edge_feature_schema=EDGE_FEATURE_SCHEMA,
    )


# 从网表提取项目统一配置下的路径管理器。
def extractNetlistManager(path):
    from GraphSAGE.PathExtraction import PathExtractConfig, PathExtractManager

    path = Path(path)
    config = PathExtractConfig()
    config.ignoreMosBulk = True
    config.collapseResChains = False
    return PathExtractManager().run(
        path.read_text(encoding="utf-8"),
        sourceId=str(path),
        config=config,
    )


def netlist_to_graph_tensor(path):
    manager = extractNetlistManager(path)
    base = ckt_graph_to_tensor(manager.getGraph())
    indexes = [
        base.node_feature_schema.index(name)
        for name in NETLIST_NODE_FEATURE_SCHEMA
    ]
    base.x = base.x[:, indexes]
    base.node_feature_schema = NETLIST_NODE_FEATURE_SCHEMA
    return base


def batch_graphs(graphs):
    xs = []
    edge_indices = []
    edge_attrs = []
    batch = []
    offset = 0

    for graph_idx, graph in enumerate(graphs):
        xs.append(graph.x)
        batch.append(
            torch.full(
                (graph.x.shape[0],),
                graph_idx,
                dtype=torch.long,
            )
        )

        if graph.edge_index.numel() > 0:
            edge_indices.append(graph.edge_index + offset)
            edge_attrs.append(graph.edge_attr)

        offset += graph.x.shape[0]

    x = torch.cat(xs, dim=0)
    batch = torch.cat(batch, dim=0)

    if edge_indices:
        edge_index = torch.cat(edge_indices, dim=1)
        edge_attr = torch.cat(edge_attrs, dim=0)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_dim = graphs[0].edge_attr.shape[1] if graphs else EDGE_FEATURE_DIM
        edge_attr = torch.empty((0, edge_dim), dtype=torch.float32)

    return x, edge_index, edge_attr, batch
