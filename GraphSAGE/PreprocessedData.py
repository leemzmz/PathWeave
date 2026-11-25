"""发布模型输入张量；仅导出阶段需要 CktPath，读取阶段只处理张量。"""

import hashlib
import json
from pathlib import Path

import torch

from GraphSAGE.graph_data import GraphTensor, NETLIST_NODE_FEATURE_SCHEMA, EDGE_FEATURE_SCHEMA
from GraphSAGE.SignalAttentionData import (
    SignalAttentionBranchTensor, SIGNAL_NODE_FEATURE_SCHEMA,
    SIGNAL_ONLY_EDGE_FEATURE_SCHEMA, VPATH_SIGNAL_EDGE_FEATURE_SCHEMA,
)

FORMAT_VERSION = "pathweave_graph_tensors_v1"
BACKENDS = ("graph_netlist", "graph_vpath_attention", "graph_path_attention")


# 对数据和模型文件生成内容摘要。
def hashFile(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# 将公开输入对象转为仅含 tensor、列表和字符串的安全序列化记录。
def packGraph(graph, backend):
    base = graph if backend == "graph_netlist" else graph.m_graph
    result = {"x": base.x.cpu(), "edge_index": base.edge_index.cpu(),
              "edge_attr": base.edge_attr.cpu(), "node_schema": base.node_feature_schema,
              "edge_schema": base.edge_feature_schema}
    if backend != "graph_netlist":
        result.update(signal_index=graph.m_signalIndex.cpu(), signal_attr=graph.m_signalAttr.cpu(),
                      signal_schema=graph.m_signalFeatureSchema)
    return result


# 检查张量形状、类型、特征契约、索引范围和有限值。
def _validateEdges(index, attr, count, width):
    if index.dtype != torch.long or index.ndim != 2 or index.shape[0] != 2:
        raise ValueError("invalid edge index")
    if attr.dtype != torch.float32 or tuple(attr.shape) != (index.shape[1], width):
        raise ValueError("invalid edge features")
    if not torch.isfinite(attr).all() or (index.numel() and (index.min() < 0 or index.max() >= count)):
        raise ValueError("invalid edge values or endpoints")


# 只恢复公开模型输入对象，不执行网表提取或任意 pickle 类加载。
def unpackGraph(value, backend):
    if backend not in BACKENDS:
        raise ValueError("unknown backend")
    schema = NETLIST_NODE_FEATURE_SCHEMA if backend == "graph_netlist" else SIGNAL_NODE_FEATURE_SCHEMA
    x = value["x"]
    if value["node_schema"] != schema or value["edge_schema"] != EDGE_FEATURE_SCHEMA:
        raise ValueError("physical schema mismatch")
    if x.dtype != torch.float32 or x.ndim != 2 or x.shape[0] < 1 or x.shape[1] != len(schema) or not torch.isfinite(x).all():
        raise ValueError("invalid node features")
    _validateEdges(value["edge_index"], value["edge_attr"], len(x), len(EDGE_FEATURE_SCHEMA))
    base = GraphTensor(x=x, edge_index=value["edge_index"], edge_attr=value["edge_attr"],
                       node_feature_schema=list(schema), edge_feature_schema=list(EDGE_FEATURE_SCHEMA))
    if backend == "graph_netlist":
        return base
    signalSchema = VPATH_SIGNAL_EDGE_FEATURE_SCHEMA if backend == "graph_vpath_attention" else SIGNAL_ONLY_EDGE_FEATURE_SCHEMA
    if value["signal_schema"] != signalSchema:
        raise ValueError("signal schema mismatch")
    _validateEdges(value["signal_index"], value["signal_attr"], len(x), len(signalSchema))
    return SignalAttentionBranchTensor(base, value["signal_index"], value["signal_attr"], [], list(signalSchema))


# 验证发布包哈希、样本覆盖和输入身份，然后恢复三种表示。
def loadPreprocessedGraphs(directory, cases):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["format"] != FORMAT_VERSION or manifest["graphs_sha256"] != hashFile(directory / "graphs.pt"):
        raise ValueError("preprocessed format or checksum mismatch")
    if manifest["core_sha256"] != {c.name: hashFile(c.netlistPath) for c in cases}:
        raise ValueError("preprocessed circuit input mismatch")
    if manifest["target_sha256"] != {c.topology: hashFile(c.targetPath) for c in cases}:
        raise ValueError("preprocessed target input mismatch")
    payload = torch.load(directory / "graphs.pt", map_location="cpu", weights_only=True)
    names = {c.name for c in cases}
    if set(payload) != set(BACKENDS) or any(set(payload[b]) != names for b in BACKENDS):
        raise ValueError("preprocessed sample coverage mismatch")
    graphs = {b: {name: unpackGraph(value, b) for name, value in payload[b].items()} for b in BACKENDS}
    return manifest, graphs
