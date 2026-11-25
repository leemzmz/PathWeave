import inspect
from pathlib import Path

import pytest

from Experiment.EncoderFewShotExperiment import ENCODER_FEW_SHOT_BACKENDS
from Experiment.encoder_backends import getGraphBackendSpec
from GraphSAGE.CircuitEncoder import buildCircuitEncoder
from GraphSAGE.SignalAttentionData import (
    SIGNAL_NODE_FEATURE_SCHEMA,
    VPATH_SIGNAL_EDGE_FEATURE_SCHEMA,
)
from GraphSAGE.graph_data import EDGE_FEATURE_SCHEMA, netlist_to_graph_tensor


ROOT = Path(__file__).resolve().parent
SAMPLE = ROOT / "_data" / "topo_NMCCB" / "core_05.cir"
# 已被移除的路径语义描述符：出现在任一活跃 schema 中即为泄漏。
FORBIDDEN_NAMES = (
    "main_position",
    "is_main_to_main",
    "main_span",
    "path_device_count",
    "path_net_count",
    "path_res_count",
    "path_cap_count",
    "path_pmos_count",
    "path_nmos_count",
    "path_gate_entry_count",
    "path_diode_count",
    "path_block_count",
    "is_key_node",
    "is_netg_key",
    "is_netg_diode",
    "is_netg_bias",
    "is_vpath",
    "is_hpath",
    "is_hpath_directed",
    "HPATH_MEMBER",
    "HPATH_SOURCE",
    "HPATH_TARGET",
)
ALLOWED_SIGNAL_NAMES = {
    "VPATH_COMEMBER",
    "HLINK",
    "HLINK_GM",
    "HLINK_CM",
    "HLINK_RC",
    "HLINK_OL",
    "HLINK_OT",
    "HLINK_DIRECTED",
    "DELTA_POSITION",
}


# 正式矩阵必须恰好是 Netlist / vpath 注意力 / pathweave 注意力。
def testActiveBackendSetIsExactlyThree():
    assert ENCODER_FEW_SHOT_BACKENDS == (
        "graph_netlist",
        "graph_vpath_attention",
        "graph_path_attention",
    )


# 活跃 backend 只能使用 physical 顶点，且 node/edge schema 不含路径描述符。
@pytest.mark.parametrize("backend", ENCODER_FEW_SHOT_BACKENDS)
def testActiveBackendUsesPhysicalVerticesOnly(backend):
    spec = getGraphBackendSpec(backend)
    graph = spec.build_tensor(SAMPLE)
    netlist = netlist_to_graph_tensor(SAMPLE)
    tensor = getattr(graph, "m_graph", graph)
    assert tensor.x.shape[0] == netlist.x.shape[0]
    assert tensor.x.shape[1] == len(SIGNAL_NODE_FEATURE_SCHEMA)
    assert spec.node_feature_schema == SIGNAL_NODE_FEATURE_SCHEMA
    assert spec.edge_feature_schema == EDGE_FEATURE_SCHEMA
    for name in FORBIDDEN_NAMES:
        assert name not in spec.node_feature_schema
        assert name not in spec.edge_feature_schema


# signal 特征必须落在允许集合内，且与声明的 schema 维度一致。
@pytest.mark.parametrize("backend", ENCODER_FEW_SHOT_BACKENDS)
def testActiveBackendSignalFeaturesAreAudited(backend):
    spec = getGraphBackendSpec(backend)
    graph = spec.build_tensor(SAMPLE)
    if not hasattr(graph, "m_signalFeatureSchema"):
        return
    signal = list(graph.m_signalFeatureSchema)
    assert signal, "signal backend must declare its feature schema"
    assert set(signal) <= ALLOWED_SIGNAL_NAMES
    for name in FORBIDDEN_NAMES:
        assert name not in signal
    assert graph.m_signalAttr.shape[1] == len(signal)
    if backend == "graph_vpath_attention":
        assert signal == VPATH_SIGNAL_EDGE_FEATURE_SCHEMA
        assert "HLINK" not in graph.m_signalLabels
        assert "VPATH_COMEMBER" in graph.m_signalLabels
    else:
        assert "VPATH_COMEMBER" in graph.m_signalLabels
        assert "HLINK" in graph.m_signalLabels


# 构造器只能读取网表路径，避免任何 target 侧信息进入输入表示。
def testActiveBackendBuildersReadOnlyTheNetlistPath():
    for backend in ENCODER_FEW_SHOT_BACKENDS:
        spec = getGraphBackendSpec(backend)
        parameters = list(inspect.signature(spec.build_tensor).parameters)
        assert parameters == ["path"], backend


# 分层交替编码器的 signal 宽度必须等于数据层声明的 signal 维度。
@pytest.mark.parametrize("backend", ENCODER_FEW_SHOT_BACKENDS)
def testActiveBackendEncoderMatchesSignalSchema(backend):
    spec = getGraphBackendSpec(backend)
    model = (spec.build_encoder or buildCircuitEncoder)(
        "graphsage",
        inputDim=spec.node_feature_dim,
        edgeFeatureDim=len(spec.edge_feature_schema),
        dropout=0,
    )
    if not hasattr(model, "m_signalAttention"):
        return
    graph = spec.build_tensor(SAMPLE)
    assert model.m_signalDim == len(graph.m_signalFeatureSchema)
    for block in model.m_signalAttention:
        assert block.m_key.in_features == 128 + model.m_signalDim
        assert block.m_value.in_features == 128 + model.m_signalDim
        assert block.m_query.out_features == 32
        assert block.m_output.out_features == 128
