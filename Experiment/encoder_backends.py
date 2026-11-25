from dataclasses import dataclass

from GraphSAGE.SignalAttentionData import (
    PATH_ATTENTION_MODEL_CONTRACT,
    SIGNAL_NODE_FEATURE_SCHEMA,
    VPATH_ATTENTION_MODEL_CONTRACT,
    VPATH_SIGNAL_EDGE_FEATURE_DIM,
    batchSignalAttentionBranches,
    encodeSignalAttentionBranches,
    netlistToSignalAttentionBranches,
    netlistToVpathAttentionBranches,
)
from GraphSAGE.SignalAttentionEncoder import (
    buildLayeredSignalAttentionEncoder,
    buildLayeredSignalAttentionEncoderWithSignalDim,
)
from GraphSAGE.graph_data import (
    EDGE_FEATURE_SCHEMA,
    GRAPH_NETLIST_MODEL_CONTRACT,
    NETLIST_NODE_FEATURE_DIM,
    NETLIST_NODE_FEATURE_SCHEMA,
    batch_graphs,
    netlist_to_graph_tensor,
)
from GraphSAGE.graphsage import encode_graph_batch


@dataclass(frozen=True)
class GraphBackendSpec:
    node_feature_dim: int
    node_feature_schema: list
    edge_feature_schema: list
    model_contract: str
    build_tensor: object
    batch_graphs: object = batch_graphs
    encode_batch: object = encode_graph_batch
    build_encoder: object = None


GRAPH_BACKEND_SPECS = {
    "graph_netlist": GraphBackendSpec(
        node_feature_dim=NETLIST_NODE_FEATURE_DIM,
        node_feature_schema=NETLIST_NODE_FEATURE_SCHEMA,
        edge_feature_schema=EDGE_FEATURE_SCHEMA,
        model_contract=GRAPH_NETLIST_MODEL_CONTRACT,
        build_tensor=netlist_to_graph_tensor,
    ),
    "graph_vpath_attention": GraphBackendSpec(
        node_feature_dim=len(SIGNAL_NODE_FEATURE_SCHEMA),
        node_feature_schema=SIGNAL_NODE_FEATURE_SCHEMA,
        edge_feature_schema=list(EDGE_FEATURE_SCHEMA),
        model_contract=VPATH_ATTENTION_MODEL_CONTRACT,
        build_tensor=netlistToVpathAttentionBranches,
        batch_graphs=batchSignalAttentionBranches,
        encode_batch=encodeSignalAttentionBranches,
        build_encoder=buildLayeredSignalAttentionEncoderWithSignalDim(
            VPATH_SIGNAL_EDGE_FEATURE_DIM
        ),
    ),
    "graph_path_attention": GraphBackendSpec(
        node_feature_dim=len(SIGNAL_NODE_FEATURE_SCHEMA),
        node_feature_schema=SIGNAL_NODE_FEATURE_SCHEMA,
        edge_feature_schema=list(EDGE_FEATURE_SCHEMA),
        model_contract=PATH_ATTENTION_MODEL_CONTRACT,
        build_tensor=netlistToSignalAttentionBranches,
        batch_graphs=batchSignalAttentionBranches,
        encode_batch=encodeSignalAttentionBranches,
        build_encoder=buildLayeredSignalAttentionEncoder,
    ),
}


def getGraphBackendSpec(encoderBackend):
    if encoderBackend not in GRAPH_BACKEND_SPECS:
        raise ValueError(f"unknown encoder backend: {encoderBackend}")
    return GRAPH_BACKEND_SPECS[encoderBackend]
