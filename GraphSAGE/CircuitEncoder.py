import torch
from torch import nn
from torch.nn import functional as F

from GraphSAGE.graphsage import GRAPH_SAGE_MESSAGE_TYPE, GraphSAGEEncoder


CIRCUIT_BACKBONES = ("graphsage", "gat", "gin")


# Describe operator choices without changing the existing GraphSAGE checkpoint.
def describeCircuitBackbone(backbone):
    if backbone not in CIRCUIT_BACKBONES:
        raise ValueError(f"unknown circuit backbone: {backbone}")
    if backbone == "graphsage":
        return {"message_type": GRAPH_SAGE_MESSAGE_TYPE}
    import torch_geometric

    common = {
        "library": "torch_geometric",
        "library_version": torch_geometric.__version__,
        "pooling": "mean",
        "normalization": "l2",
    }
    if backbone == "gat":
        return {
            **common, "message_type": "pyg_gat_edge_attention",
            "heads": 4, "concat": True, "add_self_loops": True,
            "fill_value": 0.0, "attention_dropout": 0.0, "residual": False,
        }
    return {
        **common, "message_type": "pyg_gine_sum",
        "train_eps": True, "initial_eps": 0.0,
        "mlp": "linear_relu_linear", "aggregation": "sum",
    }


class CircuitGraphEncoder(nn.Module):

    # Keep graph tensors, readout and output size shared across edge-aware operators.
    def __init__(
        self, backbone, inputDim, hiddenDim, outputDim, numLayers,
        dropout, edgeFeatureDim,
    ):
        super().__init__()
        from torch_geometric.nn import GATConv, GINEConv

        self.m_inputDim = inputDim
        self.m_edgeFeatureDim = edgeFeatureDim
        self.m_layers = nn.ModuleList()
        self.m_dropout = nn.Dropout(dropout)
        dims = [inputDim] + [hiddenDim] * (numLayers - 1) + [outputDim]
        for inDim, outDim in zip(dims[:-1], dims[1:]):
            if backbone == "gat":
                layer = GATConv(
                    inDim, outDim // 4, heads=4, concat=True,
                    edge_dim=edgeFeatureDim, add_self_loops=True,
                    fill_value=0.0, dropout=0.0, residual=False,
                )
            else:
                mlp = nn.Sequential(
                    nn.Linear(inDim, outDim), nn.ReLU(),
                    nn.Linear(outDim, outDim),
                )
                layer = GINEConv(
                    mlp, train_eps=True, eps=0.0, edge_dim=edgeFeatureDim,
                )
            self.m_layers.append(layer)

    # Require the original multi-dimensional port/relation features.
    def encodeNodes(self, x, edge_index, edge_attr=None):
        if x.ndim != 2 or x.shape[1] != self.m_inputDim:
            raise ValueError("circuit node feature dimension mismatch")
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("circuit edge index must have shape (2, E)")
        if edge_attr is None or edge_attr.shape != (
            edge_index.shape[1], self.m_edgeFeatureDim
        ):
            raise ValueError("circuit encoder requires matching edge features")
        for index, layer in enumerate(self.m_layers):
            x = layer(x, edge_index, edge_attr)
            if index + 1 < len(self.m_layers):
                x = self.m_dropout(F.relu(x))
        return x

    # Match the original GraphSAGE mean readout and L2 normalization.
    def poolNodes(self, x, batch):
        if batch.ndim != 1 or len(batch) != len(x):
            raise ValueError("circuit batch must identify every node")
        count = int(batch.max().item()) + 1 if batch.numel() else 0
        embeddings = x.new_zeros((count, x.shape[1]))
        counts = x.new_zeros(count)
        embeddings.index_add_(0, batch, x)
        counts.index_add_(0, batch, torch.ones_like(batch, dtype=x.dtype))
        return F.normalize(embeddings / counts.clamp(min=1).unsqueeze(1), dim=1)

    # Expose the existing graph batch calling convention.
    def forward(self, x, edge_index, batch, edge_attr=None):
        return self.poolNodes(self.encodeNodes(x, edge_index, edge_attr), batch)


# Preserve the default class and parameter initialization order exactly.
def buildCircuitEncoder(
    backbone, inputDim, hiddenDim=128, outputDim=128, numLayers=2,
    dropout=0.1, edgeFeatureDim=0,
):
    describeCircuitBackbone(backbone)
    if min(inputDim, hiddenDim, outputDim, numLayers) < 1:
        raise ValueError("circuit encoder dimensions and layers must be positive")
    if not 0 <= dropout < 1:
        raise ValueError("circuit dropout must be in [0, 1)")
    if backbone == "graphsage":
        return GraphSAGEEncoder(
            input_dim=inputDim, hidden_dim=hiddenDim, output_dim=outputDim,
            num_layers=numLayers, dropout=dropout,
            edge_feature_dim=edgeFeatureDim,
        )
    if edgeFeatureDim < 1:
        raise ValueError("GAT/GIN comparisons require edge features")
    if backbone == "gat" and (hiddenDim % 4 or outputDim % 4):
        raise ValueError("GAT total width must be divisible by four heads")
    return CircuitGraphEncoder(
        backbone, inputDim, hiddenDim, outputDim, numLayers, dropout, edgeFeatureDim,
    )
