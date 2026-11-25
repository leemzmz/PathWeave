import torch
import torch.nn as nn
import torch.nn.functional as F


GRAPH_SAGE_MESSAGE_TYPE = "edge_attr_concat_gated_mean"


class GraphSAGELayer(nn.Module):

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.self_linear = nn.Linear(in_dim, out_dim)
        self.neigh_linear = nn.Linear(in_dim, out_dim)

    def forward(self, x, edge_index, edge_attr=None, activate=True):
        if edge_index.numel() == 0:
            neigh = torch.zeros_like(x)
        else:
            src, dst = edge_index
            neigh_sum = torch.zeros_like(x)
            degree = torch.zeros(
                x.shape[0],
                device=x.device,
                dtype=x.dtype,
            )

            neigh_sum.index_add_(0, dst, x[src])
            degree.index_add_(0, dst, torch.ones_like(dst, dtype=x.dtype))
            neigh = neigh_sum / degree.clamp(min=1.0).unsqueeze(1)

        out = self.self_linear(x) + self.neigh_linear(neigh)
        if activate:
            return F.relu(out)
        return out


class EdgeGraphSAGELayer(nn.Module):

    def __init__(self, in_dim, out_dim, edge_feature_dim):
        super().__init__()
        self.edge_feature_dim = edge_feature_dim
        self.self_linear = nn.Linear(in_dim, out_dim)
        self.message_linear = nn.Linear(in_dim + edge_feature_dim, out_dim)
        self.gate_linear = nn.Linear(edge_feature_dim, out_dim)

    def forward(self, x, edge_index, edge_attr=None, activate=True):
        if edge_index.numel() == 0:
            neigh = x.new_zeros((x.shape[0], self.self_linear.out_features))
        else:
            src, dst = edge_index
            if edge_attr is None:
                edge_attr = x.new_zeros((src.shape[0], self.edge_feature_dim))
            edge_attr = edge_attr.to(x.device)
            msg_input = torch.cat([x[src], edge_attr], dim=1)
            messages = self.message_linear(msg_input)
            gates = torch.sigmoid(self.gate_linear(edge_attr))
            messages = messages * gates

            neigh_sum = x.new_zeros((x.shape[0], messages.shape[1]))
            gate_sum = x.new_zeros((x.shape[0], messages.shape[1]))

            neigh_sum.index_add_(0, dst, messages)
            gate_sum.index_add_(0, dst, gates)

            neigh = neigh_sum / gate_sum.clamp(min=1e-6)

        out = self.self_linear(x) + neigh
        if activate:
            return F.relu(out)
        return out


class GraphSAGEEncoder(nn.Module):

    def __init__(
        self,
        input_dim,
        hidden_dim=128,
        output_dim=128,
        num_layers=2,
        dropout=0.1,
        edge_feature_dim=0,
    ):
        super().__init__()
        self.edge_feature_dim = edge_feature_dim

        dims = [input_dim]
        for _ in range(max(0, num_layers - 1)):
            dims.append(hidden_dim)
        dims.append(output_dim)

        layer_cls = EdgeGraphSAGELayer if edge_feature_dim else GraphSAGELayer
        self.layers = nn.ModuleList()
        for i in range(len(dims) - 1):
            if edge_feature_dim:
                self.layers.append(
                    layer_cls(dims[i], dims[i + 1], edge_feature_dim)
                )
            else:
                self.layers.append(layer_cls(dims[i], dims[i + 1]))
        self.dropout = nn.Dropout(dropout)

    def encodeNodes(self, x, edge_index, edge_attr=None):
        for layer_idx, layer in enumerate(self.layers):
            is_last = layer_idx == len(self.layers) - 1
            x = layer(x, edge_index, edge_attr=edge_attr, activate=not is_last)
            if layer_idx != len(self.layers) - 1:
                x = self.dropout(x)
        return x

    def poolNodes(self, x, batch):
        num_graphs = int(batch.max().item()) + 1 if batch.numel() else 0
        graph_emb = torch.zeros(
            num_graphs,
            x.shape[1],
            device=x.device,
            dtype=x.dtype,
        )
        counts = torch.zeros(
            num_graphs,
            device=x.device,
            dtype=x.dtype,
        )

        graph_emb.index_add_(0, batch, x)
        counts.index_add_(0, batch, torch.ones_like(batch, dtype=x.dtype))
        graph_emb = graph_emb / counts.clamp(min=1.0).unsqueeze(1)

        return F.normalize(graph_emb, p=2, dim=1)

    def forward(self, x, edge_index, batch, edge_attr=None):
        node_embeddings = self.encodeNodes(
            x,
            edge_index,
            edge_attr=edge_attr,
        )
        return self.poolNodes(node_embeddings, batch)


def encode_graph_batch(model, graph_batch, device):
    if len(graph_batch) == 4:
        x, edge_index, edge_attr, batch = graph_batch
        edge_attr = edge_attr.to(device)
    else:
        x, edge_index, batch = graph_batch
        edge_attr = None
    return model(
        x.to(device),
        edge_index.to(device),
        batch.to(device),
        edge_attr=edge_attr,
    )
