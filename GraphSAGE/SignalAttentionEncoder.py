import torch
from torch import nn
from torch.nn import functional as F

from GraphSAGE.CircuitEncoder import buildCircuitEncoder
from GraphSAGE.SignalAttentionData import (
    SIGNAL_ATTENTION_DIM32,
    SIGNAL_NODE_FEATURE_SCHEMA,
    SIGNAL_ONLY_EDGE_FEATURE_SCHEMA,
)
from GraphSAGE.graph_data import EDGE_FEATURE_SCHEMA


# 对 signal 边做类型化注意力：logit 由 query/key 与 edge feature 共同决定。
# attention 宽度固定为 32；q/k/v 先投影到 32，再由 output 投影回节点宽度。
class TypedSignalAttention(nn.Module):

    def __init__(self, nodeDim, signalDim, dropout, attentionDim):
        super().__init__()
        if attentionDim < 1:
            raise ValueError("signal attention width must be positive")
        self.m_attentionDim = int(attentionDim)
        self.m_query = nn.Linear(nodeDim, self.m_attentionDim)
        self.m_key = nn.Linear(nodeDim + signalDim, self.m_attentionDim)
        self.m_value = nn.Linear(nodeDim + signalDim, self.m_attentionDim)
        self.m_output = nn.Linear(self.m_attentionDim, nodeDim)
        self.m_bias = nn.Linear(signalDim, 1)
        self.m_scale = float(self.m_attentionDim) ** 0.5
        self.m_dropout = nn.Dropout(dropout)
        self.m_norm = nn.LayerNorm(nodeDim)

    # 只对 signal 边聚合，目标节点无 signal 邻居时退化为恒等残差。
    def forward(self, x, signalIndex, signalAttr):
        if signalIndex is None or signalIndex.numel() == 0:
            return self.m_norm(x)
        src, dst = signalIndex
        signalAttr = signalAttr.to(x.dtype)
        edgeInput = torch.cat([x[src], signalAttr], dim=1)
        query = self.m_query(x)
        key = self.m_key(edgeInput)
        value = self.m_value(edgeInput)
        logits = (
            (query[dst] * key).sum(dim=1) / self.m_scale
            + self.m_bias(signalAttr).squeeze(1)
        )
        weights = _segmentSoftmax(logits, dst, x.shape[0])
        weights = self.m_dropout(weights)
        aggregated = x.new_zeros((x.shape[0], value.shape[1]))
        aggregated.index_add_(0, dst, weights.unsqueeze(1) * value)
        return self.m_norm(x + self.m_output(aggregated))


# 按目标节点做数值稳定的 softmax，避免引入额外依赖。
def _segmentSoftmax(logits, index, numNodes):
    if logits.numel() == 0:
        return logits
    maximum = logits.new_full((numNodes,), float("-inf"))
    maximum.scatter_reduce_(0, index, logits, reduce="amax", include_self=True)
    shifted = (logits - maximum[index]).exp()
    totals = logits.new_zeros(numNodes)
    totals.index_add_(0, index, shifted)
    return shifted / totals[index].clamp(min=1e-12)


# 分层交替：每层先做 physical 传播，再做 32-D signal 注意力，两套参数相互独立。
class LayeredSignalAttentionEncoder(nn.Module):

    def __init__(
        self,
        backbone,
        inputDim,
        hiddenDim,
        outputDim,
        numLayers,
        dropout,
        edgeFeatureDim,
        signalDim,
        attentionDim=SIGNAL_ATTENTION_DIM32,
    ):
        super().__init__()
        self.m_backbone = buildCircuitEncoder(
            backbone,
            inputDim=inputDim,
            hiddenDim=hiddenDim,
            outputDim=outputDim,
            numLayers=numLayers,
            dropout=dropout,
            edgeFeatureDim=edgeFeatureDim,
        )
        self.m_isGraphSage = backbone == "graphsage"
        self.m_physicalLayers = (
            self.m_backbone.layers
            if self.m_isGraphSage
            else self.m_backbone.m_layers
        )
        self.m_dropout = nn.Dropout(dropout)
        self.m_signalDim = int(signalDim)
        dims = [inputDim] + [hiddenDim] * (numLayers - 1) + [outputDim]
        self.m_signalAttention = nn.ModuleList([
            TypedSignalAttention(
                outDim, self.m_signalDim, dropout, attentionDim
            )
            for outDim in dims[1:]
        ])

    # 逐层交替，保持与单图 encoder 相同的激活与 dropout 约定。
    def encodeNodes(
        self,
        x,
        edge_index,
        edge_attr=None,
        signalIndex=None,
        signalAttr=None,
    ):
        if signalIndex is None:
            signalIndex = x.new_zeros((2, 0), dtype=torch.long)
            signalAttr = x.new_zeros(
                (0, self.m_signalDim), dtype=x.dtype
            )
        total = len(self.m_physicalLayers)
        for index, layer in enumerate(self.m_physicalLayers):
            isLast = index == total - 1
            if self.m_isGraphSage:
                x = layer(
                    x, edge_index, edge_attr=edge_attr, activate=not isLast
                )
                if not isLast:
                    x = self.m_dropout(x)
            else:
                x = layer(x, edge_index, edge_attr)
                if not isLast:
                    x = self.m_dropout(F.relu(x))
            x = self.m_signalAttention[index](x, signalIndex, signalAttr)
        return x

    def poolNodes(self, x, batch):
        return self.m_backbone.poolNodes(x, batch)

    def forward(
        self,
        x,
        edge_index,
        batch,
        edge_attr=None,
        signal_index=None,
        signal_attr=None,
    ):
        nodeEmbeddings = self.encodeNodes(
            x, edge_index, edge_attr, signal_index, signal_attr
        )
        return self.poolNodes(nodeEmbeddings, batch)


# 复用通用训练器的构造签名；signalDim 由 backend 显式给出。
def buildLayeredSignalAttentionEncoder(
    backbone,
    inputDim,
    hiddenDim=128,
    outputDim=128,
    numLayers=2,
    dropout=0.1,
    edgeFeatureDim=0,
    signalDim=None,
):
    if inputDim != len(SIGNAL_NODE_FEATURE_SCHEMA):
        raise ValueError("layered path attention node schema mismatch")
    if edgeFeatureDim != len(EDGE_FEATURE_SCHEMA):
        raise ValueError("layered path attention physical edge schema mismatch")
    if signalDim is None:
        signalDim = len(SIGNAL_ONLY_EDGE_FEATURE_SCHEMA)
    return LayeredSignalAttentionEncoder(
        backbone,
        inputDim=inputDim,
        hiddenDim=hiddenDim,
        outputDim=outputDim,
        numLayers=numLayers,
        dropout=dropout,
        edgeFeatureDim=edgeFeatureDim,
        signalDim=signalDim,
    )


# 固定 signal 维度的构造器，供 backend 注册使用。
def buildLayeredSignalAttentionEncoderWithSignalDim(signalDim):

    def _build(
        backbone,
        inputDim,
        hiddenDim=128,
        outputDim=128,
        numLayers=2,
        dropout=0.1,
        edgeFeatureDim=0,
    ):
        return buildLayeredSignalAttentionEncoder(
            backbone,
            inputDim,
            hiddenDim=hiddenDim,
            outputDim=outputDim,
            numLayers=numLayers,
            dropout=dropout,
            edgeFeatureDim=edgeFeatureDim,
            signalDim=signalDim,
        )

    return _build
