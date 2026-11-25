import json

import torch
import torch.nn as nn

from GraphSAGE.GmRcGraph import gmRcGraphToTensor, gmRcSemanticKey


GM_RC_DECODER_CONTRACT = "autoregressive_functional_gm_rc_graph"
GM_RC_TOKEN_SCHEMA = [
    "PAD",
    "BOS",
    "EOS",
    "HAS_GROUND",
    "NO_GROUND",
    "BOUNDARIES_BEGIN",
    "BOUNDARY",
    "BOUNDARIES_END",
    "ROLE_INPUT_PIN",
    "ROLE_MAIN_BOUNDARY",
    "ROLE_STAGE_OUTPUT",
    "ROLE_OUTPUT_PIN",
    "NO_STAGE",
    "STAGE",
    "MAIN_BEGIN",
    "MAIN_GM",
    "MAIN_END",
    "PATHS_BEGIN",
    "PATH_BEGIN",
    "PATH_END",
    "PATHS_END",
    "DEVICE_GM",
    "DEVICE_R",
    "DEVICE_C",
    "DEVICE_END",
    "TERM_INPUT",
    "TERM_OUTPUT",
    "TERM_PASSIVE",
    "TERM_END",
    "REF_BOUNDARY",
    "REF_INTERNAL",
    "REF_GROUND",
    "ONE",
    "NUM_END",
    "ITEM_END",
]
GM_RC_TOKEN_TO_ID = {
    token: index for index, token in enumerate(GM_RC_TOKEN_SCHEMA)
}
GM_RC_PAD_ID = GM_RC_TOKEN_TO_ID["PAD"]
GM_RC_BOS_ID = GM_RC_TOKEN_TO_ID["BOS"]
GM_RC_EOS_ID = GM_RC_TOKEN_TO_ID["EOS"]

_ROLE_TO_TOKEN = {
    "input-pin": "ROLE_INPUT_PIN",
    "main-boundary": "ROLE_MAIN_BOUNDARY",
    "stage-output": "ROLE_STAGE_OUTPUT",
    "output-pin": "ROLE_OUTPUT_PIN",
}
_TOKEN_TO_ROLE = {value: key for key, value in _ROLE_TO_TOKEN.items()}
_ROLE_ORDER = tuple(_ROLE_TO_TOKEN)
_KIND_TO_TOKEN = {
    "Gm": "DEVICE_GM",
    "R": "DEVICE_R",
    "C": "DEVICE_C",
}
_TOKEN_TO_KIND = {value: key for key, value in _KIND_TO_TOKEN.items()}
_TERMINAL_TO_TOKEN = {
    "input": "TERM_INPUT",
    "output": "TERM_OUTPUT",
    "passive": "TERM_PASSIVE",
}
_TOKEN_TO_TERMINAL = {
    value: key for key, value in _TERMINAL_TO_TOKEN.items()
}


def _appendNumber(tokens, value):
    value = int(value)
    if value < 0:
        raise ValueError("Gm+RC token numbers must be non-negative")
    tokens.extend(["ONE"] * value)
    tokens.append("NUM_END")


def _appendReference(tokens, name):
    if name == "GND":
        tokens.append("REF_GROUND")
        return
    if name.startswith("B"):
        tokens.append("REF_BOUNDARY")
        _appendNumber(tokens, int(name[1:]))
        return
    if name.startswith("X"):
        tokens.append("REF_INTERNAL")
        _appendNumber(tokens, int(name[1:]))
        return
    raise ValueError(f"unsupported Gm+RC semantic node reference: {name}")


# 将 functional graph 编码为不依赖 topology/id/path 编号的可变长 token。
def gmRcGraphToTokens(graphData):
    semantic = json.loads(gmRcSemanticKey(graphData))
    tokens = ["BOS"]
    tokens.append(
        "HAS_GROUND" if semantic["has_ground"] else "NO_GROUND"
    )
    tokens.append("BOUNDARIES_BEGIN")
    for boundary, roles, stage in semantic["boundaries"]:
        tokens.append("BOUNDARY")
        _appendNumber(tokens, boundary)
        roleSet = set(roles)
        for role in _ROLE_ORDER:
            if role in roleSet:
                tokens.append(_ROLE_TO_TOKEN[role])
        if stage is None:
            tokens.append("NO_STAGE")
        else:
            tokens.append("STAGE")
            _appendNumber(tokens, stage)
        tokens.append("ITEM_END")
    tokens.append("BOUNDARIES_END")

    tokens.append("MAIN_BEGIN")
    for unit, stage in semantic["main"]:
        tokens.append("MAIN_GM")
        _appendNumber(tokens, unit)
        _appendNumber(tokens, stage)
        tokens.append("ITEM_END")
    tokens.append("MAIN_END")

    tokens.append("PATHS_BEGIN")
    for path in semantic["paths"]:
        tokens.append("PATH_BEGIN")
        for kind, terminals in path:
            tokens.append(_KIND_TO_TOKEN[kind])
            for terminal, nodeName in terminals:
                tokens.append(_TERMINAL_TO_TOKEN[terminal])
                _appendReference(tokens, nodeName)
                tokens.append("TERM_END")
            tokens.append("DEVICE_END")
        tokens.append("PATH_END")
    tokens.extend(("PATHS_END", "EOS"))
    return tokens


# 返回 decoder 使用的整数 token id。
def gmRcGraphToTokenIds(graphData):
    return [GM_RC_TOKEN_TO_ID[token] for token in gmRcGraphToTokens(graphData)]


class _TokenReader:

    def __init__(self, tokenIds):
        self.m_tokens = [
            GM_RC_TOKEN_SCHEMA[int(token)] for token in tokenIds
        ]
        self.m_index = 0

    def peek(self):
        if self.m_index >= len(self.m_tokens):
            raise ValueError("unexpected end of Gm+RC token sequence")
        return self.m_tokens[self.m_index]

    def take(self, expected=None):
        token = self.peek()
        if expected is not None and token != expected:
            raise ValueError(
                f"expected Gm+RC token {expected}, found {token}"
            )
        self.m_index += 1
        return token

    def number(self):
        value = 0
        while self.peek() == "ONE":
            self.take("ONE")
            value += 1
        self.take("NUM_END")
        return value

    def finish(self):
        while self.m_index < len(self.m_tokens):
            if self.take() != "PAD":
                raise ValueError("tokens found after Gm+RC EOS")


def _readReference(reader):
    token = reader.take()
    if token == "REF_GROUND":
        return "GND"
    if token == "REF_BOUNDARY":
        return f"B{reader.number()}"
    if token == "REF_INTERNAL":
        return f"X{reader.number()}"
    raise ValueError(f"invalid Gm+RC node reference token: {token}")


def _readSemanticTokens(tokenIds):
    reader = _TokenReader(tokenIds)
    reader.take("BOS")
    groundToken = reader.take()
    if groundToken not in {"HAS_GROUND", "NO_GROUND"}:
        raise ValueError("Gm+RC token sequence is missing ground flag")

    boundaries = []
    reader.take("BOUNDARIES_BEGIN")
    while reader.peek() == "BOUNDARY":
        reader.take("BOUNDARY")
        boundary = reader.number()
        roles = []
        while reader.peek() in _TOKEN_TO_ROLE:
            roles.append(_TOKEN_TO_ROLE[reader.take()])
        stageToken = reader.take()
        if stageToken == "NO_STAGE":
            stage = None
        elif stageToken == "STAGE":
            stage = reader.number()
        else:
            raise ValueError(f"invalid boundary stage token: {stageToken}")
        reader.take("ITEM_END")
        boundaries.append((boundary, roles, stage))
    reader.take("BOUNDARIES_END")

    main = []
    reader.take("MAIN_BEGIN")
    while reader.peek() == "MAIN_GM":
        reader.take("MAIN_GM")
        main.append((reader.number(), reader.number()))
        reader.take("ITEM_END")
    reader.take("MAIN_END")

    paths = []
    reader.take("PATHS_BEGIN")
    while reader.peek() == "PATH_BEGIN":
        reader.take("PATH_BEGIN")
        devices = []
        while reader.peek() in _TOKEN_TO_KIND:
            kind = _TOKEN_TO_KIND[reader.take()]
            terminals = []
            while reader.peek() in _TOKEN_TO_TERMINAL:
                terminal = _TOKEN_TO_TERMINAL[reader.take()]
                nodeName = _readReference(reader)
                reader.take("TERM_END")
                terminals.append((terminal, nodeName))
            if len(terminals) != 2:
                raise ValueError("decoded Gm+RC device must have two terminals")
            reader.take("DEVICE_END")
            devices.append((kind, terminals))
        reader.take("PATH_END")
        paths.append(devices)
    reader.take("PATHS_END")
    reader.take("EOS")
    reader.finish()
    return {
        "has_ground": groundToken == "HAS_GROUND",
        "boundaries": boundaries,
        "main": main,
        "paths": paths,
    }


def _referenceToNodeId(reference, pathIndex, nodes, nodeIds):
    if reference == "GND":
        return "ground"
    if reference.startswith("B"):
        return f"boundary_{int(reference[1:])}"
    order = int(reference[1:])
    nodeId = f"comp_{pathIndex}_node_{order}"
    if nodeId not in nodeIds:
        nodes.append({
            "id": nodeId,
            "kind": "internal",
            "path": pathIndex,
            "order": order,
        })
        nodeIds.add(nodeId)
    return nodeId


# 将 decoder token 严格恢复为可由现有 target encoder 接收的 Gm+RC graph。
def gmRcTokenIdsToGraphData(tokenIds):
    semantic = _readSemanticTokens(tokenIds)
    nodes = []
    nodeIds = set()
    for boundary, roles, stage in semantic["boundaries"]:
        node = {
            "id": f"boundary_{boundary}",
            "kind": "boundary",
            "boundary": boundary,
            "roles": list(roles),
        }
        if stage is not None:
            node["stage"] = stage
        nodes.append(node)
        nodeIds.add(node["id"])
    if semantic["has_ground"]:
        nodes.append({"id": "ground", "kind": "ground"})
        nodeIds.add("ground")

    devices = []
    edges = []
    for unit, stage in semantic["main"]:
        deviceId = f"main_gm_{unit}"
        inputNode = f"boundary_{unit - 1}"
        outputNode = f"boundary_{unit}"
        if inputNode not in nodeIds or outputNode not in nodeIds:
            raise ValueError("main Gm references a missing boundary")
        devices.append({
            "id": deviceId,
            "kind": "Gm",
            "role": "main",
            "unit": unit,
            "stage": stage,
        })
        edges.extend((
            {"node": inputNode, "device": deviceId, "terminal": "input"},
            {"node": outputNode, "device": deviceId, "terminal": "output"},
        ))

    for pathIndex, path in enumerate(semantic["paths"]):
        for order, (kind, terminals) in enumerate(path):
            deviceId = f"comp_{pathIndex}_{order}_{kind.lower()}"
            devices.append({
                "id": deviceId,
                "kind": kind,
                "role": "compensation",
                "path": pathIndex,
                "order": order,
            })
            passiveIndex = 0
            for terminal, reference in terminals:
                nodeId = _referenceToNodeId(
                    reference, pathIndex, nodes, nodeIds
                )
                if nodeId not in nodeIds:
                    raise ValueError(
                        f"decoded Gm+RC edge references missing node {nodeId}"
                    )
                if kind in {"R", "C"}:
                    if terminal != "passive":
                        raise ValueError("passive device has directed terminal")
                    terminalName = ("a", "b")[passiveIndex]
                    passiveIndex += 1
                else:
                    if terminal not in {"input", "output"}:
                        raise ValueError("Gm device has passive terminal")
                    terminalName = terminal
                edges.append({
                    "node": nodeId,
                    "device": deviceId,
                    "terminal": terminalName,
                })

    graphData = {"nodes": nodes, "devices": devices, "edges": edges}
    gmRcGraphToTensor(graphData)
    return graphData


class GmRcGraphDecoder(nn.Module):

    def __init__(
        self,
        embeddingDim=128,
        tokenDim=64,
        hiddenDim=256,
        numLayers=2,
        dropout=0.1,
    ):
        super().__init__()
        self.m_embeddingDim = embeddingDim
        self.m_hiddenDim = hiddenDim
        self.m_numLayers = numLayers
        self.m_tokenEmbedding = nn.Embedding(
            len(GM_RC_TOKEN_SCHEMA), tokenDim, padding_idx=GM_RC_PAD_ID
        )
        self.m_initial = nn.Linear(
            embeddingDim, hiddenDim * numLayers
        )
        self.m_recurrent = nn.GRU(
            tokenDim + embeddingDim,
            hiddenDim,
            num_layers=numLayers,
            batch_first=True,
            dropout=dropout if numLayers > 1 else 0.0,
        )
        self.m_output = nn.Linear(hiddenDim, len(GM_RC_TOKEN_SCHEMA))

    def _initialHidden(self, embedding):
        hidden = torch.tanh(self.m_initial(embedding))
        hidden = hidden.reshape(
            embedding.shape[0], self.m_numLayers, self.m_hiddenDim
        )
        return hidden.transpose(0, 1).contiguous()

    def forward(self, embedding, inputTokens):
        tokenEmbedding = self.m_tokenEmbedding(inputTokens)
        condition = embedding.unsqueeze(1).expand(
            -1, inputTokens.shape[1], -1
        )
        recurrentInput = torch.cat((tokenEmbedding, condition), dim=-1)
        output, _ = self.m_recurrent(
            recurrentInput, self._initialHidden(embedding)
        )
        return self.m_output(output)

    # 使用 greedy autoregressive decoding 生成变长 graph token。
    def generate(self, embedding, maxSteps):
        if maxSteps < 1:
            raise ValueError("decoder maxSteps must be positive")
        batchSize = embedding.shape[0]
        hidden = self._initialHidden(embedding)
        current = torch.full(
            (batchSize, 1),
            GM_RC_BOS_ID,
            dtype=torch.long,
            device=embedding.device,
        )
        sequences = [current]
        finished = torch.zeros(
            batchSize, dtype=torch.bool, device=embedding.device
        )
        for _ in range(maxSteps):
            tokenEmbedding = self.m_tokenEmbedding(current)
            recurrentInput = torch.cat(
                (tokenEmbedding, embedding.unsqueeze(1)), dim=-1
            )
            output, hidden = self.m_recurrent(recurrentInput, hidden)
            logits = self.m_output(output[:, -1])
            logits[:, GM_RC_PAD_ID] = float("-inf")
            logits[:, GM_RC_BOS_ID] = float("-inf")
            nextToken = logits.argmax(dim=-1)
            nextToken = torch.where(
                finished,
                torch.full_like(nextToken, GM_RC_EOS_ID),
                nextToken,
            )
            current = nextToken.unsqueeze(1)
            sequences.append(current)
            finished |= nextToken == GM_RC_EOS_ID
            if finished.all():
                break
        return torch.cat(sequences, dim=1)
