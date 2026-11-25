from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import math
import random
import re

import torch
from torch.utils.data import Dataset

from GraphSAGE.GmRcGraph import loadGmRcGraph


_CORE_PATTERN = re.compile(r"^core_(\d+)\.cir$")


@dataclass(frozen=True)
class AlignmentCase:
    name: str
    topology: str
    netlistPath: Path
    targetPath: Path
    target: object
    feedbackImplementation: str


@dataclass(frozen=True)
class CircuitAlignmentSample:
    name: str
    topology: str
    targetKey: str
    targetIndex: int
    graph: object
    feedbackImplementation: str


class CircuitAlignmentDataset(Dataset):

    def __init__(self, samples):
        self.m_samples = list(samples)

    def __len__(self):
        return len(self.m_samples)

    def __getitem__(self, index):
        return self.m_samples[index]


def buildHardNegativeCandidates(
    targetSimilarity, targetIndices=None, topK=5
):
    targetSimilarity = torch.as_tensor(
        targetSimilarity, dtype=torch.float32
    ).cpu()
    if targetSimilarity.ndim != 2:
        raise ValueError("target similarity must be a matrix")
    if targetSimilarity.shape[0] != targetSimilarity.shape[1]:
        raise ValueError("target similarity matrix must be square")
    if topK < 1:
        raise ValueError("hard-negative topK must be positive")
    if targetIndices is None:
        targetIndices = range(targetSimilarity.shape[0])
    targetIndices = sorted(set(int(index) for index in targetIndices))
    candidatesByTarget = {}
    for targetIndex in targetIndices:
        candidates = [
            other for other in targetIndices if other != targetIndex
        ]
        candidates.sort(key=lambda other: (
            -float(targetSimilarity[targetIndex, other]),
            other,
        ))
        if len(candidates) < topK:
            raise ValueError(
                "hard-negative topK exceeds available target macros"
            )
        candidatesByTarget[targetIndex] = tuple(candidates[:topK])
    return candidatesByTarget


# 基于目标图相似度先采 macro，再采对应 core 正负样本。
class MacroHardNegativeSampler:

    def __init__(
        self,
        samples,
        targetSimilarity,
        topK=5,
        negativeCount=2,
        temperature=0.1,
        seed=42,
    ):
        if topK < 1:
            raise ValueError("hard-negative topK must be positive")
        if negativeCount < 1 or negativeCount > topK:
            raise ValueError(
                "hard-negative count must be between 1 and topK"
            )
        if temperature <= 0.0:
            raise ValueError(
                "hard-negative sampling temperature must be positive"
            )
        self.m_samplesByName = {sample.name: sample for sample in samples}
        self.m_samplesByTarget = defaultdict(list)
        for sample in samples:
            self.m_samplesByTarget[sample.targetIndex].append(sample)
        if any(
            len(group) < 2 for group in self.m_samplesByTarget.values()
        ):
            raise ValueError(
                "each target requires at least two cores for positive sampling"
            )
        self.m_targetSimilarity = torch.as_tensor(
            targetSimilarity, dtype=torch.float32
        ).cpu()
        self.m_topK = int(topK)
        self.m_negativeCount = int(negativeCount)
        self.m_temperature = float(temperature)
        self.m_rng = random.Random(f"{seed}:macro-hard-negative")
        self.m_candidates = buildHardNegativeCandidates(
            self.m_targetSimilarity,
            self.m_samplesByTarget,
            topK=self.m_topK,
        )

    def candidateTargets(self):
        return dict(self.m_candidates)

    def _sampleTargetIndices(self, targetIndex):
        pool = list(self.m_candidates[targetIndex])
        selected = []
        for _ in range(self.m_negativeCount):
            scores = [
                float(self.m_targetSimilarity[targetIndex, other])
                for other in pool
            ]
            maximum = max(scores)
            weights = [
                math.exp((score - maximum) / self.m_temperature)
                for score in scores
            ]
            selectedIndex = self.m_rng.choices(
                range(len(pool)), weights=weights, k=1
            )[0]
            selected.append(pool.pop(selectedIndex))
        return selected

    def sampleBatch(self, anchorNames):
        positiveSamples = []
        negativeSamples = []
        negativeTargets = []
        negativeSimilarities = []
        for anchorName in anchorNames:
            anchor = self.m_samplesByName[anchorName]
            positivePool = [
                sample
                for sample in self.m_samplesByTarget[anchor.targetIndex]
                if sample.name != anchor.name
            ]
            positiveSamples.append(self.m_rng.choice(positivePool))
            selectedTargets = self._sampleTargetIndices(
                anchor.targetIndex
            )
            negativeTargets.append(selectedTargets)
            negativeSimilarities.append([
                float(self.m_targetSimilarity[
                    anchor.targetIndex, targetIndex
                ])
                for targetIndex in selectedTargets
            ])
            negativeSamples.extend([
                self.m_rng.choice(self.m_samplesByTarget[targetIndex])
                for targetIndex in selectedTargets
            ])
        return (
            positiveSamples,
            negativeSamples,
            torch.tensor(negativeTargets, dtype=torch.long),
            torch.tensor(negativeSimilarities, dtype=torch.float32),
        )


def _naturalKey(value):
    return [
        int(part) if part.isdigit() else part
        for part in re.split(r"(\d+)", str(value))
    ]


def _feedbackImplementation(netlistPath):
    implementations = set()
    for rawLine in Path(netlistPath).read_text(encoding="utf-8").splitlines():
        line = rawLine.strip()
        if not line or line.startswith("*") or line.startswith("."):
            continue
        deviceName = line.split()[0].lower()
        if deviceName.startswith("rgf"):
            implementations.add("resistor")
        elif deviceName.startswith("mgf"):
            implementations.add("mos")
    if len(implementations) > 1:
        raise ValueError(
            f"mixed global feedback implementations in {netlistPath}"
        )
    return next(iter(implementations), "none")


# 发现全部裸 core；同一 topology 的 core 共用功能目标图。
def discoverAlignmentCases(datasetDir):
    datasetDir = Path(datasetDir)
    cases = []
    for topologyDir in sorted(datasetDir.iterdir(), key=lambda path: _naturalKey(path.name)):
        if not topologyDir.is_dir():
            continue
        targetPath = topologyDir / "gm_rc_graph.json"
        if not targetPath.exists():
            continue
        target = loadGmRcGraph(targetPath)
        corePaths = sorted(topologyDir.glob("core_*.cir"), key=lambda path: _naturalKey(path.name))
        for corePath in corePaths:
            if _CORE_PATTERN.fullmatch(corePath.name) is None:
                continue
            cases.append(AlignmentCase(
                name=corePath.relative_to(datasetDir).as_posix(),
                topology=topologyDir.name,
                netlistPath=corePath,
                targetPath=targetPath,
                target=target,
                feedbackImplementation=_feedbackImplementation(corePath),
            ))
    if not cases:
        raise ValueError(f"no core alignment cases found in {datasetDir}")
    return cases


# 按语义图去重目标，不使用 topology 名称或旧 compensation 类型。
def buildTargetBank(cases):
    byKey = {}
    for case in cases:
        byKey.setdefault(case.target.semanticKey, case.target)
    keys = sorted(byKey)
    records = [byKey[key] for key in keys]
    return records, {key: index for index, key in enumerate(keys)}


# 为一个 baseline 构造电路输入图，并关联共享目标图索引。
def buildCircuitAlignmentSamples(cases, targetIndex, tensorBuilder):
    samples = []
    for case in cases:
        samples.append(CircuitAlignmentSample(
            name=case.name,
            topology=case.topology,
            targetKey=case.target.semanticKey,
            targetIndex=targetIndex[case.target.semanticKey],
            graph=tensorBuilder(case.netlistPath),
            feedbackImplementation=case.feedbackImplementation,
        ))
    return samples


# 构造供 DataLoader 使用的 baseline-specific collate。
def makeAlignmentCollate(batchGraphs):
    def collate(samples):
        return (
            batchGraphs([sample.graph for sample in samples]),
            torch.tensor(
                [sample.targetIndex for sample in samples],
                dtype=torch.long,
            ),
            [sample.name for sample in samples],
        )
    return collate
