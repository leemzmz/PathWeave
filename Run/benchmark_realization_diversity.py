"""只读统计：同一 macro 内 20 个 transistor-level 实现的差异证据。

三个层次，全部基于 _data/ 下的物理 netlist（不涉及 path / hlink / 训练）：

1. 同构层：带属性的 Weisfeiler-Lehman 哈希（顶点类型 + 边端口），对哈希
   碰撞对再用 VF2 精确判定，给出"宏内同构对数"。
2. 距离层：复用 attributed approximate GED（直接调用
   GraphSAGE/GmRcGraph.py 的同一实现），在物理图上算宏内 C(20,2)=190 对
   的归一化相似度分布，并给跨宏抽样对作对照。
3. 组成层：宏内 distinct (V, E, MOS) 数与 MOS 计数极差，用于比较宏内的器件数量与规模跨度。

口径：
- 相似度 = 1 - min(1, GED / N)，N = 较大图的 (V + E)，与 target GED 同一归一化。
- 顶点属性 = NETLIST_NODE_FEATURE_SCHEMA one-hot；边属性 = 端口 one-hot。
- 每个无向物理边只算一次（与 GmRcGraph._attributedGraph 的取边方式一致）。

本脚本只读 _data/，只打印 stdout，不写入 _run/results/ 与 _run/models/。

用法：
    python Run/benchmark_realization_diversity.py                 # 全量 42 宏
    python Run/benchmark_realization_diversity.py --macros topo_ACBC2
    python Run/benchmark_realization_diversity.py --skip-ged      # 只做同构层
"""

import argparse
import hashlib
import itertools
import multiprocessing as mp
import random
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import networkx as nx

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from _project_paths import DATASET_DIR  # noqa: E402
from GraphSAGE.GmRcGraph import (  # noqa: E402
    _approximateGedGraphs,
    _featureLabel,
)
from GraphSAGE.graph_data import (  # noqa: E402
    EDGE_FEATURE_SCHEMA,
    NETLIST_NODE_FEATURE_SCHEMA,
    netlist_to_graph_tensor,
)


WL_ROUNDS = 3


# one-hot 行转类型名（同一行可能有多个 1，取首个）。
def readFeatureName(row, schema):
    for index, value in enumerate(row):
        if value == 1.0:
            return schema[index]
    return "UNK"


# 把一个 netlist tensor 转成两份表示：
# - gedGraph = (顶点数值特征, 无向边(源, 目标, 数值特征), 数值索引邻接)，供 GED；
# - hashGraph = (顶点类型名, 边端口名, 命名索引邻接)，供 WL 哈希与 VF2。
# 取边方式与 GmRcGraph._attributedGraph 一致（每对物理边取一次）。
def attributedNetlistGraph(tensor):
    rows = tensor.x.tolist()
    nodeFeatures = tuple(_featureLabel(row) for row in rows)
    nodeNames = tuple(
        readFeatureName(row, NETLIST_NODE_FEATURE_SCHEMA) for row in rows
    )
    edgeRows = tensor.edge_attr.tolist()
    edges = []
    edgeNames = []
    featureAdjacency = [[] for _ in nodeFeatures]
    nameAdjacency = [[] for _ in nodeFeatures]
    for index in range(0, tensor.edge_index.shape[1], 2):
        source = int(tensor.edge_index[0, index])
        target = int(tensor.edge_index[1, index])
        feature = _featureLabel(edgeRows[index])
        edgeName = readFeatureName(edgeRows[index], EDGE_FEATURE_SCHEMA)
        edges.append((source, target, feature))
        edgeNames.append(edgeName)
        featureAdjacency[source].append((feature, nodeFeatures[target]))
        featureAdjacency[target].append((feature, nodeFeatures[source]))
        nameAdjacency[source].append((edgeName, target))
        nameAdjacency[target].append((edgeName, source))
    gedGraph = (
        nodeFeatures,
        tuple(edges),
        tuple(tuple(items) for items in featureAdjacency),
    )
    hashGraph = (
        nodeNames,
        tuple(edgeNames),
        tuple(tuple(items) for items in nameAdjacency),
    )
    return gedGraph, hashGraph


# 读取 42 个宏及其 20 个 core，返回 {macro: [corePath, ...]}。
def loadMacroCores(datasetDir, macroFilter=None):
    macros = {}
    for path in sorted(datasetDir.iterdir()):
        if not path.is_dir() or not (path / "gm_rc_graph.json").exists():
            continue
        if macroFilter and path.name not in macroFilter:
            continue
        cores = sorted(path.glob("core_*.cir"))
        if cores:
            macros[path.name] = cores
    return macros


# 带属性 WL 哈希：顶点标签 = 类型名，边标签 = 端口名；多边按多重集参与。
def weisfeilerLehmanHash(hashGraph, rounds=WL_ROUNDS):
    nodeNames, _, nameAdjacency = hashGraph
    labels = list(nodeNames)
    for _ in range(rounds):
        refined = []
        for index, label in enumerate(labels):
            neighbourLabels = sorted(
                (edgeName, labels[neighbour])
                for edgeName, neighbour in nameAdjacency[index]
            )
            refined.append(hashlib.sha1(
                (label + "|" + repr(neighbourLabels)).encode("utf-8")
            ).hexdigest()[:16])
        labels = refined
    return hashlib.sha1(repr(sorted(labels)).encode("utf-8")).hexdigest()[:16]


# 构造带属性的 networkx 多重图（同端口平行边保留）。
def buildNetworkxMultiGraph(hashGraph):
    nodeNames, _, nameAdjacency = hashGraph
    multi = nx.MultiGraph()
    for index, label in enumerate(nodeNames):
        multi.add_node(index, label=label)
    for index, neighbours in enumerate(nameAdjacency):
        for edgeName, neighbour in neighbours:
            if index < neighbour:
                multi.add_edge(index, neighbour, port=edgeName)
    return multi


# VF2 精确同构判定：顶点类型与端口多重集都必须匹配。
def exactIsomorphic(firstHashGraph, secondHashGraph):
    firstGraph = buildNetworkxMultiGraph(firstHashGraph)
    secondGraph = buildNetworkxMultiGraph(secondHashGraph)
    if (firstGraph.number_of_nodes(), firstGraph.number_of_edges()) != (
        secondGraph.number_of_nodes(), secondGraph.number_of_edges()
    ):
        return False

    def nodeMatch(left, right):
        return left["label"] == right["label"]

    def edgeMatch(left, right):
        leftPorts = sorted(item["port"] for item in left.values())
        rightPorts = sorted(item["port"] for item in right.values())
        return leftPorts == rightPorts

    return nx.is_isomorphic(
        firstGraph, secondGraph, node_match=nodeMatch, edge_match=edgeMatch
    )


# 统计 core 的 (V, E, MOS) 签名与器件构成，供组成层使用。
def readCoreComposition(gedGraph, hashGraph):
    nodeFeatures, edges, _ = gedGraph
    nodeNames = hashGraph[0]
    counts = defaultdict(int)
    for label in nodeNames:
        counts[label] += 1
    return {
        "vertices": len(nodeFeatures),
        "edges": len(edges),
        "mos": counts["PMOS"] + counts["NMOS"],
        "nmos": counts["NMOS"],
        "pmos": counts["PMOS"],
        "res": counts["Res"],
        "cap": counts["Cap"],
    }


# 单对 core 的归一化 GED 相似度（多进程工作函数，参数需可 pickle）。
def pairGedSimilarity(corePair):
    firstPath, secondPath = corePair
    firstGraph = attributedNetlistGraph(netlist_to_graph_tensor(firstPath))[0]
    secondGraph = attributedNetlistGraph(netlist_to_graph_tensor(secondPath))[0]
    normalizer = max(
        len(firstGraph[0]) + len(firstGraph[1]),
        len(secondGraph[0]) + len(secondGraph[1]),
        1,
    )
    distance = _approximateGedGraphs(firstGraph, secondGraph) / normalizer
    return 1.0 - min(1.0, distance)


# 打印均值 / 标准差 / 中位数 / 四分位 / 极值。
def printDescription(label, values, digits=4):
    if not values:
        print(f"  {label}: n/a")
        return
    ordered = sorted(values)
    quartile = statistics.quantiles(ordered, n=4) if len(ordered) > 1 else None
    spread = (
        f"IQR [{quartile[0]:.{digits}f}, {quartile[2]:.{digits}f}]"
        if quartile else "IQR n/a"
    )
    deviation = statistics.stdev(values) if len(values) > 1 else 0.0
    print(
        f"  {label}: n={len(values)} "
        f"mean={statistics.mean(values):.{digits}f} "
        f"sd={deviation:.{digits}f} "
        f"median={statistics.median(values):.{digits}f} "
        f"{spread} min={ordered[0]:.{digits}f} "
        f"max={ordered[-1]:.{digits}f}"
    )


# 在进程池上跑一批 core 对，按块打印进度。
def runPairPool(pairs, workers, label):
    if not pairs:
        return []
    started = time.perf_counter()
    results = []
    with mp.Pool(processes=workers) as pool:
        for index, value in enumerate(
            pool.imap_unordered(pairGedSimilarity, pairs, chunksize=8), 1
        ):
            results.append(value)
            if index % 250 == 0 or index == len(pairs):
                elapsed = time.perf_counter() - started
                rate = index / elapsed if elapsed else 0.0
                print(
                    f"    {label}: {index}/{len(pairs)} pairs "
                    f"({elapsed:.0f}s, {rate:.1f} pairs/s)",
                    flush=True,
                )
    return results


# 用参考实现校验同一对 core 的相似度（防口径漂移）。
def validateAgainstReference(pairs, sampleCount=3):
    print("\n== 0) 与参考实现一致性校验 ==")
    for firstPath, secondPath in pairs[:sampleCount]:
        firstGraph = attributedNetlistGraph(
            netlist_to_graph_tensor(firstPath))[0]
        secondGraph = attributedNetlistGraph(
            netlist_to_graph_tensor(secondPath))[0]
        normalizer = max(
            len(firstGraph[0]) + len(firstGraph[1]),
            len(secondGraph[0]) + len(secondGraph[1]),
            1,
        )
        direct = 1.0 - min(
            1.0, _approximateGedGraphs(firstGraph, secondGraph) / normalizer
        )
        pooled = pairGedSimilarity((firstPath, secondPath))
        print(
            f"  {firstPath.name} vs {secondPath.name}: "
            f"pooled={pooled:.6f} reference={direct:.6f} "
            f"diff={abs(pooled - direct):.2e}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--macros", nargs="*", default=None,
                        help="只统计指定宏（默认全部）")
    parser.add_argument("--workers", type=int, default=8,
                        help="GED 进程数（默认 8）")
    parser.add_argument("--inter-sample", type=int, default=4000,
                        help="跨宏对照抽样对数（默认 4000，0 表示跳过）")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-ged", action="store_true",
                        help="只做同构层与组成层（秒级）")
    args = parser.parse_args()

    macros = loadMacroCores(DATASET_DIR, args.macros)
    coreCount = sum(len(cores) for cores in macros.values())
    print(
        f"macros={len(macros)} cores={coreCount} intra-macro pairs="
        f"{sum(len(c) * (len(c) - 1) // 2 for c in macros.values())}"
    )

    started = time.perf_counter()
    gedGraphs = {}
    hashGraphs = {}
    hashes = {}
    compositions = {}
    for macro, cores in macros.items():
        for corePath in cores:
            gedGraph, hashGraph = attributedNetlistGraph(
                netlist_to_graph_tensor(corePath))
            gedGraphs[corePath] = gedGraph
            hashGraphs[corePath] = hashGraph
            hashes[corePath] = weisfeilerLehmanHash(hashGraph)
            compositions[corePath] = readCoreComposition(gedGraph, hashGraph)
    print(
        f"built {len(gedGraphs)} physical graphs "
        f"in {time.perf_counter() - started:.1f}s"
    )

    print("\n== 1) 同构层：WL 哈希与 VF2 精确判定 ==")
    collisions = 0
    isomorphicPairs = []
    distinctPerMacro = []
    wlCollisionPairs = []
    for macro, cores in macros.items():
        byHash = defaultdict(list)
        for corePath in cores:
            byHash[hashes[corePath]].append(corePath)
        distinctPerMacro.append(len(byHash))
        for group in byHash.values():
            collisions += len(group) * (len(group) - 1) // 2
        for first, second in itertools.combinations(cores, 2):
            if hashes[first] != hashes[second]:
                continue
            wlCollisionPairs.append((first, second))
            if exactIsomorphic(hashGraphs[first], hashGraphs[second]):
                isomorphicPairs.append((first, second))
    printDescription("distinct WL hashes per macro", distinctPerMacro, digits=1)
    print(f"  WL-equivalent pairs (candidate isomorphism) = {collisions}")
    print(
        f"  exactly isomorphic core pairs = {len(isomorphicPairs)} "
        f"(VF2 tested on {len(wlCollisionPairs)} candidate pairs)"
    )
    for first, second in isomorphicPairs[:5]:
        print(f"    isomorphic: {first.parent.name} {first.name} = {second.name}")

    print("\n== 2) 组成层：宏内规模与器件差异 ==")
    distinctSizes = []
    mosSpans = []
    for macro, cores in macros.items():
        signatures = {
            (compositions[corePath]["vertices"],
             compositions[corePath]["edges"],
             compositions[corePath]["mos"])
            for corePath in cores
        }
        distinctSizes.append(len(signatures))
        mosValues = [compositions[corePath]["mos"] for corePath in cores]
        mosSpans.append(max(mosValues) - min(mosValues))
    printDescription("distinct (V,E,MOS) per macro", distinctSizes, digits=1)
    printDescription("MOS count span per macro", mosSpans, digits=1)
    printDescription(
        "MOS per core",
        [value["mos"] for value in compositions.values()],
        digits=1,
    )
    printDescription(
        "resistors per core",
        [value["res"] for value in compositions.values()],
        digits=1,
    )
    printDescription(
        "capacitors per core",
        [value["cap"] for value in compositions.values()],
        digits=1,
    )

    if args.skip_ged:
        print(f"\ntotal elapsed {time.perf_counter() - started:.1f}s")
        return

    print("\n== 3) 距离层：宏内近似 GED 相似度（全部 C(20,2) 对）==")
    intraPairs = [
        pair
        for cores in macros.values()
        for pair in itertools.combinations(cores, 2)
    ]
    validateAgainstReference(intraPairs)
    intraSimilarity = runPairPool(intraPairs, args.workers, "intra")
    printDescription("intra-macro GED similarity", intraSimilarity)
    zeroPairs = sum(1 for value in intraSimilarity if value >= 1.0 - 1e-12)
    print(f"  intra-macro pairs with zero edit distance = {zeroPairs}")

    print("\n== 4) 距离层：per-macro 汇总 ==")
    index = 0
    perMacroMeans = []
    perMacroMinimums = []
    for macro, cores in macros.items():
        count = len(cores) * (len(cores) - 1) // 2
        block = intraSimilarity[index:index + count]
        index += count
        perMacroMeans.append(statistics.mean(block))
        perMacroMinimums.append(min(block))
        print(
            f"  {macro:14s} pairs={count:4d} "
            f"mean={statistics.mean(block):.4f} min={min(block):.4f}"
        )
    printDescription("per-macro mean similarity", perMacroMeans)
    printDescription("per-macro minimum similarity", perMacroMinimums)

    if args.inter_sample > 0:
        print("\n== 5) 距离层：跨宏对照（抽样）==")
        rng = random.Random(args.seed)
        allCores = [core for cores in macros.values() for core in cores]
        interPairs = []
        seen = set()
        while len(interPairs) < args.inter_sample:
            first, second = rng.sample(allCores, 2)
            if first.parent == second.parent:
                continue
            key = tuple(sorted((str(first), str(second))))
            if key in seen:
                continue
            seen.add(key)
            interPairs.append((first, second))
        interSimilarity = runPairPool(interPairs, args.workers, "inter")
        printDescription("inter-macro GED similarity", interSimilarity)

    print(f"\ntotal elapsed {time.perf_counter() - started:.1f}s")


if __name__ == "__main__":
    main()
