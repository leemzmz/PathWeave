"""只读比较新旧数据的全量同构/规模与等额抽样 approximate GED。"""

import argparse
import itertools
import json
import multiprocessing as mp
import random
import statistics
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from Run.benchmark_realization_diversity import (  # noqa: E402
    attributedNetlistGraph, exactIsomorphic, loadMacroCores,
    readCoreComposition, weisfeilerLehmanHash,
)
from GraphSAGE.GmRcGraph import _approximateGedGraphs  # noqa: E402
from GraphSAGE.graph_data import netlist_to_graph_tensor  # noqa: E402


# 计算与既有诊断脚本完全一致的归一化距离，数值越高表示差异越大。
def measurePair(task):
    dataset, macro, indexes, first, second = task
    normalizer = max(len(first[0]) + len(first[1]), len(second[0]) + len(second[1]))
    distance = min(1.0, _approximateGedGraphs(first, second) / normalizer)
    return dataset, macro, indexes, distance


# 对全部 core 建图并检查宏内所有无序对，抽样距离使用相同 core 序号对。
def compareDatasets(pairsPerMacro=5, workers=12):
    if not 1 <= pairsPerMacro <= 190 or workers < 1:
        raise ValueError("pairs 必须在 1–190，workers 必须为正")
    rng = random.Random(20261124)
    allPairs = list(itertools.combinations(range(20), 2))
    sourceMacros = loadMacroCores(PROJECT_ROOT / "_data")
    newMacros = loadMacroCores(PROJECT_ROOT / "_data_new")
    if len(sourceMacros) != 42 or set(sourceMacros) != set(newMacros):
        raise ValueError("新旧数据必须为相同的 42 个宏模型")
    sampled = {macro: rng.sample(allPairs, pairsPerMacro) for macro in sourceMacros}
    report = {"seed": 20261124, "pairs_per_macro": pairsPerMacro, "datasets": {}}
    tasks = []
    for dataset, macros in (("old", sourceMacros), ("new", newMacros)):
        summaries = {}
        for macro, paths in macros.items():
            if len(paths) != 20:
                raise ValueError(f"core 数量错误: {macro}")
            graphs = [attributedNetlistGraph(netlist_to_graph_tensor(p)) for p in paths]
            hashes = [weisfeilerLehmanHash(g[1]) for g in graphs]
            composition = [readCoreComposition(*g) for g in graphs]
            duplicates = [(a, b) for a, b in allPairs if hashes[a] == hashes[b]
                          and exactIsomorphic(graphs[a][1], graphs[b][1])]
            summaries[macro] = {
                "distinct_wl_hashes": len(set(hashes)),
                "isomorphic_pairs": duplicates,
                "distinct_size_signatures": len({(c["vertices"], c["edges"], c["mos"])
                                                 for c in composition}),
                "mos_min": min(c["mos"] for c in composition),
                "mos_max": max(c["mos"] for c in composition),
                "mos_mean": statistics.mean(c["mos"] for c in composition),
                "sampled_pairs": [],
            }
            for a, b in sampled[macro]:
                tasks.append((dataset, macro, (a, b), graphs[a][0], graphs[b][0]))
        aggregate = {
            "isomorphic_pairs": sum(len(m["isomorphic_pairs"]) for m in summaries.values()),
            "mean_distinct_wl_hashes": statistics.mean(m["distinct_wl_hashes"] for m in summaries.values()),
            "mean_size_signatures": statistics.mean(m["distinct_size_signatures"] for m in summaries.values()),
            "mean_mos_span": statistics.mean(m["mos_max"] - m["mos_min"] for m in summaries.values()),
            "mean_mos": statistics.mean(m["mos_mean"] for m in summaries.values()),
        }
        report["datasets"][dataset] = {"aggregate": aggregate, "macros": summaries}
        print(dataset, json.dumps(aggregate), flush=True)
    with mp.get_context("fork").Pool(workers) as pool:
        for index, (dataset, macro, pair, distance) in enumerate(
                pool.imap_unordered(measurePair, tasks, chunksize=1), 1):
            report["datasets"][dataset]["macros"][macro]["sampled_pairs"].append(
                {"cores": pair, "distance": distance})
            if index % 42 == 0:
                print(f"GED {index}/{len(tasks)}", flush=True)
    for result in report["datasets"].values():
        values = []
        for macro in result["macros"].values():
            macro["sampled_pairs"].sort(key=lambda p: p["cores"])
            distances = [p["distance"] for p in macro["sampled_pairs"]]
            macro["mean_sampled_distance"] = statistics.mean(distances)
            values.extend(distances)
        result["aggregate"]["mean_sampled_distance"] = statistics.mean(values)
        result["aggregate"]["median_sampled_distance"] = statistics.median(values)
    differences = [report["datasets"]["new"]["macros"][m]["mean_sampled_distance"]
                   - report["datasets"]["old"]["macros"][m]["mean_sampled_distance"]
                   for m in sourceMacros]
    report["paired_macro_distance"] = {
        "mean_delta": statistics.mean(differences),
        "increased_macros": sum(d > 0 for d in differences),
        "decreased_macros": sum(d < 0 for d in differences),
    }
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs-per-macro", type=int, default=5)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = compareDatasets(args.pairs_per_macro, args.workers)
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v["aggregate"] for k, v in result["datasets"].items()}), flush=True)
    print(json.dumps(result["paired_macro_distance"]), flush=True)
