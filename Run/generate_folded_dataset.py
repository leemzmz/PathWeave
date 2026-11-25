"""调用 CktPath CLI，为现有 42 个宏模型生成并验证独立的 folded 数据集。"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CKTPATH_ROOT = PROJECT_ROOT.parent / "path"
SOURCE_ROOT = PROJECT_ROOT / "_data"
OUTPUT_ROOT = PROJECT_ROOT / "_data_new"
sys.path.insert(0, str(CKTPATH_ROOT))

from core import PathIR  # noqa: E402
from extraction import PathExtractManager  # noqa: E402
from generation.GmRcGraph import buildGmRcGraph, saveGmRcGraph  # noqa: E402
from generation.MacroModel import loadMacroModel  # noqa: E402
from generation.NetlistSanity import validateNetlistSyntax  # noqa: E402


# 计算文件内容摘要。
def _hashFile(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


# 计算完整源数据目录快照，用于确认归档输入未改动。
def _hashSourceDataset():
    records = [(p.relative_to(SOURCE_ROOT).as_posix(), _hashFile(p))
               for p in sorted(SOURCE_ROOT.rglob("*")) if p.is_file()]
    return hashlib.sha256(json.dumps(records).encode()).hexdigest()


# 保存机器可读清单。
def _saveJson(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")


# 从实际端口连接识别输入折叠对与差分负载深度，并交叉检查生成器标记。
def _readCoverage(netlist):
    rows = [line.split() for line in netlist.splitlines() if line.startswith("M")]
    inputs = [r for r in rows if r[2] in ("ipos", "ineg", "vref", "vfb")
              and r[3].startswith("n_tail")]
    if len(inputs) != 2:
        raise ValueError("无法定位完整差分输入对")
    foldedPairs = [(s, t) for s in inputs for t in rows
                   if s[1] == t[3] and s[5] != t[5] and t[2].startswith("b")]
    folded = len(foldedPairs) == 2
    namedFolded = any(r[0].startswith("Mfold") for r in rows)
    if len(foldedPairs) not in (0, 2) or folded != namedFolded:
        raise ValueError("folded 端口结构与生成器标记不一致")
    references = [r for r in rows if r[0].startswith("MloadRef")]
    mirrors = [r for r in rows if r[0].startswith("MloadMirror")]
    if len(references) != 1 or len(mirrors) != 1:
        raise ValueError("差分负载缺少 reference/mirror")
    depths = []
    for base in references + mirrors:
        depth, drain = 1, base[1]
        visited = set()
        while True:
            following = [r for r in rows if r[3] == drain and r[5] == base[5]
                         and r[2].startswith("b")]
            if not following:
                break
            if len(following) != 1 or drain in visited:
                raise ValueError("差分负载堆叠不唯一或存在环")
            visited.add(drain)
            depth += 1
            drain = following[0][1]
        depths.append(depth)
    if depths[0] != depths[1]:
        raise ValueError("两侧差分负载深度不一致")
    return folded, depths[0]


# 独立重载全部落盘网表，验证宏语义、数量、覆盖、源输入和文件哈希。
def verifyDataset():
    manifest = json.loads((OUTPUT_ROOT / "generation_manifest.json").read_text())
    sources = sorted(SOURCE_ROOT.glob("topo_*/macro_model.json"))
    if len(sources) != 42 or _hashSourceDataset() != manifest["source_dataset_sha256"]:
        raise ValueError("源宏模型数量或归档数据摘要不一致")
    if {p.name for p in OUTPUT_ROOT.glob("topo_*") if p.is_dir()} != {p.parent.name for p in sources}:
        raise ValueError("新旧宏模型目录集合不一致")
    totals, models, fingerprints = Counter(), {}, set()
    maxVpathMos = 0
    for source in sources:
        name = source.parent.name
        targetDir = OUTPUT_ROOT / name
        model = loadMacroModel(source)
        if loadMacroModel(targetDir / "macro_model.json") != model:
            raise ValueError(f"宏模型不一致: {name}")
        if _hashFile(targetDir / "gm_rc_graph.json") != _hashFile(source.parent / "gm_rc_graph.json"):
            raise ValueError(f"目标图与历史数据不一致: {name}")
        paths = sorted(targetDir.glob("*.cir"))
        if [p.name for p in paths] != [f"core_{i:02d}.cir" for i in range(20)]:
            raise ValueError(f"core 文件集合错误: {name}")
        info = json.loads((targetDir / "info.json").read_text())
        expectedHashes = {r["filename"]: r["fingerprint"] for r in info["core_variants"]}
        counts, records = Counter(), []
        for path in paths:
            netlist = path.read_text(encoding="utf-8")
            validateNetlistSyntax(netlist)
            manager = PathExtractManager().run(netlist)
            actual = manager.getPathIR()
            if (actual is None or not actual.hasCompleteAnchors()
                    or actual.semanticKey() != PathIR.fromCompactModel(model).semanticKey()):
                raise ValueError(f"PathIR 不一致: {path}")
            depth = max(sum(d.isMos() for d in v.devices) for v in manager.getVpaths())
            if depth > 4:
                raise ValueError(f"vpath 超过 4 MOS: {path}")
            maxVpathMos = max(maxVpathMos, depth)
            folded, loadDepth = _readCoverage(netlist)
            key = f"{'folded' if folded else 'non_folded'}_load_{loadDepth}"
            counts[key] += 1
            digest = _hashFile(path)
            if digest != expectedHashes[path.name] or digest in fingerprints:
                raise ValueError(f"网表摘要错误或全局重复: {path}")
            fingerprints.add(digest)
            records.append({"file": path.name, "folded": folded,
                            "load_depth": loadDepth, "sha256": digest})
        if not any(r["folded"] for r in records) or all(r["folded"] for r in records):
            raise ValueError(f"未覆盖 folded/non-folded: {name}")
        totals.update(counts)
        models[name] = {"coverage": dict(counts), "cores": records,
                        "attempts": info["attempt_count"]}
    report = {"macros": len(models), "cores": len(fingerprints),
              "coverage": dict(totals), "max_vpath_mos": maxVpathMos,
              "source_unchanged": True, "all_path_ir_consistent": True,
              "target_graphs_identical": True, "models": models}
    _saveJson(OUTPUT_ROOT / "validation_report.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "models"}), flush=True)
    return report


# 固定使用当前归档的 42 个宏模型和现有 seed 配方，拒绝覆盖已有输出。
def generateDataset():
    sources = sorted(SOURCE_ROOT.glob("topo_*/macro_model.json"))
    if len(sources) != 42:
        raise ValueError("必须恰有 42 个归档宏模型")
    if OUTPUT_ROOT.exists():
        raise FileExistsError(f"输出已存在，拒绝覆盖: {OUTPUT_ROOT}")
    for source in sources:
        loadMacroModel(source)
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "cktpath_commit": subprocess.check_output(
            ["git", "-C", str(CKTPATH_ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "source_dataset_sha256": _hashSourceDataset(),
        "base_seed": 42, "seed_step": 17, "core_count_per_macro": 20,
        "source_hashes": {p.parent.name: _hashFile(p) for p in sources},
        "commands": [],
    }
    OUTPUT_ROOT.mkdir()
    inputDir = OUTPUT_ROOT / "source_models"
    inputDir.mkdir()
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    for index, source in enumerate(sources, 1):
        name = source.parent.name
        staged = inputDir / f"{name.removeprefix('topo_')}.json"
        shutil.copyfile(source, staged)
        command = [sys.executable, str(CKTPATH_ROOT / "scripts/generate_topology.py"),
                   "--model", str(staged), "--output", str(OUTPUT_ROOT),
                   "--core_n", "20", "--peripheral_n", "0", "--bias_addon", "none",
                   "--base_seed", "42", "--seed_step", "17",
                   "--min_tail_devices", "1", "--max_tail_devices", "3",
                   "--min_diff_load_devices", "1", "--max_diff_load_devices", "3",
                   "--max_series_devices", "2", "--max_mos_per_vpath", "4",
                   "--match_mirror_vpath_stack"]
        manifest["commands"].append(command)
        _saveJson(OUTPUT_ROOT / "generation_manifest.json", manifest)
        subprocess.run(command, check=True, env=environment)
        saveGmRcGraph(buildGmRcGraph(loadMacroModel(source)),
                      OUTPUT_ROOT / name / "gm_rc_graph.json")
        print(f"[{index}/42] {name}: generated 20 cores", flush=True)
    verifyDataset()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    verifyDataset() if args.verify_only else generateDataset()
