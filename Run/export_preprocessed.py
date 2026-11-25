"""由维护者导出三表示输入图；发布文件不包含 CktPath 实现。"""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from Experiment.GraphAlignmentData import discoverAlignmentCases
from Experiment.encoder_backends import getGraphBackendSpec
from GraphSAGE.PreprocessedData import BACKENDS, FORMAT_VERSION, hashFile, packGraph, unpackGraph
from GraphSAGE.PathExtraction import CKTPATH_ROOT


# 先核验原输入和提取源码，再导出固定表示；拒绝覆盖已发布包。
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(1)
    cases = discoverAlignmentCases(args.dataset)
    if len(cases) != 840:
        raise ValueError("expected 840 circuit inputs")
    cores = {c.name: hashFile(c.netlistPath) for c in cases}
    targets = {c.topology: hashFile(c.targetPath) for c in cases}
    protocolPaths = sorted((args.run / "results/graph_alignment").glob("*/seed_*/protocol.json"))
    if len(protocolPaths) != 6 or {json.loads(p.read_text())["seed"] for p in protocolPaths} != {42,205,316,327,330,501}:
        raise ValueError("expected six seed protocols")
    for path in protocolPaths:
        protocol = json.loads(path.read_text())
        if protocol["core_sha256"] != cores or protocol["target_sha256"] != targets:
            raise ValueError("archive input hashes differ")
        for name, expected in protocol["source_sha256"].items():
            if name.startswith("CktPath/") and hashFile(CKTPATH_ROOT / name[len("CktPath/"):]) != expected:
                raise ValueError(f"CktPath source differs: {name}")
    payload = {}
    for backend in BACKENDS:
        spec = getGraphBackendSpec(backend)
        payload[backend] = {}
        for case in cases:
            value = packGraph(spec.build_tensor(case.netlistPath), backend)
            unpackGraph(value, backend)
            payload[backend][case.name] = value
        print(backend, len(cases), "graphs exported", flush=True)
    args.output.mkdir(parents=True)
    torch.save(payload, args.output / "graphs.pt")
    manifest = {
        "format": FORMAT_VERSION, "sample_count": len(cases),
        "graphs_sha256": hashFile(args.output / "graphs.pt"),
        "core_sha256": cores, "target_sha256": targets,
        "protocol_sha256": {str(json.loads(p.read_text())["seed"]): hashFile(p) for p in protocolPaths},
        "extractor_sha256": {p.relative_to(CKTPATH_ROOT).as_posix(): hashFile(p)
                              for folder in ("core", "extraction") for p in sorted((CKTPATH_ROOT / folder).glob("*.py"))},
        "tensor_builder_sha256": {name: hashFile(ROOT / name) for name in
                                  ("GraphSAGE/graph_data.py", "GraphSAGE/SignalAttentionData.py")},
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print("Published tensor data:", args.output, flush=True)


if __name__ == "__main__":
    main()
