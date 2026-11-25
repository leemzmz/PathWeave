from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import random

from _project_paths import (
    GRAPH_ALIGNMENT_MODEL_DIR,
    GRAPH_ALIGNMENT_RESULT_DIR,
    PROJECT_ROOT,
)


ENCODER_FEW_SHOT_PROTOCOL = "few_shot_path_attention_v1"
ENCODER_FEW_SHOT_SEED = 42
ENCODER_FEW_SHOTS = (2, 4, 6, 8, 10)
ENCODER_FEW_SHOT_BACKBONES = ("graphsage", "gat", "gin")
# 正式三表示：Netlist / vpath 注意力 / pathweave 注意力（注意力宽度统一 32-D）。
ENCODER_FEW_SHOT_BACKENDS = (
    "graph_netlist",
    "graph_vpath_attention",
    "graph_path_attention",
)
ALL_ENCODER_FEW_SHOT_BACKENDS = tuple(ENCODER_FEW_SHOT_BACKENDS)
ENCODER_FEW_SHOT_TRAIN_POOL = 10
ENCODER_FEW_SHOT_TEST_COUNT = 10
ENCODER_OPTIMIZER_STEPS_BY_SHOT = {
    2: 200,
    4: 400,
    6: 600,
    8: 800,
    10: 1000,
}
TARGET_ENCODER_EPOCHS = 300
# decoder 只在 10-shot 上运行；group 粒度 = (backbone, 表示)，即 9 组/seed。
DECODER_FEW_SHOT = 10
DECODER_GROUPS = tuple(
    (backbone, backend)
    for backbone in ENCODER_FEW_SHOT_BACKBONES
    for backend in ENCODER_FEW_SHOT_BACKENDS
)
# 步数由收敛决定（探针实测：400 步连 canonical target 都重构不出；3000 步 target-side 100%）。
DECODER_OPTIMIZER_STEPS = 3000
DECODER_INTERPOLATION_ALPHAS = (0.25, 0.5, 0.75, 1.0)
FORMAL_EMBEDDING_DEVICE = "cpu"
FORMAL_DECODER_DEVICE = "cuda"
ENCODER_FEW_SHOT_MODEL_ROOT = (
    GRAPH_ALIGNMENT_MODEL_DIR / ENCODER_FEW_SHOT_PROTOCOL
)
ENCODER_FEW_SHOT_RESULT_ROOT = (
    GRAPH_ALIGNMENT_RESULT_DIR / ENCODER_FEW_SHOT_PROTOCOL
)


def _seedRoot(root, seed):
    return Path(root) / f"seed_{int(seed)}"


def fewShotExperimentPaths(seed=ENCODER_FEW_SHOT_SEED):
    modelRoot = _seedRoot(ENCODER_FEW_SHOT_MODEL_ROOT, seed)
    resultRoot = _seedRoot(ENCODER_FEW_SHOT_RESULT_ROOT, seed)
    return {
        "model_root": modelRoot,
        "result_root": resultRoot,
        "manifest": resultRoot / "split_manifest.json",
        "protocol": resultRoot / "protocol.json",
        "target": modelRoot / "target_encoder.pt",
        "task_decoder_report": resultRoot / "shared_task_decoder_report.json",
        "task_decoder_summary": resultRoot / "shared_task_decoder_summary.json",
        "summary_json": resultRoot / "summary.json",
        "summary_csv": resultRoot / "summary.csv",
        "summary_md": resultRoot / "summary.md",
    }


def fewShotConfigurationPaths(
    shot, backbone, backend, seed=ENCODER_FEW_SHOT_SEED
):
    if shot not in ENCODER_FEW_SHOTS:
        raise ValueError("unsupported encoder few-shot count")
    if backbone not in ENCODER_FEW_SHOT_BACKBONES:
        raise ValueError("unsupported encoder few-shot backbone")
    if backend not in ALL_ENCODER_FEW_SHOT_BACKENDS:
        raise ValueError("unsupported encoder few-shot backend")
    roots = fewShotExperimentPaths(seed)
    relative = Path(f"shot_{shot}") / backbone / backend
    return {
        "circuit": roots["model_root"] / relative / "circuit_encoder.pt",
        "result": roots["result_root"] / relative / "result.json",
        "reload": roots["result_root"] / relative / "reload.json",
        "task_decoder_result": (
            roots["result_root"] / relative / "task_decoder_result.json"
        ),
    }


# decoder group 产物路径：checkpoint 落在该 group 的 10-shot 配置目录下，报告单独存放。
def decoderConfigurationPaths(
    backbone, backend, seed=ENCODER_FEW_SHOT_SEED
):
    if backbone not in ENCODER_FEW_SHOT_BACKBONES:
        raise ValueError("unsupported decoder group backbone")
    if backend not in ENCODER_FEW_SHOT_BACKENDS:
        raise ValueError("unsupported decoder group backend")
    roots = fewShotExperimentPaths(seed)
    relative = Path(f"shot_{DECODER_FEW_SHOT}") / backbone / backend
    group = f"{backbone}__{backend}"
    return {
        "group": group,
        "backbone": backbone,
        "backend": backend,
        "shot": DECODER_FEW_SHOT,
        "circuit": roots["model_root"] / relative / "circuit_encoder.pt",
        "decoder": roots["model_root"] / relative / "shared_task_decoder.pt",
        "result": roots["result_root"] / relative / "task_decoder_result.json",
        "report": (
            roots["result_root"] / "decoder_groups" / f"{group}.json"
        ),
    }


def optimizerStepsForShot(shot):
    if shot not in ENCODER_OPTIMIZER_STEPS_BY_SHOT:
        raise ValueError("unsupported encoder few-shot count")
    return ENCODER_OPTIMIZER_STEPS_BY_SHOT[shot]


# 为每个 topology 固定 10-core train pool 与 10-core 公共测试集。
def buildEncoderFewShotManifest(cases, seed=ENCODER_FEW_SHOT_SEED):
    grouped = defaultdict(list)
    topologyOrder = []
    for case in cases:
        if case.topology not in grouped:
            topologyOrder.append(case.topology)
        grouped[case.topology].append(case)
    rng = random.Random(int(seed))
    topologies = {}
    for topology in topologyOrder:
        values = sorted(grouped[topology], key=lambda case: case.name)
        if len(values) != ENCODER_FEW_SHOT_TRAIN_POOL + ENCODER_FEW_SHOT_TEST_COUNT:
            raise ValueError(
                f"encoder few-shot requires 20 cores for {topology}"
            )
        rng.shuffle(values)
        trainPool = [case.name for case in values[:ENCODER_FEW_SHOT_TRAIN_POOL]]
        test = [case.name for case in values[ENCODER_FEW_SHOT_TRAIN_POOL:]]
        topologies[topology] = {
            "train_pool": trainPool,
            "test": test,
            "train_by_shot": {
                str(shot): trainPool[:shot]
                for shot in ENCODER_FEW_SHOTS
            },
        }
    manifest = {
        "protocol": ENCODER_FEW_SHOT_PROTOCOL,
        "seed": int(seed),
        "shots": list(ENCODER_FEW_SHOTS),
        "train_pool_per_topology": ENCODER_FEW_SHOT_TRAIN_POOL,
        "common_test_per_topology": ENCODER_FEW_SHOT_TEST_COUNT,
        "topology_count": len(topologies),
        "case_count": len(cases),
        "topologies": topologies,
    }
    validateEncoderFewShotManifest(manifest, cases)
    return manifest


def validateEncoderFewShotManifest(manifest, cases, protocol=ENCODER_FEW_SHOT_PROTOCOL):
    if manifest.get("protocol") != protocol:
        raise ValueError("encoder few-shot manifest protocol mismatch")
    byName = {case.name: case for case in cases}
    if len(byName) != len(cases):
        raise ValueError("duplicate core name in encoder few-shot dataset")
    seenTest = set()
    seenTrainPool = set()
    for topology, entry in manifest["topologies"].items():
        trainPool = entry["train_pool"]
        test = entry["test"]
        if len(trainPool) != 10 or len(test) != 10:
            raise ValueError("encoder few-shot partition size mismatch")
        if set(trainPool) & set(test):
            raise ValueError("encoder few-shot train/test overlap")
        if any(byName[name].topology != topology for name in [*trainPool, *test]):
            raise ValueError("encoder few-shot topology membership mismatch")
        previous = set()
        for shot in ENCODER_FEW_SHOTS:
            names = entry["train_by_shot"][str(shot)]
            if len(names) != shot or names != trainPool[:shot]:
                raise ValueError("encoder few-shot prefix mismatch")
            if not previous <= set(names):
                raise ValueError("encoder few-shot nesting mismatch")
            previous = set(names)
        seenTrainPool.update(trainPool)
        seenTest.update(test)
    if seenTrainPool & seenTest or seenTrainPool | seenTest != set(byName):
        raise ValueError("encoder few-shot manifest does not cover dataset")
    if manifest["topology_count"] != 42 or manifest["case_count"] != 840:
        raise ValueError("encoder few-shot dataset cardinality mismatch")


def casesForEncoderFewShot(manifest, cases, shot=None):
    byName = {case.name: case for case in cases}
    testNames = []
    trainNames = []
    for entry in manifest["topologies"].values():
        testNames.extend(entry["test"])
        if shot is not None:
            trainNames.extend(entry["train_by_shot"][str(shot)])
    train = [byName[name] for name in sorted(trainNames)]
    test = [byName[name] for name in sorted(testNames)]
    return train, test


def encoderFewShotSplitPolicy(manifest, manifestSha256, shot, protocol=ENCODER_FEW_SHOT_PROTOCOL):
    return {
        "split_version": protocol,
        "split_strategy": f"encoder_few_shot_{shot}",
        "split_unit": "bare_core",
        "split_seed": int(manifest["seed"]),
        "train_cores_per_topology": int(shot),
        "train_pool_per_topology": ENCODER_FEW_SHOT_TRAIN_POOL,
        "common_test_cores_per_topology": ENCODER_FEW_SHOT_TEST_COUNT,
        "manifest_sha256": manifestSha256,
        "target": "functional_gm_rc_graph",
    }


def writeExperimentJson(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    temporary.replace(path)


def jsonFileSha256(path):
    return hashFile(path)


def hashFile(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# 协议中的源码键使用仓库相对路径，跨机器克隆不依赖绝对路径。
def _sourceProtocolKey(path):
    path = Path(path).resolve()
    for root, prefix in (
        (PROJECT_ROOT, ""),
        (PROJECT_ROOT.parent / "path", "CktPath/"),
    ):
        try:
            return prefix + path.relative_to(root).as_posix()
        except ValueError:
            continue
    raise ValueError(f"source is outside PathWeave and CktPath: {path}")


# 离散预测必须完全一致，连续指标仅允许浮点重放误差。
def compareReloadMetrics(reference, reloaded):
    differences = []

    def _compare(first, second, path):
        if isinstance(first, dict):
            if not isinstance(second, dict) or first.keys() != second.keys():
                raise ValueError(f"reload metric keys differ: {path}")
            for key in first:
                _compare(first[key], second[key], f"{path}.{key}")
        elif isinstance(first, list):
            if not isinstance(second, list) or len(first) != len(second):
                raise ValueError(f"reload metric list differs: {path}")
            for index, (left, right) in enumerate(zip(first, second)):
                _compare(left, right, f"{path}.{index}")
        elif isinstance(first, float):
            key = path.rsplit(".", 1)[-1]
            discrete = path.startswith("decoder_test.") or key.endswith(
                ("top1", "top3", "mrr", "exact", "rate")
            )
            if not isinstance(second, (int, float)) or not (
                math.isfinite(first) and math.isfinite(second)
            ):
                raise ValueError(f"nonfinite reload metric: {path}")
            if first != second:
                if discrete or not math.isclose(
                    first, second, rel_tol=1e-5, abs_tol=1e-5
                ):
                    raise ValueError(
                        f"reload metric differs: {path}: {first} != {second}"
                    )
                differences.append({
                    "metric": path,
                    "absolute_difference": abs(first - second),
                })
        elif first != second:
            raise ValueError(f"reload metric differs: {path}")

    groups = tuple(group for group in ("encoder_test", "decoder_test") if group in reference)
    if not groups or set(groups) != {group for group in ("encoder_test", "decoder_test") if group in reloaded}:
        raise ValueError("reload metric groups differ")
    for group in groups:
        _compare(reference[group], reloaded[group], group)
    return {
        "discrete_metrics_exact": True,
        "continuous_max_absolute_difference": max(
            (item["absolute_difference"] for item in differences),
            default=0.0,
        ),
        "continuous_differences": differences,
    }


def buildExperimentProtocol(cases, manifestSha256, sourcePaths, seed=42):
    return {
        "protocol": ENCODER_FEW_SHOT_PROTOCOL,
        "seed": int(seed),
        "shots": list(ENCODER_FEW_SHOTS),
        "backbones": list(ENCODER_FEW_SHOT_BACKBONES),
        "backends": list(ENCODER_FEW_SHOT_BACKENDS),
        "target_encoder_epochs": TARGET_ENCODER_EPOCHS,
        "circuit_optimizer_steps_by_shot": {
            str(shot): optimizerStepsForShot(shot)
            for shot in ENCODER_FEW_SHOTS
        },
        "formal_embedding_device": FORMAL_EMBEDDING_DEVICE,
        "formal_decoder_device": FORMAL_DECODER_DEVICE,
        "decoder_optimizer_steps": DECODER_OPTIMIZER_STEPS,
        "decoder_shot": DECODER_FEW_SHOT,
        "batch_size": 32,
        "learning_rate": 1e-3,
        "similarity_method": "ged",
        "hard_negative_config": {
            "mode": "manifold",
            "top_k": 5,
            "negative_count": 2,
            "sampling_temperature": 0.1,
            "margin_scale": 0.5,
            "manifold_loss_weight": 0.5,
            "ranking_loss_weight": 0.25,
        },
        "manifest_sha256": manifestSha256,
        "core_sha256": {
            case.name: hashFile(case.netlistPath)
            for case in cases
        },
        "target_sha256": {
            case.topology: hashFile(case.targetPath)
            for case in cases
        },
        "source_sha256": {
            _sourceProtocolKey(path): hashFile(path)
            for path in sourcePaths
        },
    }
