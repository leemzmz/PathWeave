"""导出 few_shot_path_attention_v1 的逐配置原始指标。

输入（只读）：
  _run/results/graph_alignment/few_shot_path_attention_v1/seed_<seed>/
    shot_<shot>/<backbone>/<backend>/result.json           encoder 指标
    shot_<shot>/<backbone>/<backend>/task_decoder_result.json  task decoder 指标
    shared_task_decoder_report.json                        seed 级 target-side 与 decoder 训练信息

输出：
  full_matrix.json  完整结构化数据（含全部 metric keys）
  full_matrix.csv   展平的一行一配置表（encoder_*/task_* 前缀列）

不做任何聚合、差值或筛选；聚合视图由 summarize_three_representation_results.py 负责。
"""

import argparse
import csv
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from _project_paths import GRAPH_ALIGNMENT_RESULT_DIR


DEFAULT_SEEDS = [42, 205, 316, 327, 330, 501]
BACKENDS = [
    "graph_netlist",
    "graph_vpath_attention",
    "graph_path_attention",
]
BACKBONES = ["graphsage", "gat", "gin"]
SHOTS = [2, 4, 6, 8, 10]


# 读取单个配置的 encoder 与 task decoder 产物；decoder 只在 10-shot 存在，缺失即为 None。
def _loadConfiguration(seedRoot, seed, shot, backbone, backend):
    configRoot = seedRoot / f"shot_{shot}" / backbone / backend
    encoderPayload = json.loads(
        (configRoot / "result.json").read_text(encoding="utf-8")
    )
    taskResultPath = configRoot / "task_decoder_result.json"
    taskPayload = (
        json.loads(taskResultPath.read_text(encoding="utf-8"))
        if taskResultPath.exists() else None
    )
    taskDecoder = (
        taskPayload["shared_task_decoder"] if taskPayload is not None else None
    )
    return {
        "seed": seed,
        "shot": shot,
        "backbone": backbone,
        "backend": backend,
        "protocol_sha256": encoderPayload["protocol_sha256"],
        "manifest_sha256": encoderPayload["manifest_sha256"],
        "artifact_sha256": {
            "encoder": encoderPayload.get("artifact_sha256", {}),
            "task_decoder": (
                taskPayload.get("artifact_sha256", {})
                if taskPayload is not None else {}
            ),
        },
        "encoder_training": encoderPayload["training"],
        "encoder_test": encoderPayload["encoder_test"],
        "task_decoder": taskDecoder,
        "task_decoder_policy": (
            taskPayload["decoder_training_policy"]
            if taskPayload is not None else None
        ),
        "calibration_shot": (
            taskPayload["calibration_shot"]
            if taskPayload is not None else None
        ),
        "test_samples": (
            taskPayload["test_samples"]
            if taskPayload is not None else None
        ),
    }


# 展平行：只保留标量指标，数组或嵌套结构留在 JSON 中。
def _flattenRow(config):
    row = {
        "seed": config["seed"],
        "shot": config["shot"],
        "backbone": config["backbone"],
        "backend": config["backend"],
        "protocol_sha256": config["protocol_sha256"],
        "encoder_optimizer_steps": config["encoder_training"]["optimizer_steps"],
        "encoder_seconds": config["encoder_training"]["seconds"],
        "test_samples": config["test_samples"],
    }
    for key, value in config["encoder_test"].items():
        if isinstance(value, (int, float)):
            row[f"encoder_{key}"] = value
    if config["task_decoder"]:
        for key, value in config["task_decoder"].items():
            if isinstance(value, (int, float)):
                row[f"task_{key}"] = value
    return row


# 收集全部配置与 seed 级 target-side / decoder 训练信息。
def exportMatrix(resultRoot, seeds):
    configurations = []
    targetSide = {}
    decoderTraining = {}
    for seed in seeds:
        seedRoot = resultRoot / f"seed_{seed}"
        report = json.loads(
            (seedRoot / "shared_task_decoder_report.json").read_text(
                encoding="utf-8"
            )
        )
        targetSide[str(seed)] = report["target_side_test"]
        decoderTraining[str(seed)] = report["training"]
        for shot in SHOTS:
            for backbone in BACKBONES:
                for backend in BACKENDS:
                    configurations.append(
                        _loadConfiguration(
                            seedRoot, seed, shot, backbone, backend
                        )
                    )
    payload = {
        "protocol_root": resultRoot.name,
        "seeds": list(seeds),
        "shots": SHOTS,
        "backbones": BACKBONES,
        "backends": BACKENDS,
        "configuration_count": len(configurations),
        "encoder_metric_keys": sorted(
            {
                key
                for config in configurations
                for key in config["encoder_test"]
            }
        ),
        "task_decoder_metric_keys": sorted(
            {
                key
                for config in configurations
                if config["task_decoder"]
                for key in config["task_decoder"]
            }
        ),
        "target_side_test_per_seed": targetSide,
        "decoder_training_per_seed": decoderTraining,
        "configurations": configurations,
    }
    return payload


# 写出 JSON 与展平 CSV。
def writeMatrix(resultRoot, payload):
    jsonPath = resultRoot / "full_matrix.json"
    jsonPath.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    rows = [_flattenRow(config) for config in payload["configurations"]]
    csvPath = resultRoot / "full_matrix.csv"
    # 只有 10-shot 行带 task_* 列，列集合必须取并集并保持出现顺序。
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with csvPath.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=fieldnames, restval=""
        )
        writer.writeheader()
        writer.writerows(rows)
    return jsonPath, csvPath


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("seeds", nargs="*", type=int, default=DEFAULT_SEEDS)
    args = parser.parse_args()
    resultRoot = GRAPH_ALIGNMENT_RESULT_DIR / "few_shot_path_attention_v1"
    payload = exportMatrix(resultRoot, args.seeds)
    jsonPath, csvPath = writeMatrix(resultRoot, payload)
    print(f"configurations={payload['configuration_count']}")
    print(jsonPath)
    print(csvPath)


if __name__ == "__main__":
    main()
