import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, stdev
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from _project_paths import GRAPH_ALIGNMENT_RESULT_DIR
from Experiment.EncoderFewShotExperiment import (
    DECODER_FEW_SHOT,
    DECODER_OPTIMIZER_STEPS,
)


DEFAULT_SEEDS = [42, 205, 316, 327, 330, 501]
BACKENDS = [
    "graph_netlist",
    "graph_vpath_attention",
    "graph_path_attention",
]
BACKBONES = ["graphsage", "gat", "gin"]
SHOTS = [2, 4, 6, 8, 10]
TEST_SAMPLES_PER_CONFIGURATION = 420
CROSS_SEED_METRICS = [
    "encoder_top1",
    "encoder_mrr",
    "target_side_graph_exact",
    "task_graph_exact",
    "task_valid_graph",
    "task_stage_exact",
    "task_ged_similarity",
]
T_CRITICAL_95 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
}


# Load and validate all per-seed artifacts without changing them.
def _loadResults(resultRoot, seeds):
    encoderRows = []
    taskRows = []
    targetRows = []
    audit = {}
    expectedTaskRows = len(BACKENDS) * len(BACKBONES)
    for seed in seeds:
        seedRoot = resultRoot / f"seed_{seed}"
        taskSummary = json.loads(
            (seedRoot / "shared_task_decoder_summary.json").read_text()
        )
        taskReport = json.loads(
            (seedRoot / "shared_task_decoder_report.json").read_text()
        )
        if len(taskSummary["rows"]) != expectedTaskRows:
            raise RuntimeError(
                f"seed {seed}: expected {expectedTaskRows} task rows"
            )
        if len(taskReport["calibration_sources"]) != expectedTaskRows:
            raise RuntimeError(
                f"seed {seed}: expected {expectedTaskRows} decoder sources"
            )
        if taskSummary.get("decoder_shot") != DECODER_FEW_SHOT:
            raise RuntimeError(f"seed {seed}: decoder shot mismatch")
        expectedGroups = {f"{backbone}__{backend}" for backbone in BACKBONES for backend in BACKENDS}
        if set(taskReport["groups"]) != expectedGroups:
            raise RuntimeError(f"seed {seed}: target-side group matrix is incomplete")
        for group, values in taskReport["groups"].items():
            if group != f"{values['backbone']}__{values['backend']}":
                raise RuntimeError(f"seed {seed}: decoder group identity mismatch")
            targetRows.append({**values["target_side_test"], "seed": seed,
                               "backbone": values["backbone"], "backend": values["backend"]})
        for path in sorted(seedRoot.glob("shot_*/*/*/result.json")):
            relative = path.relative_to(seedRoot)
            shot = int(relative.parts[0].split("_")[1])
            encoderTest = json.loads(path.read_text())["encoder_test"]
            encoderRows.append({
                "seed": seed,
                "shot": shot,
                "backbone": relative.parts[1],
                "backend": relative.parts[2],
                "encoder_top1": encoderTest["retrieval_top1"],
                "encoder_mrr": encoderTest["retrieval_mrr"],
                "encoder_stage_exact": encoderTest["stage_exact"],
            })
        taskRows.extend(
            {
                "seed": seed,
                "shot": row["shot"],
                "backbone": row["backbone"],
                "backend": row["backend"],
                "task": row["task"],
                "target_side_graph_exact": taskReport["groups"][
                    f"{row['backbone']}__{row['backend']}"
                ]["target_side_test"]["graph_exact"],
            }
            for row in taskSummary["rows"]
        )
        seedAudit = {
            "encoder_result_files": len(list(
                seedRoot.glob("shot_*/*/*/result.json")
            )),
            "task_decoder_result_files": len(list(
                seedRoot.glob("shot_*/*/*/task_decoder_result.json")
            )),
            "task_decoder_optimizer_steps": taskReport[
                "training"
            ]["completed_optimizer_steps"],
        }
        if seedAudit["encoder_result_files"] != 45:
            raise RuntimeError(f"seed {seed}: artifact matrix is incomplete")
        if seedAudit["task_decoder_result_files"] != expectedTaskRows:
            raise RuntimeError(f"seed {seed}: decoder matrix is incomplete")
        if (
            seedAudit["task_decoder_optimizer_steps"]
            != DECODER_OPTIMIZER_STEPS
        ):
            raise RuntimeError(f"seed {seed}: unexpected decoder step count")
        audit[str(seed)] = seedAudit
    encoderKeys = {
        (row["seed"], row["shot"], row["backbone"], row["backend"])
        for row in encoderRows
    }
    expectedEncoderKeys = {
        (seed, shot, backbone, backend)
        for seed in seeds
        for shot in SHOTS
        for backbone in BACKBONES
        for backend in BACKENDS
    }
    if encoderKeys != expectedEncoderKeys:
        raise RuntimeError(
            "three-representation configuration matrix is incomplete"
        )
    taskKeys = {
        (row["seed"], row["shot"], row["backbone"], row["backend"])
        for row in taskRows
    }
    expectedTaskKeys = {
        (seed, DECODER_FEW_SHOT, backbone, backend)
        for seed in seeds
        for backbone in BACKBONES
        for backend in BACKENDS
    }
    if taskKeys != expectedTaskKeys:
        raise RuntimeError("decoder group matrix is incomplete")
    return encoderRows, taskRows, targetRows, audit


def _filterRows(rows, **conditions):
    return [
        row for row in rows
        if all(row[key] == value for key, value in conditions.items())
    ]


def _meanField(rows, field):
    return mean(row[field] for row in rows)


# Aggregate encoder metrics (all shots) and decoder metrics (10-shot only).
def _summarizeGroup(encoderRows, taskRows):
    taskMetrics = [row["task"] for row in taskRows]
    sampleCount = len(taskRows) * TEST_SAMPLES_PER_CONFIGURATION
    return {
        "configuration_count": len(encoderRows),
        "encoder_configuration_count": len(encoderRows),
        "task_configuration_count": len(taskRows),
        "test_prediction_count": sampleCount,
        "encoder_top1": (
            _meanField(encoderRows, "encoder_top1") if encoderRows else None
        ),
        "encoder_mrr": (
            _meanField(encoderRows, "encoder_mrr") if encoderRows else None
        ),
        "target_side_graph_exact": (
            _meanField(taskRows, "target_side_graph_exact")
            if taskRows else None
        ),
        "task_graph_exact": (
            _meanField(taskMetrics, "graph_exact") if taskMetrics else None
        ),
        "task_graph_exact_count": sum(
            round(row["graph_exact"] * TEST_SAMPLES_PER_CONFIGURATION)
            for row in taskMetrics
        ),
        "task_valid_graph": (
            _meanField(taskMetrics, "valid_graph_rate")
            if taskMetrics else None
        ),
        "task_stage_exact": (
            _meanField(taskMetrics, "stage_exact")
            if taskMetrics else None
        ),
        "task_ged_similarity": (
            _meanField(taskMetrics, "ged_similarity_mean")
            if taskMetrics else None
        ),
    }


def _studentInterval(values):
    if len(values) < 2:
        raise ValueError("paired confidence interval requires at least two seeds")
    center = mean(values)
    degrees = len(values) - 1
    critical = T_CRITICAL_95.get(degrees, 1.960)
    radius = critical * stdev(values) / math.sqrt(len(values))
    return center, center - radius, center + radius


# Compare representations on matched seed/shot/backbone configurations.
def _summarizePair(taskRows, seeds, lhs, rhs):
    rowByKey = {
        (row["seed"], row["shot"], row["backbone"], row["backend"]): row
        for row in taskRows
    }
    seedDeltas = {}
    wins = 0
    ties = 0
    losses = 0
    for seed in seeds:
        deltas = []
        for shot in (DECODER_FEW_SHOT,):
            for backbone in BACKBONES:
                lhsValue = rowByKey[
                    (seed, shot, backbone, lhs)
                ]["task"]["graph_exact"]
                rhsValue = rowByKey[
                    (seed, shot, backbone, rhs)
                ]["task"]["graph_exact"]
                delta = lhsValue - rhsValue
                deltas.append(delta)
                if delta > 0.0:
                    wins += 1
                elif delta < 0.0:
                    losses += 1
                else:
                    ties += 1
        seedDeltas[str(seed)] = mean(deltas)
    center, lower, upper = _studentInterval(list(seedDeltas.values()))
    return {
        "lhs": lhs,
        "rhs": rhs,
        "metric": "task_graph_exact",
        "mean_delta": center,
        "seed_level_t95_ci": [lower, upper],
        "seed_deltas": seedDeltas,
        "configuration_wins": wins,
        "configuration_ties": ties,
        "configuration_losses": losses,
    }


# 按实际 decoder group 计数，避免把九组平均比例舍入成一组样本数。
def _summarizeTargetRows(rows):
    sampleCount = sum(row["sample_count"] for row in rows)
    exactCount = sum(round(row["sample_count"] * row["graph_exact"]) for row in rows)
    return {
        "sample_count": sampleCount,
        "graph_exact_count": exactCount,
        "graph_exact": exactCount / sampleCount,
        "valid_graph": sum(row["sample_count"] * row["valid_graph_rate"] for row in rows) / sampleCount,
        "stage_exact": sum(row["sample_count"] * row["stage_exact"] for row in rows) / sampleCount,
        "ged_similarity": sum(row["sample_count"] * row["ged_similarity_mean"] for row in rows) / sampleCount,
    }


# 汇总检索、任务解码与独立的 canonical target-side 诊断。
def _buildSummary(seeds, encoderRows, taskRows, targetRows, audit):
    overall = {}
    byShot = {}
    byBackbone = {}
    for backend in BACKENDS:
        overall[backend] = _summarizeGroup(
            _filterRows(encoderRows, backend=backend),
            _filterRows(taskRows, backend=backend),
        )
    for shot in SHOTS:
        byShot[str(shot)] = {}
        for backend in BACKENDS:
            byShot[str(shot)][backend] = _summarizeGroup(
                _filterRows(encoderRows, shot=shot, backend=backend),
                _filterRows(taskRows, shot=shot, backend=backend),
            )
    for backbone in BACKBONES:
        byBackbone[backbone] = {}
        for backend in BACKENDS:
            byBackbone[backbone][backend] = _summarizeGroup(
                _filterRows(
                    encoderRows, backbone=backbone, backend=backend
                ),
                _filterRows(
                    taskRows, backbone=backbone, backend=backend
                ),
            )
    return {
        "seeds": seeds,
        "backends": BACKENDS,
        "shots": SHOTS,
        "backbones": BACKBONES,
        "configuration_count": len(taskRows),
        "encoder_configuration_count": len(encoderRows),
        "task_configuration_count": len(taskRows),
        "test_samples_per_configuration": TEST_SAMPLES_PER_CONFIGURATION,
        "target_side": {
            **_summarizeTargetRows(targetRows),
            "per_seed": {
                str(seed): _summarizeTargetRows(_filterRows(targetRows, seed=seed))
                for seed in seeds
            },
        },
        "overall": overall,
        "by_shot": byShot,
        "by_backbone": byBackbone,
        "paired_task_graph_exact": [
            _summarizePair(
                taskRows, seeds, "graph_vpath_attention", "graph_netlist"
            ),
            _summarizePair(
                taskRows, seeds, "graph_path_attention",
                "graph_vpath_attention",
            ),
            _summarizePair(
                taskRows, seeds, "graph_path_attention",
                "graph_netlist",
            ),
        ],
        "audit": audit,
    }


# Compute seed means first so the reported variation is genuinely cross-seed.
def _buildCrossSeedSummary(seeds, encoderRows, taskRows):
    perSeed = {}
    for seed in seeds:
        perSeed[str(seed)] = {}
        for backend in BACKENDS:
            perSeed[str(seed)][backend] = _summarizeGroup(
                _filterRows(encoderRows, seed=seed, backend=backend),
                _filterRows(taskRows, seed=seed, backend=backend),
            )
    backendStats = {}
    for backend in BACKENDS:
        backendStats[backend] = {
            "mean": {},
            "sample_std": {},
        }
        for metric in CROSS_SEED_METRICS:
            values = [
                perSeed[str(seed)][backend][metric]
                for seed in seeds
            ]
            backendStats[backend]["mean"][metric] = mean(values)
            backendStats[backend]["sample_std"][metric] = (
                stdev(values) if len(values) > 1 else 0.0
            )
    return {
        "seeds": seeds,
        "seed_count": len(seeds),
        "metrics": CROSS_SEED_METRICS,
        "per_seed": perSeed,
        "by_backend": backendStats,
    }


# 按数据生成结论块：只在 CI 排除 0 时才声明正向增益。
def _interpretation(pairs, labels):
    lines = []
    for pair in pairs:
        lower, upper = pair["seed_level_t95_ci"]
        name = f"{labels[pair['lhs']]} - {labels[pair['rhs']]}"
        if lower > 0.0:
            verdict = "statistically positive"
        elif upper < 0.0:
            verdict = "statistically negative"
        else:
            verdict = "not separable from zero"
        lines.append(
            f"- {name}: {verdict} (mean {100.0 * pair['mean_delta']:+.2f}pp, "
            f"95% CI [{100.0 * lower:+.2f}, {100.0 * upper:+.2f}]pp, "
            f"win/tie/loss {pair['configuration_wins']}/"
            f"{pair['configuration_ties']}/{pair['configuration_losses']})."
        )
    lines.append(
        "- 每个表示都有独立的同结构 decoder；比较在 seed × backbone × 10-shot 的 18 个单元上配对。"
    )
    lines.append(
        "- Netlist 与路径表示的比较同时改变表示和编码机制；decoder 比较仅作为次级证据。"
    )
    return lines


def _percent(value):
    if value is None:
        return "n/a"
    return f"{100.0 * value:.2f}%"


def _number(value, digits=4):
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def _writeCsv(path, summary):
    fields = [
        "scope",
        "scope_value",
        "backend",
        "configuration_count",
        "test_prediction_count",
        "encoder_top1",
        "encoder_mrr",
        "target_side_graph_exact",
        "task_graph_exact",
        "task_graph_exact_count",
        "task_valid_graph",
        "task_stage_exact",
        "task_ged_similarity",
    ]
    rows = []
    for backend, metrics in summary["overall"].items():
        rows.append({"scope": "overall", "scope_value": "all", "backend": backend, **metrics})
    for shot, values in summary["by_shot"].items():
        for backend, metrics in values.items():
            rows.append({"scope": "shot", "scope_value": shot, "backend": backend, **metrics})
    for backbone, values in summary["by_backbone"].items():
        for backend, metrics in values.items():
            rows.append({"scope": "backbone", "scope_value": backbone, "backend": backend, **metrics})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            {key: row.get(key) for key in fields} for row in rows
        )


def _writeMarkdown(path, summary):
    labels = {
        "graph_netlist": "Netlist",
        "graph_vpath_attention": "Vpath Attention",
        "graph_path_attention": "Path Attention",
    }
    lines = [
        "# Three-Representation Cross-Seed Summary",
        "",
        f"- Seeds: {', '.join(map(str, summary['seeds']))}.",
        f"- Encoder configurations: {summary['encoder_configuration_count']}"
        f" (all shots); decoder configurations: {summary['configuration_count']}"
        f" ({DECODER_FEW_SHOT}-shot groups only).",
        "- Common test samples per configuration: 420.",
        "- Paired confidence intervals use seed-level two-sided 95% Student-t intervals.",
        "",
        "## Overall",
        "",
        "| Representation | Encoder Top-1 | Target-side exact | Task exact | Task valid | Stage exact | GED |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for backend in BACKENDS:
        row = summary["overall"][backend]
        lines.append(
            f"| {labels[backend]} | {_percent(row['encoder_top1'])} | "
            f"{_percent(row['target_side_graph_exact'])} | "
            f"{_percent(row['task_graph_exact'])} | "
            f"{_percent(row['task_valid_graph'])} | "
            f"{_percent(row['task_stage_exact'])} | "
            f"{_number(row['task_ged_similarity'])} |"
        )
    lines.extend([
        "",
        "## Task Exact By Shot",
        "",
        "| Shot | Netlist | Vpath Attention | Path Attention |",
        "|---:|---:|---:|---:|",
    ])
    for shot in SHOTS:
        values = summary["by_shot"][str(shot)]
        lines.append(
            f"| {shot} | {_percent(values['graph_netlist']['task_graph_exact'])} | "
            f"{_percent(values['graph_vpath_attention']['task_graph_exact'])} | "
            f"{_percent(values['graph_path_attention']['task_graph_exact'])} |"
        )
    lines.extend([
        "",
        f"- Decoder 只在 {DECODER_FEW_SHOT}-shot 上运行，其余 shot 的 task 指标为 n/a。",
        "",
        "## Task Exact By Backbone",
        "",
        "| Backbone | Netlist | Vpath Attention | Path Attention |",
        "|---|---:|---:|---:|",
    ])
    for backbone in BACKBONES:
        values = summary["by_backbone"][backbone]
        lines.append(
            f"| {backbone} | {_percent(values['graph_netlist']['task_graph_exact'])} | "
            f"{_percent(values['graph_vpath_attention']['task_graph_exact'])} | "
            f"{_percent(values['graph_path_attention']['task_graph_exact'])} |"
        )
    lines.extend([
        "",
        "## Paired Task Exact",
        "",
        "| Comparison | Mean delta | Seed-level 95% CI | Win/Tie/Loss |",
        "|---|---:|---:|---:|",
    ])
    for pair in summary["paired_task_graph_exact"]:
        lower, upper = pair["seed_level_t95_ci"]
        lines.append(
            f"| {labels[pair['lhs']]} - {labels[pair['rhs']]} | "
            f"{100.0 * pair['mean_delta']:+.2f}pp | "
            f"[{100.0 * lower:+.2f}, {100.0 * upper:+.2f}]pp | "
            f"{pair['configuration_wins']}/{pair['configuration_ties']}/"
            f"{pair['configuration_losses']} |"
        )
    target = summary["target_side"]
    lines.extend([
        "",
        "## Canonical Target-Side Diagnostic",
        "",
        f"- Canonical graph exact: {target['graph_exact_count']}/{target['sample_count']} ({_percent(target['graph_exact'])}).",
        f"- Canonical valid graph: {_percent(target['valid_graph'])}.",
        f"- Canonical stage exact: {_percent(target['stage_exact'])}.",
        "- Per-seed graph exact: " + ", ".join(
            f"{seed}={values['graph_exact_count']}/{values['sample_count']}"
            for seed, values in target["per_seed"].items()
        ) + ".",
        "",
        "## Interpretation",
        "",
    ] + _interpretation(summary["paired_task_graph_exact"], labels))
    path.write_text("\n".join(lines) + "\n")


def _writeCrossSeedCsv(path, crossSeed):
    fields = ["backend"]
    for metric in CROSS_SEED_METRICS:
        fields.extend((f"{metric}_mean", f"{metric}_sample_std"))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for backend in BACKENDS:
            row = {"backend": backend}
            for metric in CROSS_SEED_METRICS:
                row[f"{metric}_mean"] = crossSeed[
                    "by_backend"
                ][backend]["mean"][metric]
                row[f"{metric}_sample_std"] = crossSeed[
                    "by_backend"
                ][backend]["sample_std"][metric]
            writer.writerow(row)


def _meanAndStd(crossSeed, backend, metric, percent=True):
    values = crossSeed["by_backend"][backend]
    center = values["mean"][metric]
    spread = values["sample_std"][metric]
    if percent:
        return f"{100.0 * center:.2f} +/- {100.0 * spread:.2f}%"
    return f"{center:.4f} +/- {spread:.4f}"


def _writeCrossSeedMarkdown(path, crossSeed):
    labels = {
        "graph_netlist": "Netlist",
        "graph_vpath_attention": "Vpath Attention",
        "graph_path_attention": "Path Attention",
    }
    lines = [
        "# Cross-Seed Mean Summary",
        "",
        f"- Seeds: {', '.join(map(str, crossSeed['seeds']))}.",
        "- Values are mean +/- sample standard deviation across seeds.",
        "- Each seed contributes equally.",
        "",
        "| Representation | Encoder Top-1 | Encoder MRR | Target-side exact | Task exact | Task valid | Stage exact | GED |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for backend in BACKENDS:
        lines.append(
            f"| {labels[backend]} | "
            f"{_meanAndStd(crossSeed, backend, 'encoder_top1')} | "
            f"{_meanAndStd(crossSeed, backend, 'encoder_mrr')} | "
            f"{_meanAndStd(crossSeed, backend, 'target_side_graph_exact')} | "
            f"{_meanAndStd(crossSeed, backend, 'task_graph_exact')} | "
            f"{_meanAndStd(crossSeed, backend, 'task_valid_graph')} | "
            f"{_meanAndStd(crossSeed, backend, 'task_stage_exact')} | "
            f"{_meanAndStd(crossSeed, backend, 'task_ged_similarity', False)} |"
        )
    lines.extend([
        "",
        "## Per-Seed Task Exact",
        "",
        "| Seed | Netlist | Vpath Attention | Path Attention |",
        "|---:|---:|---:|---:|",
    ])
    for seed in crossSeed["seeds"]:
        values = crossSeed["per_seed"][str(seed)]
        lines.append(
            f"| {seed} | "
            f"{_percent(values['graph_netlist']['task_graph_exact'])} | "
            f"{_percent(values['graph_vpath_attention']['task_graph_exact'])} | "
            f"{_percent(values['graph_path_attention']['task_graph_exact'])} |"
        )
    path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("seeds", nargs="*", type=int, default=DEFAULT_SEEDS)
    args = parser.parse_args()
    resultRoot = GRAPH_ALIGNMENT_RESULT_DIR / "few_shot_path_attention_v1"
    encoderRows, taskRows, targetRows, audit = _loadResults(
        resultRoot, args.seeds
    )
    summary = _buildSummary(
        args.seeds, encoderRows, taskRows, targetRows, audit
    )
    crossSeed = _buildCrossSeedSummary(
        args.seeds, encoderRows, taskRows
    )
    jsonPath = resultRoot / "three_representation_summary.json"
    csvPath = resultRoot / "three_representation_summary.csv"
    markdownPath = resultRoot / "three_representation_summary.md"
    crossSeedJsonPath = resultRoot / "cross_seed_mean_summary.json"
    crossSeedCsvPath = resultRoot / "cross_seed_mean_summary.csv"
    crossSeedMarkdownPath = resultRoot / "cross_seed_mean_summary.md"
    jsonPath.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    _writeCsv(csvPath, summary)
    _writeMarkdown(markdownPath, summary)
    crossSeedJsonPath.write_text(
        json.dumps(crossSeed, indent=2, sort_keys=True) + "\n"
    )
    _writeCrossSeedCsv(crossSeedCsvPath, crossSeed)
    _writeCrossSeedMarkdown(crossSeedMarkdownPath, crossSeed)
    print(markdownPath)
    print(crossSeedMarkdownPath)


if __name__ == "__main__":
    main()
