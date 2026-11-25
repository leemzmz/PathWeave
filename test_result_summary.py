import json
from statistics import stdev

import pytest

from Run import summarize_three_representation_results as summary


# 每个 group 的 target-side 不同，seed 汇总故意保留不同值以捕捉误用。
def _writeSummaryFixture(root, seeds):
    for seedIndex, seed in enumerate(seeds):
        seedRoot = root / f"seed_{seed}"
        groups, rows = {}, []
        task = {"graph_exact": 0.5, "valid_graph_rate": 0.95,
                "stage_exact": 0.8, "ged_similarity_mean": 0.9}
        for b, backbone in enumerate(summary.BACKBONES):
            for e, backend in enumerate(summary.BACKENDS):
                group = f"{backbone}__{backend}"
                exactCount = 12 + 3*b + 7*e + seedIndex - int(b == 0 and e == 0)
                groups[group] = {"backbone": backbone, "backend": backend,
                                 "target_side_test": {"sample_count": 42,
                                 "graph_exact": exactCount / 42,
                                 "valid_graph_rate": 1.0, "stage_exact": 1.0,
                                 "ged_similarity_mean": 0.9}}
                rows.append({"group": group, "shot": 10, "backbone": backbone,
                             "backend": backend, "task": task})
                for shot in summary.SHOTS:
                    directory = seedRoot / f"shot_{shot}" / backbone / backend
                    directory.mkdir(parents=True)
                    (directory / "result.json").write_text(json.dumps({"encoder_test": {
                        "retrieval_top1": 0.5, "retrieval_mrr": 0.7, "stage_exact": 0.8}}))
                    if shot == 10:
                        (directory / "task_decoder_result.json").write_text("{}")
        report = {"groups": groups, "calibration_sources": {key:"hash" for key in groups},
                  "target_side_test": {"sample_count": 42, "graph_exact": 0.99},
                  "training": {"completed_optimizer_steps": 3000}}
        (seedRoot / "shared_task_decoder_report.json").write_text(json.dumps(report))
        (seedRoot / "shared_task_decoder_summary.json").write_text(
            json.dumps({"decoder_shot": 10, "rows": rows}))


# 从真实文件读取到 JSON/CSV/Markdown 输出，检查分组、计数和跨 seed SD。
def testCanonicalDiagnosticsUseGroupObservations(tmp_path):
    seeds = [42, 205]
    _writeSummaryFixture(tmp_path, seeds)
    encoder, task, targets, audit = summary._loadResults(tmp_path, seeds)
    assert len(targets) == 18
    result = summary._buildSummary(seeds, encoder, task, targets, audit)
    assert result["target_side"]["sample_count"] == 756
    assert result["target_side"]["graph_exact_count"] == 403
    assert result["target_side"]["per_seed"]["42"]["graph_exact_count"] == 197
    assert result["target_side"]["per_seed"]["42"]["sample_count"] == 378
    assert result["overall"]["graph_netlist"]["target_side_graph_exact"] == pytest.approx(91 / 252)
    assert result["overall"]["graph_path_attention"]["target_side_graph_exact"] == pytest.approx(177 / 252)
    assert result["by_shot"]["2"]["graph_netlist"]["target_side_graph_exact"] is None
    cross = summary._buildCrossSeedSummary(seeds, encoder, task)
    assert cross["by_backend"]["graph_netlist"]["sample_std"]["target_side_graph_exact"] == pytest.approx(stdev([44/126,47/126]))
    summary._writeCsv(tmp_path / "summary.csv", result)
    summary._writeMarkdown(tmp_path / "summary.md", result)
    summary._writeCrossSeedCsv(tmp_path / "cross.csv", cross)
    summary._writeCrossSeedMarkdown(tmp_path / "cross.md", cross)
    assert "Target-side exact" in (tmp_path / "summary.md").read_text()
    assert "target_only" not in (tmp_path / "cross.csv").read_text()
