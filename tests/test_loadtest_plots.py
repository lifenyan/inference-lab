"""Overlay plot rendering: files come out, the ≤4-series cap is enforced."""

import json
from pathlib import Path

import pytest

from inference_lab.loadtest.models import LevelSummary, PercentileStats
from inference_lab.loadtest.plots import plot_overlay

_LEVELS = [1, 4, 16]


def _stats(base: float) -> PercentileStats:
    return PercentileStats(
        mean=base, p50=base, p90=base * 1.2, p99=base * 1.5, min=base, max=base * 2
    )


def _summary(concurrency: int, scale: float) -> LevelSummary:
    return LevelSummary(
        concurrency=concurrency,
        num_requests=32,
        num_errors=0,
        error_rate=0.0,
        duration_s=10.0,
        output_tokens_total=8192,
        throughput_tok_s=100.0 * concurrency * scale,
        requests_per_s=1.0 * concurrency,
        ttft_s=_stats(0.05 * scale),
        tpot_s=_stats(0.02 / scale),
        latency_s=_stats(4.0 / scale),
    )


def _write_run(run_dir: Path, scale: float) -> Path:
    run_dir.mkdir(parents=True)
    summaries = [_summary(c, scale).model_dump() for c in _LEVELS]
    (run_dir / "summary.json").write_text(json.dumps(summaries), encoding="utf-8")
    return run_dir


def test_plot_overlay_renders_all_charts(tmp_path: Path) -> None:
    labeled = [
        ("fp16", _write_run(tmp_path / "fp16", 1.0)),
        ("awq", _write_run(tmp_path / "awq", 1.6)),
    ]
    out_dir = tmp_path / "plots"
    written = plot_overlay(labeled, out_dir)
    assert len(written) == 4
    for path in written:
        assert path.parent == out_dir
        assert path.exists() and path.stat().st_size > 0


def test_plot_overlay_rejects_more_than_four_runs(tmp_path: Path) -> None:
    labeled = [(f"run{i}", _write_run(tmp_path / f"run{i}", 1.0 + i / 10)) for i in range(5)]
    with pytest.raises(ValueError, match="at most 4"):
        plot_overlay(labeled, tmp_path / "plots")
