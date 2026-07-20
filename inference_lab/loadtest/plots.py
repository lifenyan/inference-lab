"""Plots from a run directory: the three curves the optimization report is built on.

- throughput vs concurrency — where does scaling flatten (the knee)?
- latency vs throughput — the fundamental serving trade-off curve
- TTFT percentiles vs concurrency — what users feel as load grows

Plus overlay variants of the same curves that draw several runs on shared axes
(A/B experiments: baseline vs quantization variants, M5+).

Style notes: percentile families (p50/p90/p99) are ordered magnitudes of one
metric, so they use an ordinal one-hue ramp (light→dark blue) instead of
unrelated hues — the ordering survives colorblind viewing, and each line is
direct-labeled at its end so identity never rides on color alone. Overlays
compare *runs* (identity, not magnitude), so they use a categorical palette
instead: at most four series, hues assigned in fixed caller order, every line
direct-labeled at its end in addition to the legend (two of the four hues sit
below 3:1 contrast on this surface — labels carry identity, not color alone).
"""

from collections.abc import Callable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from inference_lab.loadtest.models import LevelSummary
from inference_lab.loadtest.runner import load_summaries

# Reference palette (light mode): ordinal blue ramp for percentile series,
# text/ink roles for chrome. p50/p90/p99 keep these colors on every chart.
_SURFACE = "#fcfcfb"
_INK_PRIMARY = "#0b0b0b"
_INK_SECONDARY = "#52514e"
_INK_MUTED = "#898781"
_GRID = "#e1e0d9"
_AXIS = "#c3c2b7"
_P50 = "#86b6ef"
_P90 = "#2a78d6"
_P99 = "#104281"
# Categorical hues for run overlays (validated all-pairs on this surface, ≤4 series).
# Assigned in the caller's fixed order — for M5: baseline, awq, gptq, fp8.
_CATEGORICAL = ("#2a78d6", "#008300", "#e87ba4", "#eda100")

_LINE_KW = {"linewidth": 2, "marker": "o", "markersize": 6, "markeredgewidth": 1.5}


def _new_axes(title: str, xlabel: str, ylabel: str) -> tuple[Figure, Axes]:
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    fig.set_facecolor(_SURFACE)
    ax.set_facecolor(_SURFACE)
    ax.set_title(title, color=_INK_PRIMARY, fontsize=12, loc="left", pad=12)
    ax.set_xlabel(xlabel, color=_INK_SECONDARY, fontsize=10)
    ax.set_ylabel(ylabel, color=_INK_SECONDARY, fontsize=10)
    ax.grid(axis="y", color=_GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_AXIS)
    ax.tick_params(colors=_INK_MUTED, labelsize=9)
    return fig, ax


def _concurrency_axis(ax: Axes, levels: list[int]) -> None:
    """Log2 x-axis with a tick at each swept level (sweeps double concurrency)."""
    if len(levels) > 1 and max(levels) / max(min(levels), 1) >= 8:
        ax.set_xscale("log", base=2)
    ax.set_xticks(levels)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.minorticks_off()


def _label_line_ends(ax: Axes, x: float, labels: list[tuple[float, str]]) -> None:
    """Direct-label each line at its right end, nudging labels apart if they collide."""
    y_min, y_max = ax.get_ylim()
    min_gap = (y_max - y_min) * 0.045
    placed: list[float] = []
    for y, text in sorted(labels):
        y_label = y if not placed else max(y, placed[-1] + min_gap)
        placed.append(y_label)
        ax.annotate(
            f" {text}",
            (x, y),
            xytext=(x, y_label),
            color=_INK_SECONDARY,
            fontsize=9,
            va="center",
            annotation_clip=False,
        )


def _save(fig: Figure, path: Path) -> None:
    fig.savefig(path, bbox_inches="tight", facecolor=_SURFACE)
    plt.close(fig)


def plot_throughput_vs_concurrency(summaries: list[LevelSummary], path: Path) -> None:
    """Server throughput (output tokens/sec) at each swept concurrency level."""
    levels = [s.concurrency for s in summaries]
    throughput = [s.throughput_tok_s for s in summaries]
    fig, ax = _new_axes(
        "Throughput vs concurrency", "concurrency (closed-loop workers)", "output tokens / s"
    )
    ax.plot(levels, throughput, color=_P90, markeredgecolor=_SURFACE, **_LINE_KW)
    _concurrency_axis(ax, levels)
    ax.set_ylim(bottom=0)
    _save(fig, path)


def plot_latency_vs_throughput(summaries: list[LevelSummary], path: Path) -> None:
    """The trade-off curve: each point is one concurrency level (annotated)."""
    pts = [s for s in summaries if s.latency_s is not None]
    x = [s.throughput_tok_s for s in pts]
    fig, ax = _new_axes(
        "Latency vs throughput (each point: one concurrency level)",
        "output tokens / s",
        "request latency (s)",
    )
    series = [
        ("p50", _P50, [s.latency_s.p50 for s in pts]),
        ("p99", _P99, [s.latency_s.p99 for s in pts]),
    ]
    for name, color, ys in series:
        ax.plot(x, ys, color=color, markeredgecolor=_SURFACE, label=name, **_LINE_KW)
    ax.set_ylim(bottom=0)
    _label_line_ends(ax, x[-1], [(ys[-1], name) for name, _, ys in series])
    for s, xi in zip(pts, x, strict=True):
        ax.annotate(
            f"c={s.concurrency}",
            (xi, s.latency_s.p50),
            xytext=(0, -14),
            textcoords="offset points",
            ha="center",
            color=_INK_MUTED,
            fontsize=8,
        )
    ax.legend(frameon=False, labelcolor=_INK_SECONDARY, fontsize=9)
    _save(fig, path)


def plot_ttft_vs_concurrency(summaries: list[LevelSummary], path: Path) -> None:
    """TTFT p50/p90/p99 as concurrency grows — queueing shows up here first."""
    pts = [s for s in summaries if s.ttft_s is not None]
    levels = [s.concurrency for s in pts]
    fig, ax = _new_axes("TTFT percentiles vs concurrency", "concurrency", "time to first token (s)")
    series = [
        ("p50", _P50, [s.ttft_s.p50 for s in pts]),
        ("p90", _P90, [s.ttft_s.p90 for s in pts]),
        ("p99", _P99, [s.ttft_s.p99 for s in pts]),
    ]
    for name, color, ys in series:
        ax.plot(levels, ys, color=color, markeredgecolor=_SURFACE, label=name, **_LINE_KW)
    _concurrency_axis(ax, levels)
    ax.set_ylim(bottom=0)
    _label_line_ends(ax, levels[-1], [(ys[-1], name) for name, _, ys in series])
    ax.legend(frameon=False, labelcolor=_INK_SECONDARY, fontsize=9)
    _save(fig, path)


def plot_run(run_dir: Path) -> list[Path]:
    """Render all three plots from ``<run_dir>/summary.json``; return written paths."""
    summaries = load_summaries(run_dir)
    outputs = []
    for fn, name in (
        (plot_throughput_vs_concurrency, "throughput_vs_concurrency.png"),
        (plot_latency_vs_throughput, "latency_vs_throughput.png"),
        (plot_ttft_vs_concurrency, "ttft_vs_concurrency.png"),
    ):
        path = run_dir / name
        fn(summaries, path)
        outputs.append(path)
    return outputs


LabeledRuns = list[tuple[str, list[LevelSummary]]]


def _overlay_lines(
    runs: LabeledRuns,
    title: str,
    xlabel: str,
    ylabel: str,
    xy: Callable[[LevelSummary], tuple[float, float | None]],
) -> tuple[Figure, Axes]:
    """One line per run on shared axes; fixed hue per position, direct-labeled ends."""
    if len(runs) > len(_CATEGORICAL):
        raise ValueError(f"overlay supports at most {len(_CATEGORICAL)} runs, got {len(runs)}")
    fig, ax = _new_axes(title, xlabel, ylabel)
    ends: list[tuple[float, str]] = []
    x_end = 0.0
    for (label, summaries), color in zip(runs, _CATEGORICAL, strict=False):
        pts = [xy(s) for s in summaries]
        pts = [(x, y) for x, y in pts if y is not None]
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        ax.plot(xs, ys, color=color, markeredgecolor=_SURFACE, label=label, **_LINE_KW)
        ends.append((ys[-1], label))
        x_end = max(x_end, xs[-1])
    ax.set_ylim(bottom=0)
    _label_line_ends(ax, x_end, ends)
    ax.legend(frameon=False, labelcolor=_INK_SECONDARY, fontsize=9)
    return fig, ax


def plot_overlay_throughput(runs: LabeledRuns, path: Path) -> None:
    """Throughput vs concurrency, one line per run (window-average tok/s)."""
    fig, ax = _overlay_lines(
        runs,
        "Throughput vs concurrency",
        "concurrency (closed-loop workers)",
        "output tokens / s (window avg)",
        lambda s: (s.concurrency, s.throughput_tok_s),
    )
    _concurrency_axis(ax, [s.concurrency for s in runs[0][1]])
    _save(fig, path)


def plot_overlay_tpot(runs: LabeledRuns, path: Path) -> None:
    """TPOT p50 vs concurrency — the per-user decode-speed signature."""
    fig, ax = _overlay_lines(
        runs,
        "TPOT p50 vs concurrency",
        "concurrency (closed-loop workers)",
        "time per output token, p50 (ms)",
        lambda s: (s.concurrency, s.tpot_s.p50 * 1000 if s.tpot_s else None),
    )
    _concurrency_axis(ax, [s.concurrency for s in runs[0][1]])
    _save(fig, path)


def plot_overlay_ttft(runs: LabeledRuns, path: Path) -> None:
    """TTFT p50 vs concurrency, one line per run."""
    fig, ax = _overlay_lines(
        runs,
        "TTFT p50 vs concurrency",
        "concurrency (closed-loop workers)",
        "time to first token, p50 (ms)",
        lambda s: (s.concurrency, s.ttft_s.p50 * 1000 if s.ttft_s else None),
    )
    _concurrency_axis(ax, [s.concurrency for s in runs[0][1]])
    _save(fig, path)


def plot_overlay_latency_throughput(runs: LabeledRuns, path: Path) -> None:
    """The trade-off frontier: latency p99 vs throughput, one curve per run."""
    fig, _ = _overlay_lines(
        runs,
        "Latency p99 vs throughput (each point: one concurrency level)",
        "output tokens / s (window avg)",
        "request latency, p99 (s)",
        lambda s: (s.throughput_tok_s, s.latency_s.p99 if s.latency_s else None),
    )
    _save(fig, path)


def plot_overlay(labeled_dirs: list[tuple[str, Path]], out_dir: Path) -> list[Path]:
    """Render the four overlay plots from several run directories; return written paths."""
    runs: LabeledRuns = [(label, load_summaries(d)) for label, d in labeled_dirs]
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for fn, name in (
        (plot_overlay_throughput, "overlay_throughput_vs_concurrency.png"),
        (plot_overlay_tpot, "overlay_tpot_p50_vs_concurrency.png"),
        (plot_overlay_ttft, "overlay_ttft_p50_vs_concurrency.png"),
        (plot_overlay_latency_throughput, "overlay_latency_p99_vs_throughput.png"),
    ):
        path = out_dir / name
        fn(runs, path)
        outputs.append(path)
    return outputs
