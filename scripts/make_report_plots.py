"""Regenerate every figure referenced by report/optimization_report.md.

Reads only committed artifacts under experiments/ (summary.json per run/cell), so the
figures are reproducible from a clean checkout:

    python scripts/make_report_plots.py           # writes report/figures/*.png

Three figure families:
- M5 quantization overlays — delegated to the existing plot-overlay machinery in
  inference_lab.loadtest.plots (same four-hue categorical palette).
- M6 assemblies (prefix-cache A/B bars, KV-wall panels, batching-grid lines) — every M6
  cell is a single-level run directory (fresh seed per cell), so overlay curves don't
  apply; these figures assemble per-cell summaries, porting the session-time script that
  built the originals in experiments/.
- Cost summary — $/1M output tokens per fresh-regime config, with same-model API price
  reference lines (sources and dates in the report's cost section).

Style tokens (colors, axes, direct labeling) come from inference_lab.loadtest.plots so
every figure in the report reads as one system.
"""

import json
from pathlib import Path

from matplotlib.figure import Figure

from inference_lab.loadtest.plots import (
    _CATEGORICAL,
    _INK_MUTED,
    _INK_SECONDARY,
    _P90,
    _P99,
    _label_line_ends,
    _new_axes,
    _save,
    plot_overlay,
)

REPO = Path(__file__).resolve().parent.parent
EXPERIMENTS = REPO / "experiments"
FIGURES = REPO / "report" / "figures"

POD_USD_PER_HR = 0.69  # RunPod RTX 4090 secure cloud, constant across M4-M6


def load_level(run_dir: Path) -> dict:
    """The single concurrency level of an M6 cell (each cell is one closed-loop run)."""
    levels = json.loads((run_dir / "summary.json").read_text())
    assert len(levels) == 1, f"{run_dir} has {len(levels)} levels, expected 1"
    return levels[0]


def steady_state_tok_s(level: dict) -> float:
    """M4 convention: c x out-tokens-per-request / median latency."""
    out_per_req = level["output_tokens_total"] / level["num_requests"]
    return level["concurrency"] * out_per_req / level["latency_s"]["p50"]


def usd_per_million(tok_s: float) -> float:
    return POD_USD_PER_HR / (tok_s * 3600) * 1e6


# --- M5: quantization overlays (reuse the plot-overlay CLI machinery) ---------------


def quant_overlays() -> None:
    labeled = [
        ("fp16", EXPERIMENTS / "baseline-fp16"),
        ("awq", EXPERIMENTS / "quant-awq"),
        ("gptq", EXPERIMENTS / "quant-gptq"),
        ("fp8", EXPERIMENTS / "quant-fp8"),
    ]
    for path in plot_overlay(labeled, FIGURES):
        renamed = path.with_name(path.name.replace("overlay_", "quant_"))
        path.rename(renamed)
        print(f"wrote {renamed.relative_to(REPO)}")


# --- M6 experiment A: prefix caching on/off bars ------------------------------------


def _grouped_off_on_bars(
    title: str,
    ylabel: str,
    values: dict[str, tuple[float, float]],  # label -> (off, on)
    path: Path,
) -> None:
    fig, ax = _new_axes(title, "", ylabel)
    labels = list(values)
    x = range(len(labels))
    width = 0.36
    off_color, on_color = _CATEGORICAL[0], _CATEGORICAL[1]
    off_vals = [values[k][0] for k in labels]
    on_vals = [values[k][1] for k in labels]
    ax.bar([i - width / 2 for i in x], off_vals, width, color=off_color, label="caching off")
    ax.bar([i + width / 2 for i in x], on_vals, width, color=on_color, label="caching on")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=8)
    for i, k in enumerate(labels):
        off, on = values[k]
        pct = (on - off) / off * 100
        ax.annotate(
            f"{pct:+.0f}%",
            (i + width / 2, on),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            color=_INK_SECONDARY,
            fontsize=9,
        )
    ax.legend(frameon=False, labelcolor=_INK_SECONDARY, fontsize=9)
    _save(fig, path)
    print(f"wrote {path.relative_to(REPO)}")


def prefix_cache_figures() -> None:
    shapes = [("chat", "chat"), ("rag", "RAG"), ("none", "control")]
    ttft: dict[str, tuple[float, float]] = {}
    tput: dict[str, tuple[float, float]] = {}
    for shape, label in shapes:
        for c in (16, 64):
            off = load_level(EXPERIMENTS / f"prefix-cache-off-{shape}" / f"c{c}")
            on = load_level(EXPERIMENTS / f"prefix-cache-on-{shape}" / f"c{c}")
            key = f"{label} c={c}"
            ttft[key] = (off["ttft_s"]["p99"] * 1000, on["ttft_s"]["p99"] * 1000)
            tput[key] = (off["throughput_tok_s"], on["throughput_tok_s"])
    _grouped_off_on_bars(
        "Prefix caching: TTFT p99 by workload shape",
        "TTFT p99 (ms)",
        ttft,
        FIGURES / "prefix_cache_ttft_p99.png",
    )
    _grouped_off_on_bars(
        "Prefix caching: throughput by workload shape",
        "output tokens / s (window avg)",
        tput,
        FIGURES / "prefix_cache_throughput.png",
    )


# --- M6 experiment B2: the KV wall ---------------------------------------------------

UNIQUE_TOKENS_PER_SEQ = 2008  # 1,740 unique input + chat template + 256 output
KV_POOL = {0.90: 291_168, 0.80: 247_120}  # serve-log "GPU KV cache size" (tokens)


def kv_wall_figure() -> None:
    cells_090 = [
        ("unique-c96", 96),
        ("unique-c128", 128),
        ("unique-c160", 160),
        ("unique-c192", 192),
    ]
    wall_090 = KV_POOL[0.90] / UNIQUE_TOKENS_PER_SEQ
    wall_080 = KV_POOL[0.80] / UNIQUE_TOKENS_PER_SEQ

    def preemptions(run_dir: Path) -> int:
        def total(path: Path) -> float:
            val = 0.0
            for line in path.read_text().splitlines():
                if line.startswith("vllm:num_preemptions_total"):
                    val += float(line.rsplit(" ", 1)[1])
            return val

        return int(total(run_dir / "metrics_after.prom") - total(run_dir / "metrics_before.prom"))

    xs, preempts, tputs = [], [], []
    for name, c in cells_090:
        cell = EXPERIMENTS / "kv-pressure" / name
        xs.append(c / wall_090)
        preempts.append(preemptions(cell))
        tputs.append(load_level(cell).get("throughput_tok_s"))
    cell_080 = EXPERIMENTS / "kv-pressure" / "util0.80-c128"
    x_080 = 128 / wall_080
    p_080 = preemptions(cell_080)
    t_080 = load_level(cell_080)["throughput_tok_s"]

    fig = Figure(figsize=(7, 6.5), dpi=150)
    fig.set_facecolor("#fcfcfb")
    top, bottom = fig.subplots(2, 1, sharex=True)
    for ax, title, ylabel in (
        (top, "KV wall: preemptions vs offered load", "preemptions (count)"),
        (bottom, "KV wall: throughput vs offered load", "output tokens / s"),
    ):
        # Same chrome as _new_axes, applied to subplot axes.
        ax.set_facecolor("#fcfcfb")
        ax.set_title(title, color="#0b0b0b", fontsize=12, loc="left", pad=12)
        ax.set_ylabel(ylabel, color=_INK_SECONDARY, fontsize=10)
        ax.grid(axis="y", color="#e1e0d9", linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#c3c2b7")
        ax.tick_params(colors=_INK_MUTED, labelsize=9)

    top.bar(xs, preempts, width=0.055, color=_P99, label="util 0.90 (wall ≈ 145)")
    top.bar([x_080], [p_080], width=0.055, color=_CATEGORICAL[3], label="util 0.80 (wall ≈ 123)")
    top.axvline(1.0, color=_INK_MUTED, linewidth=1, linestyle="--")
    top.annotate(
        " the wall (c = pool ÷ unique tokens/seq)",
        (1.0, max(preempts) * 0.9),
        color=_INK_MUTED,
        fontsize=9,
    )
    top.legend(frameon=False, labelcolor=_INK_SECONDARY, fontsize=9)

    marker_kw = {
        "marker": "o", "markersize": 6, "markeredgecolor": "#fcfcfb", "markeredgewidth": 1.5,
    }
    bottom.plot(xs, tputs, color=_P90, linewidth=2, **marker_kw)
    bottom.plot([x_080], [t_080], color=_CATEGORICAL[3], **marker_kw)
    bottom.axvline(1.0, color=_INK_MUTED, linewidth=1, linestyle="--")
    bottom.set_xlabel(
        "offered load ÷ KV wall (concurrency ÷ pool capacity in sequences)",
        color=_INK_SECONDARY,
        fontsize=10,
    )
    bottom.set_ylim(bottom=0)

    path = FIGURES / "kv_wall.png"
    fig.savefig(path, bbox_inches="tight", facecolor="#fcfcfb")
    print(f"wrote {path.relative_to(REPO)}")


# --- M6 experiment B1: batching grid -------------------------------------------------


def batching_grid_figures() -> None:
    mns_values = (32, 64, 128, 256)
    concurrencies = (64, 128, 160)
    metric_specs = [
        (
            "grid_throughput_vs_c.png",
            "max-num-seqs: throughput vs offered concurrency",
            "output tokens / s (steady-state est.)",
            steady_state_tok_s,
        ),
        (
            "grid_ttft_p99_vs_c.png",
            "max-num-seqs: TTFT p99 vs offered concurrency",
            "TTFT p99 (s)",
            lambda level: level["ttft_s"]["p99"],
        ),
        (
            "grid_tpot_vs_c.png",
            "max-num-seqs: TPOT p50 vs offered concurrency",
            "TPOT p50 (ms)",
            lambda level: level["tpot_s"]["p50"] * 1000,
        ),
    ]
    for filename, title, ylabel, metric in metric_specs:
        fig, ax = _new_axes(title, "concurrency (closed-loop workers)", ylabel)
        ends = []
        for mns, color in zip(mns_values, _CATEGORICAL, strict=True):
            ys = [
                metric(load_level(EXPERIMENTS / "batching-grid" / f"mns{mns}-util0.90" / f"c{c}"))
                for c in concurrencies
            ]
            ax.plot(
                list(concurrencies), ys, color=color, marker="o", markersize=6,
                linewidth=2, markeredgecolor="#fcfcfb", markeredgewidth=1.5, label=f"mns={mns}",
            )
            ends.append((ys[-1], f"mns={mns}"))
        ax.set_xticks(list(concurrencies))
        ax.set_ylim(bottom=0)
        _label_line_ends(ax, concurrencies[-1], ends)
        ax.legend(frameon=False, labelcolor=_INK_SECONDARY, fontsize=9)
        path = FIGURES / filename
        _save(fig, path)
        print(f"wrote {path.relative_to(REPO)}")


# --- Cost summary --------------------------------------------------------------------


def cost_figure() -> None:
    # Fresh-traffic (cold) cells only; one regime per figure (M6 accounting rule).
    configs = [
        ("tuned chat\n512/256, c=160", EXPERIMENTS / "batching-grid" / "mns128-util0.90" / "c160"),
        ("RAG + cache\n86% hits, c=64", EXPERIMENTS / "prefix-cache-on-rag" / "c64"),
        ("RAG, cache off\nc=64", EXPERIMENTS / "prefix-cache-off-rag" / "c64"),
        ("unique 2k ctx\nbelow wall, c=128", EXPERIMENTS / "kv-pressure" / "unique-c128"),
        ("unique 2k ctx\npast wall, c=160", EXPERIMENTS / "kv-pressure" / "unique-c160"),
    ]
    costs = [usd_per_million(steady_state_tok_s(load_level(d))) for _, d in configs]

    fig, ax = _new_axes(
        "Cost per 1M output tokens by configuration (fresh traffic, steady-state)",
        "",
        "$ / 1M output tokens",
    )
    ax.bar(range(len(configs)), costs, width=0.6, color=_P90)
    ax.set_xticks(range(len(configs)))
    ax.set_xticklabels([label for label, _ in configs], fontsize=8)
    for i, cost in enumerate(costs):
        ax.annotate(
            f"${cost:.3f}", (i, cost), xytext=(0, 4), textcoords="offset points",
            ha="center", color=_INK_SECONDARY, fontsize=9,
        )
    # Same-model hosted API, priced at this workload's token shape (see report cost section).
    api_ref = 0.213
    ax.axhline(api_ref, color=_P99, linewidth=1.2, linestyle="--")
    ax.annotate(
        f"  hosted Qwen2.5-7B API ≈ ${api_ref:.3f} at the chat shape (2026-07-19)",
        (0, api_ref), xytext=(0, 5), textcoords="offset points",
        color=_P99, fontsize=8.5,
    )
    path = FIGURES / "cost_per_million.png"
    _save(fig, path)
    print(f"wrote {path.relative_to(REPO)}")


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    quant_overlays()
    prefix_cache_figures()
    kv_wall_figure()
    batching_grid_figures()
    cost_figure()


if __name__ == "__main__":
    main()
