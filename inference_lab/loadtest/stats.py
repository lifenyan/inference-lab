"""Aggregation of raw request records into per-concurrency-level summaries.

Percentiles use linear interpolation (numpy's default), the same convention as
vLLM's ``benchmark_serving.py`` — keeping our numbers directly comparable when
we cross-validate in M4.
"""

from collections.abc import Sequence

import numpy as np

from inference_lab.loadtest.models import LevelSummary, PercentileStats, RequestRecord


def percentile(values: Sequence[float], q: float) -> float:
    """Return the q-th percentile (0-100) of ``values`` with linear interpolation."""
    if not values:
        raise ValueError("percentile() of empty sequence")
    return float(np.percentile(values, q))


def summarize_values(values: Sequence[float]) -> PercentileStats:
    """Compute the standard distribution summary for one metric."""
    arr = np.asarray(values, dtype=float)
    return PercentileStats(
        mean=float(arr.mean()),
        p50=float(np.percentile(arr, 50)),
        p90=float(np.percentile(arr, 90)),
        p99=float(np.percentile(arr, 99)),
        min=float(arr.min()),
        max=float(arr.max()),
    )


def summarize_level(concurrency: int, records: Sequence[RequestRecord]) -> LevelSummary:
    """Aggregate one concurrency level from its measured (non-warmup) records.

    The throughput window runs from the first measured request's start to the
    last measured request's end, so it reflects the server's sustained rate
    while the level was active. Failed requests count toward the error rate but
    contribute no latency samples or tokens.
    """
    measured = [r for r in records if not r.warmup]
    if not measured:
        raise ValueError(f"no measured records for concurrency level {concurrency}")

    ok = [r for r in measured if r.ok]
    num_errors = len(measured) - len(ok)

    window_start = min(r.start_s for r in measured)
    window_end = max(r.start_s + r.latency_s for r in measured)
    duration_s = window_end - window_start

    output_tokens_total = sum(r.output_tokens or 0 for r in ok)

    ttfts = [r.ttft_s for r in ok if r.ttft_s is not None]
    tpots = [r.tpot_s for r in ok if r.tpot_s is not None]
    latencies = [r.latency_s for r in ok]

    return LevelSummary(
        concurrency=concurrency,
        num_requests=len(measured),
        num_errors=num_errors,
        error_rate=num_errors / len(measured),
        duration_s=duration_s,
        output_tokens_total=output_tokens_total,
        throughput_tok_s=output_tokens_total / duration_s if duration_s > 0 else 0.0,
        requests_per_s=len(ok) / duration_s if duration_s > 0 else 0.0,
        ttft_s=summarize_values(ttfts) if ttfts else None,
        tpot_s=summarize_values(tpots) if tpots else None,
        latency_s=summarize_values(latencies) if latencies else None,
    )
