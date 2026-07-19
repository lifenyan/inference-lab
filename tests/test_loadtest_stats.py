"""Stats aggregation: known inputs must produce exact percentiles and aggregates."""

import pytest

from inference_lab.loadtest.models import RequestRecord
from inference_lab.loadtest.stats import percentile, summarize_level, summarize_values


def _record(
    *,
    start_s: float,
    latency_s: float,
    ttft_s: float | None = 0.1,
    tpot_s: float | None = 0.01,
    output_tokens: int | None = 10,
    ok: bool = True,
    warmup: bool = False,
) -> RequestRecord:
    return RequestRecord(
        request_id="r",
        concurrency=4,
        warmup=warmup,
        ok=ok,
        error=None if ok else "boom",
        start_s=start_s,
        latency_s=latency_s,
        ttft_s=ttft_s if ok else None,
        tpot_s=tpot_s if ok else None,
        output_tokens=output_tokens if ok else None,
    )


class TestPercentile:
    def test_exact_values_linear_interpolation(self):
        values = list(range(1, 101))  # 1..100
        assert percentile(values, 50) == pytest.approx(50.5)
        assert percentile(values, 90) == pytest.approx(90.1)
        assert percentile(values, 99) == pytest.approx(99.01)
        assert percentile(values, 0) == 1.0
        assert percentile(values, 100) == 100.0

    def test_single_value(self):
        assert percentile([7.0], 50) == 7.0
        assert percentile([7.0], 99) == 7.0

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            percentile([], 50)

    def test_summarize_values(self):
        stats = summarize_values([1.0, 2.0, 3.0, 4.0])
        assert stats.mean == pytest.approx(2.5)
        assert stats.p50 == pytest.approx(2.5)
        assert stats.min == 1.0
        assert stats.max == 4.0


class TestSummarizeLevel:
    def test_throughput_and_window(self):
        # Two requests: window from earliest start (1.0) to latest end (3.0 + 1.0).
        records = [
            _record(start_s=1.0, latency_s=2.0, output_tokens=30),
            _record(start_s=3.0, latency_s=1.0, output_tokens=30),
        ]
        summary = summarize_level(4, records)
        assert summary.duration_s == pytest.approx(3.0)
        assert summary.output_tokens_total == 60
        assert summary.throughput_tok_s == pytest.approx(20.0)
        assert summary.requests_per_s == pytest.approx(2 / 3.0)
        assert summary.error_rate == 0.0

    def test_warmup_excluded_from_everything(self):
        records = [
            _record(start_s=0.0, latency_s=5.0, output_tokens=999, warmup=True),
            _record(start_s=10.0, latency_s=1.0, output_tokens=10),
            _record(start_s=10.0, latency_s=2.0, output_tokens=10),
        ]
        summary = summarize_level(4, records)
        assert summary.num_requests == 2
        assert summary.output_tokens_total == 20
        assert summary.duration_s == pytest.approx(2.0)  # warmup start at 0.0 ignored

    def test_errors_count_but_contribute_no_samples(self):
        records = [
            _record(start_s=0.0, latency_s=1.0, ttft_s=0.5, output_tokens=10),
            _record(start_s=0.0, latency_s=9.0, ok=False),
        ]
        summary = summarize_level(4, records)
        assert summary.num_errors == 1
        assert summary.error_rate == 0.5
        assert summary.output_tokens_total == 10
        assert summary.ttft_s is not None and summary.ttft_s.max == pytest.approx(0.5)
        # Latency percentiles only cover successful requests.
        assert summary.latency_s is not None and summary.latency_s.max == pytest.approx(1.0)

    def test_all_failed_has_no_percentiles(self):
        records = [_record(start_s=0.0, latency_s=1.0, ok=False)]
        summary = summarize_level(2, records)
        assert summary.error_rate == 1.0
        assert summary.ttft_s is None and summary.tpot_s is None and summary.latency_s is None
        assert summary.throughput_tok_s == 0.0

    def test_no_measured_records_raises(self):
        with pytest.raises(ValueError):
            summarize_level(1, [_record(start_s=0.0, latency_s=1.0, warmup=True)])
