"""Integration: run the harness against the mock server with known injected delays.

The mock injects a fixed TTFT delay and per-token delay; the harness must
measure them within tolerance. Bounds are deliberately generous — CI machines
are noisy, and asyncio timer resolution adds jitter on top of the configured
sleeps — but tight enough to catch systematic mistakes (e.g. counting the
role-only delta as the first token would report TTFT ≈ 0).
"""

import json
import socket
import threading
import time

import pytest
import uvicorn

from inference_lab.common.config import EndpointConfig
from inference_lab.loadtest.mockserver import MockServerConfig, create_app
from inference_lab.loadtest.models import GeneratedPrompt, SyntheticWorkload
from inference_lab.loadtest.runner import SweepConfig, run_sweep
from inference_lab.loadtest.stats import summarize_level

TTFT_S = 0.15
TPOT_S = 0.02
OUTPUT_TOKENS = 8


@pytest.fixture(scope="module")
def mock_server():
    """Run the mock in a background thread on an ephemeral port; yield its base URL."""
    config = MockServerConfig(ttft_s=TTFT_S, tpot_s=TPOT_S, output_tokens=OUTPUT_TOKENS)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    server = uvicorn.Server(
        uvicorn.Config(create_app(config), host="127.0.0.1", port=port, log_level="warning")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started:
        if time.monotonic() > deadline:
            raise RuntimeError("mock server failed to start within 10s")
        time.sleep(0.01)
    yield f"http://127.0.0.1:{port}/v1"
    server.should_exit = True
    thread.join(timeout=5)


@pytest.fixture
def prompts():
    return [
        GeneratedPrompt(
            messages=[{"role": "user", "content": f"prompt {i} lorem ipsum"}],
            max_tokens=OUTPUT_TOKENS,
            est_input_tokens=4,
        )
        for i in range(8)
    ]


async def test_sweep_measures_injected_delays(mock_server, prompts, tmp_path):
    endpoint = EndpointConfig(base_url=mock_server, model="mock-model", timeout_s=10)
    workload = SyntheticWorkload(
        tokenizer="unused", input_tokens=4, output_tokens=OUTPUT_TOKENS, num_prompts=len(prompts)
    )
    sweep = SweepConfig(concurrency_levels=[1, 4], num_requests=8, warmup_requests=2)
    out_dir = tmp_path / "run"

    summaries = await run_sweep(endpoint, workload, prompts, sweep, out_dir)

    assert [s.concurrency for s in summaries] == [1, 4]
    for summary in summaries:
        assert summary.error_rate == 0.0
        assert summary.num_requests == 8

        # TTFT: injected delay, plus scheduling noise, never less than injected.
        assert summary.ttft_s is not None
        assert TTFT_S <= summary.ttft_s.p50 <= TTFT_S + 0.25
        # TPOT: mean inter-token gap should track the injected per-token delay.
        assert summary.tpot_s is not None
        assert TPOT_S * 0.8 <= summary.tpot_s.p50 <= TPOT_S + 0.05
        # Latency: at least TTFT + (n-1) gaps; generous upper bound.
        floor = TTFT_S + (OUTPUT_TOKENS - 1) * TPOT_S
        assert summary.latency_s is not None
        assert floor <= summary.latency_s.p50 <= floor + 0.5
        # Usage block flows through: exact token counts, not chunk-count fallback.
        assert summary.output_tokens_total == 8 * OUTPUT_TOKENS
        assert summary.throughput_tok_s > 0


async def test_run_dir_contents_allow_recomputing_summary(mock_server, prompts, tmp_path):
    endpoint = EndpointConfig(base_url=mock_server, model="mock-model", timeout_s=10)
    workload = SyntheticWorkload(
        tokenizer="unused", input_tokens=4, output_tokens=OUTPUT_TOKENS, num_prompts=len(prompts)
    )
    sweep = SweepConfig(concurrency_levels=[2], num_requests=6, warmup_requests=2)
    out_dir = tmp_path / "run"
    summaries = await run_sweep(endpoint, workload, prompts, sweep, out_dir)

    for name in ("workload.json", "meta.json", "requests.jsonl", "summary.json"):
        assert (out_dir / name).exists(), name

    meta = json.loads((out_dir / "meta.json").read_text())
    assert meta["endpoint"]["model"] == "mock-model"
    assert meta["finished_at"] is not None
    assert "api_key" not in json.dumps(meta)  # secrets never land in run metadata

    # Raw records are sufficient to recompute the published aggregates exactly.
    from inference_lab.loadtest.models import RequestRecord

    records = [
        RequestRecord.model_validate_json(line)
        for line in (out_dir / "requests.jsonl").read_text().splitlines()
    ]
    assert len(records) == 8  # 2 warmup + 6 measured
    assert sum(r.warmup for r in records) == 2
    recomputed = summarize_level(2, records)
    assert recomputed == summaries[0]


async def test_error_endpoint_is_recorded_not_raised(prompts, tmp_path):
    endpoint = EndpointConfig(base_url="http://127.0.0.1:9", model="mock-model", timeout_s=2)
    workload = SyntheticWorkload(
        tokenizer="unused", input_tokens=4, output_tokens=OUTPUT_TOKENS, num_prompts=len(prompts)
    )
    sweep = SweepConfig(concurrency_levels=[2], num_requests=2, warmup_requests=0)
    summaries = await run_sweep(endpoint, workload, prompts, sweep, tmp_path / "run")
    assert summaries[0].error_rate == 1.0
    assert summaries[0].ttft_s is None
