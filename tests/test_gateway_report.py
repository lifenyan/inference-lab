"""The log summarizer: totals, routing counts, and cost-basis restatement."""

import json

from inference_lab.gateway.report import load_requests, summarize

LOCAL_BASIS = "local $100.0/1M output tokens (test assumption: full utilization)"
FALLBACK_BASIS = "fallback fb-model list prices $10.0/1M in + $20.0/1M out (fetched 2026-07-19)"


def make_event(**overrides) -> dict:
    return {
        "event": "request",
        "request_id": "gw-x",
        "backend": "local",
        "decided_backend": "local",
        "route_reason": "default_local",
        "model_requested": "m",
        "model_served": "m",
        "stream": False,
        "est_prompt_tokens": 10,
        "input_tokens": 500,
        "output_tokens": 200,
        "ttft_s": 0.1,
        "latency_s": 1.5,
        "http_status": 200,
        "status": "ok",
        "error": None,
        "cost_usd": 0.02,
        "cost_basis": LOCAL_BASIS,
    } | overrides


def write_log(path, events) -> None:
    path.write_text("\n".join(json.dumps(e) for e in events) + "\n")


def test_load_requests_filters_non_request_events(tmp_path):
    path = tmp_path / "log.jsonl"
    write_log(path, [{"event": "startup"}, make_event(), make_event()])
    assert len(load_requests(path)) == 2


def test_summary_totals_reasons_and_bases(tmp_path):
    events = [
        make_event(cost_usd=0.02),
        make_event(cost_usd=0.03, route_reason="over_token_threshold", backend="fallback",
                   cost_basis=FALLBACK_BASIS, input_tokens=100, output_tokens=50),
        make_event(status="error", error="HTTP 502", cost_usd=None, ttft_s=None,
                   route_reason="local_error_fallback", backend="fallback",
                   cost_basis=FALLBACK_BASIS, input_tokens=None, output_tokens=None),
    ]
    text = summarize(events)

    assert "3 requests (2 ok, 1 errors)" in text
    # Per-backend rows carry token and cost totals.
    assert "local" in text and "fallback" in text
    assert "0.020000" in text  # local cost total
    assert "0.030000" in text  # fallback cost total (error request contributes nothing)
    # Every routing reason is counted.
    assert "default_local" in text
    assert "over_token_threshold" in text
    assert "local_error_fallback" in text
    # Cost totals restate the assumption they rest on, verbatim.
    assert LOCAL_BASIS in text
    assert FALLBACK_BASIS in text


def test_summary_handles_all_error_log(tmp_path):
    events = [make_event(status="error", cost_usd=None, ttft_s=None, latency_s=None)]
    text = summarize(events)
    assert "1 requests (0 ok, 1 errors)" in text
