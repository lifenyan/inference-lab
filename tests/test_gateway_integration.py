"""Integration: the gateway routing between two mock backends over real HTTP.

The local and fallback mocks reply with distinct canned texts, so every test
can assert *who* actually served a request from the response body — plus the
``x-gateway-*`` headers and the JSONL log line, which must all agree.
"""

import json
import os
import socket
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import httpx
import pytest
import uvicorn

from inference_lab.gateway.app import create_app
from inference_lab.gateway.config import (
    FallbackBackendConfig,
    GatewayConfig,
    LocalBackendConfig,
    RoutingConfig,
)
from inference_lab.loadtest.mockserver import MockServerConfig
from inference_lab.loadtest.mockserver import create_app as create_mock_app

LOCAL_TEXT = "alpha beta gamma delta epsilon zeta"
FALLBACK_TEXT = "fallback served this request"
LOCAL_TTFT_S = 0.12
LOCAL_TPOT_S = 0.02
KEY_ENV = "GW_TEST_FALLBACK_KEY"
KEY_VALUE = "sk-test-secret-never-in-logs"


@contextmanager
def run_app(app):
    """Serve an ASGI app in a background thread on an ephemeral port; yield its URL."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started:
        if time.monotonic() > deadline:
            raise RuntimeError("server failed to start within 10s")
        time.sleep(0.01)
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    thread.join(timeout=5)


@pytest.fixture(scope="module")
def local_mock():
    config = MockServerConfig(ttft_s=LOCAL_TTFT_S, tpot_s=LOCAL_TPOT_S, canned_text=LOCAL_TEXT)
    with run_app(create_mock_app(config)) as url:
        yield url


@pytest.fixture(scope="module")
def fallback_mock():
    config = MockServerConfig(ttft_s=0.01, tpot_s=0.001, canned_text=FALLBACK_TEXT)
    with run_app(create_mock_app(config)) as url:
        yield url


def make_gateway_config(
    local_url: str, fallback_url: str, log_path: Path, **routing_overrides
) -> GatewayConfig:
    routing = {
        "prompt_token_threshold": 50,
        "health_check_interval_s": 30.0,
        "circuit_breaker_failures": 3,
        "circuit_breaker_open_s": 30.0,
    } | routing_overrides
    return GatewayConfig(
        local=LocalBackendConfig(
            base_url=f"{local_url}/v1",
            served_models=["mock-model"],
            timeout_s=10.0,
            usd_per_1m_output_tokens=100.0,
            cost_assumption="test assumption: full utilization, fresh traffic",
        ),
        fallback=FallbackBackendConfig(
            base_url=f"{fallback_url}/v1",
            model="fallback-model",
            api_key_env=KEY_ENV,
            timeout_s=10.0,
            usd_per_1m_input_tokens=10.0,
            usd_per_1m_output_tokens=20.0,
            prices_fetched_on="2026-07-19",
        ),
        routing=RoutingConfig(**routing),
        log_path=log_path,
    )


@contextmanager
def run_gateway(config: GatewayConfig):
    os.environ[KEY_ENV] = KEY_VALUE
    try:
        with run_app(create_app(config)) as url:
            yield url
    finally:
        os.environ.pop(KEY_ENV, None)


@pytest.fixture(scope="module")
def gateway(local_mock, fallback_mock, tmp_path_factory):
    """A gateway with both backends healthy; yields (url, log_path)."""
    log_path = tmp_path_factory.mktemp("gateway") / "gateway_log.jsonl"
    config = make_gateway_config(local_mock, fallback_mock, log_path)
    with run_gateway(config) as url:
        yield url, log_path


def logged_event(log_path: Path, request_id: str, timeout_s: float = 3.0) -> dict:
    """Fetch the log line for one request id (streams log after the response ends)."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if log_path.exists():
            for line in log_path.read_text().splitlines():
                event = json.loads(line)
                if event.get("request_id") == request_id:
                    return event
        time.sleep(0.02)
    raise AssertionError(f"no log event for request {request_id}")


def post(url: str, body: dict) -> httpx.Response:
    return httpx.post(f"{url}/v1/chat/completions", json=body, timeout=10.0)


def test_default_request_served_locally_and_logged(gateway):
    url, log_path = gateway
    response = post(url, {"model": "mock-model", "messages": [{"role": "user", "content": "hi"}]})
    assert response.status_code == 200
    data = response.json()
    assert data["choices"][0]["message"]["content"].strip() == LOCAL_TEXT
    assert response.headers["x-gateway-backend"] == "local"
    assert response.headers["x-gateway-route-reason"] == "default_local"

    event = logged_event(log_path, response.headers["x-gateway-request-id"])
    assert event["backend"] == "local"
    assert event["route_reason"] == "default_local"
    assert event["status"] == "ok"
    assert event["model_served"] == "mock-model"
    assert event["stream"] is False
    assert isinstance(event["est_prompt_tokens"], int)
    assert event["input_tokens"] == data["usage"]["prompt_tokens"]
    assert event["output_tokens"] == data["usage"]["completion_tokens"]
    # Local cost: output_tokens × $100/1M, basis restates the assumption.
    assert abs(event["cost_usd"] - event["output_tokens"] * 100.0 / 1e6) < 1e-12
    assert "test assumption" in event["cost_basis"]
    assert event["latency_s"] > 0


def test_long_prompt_routes_to_fallback_with_model_rewrite(gateway):
    url, log_path = gateway
    # ~300 chars ≈ 79 estimated tokens > threshold 50.
    response = post(
        url, {"model": "mock-model", "messages": [{"role": "user", "content": "x" * 300}]}
    )
    assert response.headers["x-gateway-backend"] == "fallback"
    assert response.headers["x-gateway-route-reason"] == "over_token_threshold"
    data = response.json()
    assert data["choices"][0]["message"]["content"].strip() == FALLBACK_TEXT
    assert data["model"] == "fallback-model"  # local-model request rewritten for the fallback

    event = logged_event(log_path, response.headers["x-gateway-request-id"])
    assert event["model_requested"] == "mock-model"
    assert event["model_served"] == "fallback-model"
    # Fallback cost: in × $10/1M + out × $20/1M from exact usage counts.
    expected = event["input_tokens"] * 10.0 / 1e6 + event["output_tokens"] * 20.0 / 1e6
    assert abs(event["cost_usd"] - expected) < 1e-12
    assert "fetched 2026-07-19" in event["cost_basis"]


def test_unknown_model_routes_to_fallback_unchanged(gateway):
    url, log_path = gateway
    response = post(
        url, {"model": "some-other-model", "messages": [{"role": "user", "content": "hi"}]}
    )
    assert response.headers["x-gateway-backend"] == "fallback"
    assert response.headers["x-gateway-route-reason"] == "model_not_local"
    assert response.json()["model"] == "some-other-model"  # passed through, not rewritten

    event = logged_event(log_path, response.headers["x-gateway-request-id"])
    assert event["model_served"] == "some-other-model"


def test_streaming_passthrough_is_incremental_and_intact(gateway):
    url, log_path = gateway
    body = {
        "model": "mock-model",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
    }
    chunks: list[dict] = []
    arrival_s: list[float] = []
    saw_done = False
    with httpx.stream(
        "POST", f"{url}/v1/chat/completions", json=body, timeout=10.0
    ) as response:
        assert response.headers["x-gateway-backend"] == "local"
        request_id = response.headers["x-gateway-request-id"]
        for line in response.iter_lines():
            if not line.startswith("data: "):
                continue
            data = line[len("data: ") :]
            if data.strip() == "[DONE]":
                saw_done = True
                break
            chunks.append(json.loads(data))
            arrival_s.append(time.perf_counter())

    assert saw_done
    # Role-only chunk first — forwarded untouched, faithful to the upstream stream shape.
    assert chunks[0]["choices"][0]["delta"] == {"role": "assistant"}
    content_chunks = [
        (c["choices"][0]["delta"].get("content"), t)
        for c, t in zip(chunks, arrival_s, strict=True)
        if c.get("choices") and c["choices"][0]["delta"].get("content")
    ]
    text = "".join(piece for piece, _ in content_chunks)
    assert text.strip() == LOCAL_TEXT  # content intact
    # Chunks arrived incrementally (6 tokens at 20 ms spacing), not in one buffered burst.
    spread = content_chunks[-1][1] - content_chunks[0][1]
    assert spread >= (len(content_chunks) - 1) * LOCAL_TPOT_S * 0.5
    # The usage chunk the gateway requested for itself was NOT forwarded...
    assert not any(c.get("usage") for c in chunks)

    # ...yet the log still has exact token counts, plus a TTFT ≥ the injected delay.
    event = logged_event(log_path, request_id)
    assert event["stream"] is True
    assert event["output_tokens"] == len(content_chunks)
    assert event["input_tokens"] is not None
    assert LOCAL_TTFT_S <= event["ttft_s"] <= LOCAL_TTFT_S + 1.0
    assert event["cost_usd"] is not None


def test_streaming_forwards_usage_when_client_asks(gateway):
    url, _ = gateway
    body = {
        "model": "mock-model",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    usage_chunks = []
    with httpx.stream(
        "POST", f"{url}/v1/chat/completions", json=body, timeout=10.0
    ) as response:
        for line in response.iter_lines():
            if not line.startswith("data: ") or line[6:].strip() == "[DONE]":
                continue
            chunk = json.loads(line[6:])
            if chunk.get("usage"):
                usage_chunks.append(chunk)
    assert len(usage_chunks) == 1
    assert usage_chunks[0]["usage"]["completion_tokens"] > 0


def test_gateway_health_endpoint(gateway):
    url, _ = gateway
    data = httpx.get(f"{url}/health", timeout=5.0).json()
    assert data == {"status": "ok", "local_healthy": True, "circuit_open": False}


def test_no_secret_leaks_into_log(gateway):
    url, log_path = gateway
    post(url, {"model": "mock-model", "messages": [{"role": "user", "content": "hi"}]})
    assert KEY_VALUE not in log_path.read_text()


def test_local_failure_falls_back_then_circuit_opens(fallback_mock, tmp_path):
    """Dead local backend: per-request fallback, then breaker opens, then a half-open trial."""
    log_path = tmp_path / "log.jsonl"
    config = make_gateway_config(
        "http://127.0.0.1:9",  # nothing listens here
        fallback_mock,
        log_path,
        circuit_breaker_failures=2,
        circuit_breaker_open_s=0.4,
    )
    with run_gateway(config) as url:
        body = {"model": "mock-model", "messages": [{"role": "user", "content": "hi"}]}
        reasons = []

        response = post(url, body)  # failure 1: local attempted, fell back
        assert response.json()["choices"][0]["message"]["content"].strip() == FALLBACK_TEXT
        reasons.append(response.headers["x-gateway-route-reason"])

        with httpx.stream(  # failure 2 (streaming path): opens the circuit
            "POST", f"{url}/v1/chat/completions", json=body | {"stream": True}, timeout=10.0
        ) as stream_response:
            reasons.append(stream_response.headers["x-gateway-route-reason"])
            text = "".join(
                json.loads(line[6:])["choices"][0]["delta"].get("content") or ""
                for line in stream_response.iter_lines()
                if line.startswith("data: ")
                and line[6:].strip() != "[DONE]"
                and json.loads(line[6:]).get("choices")
            )
            assert text.strip() == FALLBACK_TEXT

        t_open = time.perf_counter()
        response = post(url, body)  # circuit open: straight to fallback, no local attempt
        reasons.append(response.headers["x-gateway-route-reason"])
        assert time.perf_counter() - t_open < 0.35  # no connect-timeout stall on local

        time.sleep(0.5)  # let the open window (0.4 s) expire
        response = post(url, body)  # half-open trial: local attempted again, still dead
        reasons.append(response.headers["x-gateway-route-reason"])
        response = post(url, body)  # trial failure re-opened the circuit
        reasons.append(response.headers["x-gateway-route-reason"])

        assert reasons == [
            "local_error_fallback",
            "local_error_fallback",
            "circuit_open",
            "local_error_fallback",
            "circuit_open",
        ]
        assert all(r.status_code == 200 for r in [response])


def test_unhealthy_local_detected_by_monitor(fallback_mock, tmp_path):
    """With a fast health interval, the monitor reroutes before any request fails."""
    config = make_gateway_config(
        "http://127.0.0.1:9",
        fallback_mock,
        tmp_path / "log.jsonl",
        health_check_interval_s=0.1,
        circuit_breaker_failures=100,  # keep the breaker out of this test
    )
    with run_gateway(config) as url:
        time.sleep(0.4)  # a few probe intervals
        response = post(
            url, {"model": "mock-model", "messages": [{"role": "user", "content": "hi"}]}
        )
        assert response.headers["x-gateway-backend"] == "fallback"
        assert response.headers["x-gateway-route-reason"] == "local_unhealthy"
        health = httpx.get(f"{url}/health", timeout=5.0).json()
        assert health["local_healthy"] is False
