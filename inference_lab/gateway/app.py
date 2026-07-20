"""OpenAI-compatible gateway: FastAPI app factory.

``create_app(config)`` builds an app exposing ``/v1/chat/completions``
(streaming and non-streaming) and ``/health``. Each request is routed by
``routing.Router``, forwarded with httpx, and logged as one JSONL event via
``common.logging.log_event``; responses carry ``x-gateway-*`` headers naming
the backend, route reason, and request id.

Conventions shared with the loadtest harness: TTFT is gateway-receipt to
first non-empty *content* token (role-only deltas don't count), and token
counts come from the server's ``usage`` block — the gateway always requests
``stream_options.include_usage`` upstream and swallows the usage chunk if the
client didn't ask for it, so logging is exact without changing what the
client sees.
"""

import asyncio
import json
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from inference_lab.common.logging import get_logger, log_event
from inference_lab.gateway.config import GatewayConfig
from inference_lab.gateway.cost import request_cost
from inference_lab.gateway.routing import (
    FALLBACK,
    LOCAL,
    REASON_LOCAL_ERROR,
    CircuitBreaker,
    HealthMonitor,
    Router,
    estimate_prompt_tokens,
    health_url,
)

logger = get_logger("gateway")

_SSE_DATA_PREFIX = "data: "
_SSE_DONE = "[DONE]"


class UpstreamError(Exception):
    """An upstream attempt failed in a way that justifies trying elsewhere."""


def prepare_fallback_payload(
    payload: dict[str, Any], fallback_model: str, served_models: list[str]
) -> dict[str, Any]:
    """Adapt a request body for the commercial fallback API.

    - Requests for a locally served model are rewritten to the fallback model.
    - ``max_tokens`` becomes ``max_completion_tokens``: OpenAI's newer models
      reject the former (found live in the M8 demo — gpt-5.4-nano 400s on it),
      while vLLM accepts the standard name.
    - ``ignore_eos`` (a vLLM extension the loadtest harness uses) is dropped;
      commercial APIs reject unknown parameters.
    """
    adapted = dict(payload)
    if adapted.get("model") in served_models:
        adapted["model"] = fallback_model
    if "max_tokens" in adapted:
        adapted.setdefault("max_completion_tokens", adapted.pop("max_tokens"))
    adapted.pop("ignore_eos", None)
    return adapted


def create_app(config: GatewayConfig) -> FastAPI:
    """Build the gateway app for one configuration."""
    fallback_key = config.fallback.resolve_api_key()  # fail fast if the env var is missing
    breaker = CircuitBreaker(
        config.routing.circuit_breaker_failures, config.routing.circuit_breaker_open_s
    )
    monitor = HealthMonitor(
        health_url(config.local.base_url), config.routing.health_check_interval_s
    )
    router = Router(config.routing, config.local.served_models, breaker, monitor)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        client = httpx.AsyncClient()
        app.state.client = client
        probe_task = asyncio.create_task(monitor.run(client))
        yield
        probe_task.cancel()
        await client.aclose()

    app = FastAPI(title="inference_lab gateway", lifespan=lifespan)

    def build_upstream_request(backend: str, body: dict[str, Any], stream: bool) -> httpx.Request:
        """Build the outbound request, rewriting local-model names for the fallback."""
        if backend == LOCAL:
            payload = dict(body)
            base, key, timeout = config.local.base_url, config.local.api_key, config.local.timeout_s
        else:
            payload = prepare_fallback_payload(
                body, config.fallback.model, config.local.served_models
            )
            base = config.fallback.base_url
            key, timeout = fallback_key, config.fallback.timeout_s
        if stream:
            payload["stream_options"] = {"include_usage": True}
        client: httpx.AsyncClient = app.state.client
        return client.build_request(
            "POST",
            f"{base.rstrip('/')}/chat/completions",
            json=payload,
            headers={"Authorization": f"Bearer {key}"},
            timeout=timeout,
        )

    async def send_upstream(backend: str, body: dict[str, Any], stream: bool) -> httpx.Response:
        """Forward to one backend; connection errors and 5xx raise UpstreamError.

        4xx responses are returned as-is — the request itself is bad, and
        retrying it on another backend would neither help nor be honest.
        """
        client: httpx.AsyncClient = app.state.client
        upstream_request = build_upstream_request(backend, body, stream)
        try:
            response = await client.send(upstream_request, stream=stream)
        except httpx.HTTPError as exc:
            raise UpstreamError(f"{type(exc).__name__}: {exc}") from exc
        if response.status_code >= 500:
            detail = (await response.aread()).decode("utf-8", errors="replace")[:200]
            await response.aclose()
            raise UpstreamError(f"HTTP {response.status_code}: {detail}")
        return response

    def finish_event(
        base_event: dict[str, Any],
        backend: str,
        t0: float,
        usage: dict[str, Any] | None,
        ttft_s: float | None,
        http_status: int,
        error: str | None,
    ) -> dict[str, Any]:
        """Complete one request's log event with tokens, latency, and cost."""
        usage = usage or {}
        input_tokens = usage.get("prompt_tokens")
        output_tokens = usage.get("completion_tokens")
        cost_usd, cost_basis = request_cost(config, backend, input_tokens, output_tokens)
        if error is None and http_status >= 400:
            error = f"HTTP {http_status}"
        return base_event | {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "ttft_s": ttft_s,
            "latency_s": time.perf_counter() - t0,
            "http_status": http_status,
            "status": "ok" if error is None else "error",
            "error": error,
            "cost_usd": cost_usd,
            "cost_basis": cost_basis,
        }

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "local_healthy": monitor.healthy,
            "circuit_open": breaker.is_open,
        }

    @app.post("/v1/chat/completions", response_model=None)
    async def chat_completions(request: Request) -> JSONResponse | StreamingResponse:
        t0 = time.perf_counter()
        request_id = f"gw-{uuid.uuid4().hex[:12]}"
        try:
            body = await request.json()
        except json.JSONDecodeError:
            return JSONResponse({"error": "request body is not valid JSON"}, status_code=400)

        model = str(body.get("model", ""))
        stream = bool(body.get("stream", False))
        client_wants_usage = bool((body.get("stream_options") or {}).get("include_usage"))
        est_tokens = estimate_prompt_tokens(body.get("messages") or [])
        decision = router.decide(model, est_tokens)

        backend, reason = decision.backend, decision.reason
        upstream: httpx.Response | None = None
        error: str | None = None
        if backend == LOCAL:
            try:
                upstream = await send_upstream(LOCAL, body, stream)
                breaker.record_success()
            except UpstreamError as exc:
                breaker.record_failure()
                logger.warning("local backend failed (%s); retrying on fallback", exc)
                backend, reason = FALLBACK, REASON_LOCAL_ERROR
        if upstream is None:
            try:
                upstream = await send_upstream(FALLBACK, body, stream)
            except UpstreamError as exc:
                error = str(exc)

        model_served = model
        if backend == FALLBACK and model in config.local.served_models:
            model_served = config.fallback.model
        headers = {
            "x-gateway-request-id": request_id,
            "x-gateway-backend": backend,
            "x-gateway-route-reason": reason,
        }
        base_event = {
            "event": "request",
            "request_id": request_id,
            "backend": backend,
            "decided_backend": decision.backend,
            "route_reason": reason,
            "model_requested": model,
            "model_served": model_served,
            "stream": stream,
            "est_prompt_tokens": est_tokens,
        }

        if upstream is None:  # both backends failed
            event = finish_event(base_event, backend, t0, None, None, 502, error)
            log_event(config.log_path, event)
            return JSONResponse(
                {"error": f"all backends failed: {error}"}, status_code=502, headers=headers
            )

        if not stream:
            data = upstream.json()
            usage = data.get("usage") if isinstance(data, dict) else None
            event = finish_event(base_event, backend, t0, usage, None, upstream.status_code, None)
            log_event(config.log_path, event)
            return JSONResponse(data, status_code=upstream.status_code, headers=headers)

        if upstream.status_code != 200:  # 4xx from upstream on a streaming request
            detail = (await upstream.aread()).decode("utf-8", errors="replace")
            await upstream.aclose()
            try:
                payload = json.loads(detail)
            except json.JSONDecodeError:
                payload = {"error": detail[:500]}
            event = finish_event(base_event, backend, t0, None, None, upstream.status_code, None)
            log_event(config.log_path, event)
            return JSONResponse(payload, status_code=upstream.status_code, headers=headers)

        async def relay() -> AsyncIterator[str]:
            """Pass SSE lines through untouched, observing TTFT and usage on the way."""
            usage: dict[str, Any] | None = None
            ttft_s: float | None = None
            relay_error: str | None = None
            try:
                async for line in upstream.aiter_lines():
                    if not line:
                        continue
                    if not line.startswith(_SSE_DATA_PREFIX):
                        yield line + "\n\n"
                        continue
                    data_str = line[len(_SSE_DATA_PREFIX) :]
                    if data_str.strip() == _SSE_DONE:
                        yield line + "\n\n"
                        break
                    try:
                        chunk = json.loads(data_str)
                    except json.JSONDecodeError:
                        yield line + "\n\n"
                        continue
                    if chunk.get("usage"):
                        usage = chunk["usage"]
                        if not chunk.get("choices") and not client_wants_usage:
                            # Usage-only chunk we requested on the client's behalf.
                            continue
                    choices = chunk.get("choices") or []
                    if (
                        ttft_s is None
                        and choices
                        and (choices[0].get("delta") or {}).get("content")
                    ):
                        ttft_s = time.perf_counter() - t0
                    yield line + "\n\n"
            except httpx.HTTPError as exc:
                relay_error = f"{type(exc).__name__}: {exc}"
                if backend == LOCAL:
                    breaker.record_failure()
                logger.warning("stream from %s aborted: %s", backend, relay_error)
            finally:
                await upstream.aclose()
                event = finish_event(base_event, backend, t0, usage, ttft_s, 200, relay_error)
                log_event(config.log_path, event)

        return StreamingResponse(relay(), media_type="text/event-stream", headers=headers)

    return app
