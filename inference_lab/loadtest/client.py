"""Async streaming client for OpenAI-compatible ``/v1/chat/completions`` endpoints.

Measurement conventions (the subtle parts):

- All timestamps are ``time.perf_counter()`` — monotonic, immune to wall-clock
  adjustments. Only differences are meaningful.
- **TTFT** is send-to-first-*content*-token. Servers commonly open the stream
  with a role-only delta (``{"delta": {"role": "assistant"}}``) before any real
  token; counting that chunk would report a flattering, fake TTFT.
- **TPOT** is the mean inter-token gap ``(t_last - t_first) / (n_tokens - 1)``
  — first-token time belongs to TTFT (prefill), not TPOT (decode).
- Token counts prefer the server's ``usage`` block (requested via
  ``stream_options.include_usage``); SSE chunk count is the fallback, which
  undercounts if the server coalesces tokens into one chunk.
"""

import json
import time

import httpx

from inference_lab.common.config import EndpointConfig
from inference_lab.loadtest.models import GeneratedPrompt, RequestRecord

_SSE_DATA_PREFIX = "data: "
_SSE_DONE = "[DONE]"


def build_payload(endpoint: EndpointConfig, prompt: GeneratedPrompt) -> dict:
    """Build the chat-completions request body for one prompt."""
    payload = {
        "model": endpoint.model,
        "messages": prompt.messages,
        "max_tokens": prompt.max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if prompt.ignore_eos:
        payload["ignore_eos"] = True  # vLLM extension; OpenAI-incompatible servers may reject it
    return payload


async def run_streaming_request(
    client: httpx.AsyncClient,
    endpoint: EndpointConfig,
    prompt: GeneratedPrompt,
    *,
    request_id: str,
    concurrency: int,
    warmup: bool,
    origin_s: float = 0.0,
) -> RequestRecord:
    """Send one streaming request and measure it; never raises on request failure.

    ``origin_s`` is the run's perf_counter origin: recorded ``start_s`` values
    are relative to it so records from one run share a common timeline.
    """
    url = f"{endpoint.base_url.rstrip('/')}/chat/completions"
    payload = build_payload(endpoint, prompt)

    first_token_s: float | None = None
    last_token_s: float | None = None
    num_chunks = 0
    finish_reason: str | None = None
    usage: dict | None = None
    error: str | None = None

    t_start = time.perf_counter()
    try:
        async with client.stream(
            "POST",
            url,
            json=payload,
            headers={"Authorization": f"Bearer {endpoint.api_key}"},
            timeout=endpoint.timeout_s,
        ) as response:
            if response.status_code != 200:
                body = (await response.aread()).decode("utf-8", errors="replace")
                error = f"HTTP {response.status_code}: {body[:200]}"
            else:
                async for line in response.aiter_lines():
                    if not line.startswith(_SSE_DATA_PREFIX):
                        continue
                    data = line[len(_SSE_DATA_PREFIX) :]
                    if data.strip() == _SSE_DONE:
                        break
                    now = time.perf_counter()
                    chunk = json.loads(data)
                    if chunk.get("usage"):
                        usage = chunk["usage"]
                    choices = chunk.get("choices") or []
                    if not choices:
                        continue
                    if choices[0].get("finish_reason"):
                        finish_reason = choices[0]["finish_reason"]
                    if (choices[0].get("delta") or {}).get("content"):
                        num_chunks += 1
                        if first_token_s is None:
                            first_token_s = now
                        last_token_s = now
    except httpx.HTTPError as exc:
        error = f"{type(exc).__name__}: {exc}"
    t_end = time.perf_counter()

    ok = error is None and first_token_s is not None
    if error is None and first_token_s is None:
        error = "stream ended without any content token"

    output_tokens: int | None = None
    input_tokens: int | None = prompt.est_input_tokens
    if usage is not None:
        output_tokens = usage.get("completion_tokens")
        input_tokens = usage.get("prompt_tokens", input_tokens)
    if output_tokens is None and num_chunks > 0:
        output_tokens = num_chunks

    tpot_s: float | None = None
    if (
        ok
        and output_tokens is not None
        and output_tokens > 1
        and last_token_s is not None
        and first_token_s is not None
    ):
        tpot_s = (last_token_s - first_token_s) / (output_tokens - 1)

    return RequestRecord(
        request_id=request_id,
        concurrency=concurrency,
        warmup=warmup,
        ok=ok,
        error=error,
        start_s=t_start - origin_s,
        latency_s=t_end - t_start,
        ttft_s=(first_token_s - t_start) if first_token_s is not None else None,
        tpot_s=tpot_s,
        input_tokens=input_tokens,
        output_tokens=output_tokens if ok else None,
        num_chunks=num_chunks,
        finish_reason=finish_reason,
    )
