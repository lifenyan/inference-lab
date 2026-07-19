"""Mock OpenAI-compatible server with configurable artificial latencies.

Lets the whole harness (and later the M3 evals and M8 gateway tests) run
offline: the server injects a known TTFT delay and per-token delay, so tests
can assert that the harness measures what was injected.

Faithful quirks of real servers, kept on purpose:

- the stream opens with a role-only delta *before* the TTFT delay — a correct
  client must not count it as the first token;
- the final content chunk is followed by a finish-reason chunk, a usage chunk
  (when ``stream_options.include_usage`` is set), then ``data: [DONE]``.

Run standalone: ``python -m inference_lab.loadtest.mockserver --port 8100``.
FastAPI/uvicorn are dev dependencies — this module is not imported by the
harness itself.
"""

import argparse
import asyncio
import json
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field


class MockServerConfig(BaseModel):
    """Artificial serving behavior; canned_text overrides token generation (M3 evals)."""

    ttft_s: float = Field(default=0.05, ge=0, description="Delay before the first content token")
    tpot_s: float = Field(default=0.01, ge=0, description="Delay between subsequent tokens")
    output_tokens: int = Field(
        default=32, gt=0, description="Natural reply length; capped by the request's max_tokens"
    )
    model: str = "mock-model"
    canned_text: str | None = Field(
        default=None,
        description="If set, reply with exactly this text (split on spaces) instead of filler",
    )


def _reply_tokens(config: MockServerConfig, max_tokens: int | None) -> tuple[list[str], str]:
    """Return (tokens, finish_reason) for one completion."""
    if config.canned_text is not None:
        tokens = [w + " " for w in config.canned_text.split(" ")]
    else:
        tokens = [f"tok{i} " for i in range(config.output_tokens)]
    if max_tokens is not None and len(tokens) > max_tokens:
        return tokens[:max_tokens], "length"
    return tokens, "stop"


def _estimate_prompt_tokens(messages: list[dict[str, Any]]) -> int:
    return sum(len(str(m.get("content", "")).split()) for m in messages)


def _chunk(completion_id: str, model: str, created: int, **choice_fields: Any) -> str:
    body = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": None} | choice_fields],
    }
    return f"data: {json.dumps(body)}\n\n"


def create_app(config: MockServerConfig) -> FastAPI:
    """Build the mock server app for a given latency/length configuration."""
    app = FastAPI(title="inference_lab mock server")

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/chat/completions", response_model=None)
    async def chat_completions(request: dict[str, Any]) -> JSONResponse | StreamingResponse:
        completion_id = f"chatcmpl-mock-{uuid.uuid4().hex[:12]}"
        created = int(time.time())
        model = request.get("model", config.model)
        tokens, finish_reason = _reply_tokens(config, request.get("max_tokens"))
        usage = {
            "prompt_tokens": _estimate_prompt_tokens(request.get("messages", [])),
            "completion_tokens": len(tokens),
            "total_tokens": _estimate_prompt_tokens(request.get("messages", [])) + len(tokens),
        }

        if not request.get("stream", False):
            await asyncio.sleep(config.ttft_s + config.tpot_s * max(0, len(tokens) - 1))
            return JSONResponse(
                {
                    "id": completion_id,
                    "object": "chat.completion",
                    "created": created,
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "".join(tokens)},
                            "finish_reason": finish_reason,
                        }
                    ],
                    "usage": usage,
                }
            )

        include_usage = bool((request.get("stream_options") or {}).get("include_usage"))

        async def stream() -> AsyncIterator[str]:
            # Role-only delta before any delay, like real servers.
            yield _chunk(completion_id, model, created, delta={"role": "assistant"})
            await asyncio.sleep(config.ttft_s)
            for i, token in enumerate(tokens):
                if i > 0:
                    await asyncio.sleep(config.tpot_s)
                yield _chunk(completion_id, model, created, delta={"content": token})
            yield _chunk(completion_id, model, created, finish_reason=finish_reason)
            if include_usage:
                body = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [],
                    "usage": usage,
                }
                yield f"data: {json.dumps(body)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(stream(), media_type="text/event-stream")

    return app


def main() -> None:
    """CLI entry point: run the mock server standalone for demos and manual testing."""
    import uvicorn

    parser = argparse.ArgumentParser(description="Mock OpenAI-compatible streaming server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument("--ttft", type=float, default=0.05, help="TTFT delay in seconds")
    parser.add_argument("--tpot", type=float, default=0.01, help="Per-token delay in seconds")
    parser.add_argument("--output-tokens", type=int, default=32, help="Natural reply length")
    args = parser.parse_args()

    config = MockServerConfig(ttft_s=args.ttft, tpot_s=args.tpot, output_tokens=args.output_tokens)
    uvicorn.run(create_app(config), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
