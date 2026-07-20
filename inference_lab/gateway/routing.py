"""Routing policy: which backend serves a request, and why.

The rules are deliberately simple and explainable, applied in precedence
order:

1. requested model not served locally      → fallback  (``model_not_local``)
2. circuit breaker open                    → fallback  (``circuit_open``)
3. local health probe failing              → fallback  (``local_unhealthy``)
4. estimated prompt tokens over threshold  → fallback  (``over_token_threshold``)
5. otherwise                               → local     (``default_local``)

A local attempt that fails mid-flight is retried on the fallback and logged
as ``local_error_fallback``.

Known limitation (measured in M6, documented rather than engineered around):
a raw prompt-token threshold is a crude cost proxy. What is expensive locally
is *unique* (non-cached) tokens — shared-prefix long prompts measured as the
cheapest traffic to serve (86% prefix-cache hit rate, $0.062/1M output) while
unique long contexts are the most expensive ($0.213/1M), and the KV preemption
wall is counted in unique tokens per sequence. The threshold's real job is
guarding that wall; a threshold on an estimate of unique tokens is the noted
future work.
"""

import asyncio
import math
import time
from dataclasses import dataclass
from typing import Any

import httpx

from inference_lab.gateway.config import RoutingConfig

LOCAL = "local"
FALLBACK = "fallback"

REASON_MODEL_NOT_LOCAL = "model_not_local"
REASON_CIRCUIT_OPEN = "circuit_open"
REASON_LOCAL_UNHEALTHY = "local_unhealthy"
REASON_OVER_THRESHOLD = "over_token_threshold"
REASON_DEFAULT = "default_local"
REASON_LOCAL_ERROR = "local_error_fallback"

_CHARS_PER_TOKEN = 4.0
_TOKENS_PER_MESSAGE = 4  # chat-template framing overhead per message


def estimate_prompt_tokens(messages: list[dict[str, Any]]) -> int:
    """Cheap deterministic estimate of prompt tokens (~4 chars/token + framing).

    Routing needs a token count *before* any backend is contacted, so the
    exact tokenizer count (which the server reports afterwards in ``usage``)
    is not available; both the estimate and the exact count are logged per
    request so the estimator's error stays visible.
    """
    total = 0
    for message in messages:
        content = message.get("content") or ""
        if isinstance(content, list):  # multimodal-style content parts
            content = " ".join(
                str(part.get("text", "")) for part in content if isinstance(part, dict)
            )
        total += _TOKENS_PER_MESSAGE + math.ceil(len(str(content)) / _CHARS_PER_TOKEN)
    return total


def health_url(base_url: str) -> str:
    """Derive the server-root ``/health`` URL from an OpenAI-style ``.../v1`` base URL."""
    root = base_url.rstrip("/").removesuffix("/v1")
    return f"{root}/health"


class CircuitBreaker:
    """Open after N consecutive local failures; try again T seconds later.

    After the open window expires the next local request is the trial: a
    success resets the failure count, a failure re-opens the circuit
    immediately (the consecutive count is still at the threshold).
    """

    def __init__(self, failure_threshold: int, open_s: float) -> None:
        self._threshold = failure_threshold
        self._open_s = open_s
        self._consecutive_failures = 0
        self._open_until = 0.0  # time.monotonic() deadline

    @property
    def is_open(self) -> bool:
        return time.monotonic() < self._open_until

    def record_success(self) -> None:
        self._consecutive_failures = 0
        self._open_until = 0.0

    def record_failure(self) -> None:
        self._consecutive_failures += 1
        if self._consecutive_failures >= self._threshold:
            self._open_until = time.monotonic() + self._open_s


class HealthMonitor:
    """Background poller of the local backend's ``/health`` endpoint.

    Starts optimistic (healthy until the first probe, which runs after one
    interval): instant failure detection is the circuit breaker's job; the
    monitor's job is to stop offering traffic to a dead backend *between*
    request failures and to bring it back once it recovers.
    """

    def __init__(self, url: str, interval_s: float) -> None:
        self.healthy = True
        self._url = url
        self._interval_s = interval_s

    async def run(self, client: httpx.AsyncClient) -> None:
        while True:
            await asyncio.sleep(self._interval_s)
            try:
                response = await client.get(self._url, timeout=5.0)
                self.healthy = response.status_code == 200
            except httpx.HTTPError:
                self.healthy = False


@dataclass(frozen=True)
class RouteDecision:
    backend: str  # LOCAL or FALLBACK
    reason: str


class Router:
    """Applies the routing policy. Owns no I/O — state comes from its collaborators."""

    def __init__(
        self,
        config: RoutingConfig,
        served_models: list[str],
        breaker: CircuitBreaker,
        monitor: HealthMonitor,
    ) -> None:
        self._config = config
        self._served_models = set(served_models)
        self._breaker = breaker
        self._monitor = monitor

    def decide(self, model: str, est_prompt_tokens: int) -> RouteDecision:
        if model not in self._served_models:
            return RouteDecision(FALLBACK, REASON_MODEL_NOT_LOCAL)
        if self._breaker.is_open:
            return RouteDecision(FALLBACK, REASON_CIRCUIT_OPEN)
        if not self._monitor.healthy:
            return RouteDecision(FALLBACK, REASON_LOCAL_UNHEALTHY)
        if est_prompt_tokens > self._config.prompt_token_threshold:
            return RouteDecision(FALLBACK, REASON_OVER_THRESHOLD)
        return RouteDecision(LOCAL, REASON_DEFAULT)
