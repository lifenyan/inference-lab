"""Gateway configuration.

The gateway never hardcodes endpoints, model names, prices, or thresholds —
everything arrives through ``GatewayConfig``, loaded from a JSON file (see
``configs/gateway.example.json``). Two cost-accounting rules from the M7
report are enforced structurally:

- A local $/1M-output figure is meaningless without its utilization and
  traffic-regime assumption (report §7: ~25× spread between c=1 and c=64
  steady-state on the same pod, 1.65× between fresh and cache-warm traffic),
  so ``cost_assumption`` is a required field, logged verbatim next to every
  cost derived from the figure.
- API list prices drift, so ``prices_fetched_on`` is a required date carried
  alongside the fallback prices.

Secrets never live in config files: the fallback API key is *named* by
environment variable (``api_key_env``) and resolved at startup.
"""

import json
import os
from pathlib import Path

from pydantic import BaseModel, Field


class LocalBackendConfig(BaseModel):
    """The self-hosted vLLM endpoint."""

    base_url: str = Field(description="OpenAI-compatible base URL, e.g. http://pod:8000/v1")
    served_models: list[str] = Field(
        min_length=1, description="Model names served locally; anything else routes to fallback"
    )
    api_key: str = Field(default="EMPTY", description="vLLM accepts any bearer token")
    timeout_s: float = Field(default=120.0, gt=0)
    usd_per_1m_output_tokens: float = Field(
        ge=0, description="Self-hosted cost per 1M output tokens under cost_assumption"
    )
    cost_assumption: str = Field(
        description="Utilization + traffic-regime assumption behind the $/1M figure, verbatim"
    )


class FallbackBackendConfig(BaseModel):
    """The commercial OpenAI-compatible API."""

    base_url: str
    model: str = Field(description="Model substituted when a local-model request is rerouted here")
    api_key_env: str = Field(description="Name of the environment variable holding the API key")
    timeout_s: float = Field(default=120.0, gt=0)
    usd_per_1m_input_tokens: float = Field(ge=0)
    usd_per_1m_output_tokens: float = Field(ge=0)
    prices_fetched_on: str = Field(description="ISO date the list prices were fetched")

    def resolve_api_key(self) -> str:
        """Read the API key from the environment; fail fast if it is missing."""
        try:
            return os.environ[self.api_key_env]
        except KeyError as exc:
            raise RuntimeError(
                f"fallback API key environment variable {self.api_key_env!r} is not set"
            ) from exc


class RoutingConfig(BaseModel):
    """Thresholds and failure-handling knobs for the routing policy."""

    prompt_token_threshold: int = Field(
        gt=0,
        description="Estimated prompt tokens above which a request routes to fallback "
        "(guards the KV wall; see routing module docstring for the unique-token caveat)",
    )
    health_check_interval_s: float = Field(default=5.0, gt=0)
    circuit_breaker_failures: int = Field(
        default=3, gt=0, description="Consecutive local failures that open the circuit"
    )
    circuit_breaker_open_s: float = Field(
        default=30.0, gt=0, description="Seconds the circuit stays open before a retrial"
    )


class GatewayConfig(BaseModel):
    """Everything the gateway needs; loaded from one JSON file."""

    local: LocalBackendConfig
    fallback: FallbackBackendConfig
    routing: RoutingConfig
    log_path: Path = Field(description="JSONL request log, written via common.logging.log_event")

    @classmethod
    def from_file(cls, path: str | Path) -> "GatewayConfig":
        return cls.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))
