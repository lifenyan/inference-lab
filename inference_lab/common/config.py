"""Configuration models shared across the harness, evals, and gateway.

Convention (see CLAUDE.md): model names and endpoints are never hardcoded in
application code — they always arrive through these config objects, loaded from
a file, CLI flags, or environment variables.
"""

import os

from pydantic import BaseModel, Field


class EndpointConfig(BaseModel):
    """An OpenAI-compatible inference endpoint to benchmark, evaluate, or route to."""

    base_url: str = Field(description="Base URL, e.g. http://localhost:8000/v1")
    model: str = Field(description="Model name as served, e.g. Qwen/Qwen2.5-7B-Instruct")
    api_key: str = Field(default="EMPTY", description="Bearer token; vLLM accepts any value")
    timeout_s: float = Field(default=120.0, gt=0)

    @classmethod
    def from_env(cls, prefix: str = "INFERENCE_LAB") -> "EndpointConfig":
        """Build from environment variables: <prefix>_BASE_URL, <prefix>_MODEL, <prefix>_API_KEY."""
        return cls(
            base_url=os.environ[f"{prefix}_BASE_URL"],
            model=os.environ[f"{prefix}_MODEL"],
            api_key=os.environ.get(f"{prefix}_API_KEY", "EMPTY"),
        )
