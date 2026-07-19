"""Pydantic models for the load-test harness.

Three groups:

- **Workload specs** (`SyntheticWorkload`, `ShareGPTWorkload`) — the small config
  that fully determines what traffic a run sends. Saved verbatim into the run
  directory (``workload.json``) so any run can be reproduced or reused for an
  identical A/B comparison.
- **Per-request records** (`RequestRecord`) — the raw measurements, one JSONL
  line per request. Every aggregate in ``summary.json`` is recomputable from
  these; we never store only aggregates.
- **Aggregates** (`PercentileStats`, `LevelSummary`) — per-concurrency-level
  summaries derived from the records.
"""

from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, Field, TypeAdapter


class SyntheticWorkload(BaseModel):
    """Controlled-shape prompts built with the model's tokenizer to hit token budgets.

    Every prompt starts with a unique marker (so servers cannot prefix-cache
    across requests) and is padded with seeded random words to
    ``input_tokens``. An optional shared system prompt of
    ``shared_prefix_tokens`` tokens is identical across all prompts — the
    intended target for prefix-caching experiments.
    """

    mode: Literal["synthetic"] = "synthetic"
    tokenizer: str | None = Field(
        default=None,
        description="HF repo id of the tokenizer used to hit token budgets; "
        "defaults to the endpoint's model name at generation time",
    )
    input_tokens: int = Field(gt=0, description="Target tokens in the user message")
    output_tokens: int = Field(gt=0, description="max_tokens requested per completion")
    shared_prefix_tokens: int = Field(
        default=0, ge=0, description="Length of the shared system-prompt prefix, in tokens"
    )
    num_prompts: int = Field(
        default=128, gt=0, description="Distinct prompts to generate (cycled if a sweep needs more)"
    )
    seed: int = 0


class ShareGPTWorkload(BaseModel):
    """Real conversations sampled (seeded) from the ShareGPT dataset on HuggingFace.

    Follows vLLM's benchmark convention: each sample is the first human turn of
    a conversation, with ``max_tokens`` set to the length of the assistant
    reply that actually followed (capped at ``max_output_tokens``).
    """

    mode: Literal["sharegpt"] = "sharegpt"
    tokenizer: str | None = Field(
        default=None,
        description="HF repo id of the tokenizer used to measure lengths; "
        "defaults to the endpoint's model name at generation time",
    )
    num_prompts: int = Field(default=128, gt=0)
    min_input_tokens: int = Field(default=4, ge=1, description="Skip degenerate tiny prompts")
    max_input_tokens: int = Field(default=2048, ge=1, description="Skip prompts longer than this")
    min_output_tokens: int = Field(default=4, ge=1, description="Skip near-empty replies")
    max_output_tokens: int = Field(default=1024, ge=1, description="Cap on max_tokens per request")
    seed: int = 0
    dataset_repo: str = "anon8231489123/ShareGPT_Vicuna_unfiltered"
    dataset_file: str = "ShareGPT_V3_unfiltered_cleaned_split.json"
    dataset_path: str | None = Field(
        default=None, description="Local JSON path override; skips the HF download when set"
    )


WorkloadSpec = Annotated[SyntheticWorkload | ShareGPTWorkload, Field(discriminator="mode")]

_workload_adapter: TypeAdapter[WorkloadSpec] = TypeAdapter(WorkloadSpec)


def load_workload(path: Path) -> WorkloadSpec:
    """Load and validate a workload spec from a JSON file."""
    return _workload_adapter.validate_json(path.read_text(encoding="utf-8"))


def dump_workload(spec: WorkloadSpec) -> str:
    """Serialize a workload spec to pretty JSON (for saving alongside results)."""
    return _workload_adapter.dump_json(spec, indent=2).decode("utf-8")


class GeneratedPrompt(BaseModel):
    """One ready-to-send request payload produced by workload generation."""

    messages: list[dict[str, str]]
    max_tokens: int = Field(gt=0)
    est_input_tokens: int = Field(
        gt=0, description="Input length measured with the workload tokenizer (client-side estimate)"
    )


class RequestRecord(BaseModel):
    """Raw measurements for a single request (one JSONL line in ``requests.jsonl``).

    Timing fields come from ``time.perf_counter()`` and are relative to a
    per-run monotonic origin — only differences between them are meaningful.
    """

    request_id: str
    concurrency: int = Field(description="Concurrency level this request was issued under")
    warmup: bool = Field(description="Warmup requests are recorded but excluded from aggregates")
    ok: bool
    error: str | None = None
    start_s: float = Field(description="perf_counter at send, relative to the run origin")
    latency_s: float = Field(ge=0, description="Send to final byte of the response")
    ttft_s: float | None = Field(
        default=None, ge=0, description="Send to first *content* token (role-only deltas skipped)"
    )
    tpot_s: float | None = Field(
        default=None,
        ge=0,
        description="Mean inter-token gap: (last_token_ts - first_token_ts) / (output_tokens - 1)",
    )
    input_tokens: int | None = Field(
        default=None, description="From the server's usage block when available, else estimated"
    )
    output_tokens: int | None = Field(
        default=None, description="From usage when available, else the streamed chunk count"
    )
    num_chunks: int = Field(default=0, description="SSE chunks with non-empty content")
    finish_reason: str | None = None


class PercentileStats(BaseModel):
    """Distribution summary of one metric (all values in seconds)."""

    mean: float
    p50: float
    p90: float
    p99: float
    min: float
    max: float


class LevelSummary(BaseModel):
    """Aggregates for one concurrency level, computed from measured (non-warmup) records."""

    concurrency: int
    num_requests: int = Field(description="Measured requests (warmup excluded)")
    num_errors: int
    error_rate: float
    duration_s: float = Field(
        description="Wall-clock window: first measured request start to last measured request end"
    )
    output_tokens_total: int
    throughput_tok_s: float = Field(description="Server throughput: total output tokens / window")
    requests_per_s: float
    ttft_s: PercentileStats | None = Field(default=None, description="None if all requests failed")
    tpot_s: PercentileStats | None = None
    latency_s: PercentileStats | None = None
