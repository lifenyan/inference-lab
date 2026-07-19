"""Pydantic models for the quality eval runner.

Three groups, mirroring the load-test harness:

- **Task configs** (`GSM8KConfig`) — the small spec that fully determines an
  eval: which dataset, how many questions, which seed, how to prompt. Saved
  verbatim into the run's ``meta.json`` so any eval can be reproduced.
- **Per-question records** (`Question`, `QuestionRecord`) — the raw material,
  one JSONL line per question in ``questions.jsonl``. The score is recomputable
  from these; we never store only the aggregate. The shared few-shot prompt
  prefix is recorded once in ``meta.json``; each record carries only the
  per-question user turn.
- **Aggregate** (`EvalScore`) — the summary written to ``score.json``.
"""

from typing import Literal

from pydantic import BaseModel, Field


class GSM8KConfig(BaseModel):
    """Spec for a GSM8K eval: a fixed seeded subset of the test split.

    The (dataset revision, ``num_questions``, ``seed``) triple selects the same
    questions forever — comparability across experiments depends on it.
    """

    task: Literal["gsm8k"] = "gsm8k"
    num_questions: int = Field(default=300, gt=0, description="Subset size sampled from the split")
    seed: int = 0
    few_shot: int = Field(default=5, ge=1, le=8, description="Number of few-shot exemplars")
    max_tokens: int = Field(default=512, gt=0, description="Completion budget per question")
    dataset_repo: str = "openai/gsm8k"
    dataset_file: str = "main/test-00000-of-00001.parquet"
    dataset_path: str | None = Field(
        default=None, description="Local parquet path override; skips the HF download when set"
    )


# Becomes a discriminated union (like WorkloadSpec) when a second task lands.
TaskConfig = GSM8KConfig


class Question(BaseModel):
    """One benchmark question with its canonical gold answer."""

    id: str = Field(description="Stable id, e.g. gsm8k-test-0042 (row index in the split)")
    question: str
    gold: str = Field(description="Canonical answer string (normalized, e.g. '72' or '3.5')")


class QuestionRecord(BaseModel):
    """Raw result for one question (one JSONL line in ``questions.jsonl``)."""

    id: str
    question: str = Field(description="The per-question user turn; shared prefix is in meta.json")
    raw_response: str | None = Field(description="Full model output; None if the request failed")
    parsed: str | None = Field(description="Canonical answer extracted from the response")
    gold: str
    correct: bool
    error: str | None = Field(default=None, description="Final error if all attempts failed")
    attempts: int = Field(default=1, ge=1, description="Request attempts, including retries")
    latency_s: float | None = Field(default=None, ge=0)


class EvalScore(BaseModel):
    """Aggregate score written to ``score.json``.

    Failed requests count as wrong (never dropped), so accuracy denominators
    are identical across A/B runs; ``num_errors`` makes a run degraded by
    endpoint failures visible rather than silently lower-scoring.
    """

    task: str
    num_questions: int
    num_correct: int
    num_errors: int
    accuracy: float = Field(description="num_correct / num_questions; errors count as wrong")
