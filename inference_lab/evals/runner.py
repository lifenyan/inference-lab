"""Async eval runner: bounded concurrency, retries on transient errors.

Requests are non-streaming (only the final text matters) at temperature 0 —
the eval detects regressions between two configurations of the same model, so
sampling noise is the enemy.

Output layout (``experiments/<run-name>/eval/``):

- ``meta.json``       — endpoint, task config, subset ids + content hash,
  shared prompt prefix, runner settings, timestamps, library versions
- ``questions.jsonl`` — raw per-question records (the score is recomputable)
- ``score.json``      — the aggregate :class:`EvalScore`

Scoring convention: a question whose request ultimately fails counts as wrong,
never dropped — accuracy denominators stay identical across A/B runs — and
``num_errors`` is reported so a run degraded by endpoint failures is visible.
"""

import asyncio
import datetime
import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from inference_lab.common.config import EndpointConfig
from inference_lab.common.logging import get_logger, log_event
from inference_lab.common.versions import collect_versions
from inference_lab.evals.models import EvalScore, Question, QuestionRecord
from inference_lab.evals.tasks import EvalTask, subset_hash

logger = get_logger("evals.runner")

_TRANSIENT_STATUS = frozenset({408, 429, 500, 502, 503, 504})


@dataclass
class EvalRunConfig:
    """Runner parameters (recorded into ``meta.json``)."""

    concurrency: int = 8
    max_retries: int = 3
    retry_backoff_s: float = 0.5
    temperature: float = 0.0
    run_name: str = field(default_factory=lambda: f"eval-{uuid.uuid4().hex[:8]}")


async def _request_answer(
    client: httpx.AsyncClient,
    endpoint: EndpointConfig,
    messages: list[dict[str, str]],
    run_cfg: EvalRunConfig,
    *,
    max_tokens: int,
) -> tuple[str | None, str | None, int]:
    """Request one completion; returns (content, final_error, attempts).

    Transport errors and transient HTTP statuses are retried with exponential
    backoff; other HTTP errors (e.g. 400) fail immediately. Never raises.
    """
    url = f"{endpoint.base_url.rstrip('/')}/chat/completions"
    payload = {
        "model": endpoint.model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": run_cfg.temperature,
    }
    error = "no attempts made"
    for attempt in range(1, run_cfg.max_retries + 2):
        if attempt > 1:
            await asyncio.sleep(run_cfg.retry_backoff_s * 2 ** (attempt - 2))
        try:
            response = await client.post(
                url,
                json=payload,
                headers={"Authorization": f"Bearer {endpoint.api_key}"},
                timeout=endpoint.timeout_s,
            )
        except httpx.HTTPError as exc:
            error = f"{type(exc).__name__}: {exc}"
            continue
        if response.status_code in _TRANSIENT_STATUS:
            error = f"HTTP {response.status_code}: {response.text[:200]}"
            continue
        if response.status_code != 200:
            return None, f"HTTP {response.status_code}: {response.text[:200]}", attempt
        try:
            content = response.json()["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError, ValueError):
            return None, f"malformed response body: {response.text[:200]}", attempt
        return content or "", None, attempt
    return None, error, run_cfg.max_retries + 1


async def run_eval(
    endpoint: EndpointConfig,
    task: EvalTask,
    run_cfg: EvalRunConfig,
    out_dir: Path,
) -> EvalScore:
    """Run the task's full subset against the endpoint and write the eval dir.

    ``meta.json`` is written up front so a crashed run still leaves its
    context (including the exact subset ids) next to whatever records it
    produced.
    """
    eval_dir = Path(out_dir) / "eval"
    records_path = eval_dir / "questions.jsonl"
    if records_path.exists():
        raise FileExistsError(f"{records_path} already exists; refusing to mix runs")
    eval_dir.mkdir(parents=True, exist_ok=True)

    questions = task.load_questions()
    meta = {
        "run_name": run_cfg.run_name,
        "endpoint": {"base_url": endpoint.base_url, "model": endpoint.model},
        "task": task.config.model_dump(),
        "subset": {
            "size": len(questions),
            "hash": subset_hash(questions),
            "ids": [q.id for q in questions],
        },
        "prompt_prefix": task.prompt_prefix(),
        "runner": {
            "concurrency": run_cfg.concurrency,
            "max_retries": run_cfg.max_retries,
            "retry_backoff_s": run_cfg.retry_backoff_s,
            "temperature": run_cfg.temperature,
        },
        "started_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "finished_at": None,
        "versions": collect_versions(
            ("inference-lab", "httpx", "pydantic", "pyarrow", "huggingface-hub")
        ),
    }
    meta_path = eval_dir / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    logger.info(
        "eval %s: %d questions against %s (concurrency %d)",
        task.name,
        len(questions),
        endpoint.base_url,
        run_cfg.concurrency,
    )

    records: list[QuestionRecord] = []
    semaphore = asyncio.Semaphore(run_cfg.concurrency)
    limits = httpx.Limits(max_connections=run_cfg.concurrency + 4)
    async with httpx.AsyncClient(limits=limits) as client:

        async def evaluate(question: Question) -> None:
            async with semaphore:
                t_start = time.perf_counter()
                content, error, attempts = await _request_answer(
                    client,
                    endpoint,
                    task.build_messages(question),
                    run_cfg,
                    max_tokens=task.config.max_tokens,
                )
                latency_s = time.perf_counter() - t_start
            parsed = task.parse_answer(content) if content is not None else None
            record = QuestionRecord(
                id=question.id,
                question=question.question,
                raw_response=content,
                parsed=parsed,
                gold=question.gold,
                correct=task.is_correct(parsed, question.gold),
                error=error,
                attempts=attempts,
                latency_s=latency_s,
            )
            records.append(record)
            log_event(records_path, record.model_dump())
            if len(records) % 25 == 0 or len(records) == len(questions):
                logger.info(
                    "progress: %d/%d answered, %d correct, %d errors",
                    len(records),
                    len(questions),
                    sum(r.correct for r in records),
                    sum(r.error is not None for r in records),
                )

        async with asyncio.TaskGroup() as tg:
            for question in questions:
                tg.create_task(evaluate(question))

    score = EvalScore(
        task=task.name,
        num_questions=len(questions),
        num_correct=sum(r.correct for r in records),
        num_errors=sum(r.error is not None for r in records),
        accuracy=sum(r.correct for r in records) / len(questions),
    )
    (eval_dir / "score.json").write_text(
        score.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    meta["finished_at"] = datetime.datetime.now(datetime.UTC).isoformat()
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    logger.info(
        "score: %d/%d = %.1f%% (%d errors) -> %s",
        score.num_correct,
        score.num_questions,
        100 * score.accuracy,
        score.num_errors,
        eval_dir / "score.json",
    )
    return score
