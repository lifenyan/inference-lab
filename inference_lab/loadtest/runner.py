"""Closed-loop concurrency sweep runner and run-directory output layout.

Closed loop means N workers each keep exactly one request in flight — a new
request is issued only when the previous one finishes. This measures the
server's behavior *at* a given concurrency; it does not model open-loop
(Poisson-arrival) traffic where load is independent of server speed. Open-loop
generation is noted as future work.

Output layout (``experiments/<run-name>/``):

- ``workload.json``  — the workload spec (regenerates the identical prompts)
- ``meta.json``      — endpoint, sweep parameters, timestamps, library versions
- ``requests.jsonl`` — raw per-request records (every aggregate is recomputable)
- ``summary.json``   — per-concurrency-level aggregates
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
from inference_lab.loadtest.client import run_streaming_request
from inference_lab.loadtest.models import (
    GeneratedPrompt,
    LevelSummary,
    RequestRecord,
    WorkloadSpec,
    dump_workload,
)
from inference_lab.loadtest.stats import summarize_level

logger = get_logger("loadtest.runner")


@dataclass
class SweepConfig:
    """Parameters of a concurrency sweep (recorded into ``meta.json``)."""

    concurrency_levels: list[int]
    num_requests: int = 32
    warmup_requests: int = 4
    run_name: str = field(default_factory=lambda: f"run-{uuid.uuid4().hex[:8]}")


def _versions() -> dict[str, str]:
    """Library/platform versions recorded for reproducibility."""
    return collect_versions(("inference-lab", "httpx", "pydantic", "numpy", "tokenizers"))


async def run_level(
    endpoint: EndpointConfig,
    prompts: list[GeneratedPrompt],
    *,
    concurrency: int,
    num_requests: int,
    warmup_requests: int,
    origin_s: float,
    records_path: Path,
) -> list[RequestRecord]:
    """Run one closed-loop level: ``concurrency`` workers, fixed request count.

    The first ``warmup_requests`` issued are flagged as warmup (recorded, but
    excluded from aggregates); prompts are consumed round-robin so the same
    workload list serves any request count.
    """
    total = warmup_requests + num_requests
    next_index = 0
    records: list[RequestRecord] = []

    limits = httpx.Limits(max_connections=concurrency + 4)
    async with httpx.AsyncClient(limits=limits) as client:

        async def worker() -> None:
            nonlocal next_index
            while next_index < total:
                i = next_index
                next_index += 1  # single-threaded event loop: no await between read and write
                record = await run_streaming_request(
                    client,
                    endpoint,
                    prompts[i % len(prompts)],
                    request_id=f"c{concurrency}-{i:05d}",
                    concurrency=concurrency,
                    warmup=i < warmup_requests,
                    origin_s=origin_s,
                )
                records.append(record)
                log_event(records_path, record.model_dump())

        async with asyncio.TaskGroup() as tg:
            for _ in range(min(concurrency, total)):
                tg.create_task(worker())

    return records


async def run_sweep(
    endpoint: EndpointConfig,
    workload: WorkloadSpec,
    prompts: list[GeneratedPrompt],
    sweep: SweepConfig,
    out_dir: Path,
) -> list[LevelSummary]:
    """Run the full concurrency sweep and write the run directory.

    ``workload.json`` and ``meta.json`` are written up front so a crashed run
    still leaves its context on disk next to whatever records it produced.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    records_path = out_dir / "requests.jsonl"
    if records_path.exists():
        raise FileExistsError(f"{records_path} already exists; refusing to mix runs")

    (out_dir / "workload.json").write_text(dump_workload(workload) + "\n", encoding="utf-8")
    meta = {
        "run_name": sweep.run_name,
        "endpoint": {"base_url": endpoint.base_url, "model": endpoint.model},
        "sweep": {
            "concurrency_levels": sweep.concurrency_levels,
            "num_requests": sweep.num_requests,
            "warmup_requests": sweep.warmup_requests,
        },
        "started_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "finished_at": None,
        "versions": _versions(),
    }
    meta_path = out_dir / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    origin_s = time.perf_counter()
    summaries: list[LevelSummary] = []
    for concurrency in sweep.concurrency_levels:
        logger.info(
            "level c=%d: %d warmup + %d measured requests",
            concurrency,
            sweep.warmup_requests,
            sweep.num_requests,
        )
        records = await run_level(
            endpoint,
            prompts,
            concurrency=concurrency,
            num_requests=sweep.num_requests,
            warmup_requests=sweep.warmup_requests,
            origin_s=origin_s,
            records_path=records_path,
        )
        summary = summarize_level(concurrency, records)
        summaries.append(summary)
        logger.info(
            "level c=%d: %.1f tok/s, TTFT p50=%.3fs p99=%.3fs, errors=%d/%d",
            concurrency,
            summary.throughput_tok_s,
            summary.ttft_s.p50 if summary.ttft_s else float("nan"),
            summary.ttft_s.p99 if summary.ttft_s else float("nan"),
            summary.num_errors,
            summary.num_requests,
        )

    (out_dir / "summary.json").write_text(
        json.dumps([s.model_dump() for s in summaries], indent=2) + "\n", encoding="utf-8"
    )
    meta["finished_at"] = datetime.datetime.now(datetime.UTC).isoformat()
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return summaries


def load_summaries(run_dir: Path) -> list[LevelSummary]:
    """Load ``summary.json`` from a run directory."""
    raw = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    return [LevelSummary.model_validate(item) for item in raw]
