"""Integration: eval runner against the mock server rigged with canned answers.

The mock's ``canned_map`` keys on a marker word in each question's user turn,
so each of the four questions gets a scripted reply: three parse to the gold
answer, one is wrong -> the runner must report exactly 0.75. A second rig flips
one question each way and the comparison utility must name the exact flips.
"""

import json
import socket
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from inference_lab.common.config import EndpointConfig
from inference_lab.evals.compare import compare_runs, format_comparison
from inference_lab.evals.gsm8k import GSM8KTask
from inference_lab.evals.models import GSM8KConfig, Question, QuestionRecord
from inference_lab.evals.runner import EvalRunConfig, run_eval
from inference_lab.loadtest.mockserver import MockServerConfig, create_app

QUESTIONS = [
    Question(id="q-000", question="alpha: A farmer has 3 hens, buys 4 more. How many?", gold="7"),
    Question(id="q-001", question="beta: Two sixes of eggs. How many eggs?", gold="12"),
    Question(id="q-002", question="gamma: Ten trios of cars. How many cars?", gold="30"),
    Question(id="q-003", question="delta: Three trios of cats. How many cats?", gold="9"),
]

# Three correct, one wrong (delta) -> accuracy 0.75. gamma has no '####' marker
# on purpose: it exercises the last-number fallback end-to-end.
CANNED_A = {
    "alpha": "3 + 4 = 7 hens.\n#### 7",
    "beta": "6 * 2 = 12 eggs.\n#### 12",
    "gamma": "The answer is $30.",
    "delta": "3 * 3 = 8 cats.\n#### 8",
}

# Also 0.75, but beta flips correct->wrong and delta flips wrong->correct.
CANNED_B = CANNED_A | {"beta": "6 + 2 = 8 eggs.\n#### 8", "delta": "3 * 3 = 9 cats.\n#### 9"}


class CannedGSM8KTask(GSM8KTask):
    """GSM8K prompting/parsing over a fixed local question set (no download)."""

    def load_questions(self) -> list[Question]:
        return list(QUESTIONS)


@contextmanager
def run_app(app: FastAPI) -> Iterator[str]:
    """Serve an ASGI app in a background thread on an ephemeral port."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started:
        if time.monotonic() > deadline:
            raise RuntimeError("mock server failed to start within 10s")
        time.sleep(0.01)
    try:
        yield f"http://127.0.0.1:{port}/v1"
    finally:
        server.should_exit = True
        thread.join(timeout=5)


def _endpoint(base_url: str) -> EndpointConfig:
    return EndpointConfig(base_url=base_url, model="mock-model", timeout_s=10)


def _task() -> CannedGSM8KTask:
    return CannedGSM8KTask(GSM8KConfig(num_questions=4, few_shot=2))


async def test_eval_measures_known_score(tmp_path):
    with run_app(create_app(MockServerConfig(canned_map=CANNED_A))) as base_url:
        score = await run_eval(
            _endpoint(base_url), _task(), EvalRunConfig(concurrency=2), tmp_path / "run"
        )

    assert score.num_questions == 4
    assert score.num_correct == 3
    assert score.num_errors == 0
    assert score.accuracy == 0.75

    eval_dir = tmp_path / "run" / "eval"
    for name in ("meta.json", "questions.jsonl", "score.json"):
        assert (eval_dir / name).exists(), name

    meta = json.loads((eval_dir / "meta.json").read_text())
    assert meta["subset"]["ids"] == [q.id for q in QUESTIONS]
    assert meta["subset"]["hash"]
    assert meta["finished_at"] is not None
    assert "api_key" not in json.dumps(meta)  # secrets never land in run metadata

    # Raw records are sufficient to recompute the published score.
    records = {
        r.id: r
        for line in (eval_dir / "questions.jsonl").read_text().splitlines()
        if (r := QuestionRecord.model_validate_json(line))
    }
    assert len(records) == 4
    assert sum(r.correct for r in records.values()) == score.num_correct
    assert records["q-002"].parsed == "30"  # no-marker fallback path
    assert records["q-003"].parsed == "8" and not records["q-003"].correct


async def test_compare_reports_exact_flips(tmp_path):
    for name, canned in (("a", CANNED_A), ("b", CANNED_B)):
        with run_app(create_app(MockServerConfig(canned_map=canned))) as base_url:
            await run_eval(
                _endpoint(base_url), _task(), EvalRunConfig(concurrency=2), tmp_path / name
            )

    cmp = compare_runs(tmp_path / "a", tmp_path / "b")
    assert cmp.subset_match
    assert cmp.num_common == 4
    assert cmp.accuracy_a == cmp.accuracy_b == 0.75
    assert cmp.delta == 0.0  # flat delta hides the churn below — the point of compare
    assert [f.id for f in cmp.flipped_to_wrong] == ["q-001"]
    assert [f.id for f in cmp.flipped_to_correct] == ["q-003"]

    report = format_comparison(cmp)
    assert "q-001" in report and "q-003" in report


async def test_endpoint_down_records_errors_not_raise(tmp_path):
    endpoint = EndpointConfig(base_url="http://127.0.0.1:9", model="mock-model", timeout_s=2)
    run_cfg = EvalRunConfig(concurrency=2, max_retries=1, retry_backoff_s=0.01)

    score = await run_eval(endpoint, _task(), run_cfg, tmp_path / "run")

    assert score.num_errors == 4
    assert score.num_correct == 0
    assert score.accuracy == 0.0  # errors count as wrong, never dropped


def _flaky_app(fail_first: int) -> FastAPI:
    """Return 503 for the first ``fail_first`` requests, then answer normally."""
    app = FastAPI()
    state = {"failures": 0}

    @app.post("/v1/chat/completions")
    async def chat(request: dict) -> JSONResponse:
        if state["failures"] < fail_first:
            state["failures"] += 1
            return JSONResponse({"error": "overloaded"}, status_code=503)
        return JSONResponse(
            {
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "3 + 4 = 7 hens.\n#### 7"},
                        "finish_reason": "stop",
                    }
                ]
            }
        )

    return app


async def test_transient_errors_are_retried(tmp_path):
    task = CannedGSM8KTask(GSM8KConfig(num_questions=1, few_shot=2))
    task.load_questions = lambda: [QUESTIONS[0]]  # type: ignore[method-assign]
    run_cfg = EvalRunConfig(concurrency=1, max_retries=3, retry_backoff_s=0.01)

    with run_app(_flaky_app(fail_first=2)) as base_url:
        score = await run_eval(_endpoint(base_url), task, run_cfg, tmp_path / "run")

    assert score.num_errors == 0
    assert score.accuracy == 1.0
    record = QuestionRecord.model_validate_json(
        (tmp_path / "run" / "eval" / "questions.jsonl").read_text().splitlines()[0]
    )
    assert record.attempts == 3  # two 503s, then success
