"""Task abstraction: what an eval task must provide, plus subset utilities.

A task owns three things: which questions (a fixed seeded subset), how to
prompt (few-shot messages), and how to parse/score a response. The runner is
task-agnostic, so a second task (e.g. an MMLU multiple-choice subset) slots in
as one new ``TaskConfig`` variant plus one ``EvalTask`` subclass.
"""

import hashlib
import json
import random
from abc import ABC, abstractmethod

from inference_lab.evals.models import Question, TaskConfig


class EvalTask(ABC):
    """One benchmark task: question source, prompting, parsing, and scoring."""

    name: str

    def __init__(self, config: TaskConfig) -> None:
        self.config = config

    @abstractmethod
    def load_questions(self) -> list[Question]:
        """Return the fixed seeded subset this eval runs on (same forever)."""

    @abstractmethod
    def prompt_prefix(self) -> list[dict[str, str]]:
        """Messages shared by every question (system prompt + few-shot turns)."""

    @abstractmethod
    def build_messages(self, question: Question) -> list[dict[str, str]]:
        """Full message list for one question (prefix + the question turn)."""

    @abstractmethod
    def parse_answer(self, text: str) -> str | None:
        """Extract the canonical answer from a model response, or None."""

    def is_correct(self, parsed: str | None, gold: str) -> bool:
        """Exact match on canonical answer strings; an unparseable answer is wrong."""
        return parsed is not None and parsed == gold


def sample_questions(questions: list[Question], num: int, seed: int) -> list[Question]:
    """Sample a deterministic subset: same (questions, num, seed) -> same ids, forever.

    Candidates are sorted by id before sampling so the result is independent of
    the order the dataset happened to arrive in, and the sample is re-sorted so
    run order is deterministic too.
    """
    ordered = sorted(questions, key=lambda q: q.id)
    if num > len(ordered):
        raise ValueError(f"cannot sample {num} questions from {len(ordered)}")
    picked = random.Random(seed).sample(ordered, num)
    return sorted(picked, key=lambda q: q.id)


def subset_hash(questions: list[Question]) -> str:
    """Content hash of a subset; detects dataset drift between two runs."""
    payload = json.dumps(
        [[q.id, q.question, q.gold] for q in questions], ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def build_task(config: TaskConfig) -> EvalTask:
    """Instantiate the task for a config."""
    from inference_lab.evals.gsm8k import GSM8KTask

    if config.task == "gsm8k":
        return GSM8KTask(config)
    raise ValueError(f"unknown task: {config.task}")
