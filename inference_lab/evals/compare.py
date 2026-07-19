"""Comparison utility: make quality regressions inspectable, not just a delta.

Given two eval run dirs (A = reference/baseline, B = candidate), report the
score delta plus the exact questions that flipped correct→wrong and
wrong→correct. A flat delta hides churn: an optimization can score ±0 while
flipping a dozen questions each way, and *which* questions broke (long
arithmetic chains vs parsing quirks) is what makes quantization damage a
finding instead of a number.
"""

import json
from pathlib import Path

from pydantic import BaseModel, Field

from inference_lab.evals.models import QuestionRecord

_SNIPPET_LEN = 60


class FlippedQuestion(BaseModel):
    """One question whose correctness differs between the two runs."""

    id: str
    question: str
    gold: str
    parsed_a: str | None
    parsed_b: str | None


class RunComparison(BaseModel):
    """Result of comparing two eval runs on their common question set."""

    run_a: str
    run_b: str
    task: str
    accuracy_a: float
    accuracy_b: float
    delta: float = Field(description="accuracy_b - accuracy_a")
    num_common: int = Field(description="Questions present in both runs")
    subset_match: bool = Field(
        description="True if both runs evaluated the identical subset (content hash equal); "
        "a mismatch means the comparison is not a controlled A/B"
    )
    flipped_to_wrong: list[FlippedQuestion]
    flipped_to_correct: list[FlippedQuestion]


def _eval_dir(run_dir: Path) -> Path:
    """Accept either the run dir (``experiments/x``) or its ``eval/`` subdir."""
    if (run_dir / "questions.jsonl").exists():
        return run_dir
    return run_dir / "eval"


def load_eval_run(run_dir: Path) -> tuple[dict, dict[str, QuestionRecord]]:
    """Load (meta, records-by-id) from an eval run directory."""
    eval_dir = _eval_dir(run_dir)
    meta = json.loads((eval_dir / "meta.json").read_text(encoding="utf-8"))
    records = {}
    for line in (eval_dir / "questions.jsonl").read_text(encoding="utf-8").splitlines():
        record = QuestionRecord.model_validate_json(line)
        records[record.id] = record
    return meta, records


def compare_runs(dir_a: Path, dir_b: Path) -> RunComparison:
    """Compare two eval runs; accuracies are recomputed from the raw records."""
    meta_a, records_a = load_eval_run(dir_a)
    meta_b, records_b = load_eval_run(dir_b)

    common_ids = sorted(records_a.keys() & records_b.keys())
    flipped_to_wrong: list[FlippedQuestion] = []
    flipped_to_correct: list[FlippedQuestion] = []
    for qid in common_ids:
        a, b = records_a[qid], records_b[qid]
        if a.correct == b.correct:
            continue
        flip = FlippedQuestion(
            id=qid, question=a.question, gold=a.gold, parsed_a=a.parsed, parsed_b=b.parsed
        )
        (flipped_to_wrong if a.correct else flipped_to_correct).append(flip)

    accuracy_a = sum(r.correct for r in records_a.values()) / max(1, len(records_a))
    accuracy_b = sum(r.correct for r in records_b.values()) / max(1, len(records_b))
    return RunComparison(
        run_a=str(dir_a),
        run_b=str(dir_b),
        task=meta_a.get("task", {}).get("task", "unknown"),
        accuracy_a=accuracy_a,
        accuracy_b=accuracy_b,
        delta=accuracy_b - accuracy_a,
        num_common=len(common_ids),
        subset_match=meta_a.get("subset", {}).get("hash") == meta_b.get("subset", {}).get("hash"),
        flipped_to_wrong=flipped_to_wrong,
        flipped_to_correct=flipped_to_correct,
    )


def _flip_lines(title: str, flips: list[FlippedQuestion]) -> list[str]:
    lines = [f"{title} ({len(flips)}):"]
    if not flips:
        return [f"{title}: none"]
    for flip in flips:
        snippet = flip.question[:_SNIPPET_LEN] + ("…" if len(flip.question) > _SNIPPET_LEN else "")
        lines.append(
            f"    {flip.id}  gold={flip.gold}  A={flip.parsed_a}  B={flip.parsed_b}  {snippet!r}"
        )
    return lines


def format_comparison(cmp: RunComparison) -> str:
    """Human-readable comparison report."""
    subset_note = (
        "subsets match"
        if cmp.subset_match
        else "SUBSETS DIFFER — this is not a controlled comparison"
    )
    lines = [
        f"eval comparison ({cmp.task}): A={cmp.run_a}  B={cmp.run_b}",
        f"  {cmp.num_common} common questions ({subset_note})",
        f"  accuracy: {cmp.accuracy_a:.3f} -> {cmp.accuracy_b:.3f}  (delta {cmp.delta:+.3f})",
        *_flip_lines("  flipped correct->wrong", cmp.flipped_to_wrong),
        *_flip_lines("  flipped wrong->correct", cmp.flipped_to_correct),
    ]
    return "\n".join(lines)
