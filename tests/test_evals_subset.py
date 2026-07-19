"""Subset determinism tests.

Comparability across experiments is sacred: the same (questions, size, seed)
must select the identical subset forever, regardless of the order the dataset
happened to arrive in.
"""

import random

import pytest

from inference_lab.evals.models import Question
from inference_lab.evals.tasks import sample_questions, subset_hash


def _questions(n: int = 50) -> list[Question]:
    return [
        Question(id=f"gsm8k-test-{i:04d}", question=f"question {i}?", gold=str(i)) for i in range(n)
    ]


def test_same_seed_same_subset() -> None:
    a = sample_questions(_questions(), 10, seed=0)
    b = sample_questions(_questions(), 10, seed=0)
    assert [q.id for q in a] == [q.id for q in b]


def test_input_order_does_not_matter() -> None:
    shuffled = _questions()
    random.Random(99).shuffle(shuffled)
    a = sample_questions(_questions(), 10, seed=0)
    b = sample_questions(shuffled, 10, seed=0)
    assert [q.id for q in a] == [q.id for q in b]


def test_different_seed_different_subset() -> None:
    a = sample_questions(_questions(), 10, seed=0)
    b = sample_questions(_questions(), 10, seed=1)
    assert [q.id for q in a] != [q.id for q in b]


def test_subset_is_sorted_by_id() -> None:
    ids = [q.id for q in sample_questions(_questions(), 10, seed=0)]
    assert ids == sorted(ids)


def test_oversized_sample_raises() -> None:
    with pytest.raises(ValueError, match="cannot sample"):
        sample_questions(_questions(5), 6, seed=0)


def test_subset_hash_detects_content_drift() -> None:
    subset = sample_questions(_questions(), 10, seed=0)
    assert subset_hash(subset) == subset_hash([q.model_copy() for q in subset])

    drifted = [q.model_copy() for q in subset]
    drifted[3] = drifted[3].model_copy(update={"gold": "changed"})
    assert subset_hash(subset) != subset_hash(drifted)
