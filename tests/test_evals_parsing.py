"""Unit tests for GSM8K answer parsing and prompt construction.

The parser is where eval bugs hide: a format it silently mishandles becomes a
phantom quality regression (or masks a real one) in every experiment that
follows, so the nasty formats get exhaustive coverage.
"""

import pytest

from inference_lab.evals.gsm8k import (
    FEW_SHOT_EXAMPLES,
    GSM8KTask,
    normalize_number,
    parse_numeric_answer,
)
from inference_lab.evals.models import GSM8KConfig, Question


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # The instructed format.
        ("#### 72", "72"),
        ("She sold 48 / 2 = 24 clips in May.\n#### 24", "24"),
        # Marker beats earlier calculations.
        ("First I computed 10, then corrected it. #### 12", "12"),
        ("First try #### 1 was wrong. #### 2", "2"),  # last marker wins
        ("#### 72 eggs remain", "72"),  # trailing words after the marker's number
        # Currency, separators, units, sentence punctuation.
        ("The answer is $1,000.", "1000"),
        ("#### $18", "18"),
        ("#### 1,234.50", "1234.5"),
        ("She has 72 eggs left.", "72"),
        ("So the total is 1,000,000 dollars", "1000000"),
        # Numeric canonicalization: integral floats collapse to ints.
        ("The answer is 72.0", "72"),
        ("3.50", "3.5"),
        ("#### -5", "-5"),
        # No-marker fallback: last number in the text.
        ("Step 1: 3 + 4 = 7. Step 2: 7 * 2 = 14. So 14.", "14"),
        # Unparseable responses.
        ("no numbers here at all", None),
        ("", None),
        ("#### unknown", None),  # marker present but no number after it -> no rescue
    ],
)
def test_parse_numeric_answer(text: str, expected: str | None) -> None:
    assert parse_numeric_answer(text) == expected


def test_normalize_number_rejects_garbage() -> None:
    assert normalize_number("$,") is None


def test_gold_answers_use_the_marker_path() -> None:
    gold = "Natalia sold 48/2 = <<48/2=24>>24 clips in May.\n#### 72"
    assert parse_numeric_answer(gold) == "72"


def test_equivalent_forms_compare_equal() -> None:
    task = GSM8KTask(GSM8KConfig())
    assert task.is_correct(parse_numeric_answer("The answer is $72.00"), "72")
    assert not task.is_correct(None, "0")  # unparseable is wrong, even vs gold '0'


def test_few_shot_exemplars_parse_to_a_number() -> None:
    for _, answer in FEW_SHOT_EXAMPLES:
        assert parse_numeric_answer(answer) is not None


def test_build_messages_shape() -> None:
    task = GSM8KTask(GSM8KConfig(few_shot=3))
    question = Question(id="q", question="What is 2 + 2?", gold="4")
    messages = task.build_messages(question)

    assert len(messages) == 1 + 2 * 3 + 1  # system + 3 exemplar pairs + question
    assert messages[0]["role"] == "system"
    roles = [m["role"] for m in messages[1:]]
    assert roles == ["user", "assistant"] * 3 + ["user"]
    # The real question must be the last user turn (canned_map keys on it, and
    # it keeps the shared prefix identical across questions).
    assert messages[-1] == {"role": "user", "content": "What is 2 + 2?"}
