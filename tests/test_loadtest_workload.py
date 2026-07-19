"""Workload generation: token budgets hit within tolerance, seeded reproducibility.

Uses a character-level fake tokenizer (1 char = 1 token) so tests are exact,
offline, and independent of any HF download.
"""

import json

import pytest

from inference_lab.loadtest.models import ShareGPTWorkload, SyntheticWorkload
from inference_lab.loadtest.workload import generate_prompts, generate_sharegpt, generate_synthetic


class CharTokenizer:
    """1 token per character: reversible and exact, ideal for budget assertions."""

    def encode(self, text: str) -> list[int]:
        return [ord(c) for c in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(chr(i) for i in ids)


TOKENIZER = CharTokenizer()
BUDGET_TOLERANCE = 0.05  # ±5% of target input tokens


class TestSynthetic:
    def test_token_budgets_within_tolerance(self):
        spec = SyntheticWorkload(input_tokens=200, output_tokens=64, num_prompts=8, seed=1)
        prompts = generate_synthetic(spec, TOKENIZER)
        assert len(prompts) == 8
        for p in prompts:
            n = len(TOKENIZER.encode(p.messages[-1]["content"]))
            assert abs(n - 200) <= 200 * BUDGET_TOLERANCE
            assert p.max_tokens == 64
            assert p.est_input_tokens == pytest.approx(200, rel=BUDGET_TOLERANCE)

    def test_shared_prefix_is_identical_and_budgeted(self):
        spec = SyntheticWorkload(
            input_tokens=100, output_tokens=16, shared_prefix_tokens=50, num_prompts=4, seed=2
        )
        prompts = generate_synthetic(spec, TOKENIZER)
        systems = {p.messages[0]["content"] for p in prompts}
        assert all(p.messages[0]["role"] == "system" for p in prompts)
        assert len(systems) == 1  # shared across the whole workload
        n = len(TOKENIZER.encode(systems.pop()))
        assert abs(n - 50) <= max(2, 50 * BUDGET_TOLERANCE)

    def test_no_prefix_means_no_system_message(self):
        spec = SyntheticWorkload(input_tokens=50, output_tokens=16, num_prompts=2, seed=0)
        prompts = generate_synthetic(spec, TOKENIZER)
        assert all(len(p.messages) == 1 and p.messages[0]["role"] == "user" for p in prompts)

    def test_user_messages_are_unique_across_prompts(self):
        spec = SyntheticWorkload(input_tokens=64, output_tokens=16, num_prompts=16, seed=3)
        prompts = generate_synthetic(spec, TOKENIZER)
        contents = {p.messages[-1]["content"] for p in prompts}
        assert len(contents) == 16  # unique marker defeats cross-request prefix caching

    def test_ignore_eos_flows_from_spec_to_prompt_and_payload(self):
        from inference_lab.common.config import EndpointConfig
        from inference_lab.loadtest.client import build_payload

        base = dict(input_tokens=50, output_tokens=16, num_prompts=2, seed=0)
        endpoint = EndpointConfig(base_url="http://x/v1", model="m")
        on = generate_synthetic(SyntheticWorkload(ignore_eos=True, **base), TOKENIZER)
        off = generate_synthetic(SyntheticWorkload(**base), TOKENIZER)
        assert all(p.ignore_eos for p in on)
        assert build_payload(endpoint, on[0])["ignore_eos"] is True
        # default stays off, and the key is absent (not False) for OpenAI compatibility
        assert "ignore_eos" not in build_payload(endpoint, off[0])

    def test_same_seed_reproduces_identical_prompts(self):
        spec = SyntheticWorkload(input_tokens=80, output_tokens=16, num_prompts=4, seed=42)
        a = generate_synthetic(spec, TOKENIZER)
        b = generate_synthetic(spec, TOKENIZER)
        assert [p.messages for p in a] == [p.messages for p in b]

    def test_different_seed_differs(self):
        base = dict(input_tokens=80, output_tokens=16, num_prompts=4)
        a = generate_synthetic(SyntheticWorkload(seed=1, **base), TOKENIZER)
        b = generate_synthetic(SyntheticWorkload(seed=2, **base), TOKENIZER)
        assert [p.messages for p in a] != [p.messages for p in b]


@pytest.fixture
def sharegpt_file(tmp_path):
    """A small ShareGPT-shaped fixture with valid and filterable conversations."""
    good = [
        {
            "id": f"conv{i}",
            "conversations": [
                {"from": "human", "value": f"question {i} " + "x" * 40},
                {"from": "gpt", "value": f"answer {i} " + "y" * 60},
            ],
        }
        for i in range(8)
    ]
    bad = [
        # Starts with gpt: skipped.
        {"id": "gpt-first", "conversations": [{"from": "gpt", "value": "hi"}]},
        # Input too short (< min_input_tokens).
        {
            "id": "tiny",
            "conversations": [
                {"from": "human", "value": "?"},
                {"from": "gpt", "value": "z" * 50},
            ],
        },
        # Input too long (> max_input_tokens with char tokenizer).
        {
            "id": "huge",
            "conversations": [
                {"from": "human", "value": "x" * 5000},
                {"from": "gpt", "value": "z" * 50},
            ],
        },
        # Reply too short.
        {
            "id": "shortreply",
            "conversations": [
                {"from": "human", "value": "y" * 50},
                {"from": "gpt", "value": "ok"},
            ],
        },
    ]
    path = tmp_path / "sharegpt.json"
    path.write_text(json.dumps(good + bad), encoding="utf-8")
    return path


class TestShareGPT:
    def _spec(self, path, **overrides) -> ShareGPTWorkload:
        defaults = dict(
            num_prompts=5,
            min_input_tokens=10,
            max_input_tokens=2048,
            min_output_tokens=10,
            max_output_tokens=1024,
            seed=7,
            dataset_path=str(path),
        )
        return ShareGPTWorkload(**{**defaults, **overrides})

    def test_filters_and_samples(self, sharegpt_file):
        prompts = generate_sharegpt(self._spec(sharegpt_file), TOKENIZER)
        assert len(prompts) == 5
        for p in prompts:
            assert p.messages == [{"role": "user", "content": p.messages[0]["content"]}]
            assert p.messages[0]["content"].startswith("question ")  # bad convs filtered out

    def test_max_tokens_matches_real_reply_length_capped(self, sharegpt_file):
        capped = generate_sharegpt(self._spec(sharegpt_file, max_output_tokens=20), TOKENIZER)
        assert all(p.max_tokens == 20 for p in capped)
        uncapped = generate_sharegpt(self._spec(sharegpt_file), TOKENIZER)
        # Reply "answer {i} " + 60*"y" is ~69-70 chars => max_tokens mirrors it.
        assert all(60 < p.max_tokens < 75 for p in uncapped)

    def test_seeded_sampling_is_reproducible(self, sharegpt_file):
        a = generate_sharegpt(self._spec(sharegpt_file), TOKENIZER)
        b = generate_sharegpt(self._spec(sharegpt_file), TOKENIZER)
        assert [p.messages for p in a] == [p.messages for p in b]
        c = generate_sharegpt(self._spec(sharegpt_file, seed=8), TOKENIZER)
        assert [p.messages for p in a] != [p.messages for p in c]

    def test_too_few_candidates_raises(self, sharegpt_file):
        with pytest.raises(ValueError, match="pass the filters"):
            generate_sharegpt(self._spec(sharegpt_file, num_prompts=100), TOKENIZER)


def test_generate_prompts_dispatches(sharegpt_file):
    synthetic = SyntheticWorkload(input_tokens=32, output_tokens=8, num_prompts=2)
    assert len(generate_prompts(synthetic, TOKENIZER)) == 2
    sharegpt = ShareGPTWorkload(
        num_prompts=3, min_input_tokens=10, min_output_tokens=10, dataset_path=str(sharegpt_file)
    )
    assert len(generate_prompts(sharegpt, TOKENIZER)) == 3
