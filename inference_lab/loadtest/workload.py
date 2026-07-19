"""Workload generation: synthetic controlled-shape prompts and ShareGPT sampling.

Both modes are fully determined by their spec (including the seed), so the same
``workload.json`` always regenerates the identical prompt list — the property
that makes A/B experiment runs comparable.

The tokenizer is injected behind a small protocol so unit tests can use a fake;
production code loads the real HF tokenizer for the served model.
"""

import json
import random
from pathlib import Path
from typing import Protocol

from inference_lab.common.logging import get_logger
from inference_lab.loadtest.models import (
    GeneratedPrompt,
    ShareGPTWorkload,
    SyntheticWorkload,
    WorkloadSpec,
)

logger = get_logger("loadtest.workload")

# Small fixed vocabulary for synthetic filler text. The words themselves are
# irrelevant — only the token-count shape of the prompt matters for serving
# performance (see LEARNING.md §5).
_FILLER_WORDS = (
    "system latency throughput batch cache memory tensor kernel stream decode prefill "
    "token request server metric sample vector matrix schedule queue worker buffer page "
    "block index shard replica cluster node graph model weight layer head context window"
).split()

_MAX_ADJUST_ITERATIONS = 10


class TokenizerLike(Protocol):
    """The minimal tokenizer interface workload generation needs."""

    def encode(self, text: str) -> list[int]:
        """Tokenize ``text`` (no special tokens) to ids."""
        ...

    def decode(self, ids: list[int]) -> str:
        """Inverse of :meth:`encode`."""
        ...


class HFTokenizer:
    """`tokenizers`-backed adapter satisfying :class:`TokenizerLike`."""

    def __init__(self, repo_id: str) -> None:
        from tokenizers import Tokenizer

        self._tok = Tokenizer.from_pretrained(repo_id)

    def encode(self, text: str) -> list[int]:
        return self._tok.encode(text, add_special_tokens=False).ids

    def decode(self, ids: list[int]) -> str:
        return self._tok.decode(ids, skip_special_tokens=False)


def load_tokenizer(repo_id: str) -> TokenizerLike:
    """Load the HF tokenizer for ``repo_id`` (downloads and caches on first use)."""
    logger.info("loading tokenizer %s", repo_id)
    return HFTokenizer(repo_id)


def _text_of_token_length(target: int, rng: random.Random, tokenizer: TokenizerLike) -> str:
    """Build seeded filler text that tokenizes to ~``target`` tokens.

    Strategy: oversample words, truncate the id sequence to ``target``, decode,
    and re-encode to check — decode/encode round-trips can merge or split
    tokens at the cut point, so we iterate a few times and accept the closest
    result (always within a token or two of the target in practice).
    """
    words: list[str] = []
    text = ""
    # Oversample: filler words are short, so ~1.5 tokens/word is a safe overshoot.
    while len(tokenizer.encode(text)) < target:
        words.extend(rng.choices(_FILLER_WORDS, k=max(16, target)))
        text = " ".join(words)

    for _ in range(_MAX_ADJUST_ITERATIONS):
        ids = tokenizer.encode(text)
        if len(ids) <= target:
            break
        text = tokenizer.decode(ids[:target])
    return text


def generate_synthetic(
    spec: SyntheticWorkload, tokenizer: TokenizerLike
) -> list[GeneratedPrompt]:
    """Generate ``spec.num_prompts`` prompts hitting the spec's token budgets.

    Each user message begins with a unique per-prompt marker so a server can
    never prefix-cache across requests; only the (optional) shared system
    prompt is identical across the workload, by design.
    """
    rng = random.Random(spec.seed)

    system_prompt: str | None = None
    if spec.shared_prefix_tokens > 0:
        system_prompt = _text_of_token_length(spec.shared_prefix_tokens, rng, tokenizer)

    prompts: list[GeneratedPrompt] = []
    for i in range(spec.num_prompts):
        marker = f"request {i:05d} :: "
        marker_tokens = len(tokenizer.encode(marker))
        filler_budget = max(1, spec.input_tokens - marker_tokens)
        content = marker + _text_of_token_length(filler_budget, rng, tokenizer)

        messages = []
        if system_prompt is not None:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": content})

        est = sum(len(tokenizer.encode(m["content"])) for m in messages)
        prompts.append(
            GeneratedPrompt(
                messages=messages,
                max_tokens=spec.output_tokens,
                est_input_tokens=est,
                ignore_eos=spec.ignore_eos,
            )
        )
    return prompts


def _sharegpt_dataset_path(spec: ShareGPTWorkload) -> Path:
    """Resolve the ShareGPT JSON file, downloading (with HF cache) if needed."""
    if spec.dataset_path is not None:
        return Path(spec.dataset_path)
    from huggingface_hub import hf_hub_download

    logger.info(
        "fetching %s/%s (cached after first download)", spec.dataset_repo, spec.dataset_file
    )
    return Path(
        hf_hub_download(spec.dataset_repo, spec.dataset_file, repo_type="dataset")
    )


def generate_sharegpt(
    spec: ShareGPTWorkload, tokenizer: TokenizerLike
) -> list[GeneratedPrompt]:
    """Sample real prompts from ShareGPT (seeded, so identical across runs).

    Convention follows vLLM's ``benchmark_serving.py``: take the first
    human→assistant exchange of each conversation; the prompt is the human
    turn, and ``max_tokens`` is the tokenized length of the real assistant
    reply (capped at ``spec.max_output_tokens``) so output lengths mirror real
    traffic instead of a fixed budget.
    """
    path = _sharegpt_dataset_path(spec)
    conversations = json.loads(path.read_text(encoding="utf-8"))

    candidates: list[GeneratedPrompt] = []
    for conv in conversations:
        turns = conv.get("conversations", [])
        if len(turns) < 2 or turns[0].get("from") != "human" or turns[1].get("from") != "gpt":
            continue
        prompt_text = turns[0].get("value", "")
        reply_text = turns[1].get("value", "")
        n_in = len(tokenizer.encode(prompt_text))
        n_out = len(tokenizer.encode(reply_text))
        if not (spec.min_input_tokens <= n_in <= spec.max_input_tokens):
            continue
        if n_out < spec.min_output_tokens:
            continue
        candidates.append(
            GeneratedPrompt(
                messages=[{"role": "user", "content": prompt_text}],
                max_tokens=min(n_out, spec.max_output_tokens),
                est_input_tokens=n_in,
            )
        )

    if len(candidates) < spec.num_prompts:
        raise ValueError(
            f"only {len(candidates)} ShareGPT conversations pass the filters; "
            f"{spec.num_prompts} requested"
        )
    rng = random.Random(spec.seed)
    return rng.sample(candidates, spec.num_prompts)


def generate_prompts(spec: WorkloadSpec, tokenizer: TokenizerLike) -> list[GeneratedPrompt]:
    """Dispatch to the generator for ``spec.mode``."""
    if isinstance(spec, SyntheticWorkload):
        return generate_synthetic(spec, tokenizer)
    return generate_sharegpt(spec, tokenizer)
