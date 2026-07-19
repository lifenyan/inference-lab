"""GSM8K task: seeded test-split subset, few-shot CoT prompting, numeric parsing.

The answer parser is where eval bugs hide, so the format is kept uniform
end-to-end: gold answers in the dataset end with ``#### <number>``, the
few-shot exemplars end with ``#### <number>``, and the system prompt asks the
model to do the same. Parsing prefers the text after the last ``####`` marker
and falls back to the last number anywhere in the response (models under
quantization damage often drop the marker before they drop the arithmetic).

Numbers are canonicalized before comparison so ``72``, ``72.0``, ``$72`` and
``1,000`` vs ``1000`` compare equal. Fractions and units are out of scope —
GSM8K golds are plain integers/decimals.
"""

import re
from pathlib import Path

from inference_lab.common.logging import get_logger
from inference_lab.evals.models import GSM8KConfig, Question
from inference_lab.evals.tasks import EvalTask, sample_questions

logger = get_logger("evals.gsm8k")

SYSTEM_PROMPT = (
    "You are solving grade-school math word problems. Reason step by step, then "
    "give the final numeric answer on its own last line in the form '#### <answer>'."
)

# Standard chain-of-thought exemplars (Wei et al. 2022, drawn from the GSM8K
# train split), reformatted to end with the dataset's own '#### <answer>'
# marker so the parser sees one format everywhere.
FEW_SHOT_EXAMPLES: tuple[tuple[str, str], ...] = (
    (
        "There are 15 trees in the grove. Grove workers will plant trees in the grove "
        "today. After they are done, there will be 21 trees. How many trees did the "
        "grove workers plant today?",
        "There are 15 trees originally. After planting there are 21 trees, so the workers "
        "planted 21 - 15 = 6 trees.\n#### 6",
    ),
    (
        "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars "
        "are in the parking lot?",
        "There are 3 cars originally and 2 more arrive, so there are 3 + 2 = 5 cars.\n#### 5",
    ),
    (
        "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces "
        "do they have left in total?",
        "Together they had 32 + 42 = 74 chocolates. After eating 35, they have "
        "74 - 35 = 39 pieces left.\n#### 39",
    ),
    (
        "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 "
        "lollipops. How many lollipops did Jason give to Denny?",
        "Jason started with 20 lollipops and has 12 left, so he gave Denny "
        "20 - 12 = 8 lollipops.\n#### 8",
    ),
    (
        "Shawn has five toys. For Christmas, he got two toys each from his mom and dad. "
        "How many toys does he have now?",
        "Shawn started with 5 toys. He got 2 toys from his mom and 2 from his dad, so he "
        "has 5 + 2 + 2 = 9 toys.\n#### 9",
    ),
)

_NUMBER_RE = re.compile(r"-?\$?\d[\d,]*(?:\.\d+)?")


def normalize_number(raw: str) -> str | None:
    """Canonicalize a matched number: strip $/commas, drop a trailing period,
    and render integral values without a decimal point ('72.0' -> '72')."""
    cleaned = raw.replace("$", "").replace(",", "").rstrip(".")
    try:
        value = float(cleaned)
    except ValueError:
        return None
    if value == int(value):
        return str(int(value))
    return str(value)


def parse_numeric_answer(text: str) -> str | None:
    """Extract the final numeric answer from a response (or a gold answer).

    If a ``####`` marker is present, only the text after the *last* marker
    counts — a marker followed by no number is an unparseable answer, not a
    license to grab an earlier calculation. Without a marker, the last number
    anywhere in the text is taken (handles '$1,000.', '72 eggs', '...is 14.').
    """
    if "####" in text:
        match = _NUMBER_RE.search(text.rsplit("####", 1)[1])
        return normalize_number(match.group()) if match else None
    matches = _NUMBER_RE.findall(text)
    return normalize_number(matches[-1]) if matches else None


class GSM8KTask(EvalTask):
    """GSM8K exact-match eval over a fixed seeded subset of the test split."""

    name = "gsm8k"

    def __init__(self, config: GSM8KConfig) -> None:
        super().__init__(config)

    def _dataset_path(self) -> Path:
        """Resolve the parquet file, downloading (with HF cache) if needed."""
        if self.config.dataset_path is not None:
            return Path(self.config.dataset_path)
        from huggingface_hub import hf_hub_download

        logger.info(
            "fetching %s/%s (cached after first download)",
            self.config.dataset_repo,
            self.config.dataset_file,
        )
        return Path(
            hf_hub_download(
                self.config.dataset_repo, self.config.dataset_file, repo_type="dataset"
            )
        )

    def load_questions(self) -> list[Question]:
        """Load the test split and return the fixed seeded subset.

        Ids are the row index in the canonical split order, so the same
        (dataset revision, num_questions, seed) selects the same questions
        forever; the runner additionally records a content hash of the subset
        in ``meta.json`` so any dataset drift is detectable.
        """
        import pyarrow.parquet as pq

        table = pq.read_table(self._dataset_path())
        questions: list[Question] = []
        rows = zip(
            table.column("question").to_pylist(), table.column("answer").to_pylist(), strict=True
        )
        for i, (question, answer) in enumerate(rows):
            gold = parse_numeric_answer(answer)
            if gold is None:
                raise ValueError(f"row {i}: gold answer not parseable: {answer[-80:]!r}")
            questions.append(Question(id=f"gsm8k-test-{i:04d}", question=question, gold=gold))
        return sample_questions(questions, self.config.num_questions, self.config.seed)

    def prompt_prefix(self) -> list[dict[str, str]]:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for question, answer in FEW_SHOT_EXAMPLES[: self.config.few_shot]:
            messages.append({"role": "user", "content": question})
            messages.append({"role": "assistant", "content": answer})
        return messages

    def build_messages(self, question: Question) -> list[dict[str, str]]:
        return [*self.prompt_prefix(), {"role": "user", "content": question.question}]

    def parse_answer(self, text: str) -> str | None:
        return parse_numeric_answer(text)
