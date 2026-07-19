"""CLI for the quality eval runner.

Run an eval (``run`` may be omitted — it is the default subcommand)::

    python -m inference_lab.evals \\
        --endpoint http://localhost:8000/v1 --model Qwen/Qwen2.5-7B-Instruct \\
        --task gsm8k --num-questions 300 --out experiments/baseline

Compare two runs (score delta + flipped questions)::

    python -m inference_lab.evals compare experiments/baseline experiments/quant-awq
"""

import argparse
import asyncio
import sys
from pathlib import Path

from inference_lab.common.config import EndpointConfig
from inference_lab.evals.compare import compare_runs, format_comparison
from inference_lab.evals.models import GSM8KConfig
from inference_lab.evals.runner import EvalRunConfig, run_eval
from inference_lab.evals.tasks import build_task


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m inference_lab.evals", description="LLM endpoint quality eval runner"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run an eval against an endpoint")
    run.add_argument("--endpoint", required=True, help="Base URL, e.g. http://localhost:8000/v1")
    run.add_argument("--model", required=True, help="Model name as served")
    run.add_argument("--api-key", default="EMPTY")
    run.add_argument("--timeout", type=float, default=120.0, help="Per-request timeout (s)")
    run.add_argument("--task", choices=["gsm8k"], default="gsm8k")
    run.add_argument(
        "--num-questions", type=int, default=300, help="Seeded subset size (default: 300)"
    )
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--few-shot", type=int, default=5, help="Few-shot exemplars (default: 5)")
    run.add_argument("--max-tokens", type=int, default=512)
    run.add_argument(
        "--dataset-path", default=None, help="Local dataset file override (skips HF download)"
    )
    run.add_argument(
        "--concurrency", type=int, default=8, help="Concurrent requests (default: 8)"
    )
    run.add_argument("--out", required=True, type=Path, help="Run directory, e.g. experiments/x")

    compare = sub.add_parser("compare", help="Compare two eval runs (delta + flipped questions)")
    compare.add_argument("run_a", type=Path, help="Reference run directory")
    compare.add_argument("run_b", type=Path, help="Candidate run directory")
    return parser


def _cmd_run(args: argparse.Namespace) -> int:
    endpoint = EndpointConfig(
        base_url=args.endpoint, model=args.model, api_key=args.api_key, timeout_s=args.timeout
    )
    task_config = GSM8KConfig(
        num_questions=args.num_questions,
        seed=args.seed,
        few_shot=args.few_shot,
        max_tokens=args.max_tokens,
        dataset_path=args.dataset_path,
    )
    run_cfg = EvalRunConfig(concurrency=args.concurrency, run_name=args.out.name)
    score = asyncio.run(run_eval(endpoint, build_task(task_config), run_cfg, args.out))
    return 1 if score.num_errors > 0 else 0


def _cmd_compare(args: argparse.Namespace) -> int:
    print(format_comparison(compare_runs(args.run_a, args.run_b)))
    return 0


def main(argv: list[str] | None = None) -> int:
    """Entry point; bare flags default to the ``run`` subcommand."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0].startswith("-") and argv[0] not in ("-h", "--help"):
        argv.insert(0, "run")
    args = _build_parser().parse_args(argv)
    return _cmd_run(args) if args.command == "run" else _cmd_compare(args)


if __name__ == "__main__":
    sys.exit(main())
