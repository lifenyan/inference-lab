"""CLI for the load-test harness.

Run a sweep (``run`` may be omitted — it is the default subcommand)::

    python -m inference_lab.loadtest \\
        --endpoint http://localhost:8000/v1 --model Qwen/Qwen2.5-7B-Instruct \\
        --workload workloads/chat.json --concurrency 1,4,8,16 \\
        --out experiments/baseline

Re-plot an existing run directory::

    python -m inference_lab.loadtest plot experiments/baseline
"""

import argparse
import asyncio
import sys
from pathlib import Path

from inference_lab.common.config import EndpointConfig
from inference_lab.common.logging import get_logger
from inference_lab.loadtest.models import load_workload
from inference_lab.loadtest.plots import plot_run
from inference_lab.loadtest.runner import SweepConfig, run_sweep
from inference_lab.loadtest.workload import generate_prompts, load_tokenizer

logger = get_logger("loadtest.cli")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m inference_lab.loadtest", description="LLM endpoint load-testing harness"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run a concurrency sweep against an endpoint")
    run.add_argument("--endpoint", required=True, help="Base URL, e.g. http://localhost:8000/v1")
    run.add_argument("--model", required=True, help="Model name as served")
    run.add_argument("--api-key", default="EMPTY")
    run.add_argument("--timeout", type=float, default=120.0, help="Per-request timeout (s)")
    run.add_argument("--workload", required=True, type=Path, help="Workload spec JSON path")
    run.add_argument(
        "--concurrency",
        default="1,4,8,16,32,64",
        help="Comma-separated concurrency levels (default: 1,4,8,16,32,64)",
    )
    run.add_argument(
        "--num-requests", type=int, default=32, help="Measured requests per level (default: 32)"
    )
    run.add_argument(
        "--warmup", type=int, default=4, help="Warmup requests per level, excluded (default: 4)"
    )
    run.add_argument("--out", required=True, type=Path, help="Run directory, e.g. experiments/x")
    run.add_argument("--no-plots", action="store_true", help="Skip rendering PNGs after the sweep")

    plot = sub.add_parser("plot", help="Render plots from an existing run directory")
    plot.add_argument("run_dir", type=Path)
    return parser


def _cmd_run(args: argparse.Namespace) -> int:
    endpoint = EndpointConfig(
        base_url=args.endpoint, model=args.model, api_key=args.api_key, timeout_s=args.timeout
    )
    workload = load_workload(args.workload)
    if workload.tokenizer is None:
        workload = workload.model_copy(update={"tokenizer": endpoint.model})

    prompts = generate_prompts(workload, load_tokenizer(workload.tokenizer))
    logger.info("generated %d prompts (%s workload)", len(prompts), workload.mode)

    levels = [int(x) for x in args.concurrency.split(",") if x.strip()]
    sweep = SweepConfig(
        concurrency_levels=levels,
        num_requests=args.num_requests,
        warmup_requests=args.warmup,
        run_name=args.out.name,
    )
    summaries = asyncio.run(run_sweep(endpoint, workload, prompts, sweep, args.out))

    if not args.no_plots:
        for path in plot_run(args.out):
            logger.info("wrote %s", path)
    logger.info("run complete: %s", args.out)
    return 1 if any(s.error_rate > 0 for s in summaries) else 0


def _cmd_plot(args: argparse.Namespace) -> int:
    for path in plot_run(args.run_dir):
        logger.info("wrote %s", path)
    return 0


def main(argv: list[str] | None = None) -> int:
    """Entry point; bare flags default to the ``run`` subcommand."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0].startswith("-") and argv[0] not in ("-h", "--help"):
        argv.insert(0, "run")
    args = _build_parser().parse_args(argv)
    return _cmd_run(args) if args.command == "run" else _cmd_plot(args)


if __name__ == "__main__":
    sys.exit(main())
