# CLAUDE.md — repo conventions for inference_lab

Read ignore/PROJECT.md (what & why), ignore/MILESTONES.md (execution plan M0–M8), ignore/LEARNING.md (accumulated insights) before starting work. The `ignore/` folder is local-only (gitignored, never pushed) — it holds the user's personal planning docs; any prompt that references these files at the repo root means this folder.

## Layout

- `inference_lab/` — the Python package; **all source code lives here** (loadtest, evals, gateway, common).
- `docs/` — performance ledger, per-experiment writeups. `report/` — the final optimization report.
- `experiments/` — one subdir per run: workload spec + server config + environment versions + raw results (JSONL/CSV). Committed (raw model weights never are).
- `scripts/` — GPU-pod setup/run scripts. `tests/` — pytest suite.
- This repo root is `~/LLM_projects/inference_lab/inference_lab/`; the **outer** folder is an untracked workspace for scratch files (`tem/`, brochures, etc.) — never write project files there.

## Rules

- **Model names and endpoints are never hardcoded** — they flow through `inference_lab.common.config` objects (file/CLI/env).
- **Every experiment run records**: workload spec, server flags, versions (vLLM/CUDA/driver/GPU), seeds, and raw per-request records — reproducibility is the product.
- **Comparability is sacred**: A/B runs use identical workload specs, seeds, and GPU class.
- **Local Mac = development only** (no GPU): everything must be testable against the mock server in tests. GPU pods are rented by the user, execution-only; always remind the user to shut pods down and record approximate session cost.
- Machine-readable output is JSONL via `common.logging.log_event`; never parse log text downstream.
- End every milestone: update README checklist, append insights to ignore/LEARNING.md.

## Commands

- Test: `python3 -m pytest` (venv: `.venv`)
- Lint: `python3 -m ruff check .`
- Install dev: `pip install -e '.[dev]'`
