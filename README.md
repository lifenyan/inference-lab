# inference_lab

Self-hosted LLM inference benchmarking and optimization lab. For a chosen open-source model (Qwen/Llama, 7B–8B) served with vLLM on rented cloud GPUs, this project measures — with a self-built load-testing harness — how each serving optimization (**AWQ/GPTQ/FP8 quantization, prefix caching, continuous-batching parameters, KV-cache memory allocation, speculative decoding**) trades speed and $/1M-token cost against answer quality, and turns the results into a data-backed decision map a team running these models could actually use. A thin OpenAI-compatible gateway (routing + per-request cost/latency logging) fronts the tuned deployment.

Planning and learning docs (PROJECT.md, MILESTONES.md, LEARNING.md) are kept in the local-only `ignore/` folder and are not published with this repo.

## Repo layout

```
inference_lab/     Python package — all source code
  loadtest/        async load-test harness: workloads, sweeps, TTFT/TPOT/P99 stats (M2)
  evals/           quality eval runner: seeded GSM8K/MMLU subsets (M3)
  gateway/         OpenAI-compatible routing gateway with cost logging (M8)
  common/          shared config models + structured JSONL logging
docs/              performance ledger, per-experiment writeups
experiments/       one dir per run: configs, environment versions, raw results
report/            the final optimization report (key deliverable)
scripts/           GPU-pod setup/run scripts
tests/             pytest suite (everything testable locally against a mock server)
```

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'
pytest && ruff check .
```

## Load-test harness (M2)

An async closed-loop load tester for any OpenAI-compatible streaming endpoint,
measuring TTFT, TPOT, throughput, and P50/P90/P99 latency across concurrency levels.

```bash
# Workload spec (saved into every run dir for reproducibility):
cat > chat_workload.json <<'EOF'
{"mode": "synthetic", "input_tokens": 512, "output_tokens": 256,
 "shared_prefix_tokens": 200, "num_prompts": 128, "seed": 0}
EOF
# ({"mode": "sharegpt", "num_prompts": 128, "seed": 0} samples real conversations instead)

python -m inference_lab.loadtest \
  --endpoint http://localhost:8000/v1 --model Qwen/Qwen2.5-7B-Instruct \
  --workload chat_workload.json --concurrency 1,4,8,16,32,64 \
  --out experiments/baseline

python -m inference_lab.loadtest plot experiments/baseline   # re-render PNGs
```

Each run directory contains `workload.json`, `meta.json` (endpoint, versions,
timestamps), `requests.jsonl` (raw per-request records — every aggregate is
recomputable), `summary.json` (per-concurrency aggregates), and three plots.
For offline development there is a mock OpenAI-compatible server with
configurable artificial delays: `python -m inference_lab.loadtest.mockserver
--ttft 0.08 --tpot 0.005` (see `experiments/demo-mock-server/` for a sample
run against it). Closed-loop only; open-loop (Poisson-arrival) generation is
future work.

## Milestones

- [x] **M0** — Repo scaffold & tooling
- [x] **M1** — Performance ledger (theoretical TTFT/throughput limits)
- [x] **M2** — Load-test harness
- [ ] **M3** — Quality eval runner
- [ ] **M4** — Baseline deployment & measurement (GPU)
- [ ] **M5** — Experiment: quantization (GPU)
- [ ] **M6** — Experiments: prefix caching & batching/KV params (GPU)
- [ ] **M7** — Optimization report & cost analysis
- [ ] **M8** — Gateway & end-to-end demo
