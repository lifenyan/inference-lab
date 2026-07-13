# inference_lab

Self-hosted LLM inference benchmarking and optimization lab. For a chosen open-source model (Qwen/Llama, 7B–8B) served with vLLM on rented cloud GPUs, this project measures — with a self-built load-testing harness — how each serving optimization (**AWQ/GPTQ/FP8 quantization, prefix caching, continuous-batching parameters, KV-cache memory allocation, speculative decoding**) trades speed and $/1M-token cost against answer quality, and turns the results into a data-backed decision map a team running these models could actually use. A thin OpenAI-compatible gateway (routing + per-request cost/latency logging) fronts the tuned deployment.

Full context: [PROJECT.md](PROJECT.md) (what & why) · [MILESTONES.md](MILESTONES.md) (execution plan) · [LEARNING.md](LEARNING.md) (notes & insights).

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

## Milestones

- [x] **M0** — Repo scaffold & tooling
- [ ] **M1** — Performance ledger (theoretical TTFT/throughput limits)
- [ ] **M2** — Load-test harness
- [ ] **M3** — Quality eval runner
- [ ] **M4** — Baseline deployment & measurement (GPU)
- [ ] **M5** — Experiment: quantization (GPU)
- [ ] **M6** — Experiments: prefix caching & batching/KV params (GPU)
- [ ] **M7** — Optimization report & cost analysis
- [ ] **M8** — Gateway & end-to-end demo
