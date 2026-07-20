# End-to-End Demo: Gateway → vLLM Pod + Commercial Fallback (M8)

> The tuned deployment from the [optimization report](../report/optimization_report.md),
> served as a service: the gateway running on the dev machine routes an OpenAI-compatible
> mixed workload between a vLLM pod (the report's recommended configuration) and a real
> commercial API — every routing rule fires live, every request logged with cost.
> Raw artifacts: [`experiments/demo-gateway-e2e/`](../experiments/demo-gateway-e2e/).

## Topology

```
client (scripts/run_m8_demo.py, Mac)
  → gateway :8080 (Mac)                              ← inference_lab/gateway/
      ├─ local:    vLLM on RunPod RTX 4090 over exposed TCP (WAN)
      │            Qwen2.5-7B-Instruct-GPTQ-Int4, --max-num-seqs 128,
      │            --gpu-memory-utilization 0.90, prefix caching ON (default)
      └─ fallback: api.openai.com, gpt-5.4-nano ($0.20/$1.25 per 1M, fetched 2026-07-19)
```

Run 2026-07-19 (UTC), vLLM 0.25.1+cu129 / driver 570.195.03. Serve-config evidence
([`serve-evidence/`](../experiments/demo-gateway-e2e/serve-evidence/serve_log_extract.txt)):
`MarlinLinearKernel` (GPTQ), `enable_prefix_caching=True`, KV pool **276,480 tokens** at
util 0.90. Routing config ([`gateway.json`](../experiments/demo-gateway-e2e/gateway.json)):
prompt-token threshold 1,600, circuit breaker 3 failures / 30 s, health probe every 5 s.

## The workload, and what routed where

| traffic | requests | backend, reason | why |
|---|---|---|---|
| short chat (stream + JSON) | 3 | `local` / `default_local` | under threshold, model served locally |
| shared 1.2k-token system prefix + short questions | 8×2 passes | `local` / `default_local` | ~1,346 estimated tokens < 1,600: shared-prefix traffic is the *cheapest* to serve locally (M6) — the threshold must not evict it |
| unique ~2k-token documents | 2×2 passes | `fallback` / `over_token_threshold` | the KV-wall-risk traffic the threshold exists for |
| request for `gpt-5.4-nano` | 1×3 | `fallback` / `model_not_local` | model not in `served_models` |
| after `pkill vllm` on the pod | 5 | `fallback` / `local_unhealthy` | health probe (5 s) caught the dead backend before any request failed |
| same, with the probe slowed to 300 s | 5 | `fallback` / `local_error_fallback` ×3 → **`circuit_open`** ×2 | per-request failover trips the breaker after 3 consecutive failures |

All six routing reasons appear in one log; **zero client-visible failures during the
outage** — every request returned 200, served by the fallback. The full summary
([`gateway_report.txt`](../experiments/demo-gateway-e2e/gateway_report.txt)):
39 requests — 22 local ($0.000082 total) and 17 fallback ($0.001855 total), each cost
total printed next to the assumption it rests on.

## The prefix-cache story, measured server-side

The report's cheapest-traffic finding (§5) shows up in the serving path exactly as
measured in M6: across the demo the pod's counters recorded a
**92.1% prefix-cache hit rate** (16,416 of 17,830 prompt tokens,
`metrics_before_demo.prom` → `metrics_after_demo.prom`) — the shared-prefix batch
dominates the local traffic, and its ~1.1k-token prefills were nearly free: vLLM's
server-side TTFT averaged **16 ms** over the 22 local requests.

**WAN caveat (do not compare these to the report's numbers).** Clients measured TTFT of
0.20–0.42 s — that is ~200–400 ms of public-internet round trip between the Mac and the
pod, an order of magnitude larger than the on-pod prefill cost it wraps. Demo latencies
are WAN-inclusive service numbers; every benchmark number in the report was measured
on-pod for exactly this reason (LEARNING.md §3). The demo's evidence for cache behavior
is the server counters, not the client stopwatch.

## Found live, fixed, re-run

The first pass's fallback-routed requests all returned **400**: OpenAI's newer models
reject `max_tokens` (requiring `max_completion_tokens`), while vLLM accepts the standard
name. The gateway now adapts the payload when forwarding to the fallback (renames
`max_tokens`, drops vLLM-only `ignore_eos`) — `prepare_fallback_payload()` in
`gateway/app.py`, with tests. The 4 error events stay in the demo log; a translation
layer earning its keep on a provider quirk is the realistic part of the demo.

## Known limitation (stated, not engineered around)

The raw prompt-token threshold misjudges cost: what is expensive locally is *unique*
(non-cached) tokens. M6 measured shared-prefix long prompts as the cheapest traffic
($0.062/1M, 86% cache hit) and unique long contexts as the most expensive ($0.213/1M,
KV-wall risk). This demo's threshold keeps the 1.2k-token *shared-prefix* batch local
(correct — 92% cached) and evicts 2k-token *unique* documents (correct — KV-wall risk),
but only because the workload was designed to sit on the right sides of one number; a
prefix-heavy request over 1,600 tokens would be evicted despite being cheap. The honest
future-work answer: threshold on an estimate of unique tokens. Also visible in the log:
the ~4-chars/token estimator overshot the tokenizer's count by ~19–22% (est 2,168 vs
actual 1,822) — both numbers are logged per request, so the estimator's error is
auditable.

## Reproducing

```bash
# pod: bash scripts/setup_pod.sh (MODEL_ID=Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4), then
#      bash scripts/run_m8_serve.sh   # + expose TCP port 8000 in RunPod
# mac:
export OPENAI_API_KEY=...
python -m inference_lab.gateway --config experiments/demo-gateway-e2e/gateway.json --port 8080
python scripts/run_m8_demo.py --stage main
# stop vLLM on the pod, then:
python scripts/run_m8_demo.py --stage failure
python scripts/gateway_report.py experiments/demo-gateway-e2e/gateway_log.jsonl
```

GPU session: ~40 min ≈ **$0.46** (setup 5 min on a fresh pod — cu129 wheel + NVMe venv
playbook — serve up in 90 s, demo ~15 min). Project GPU total: **~$5.18** of the $50–150
budget.
