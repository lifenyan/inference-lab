# inference_lab — One-Page Summary

**What:** Self-hosted Qwen2.5-7B-Instruct on vLLM (RTX 4090 24 GB, rented at $0.69/hr) and
measured — with a self-built async load-test harness and a seeded GSM8K quality gate —
what quantization (AWQ/GPTQ/FP8), automatic prefix caching, and the batching/KV knobs
each buy in speed, dollars, and answer quality. Full analysis:
[optimization_report.md](optimization_report.md).

**Method in one paragraph:** Predictions first — a first-principles performance ledger
(roofline: prefill compute-bound, decode bandwidth-bound; KV arithmetic from the model
config) written before renting a GPU, then confronted with measurement: 8 of the 9 predictions
tested landed, and the miss (TPOT degradation under batch) was root-caused and fed back
into the model. The harness was cross-validated against vLLM's own benchmark (TPOT
within 2%; every residual gap decomposed to a named convention). Every experiment is a
controlled A/B: identical seeded workloads, same GPU class, raw per-request records
committed. Total GPU spend for the entire evidence base: **$4.72**.

## Top findings

1. **4-bit quantization: 2.6× faster decode per user, 1.5× peak throughput, 35% cheaper
   tokens** (TPOT 15.9→6.0 ms; $0.069→$0.045 per 1M output tokens; GSM8K −2.3 pts) — the
   gap between 2.6× and 1.5× exists because batching amortizes exactly the weight reads
   quantization shrinks. It never buys TTFT; under load it *costs* TTFT (210→530 ms p50).
2. **Prefix caching pays back exactly the prefix's share of prefill** — measured cache
   hit rates equal the share (28%/86%/0% across shapes) and TTFT p99 follows
   (−22%/−82%/0%). On RAG-shaped traffic it also multiplies effective KV capacity ~3.9×
   and turns the most expensive workload into the cheapest ($0.173→$0.062 per 1M).
   (The 3.9× is total ÷ *unique* tokens per sequence, 2,008 ÷ ~508 — the shared prefix
   is stored once across all sequences, moving the preemption wall ~145 → ~560
   concurrent; report §6.1.)
3. **The KV preemption wall sits exactly at pool ÷ unique-tokens-per-sequence** (0
   preemptions at 0.88× the computed wall, 37 at 1.10×), and crossing it is strictly bad:
   throughput falls and p99 blows up (57→92 s). Capacity-plan long-context services in
   KV-tokens, not requests/sec. `max-num-seqs` doesn't create throughput — it only picks
   whether excess load waits outside (TTFT suffers) or inside (TPOT suffers) the engine.

**Cost & break-even (prices dated 2026-07-19):** full-utilization self-hosting =
**$0.062 per 1M output tokens** vs $0.21 for the same model behind a hosted API — but the
API wins below **~29% sustained utilization**, since the pod bills wall-clock hours.

## Decision map (miniature)

| Workload | Pick | Because |
|---|---|---|
| Latency-sensitive chat | GPTQ 4-bit | 2.6× streaming speed; TTFT needs caching + admission tuning, not quantization |
| Batch / cost-driven | GPTQ, mns=128, util 0.90 | Cheapest measured tokens ($0.045–0.062/1M); saturates by mns≈128 |
| Quality-critical | FP8 (or FP16) | Same score delta, half the churn of AWQ (11 vs 20 flips); failures are misreadings, not math |
| Prefix-heavy agents | GPTQ + prefix caching | Hit rate = prefix share; −82% TTFT p99, 2.6× throughput, ~3.9× KV capacity |
| Unique long-context RAG | Plan in KV-tokens, cap admission below the wall | 3.4× cost of tuned config; throughput falls with concurrency; the wall is arithmetic |

**Artifacts:** load-test harness + eval runner (fully unit/integration-tested against a
mock server), reproducible experiment scripts, raw data for every number, and the
report. Repo: `inference_lab`.
