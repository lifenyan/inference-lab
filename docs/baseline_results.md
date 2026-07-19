# Baseline Results — Qwen2.5-7B-Instruct, FP16, default vLLM (M4)

> First GPU milestone: the control every M5/M6 experiment compares against, and the
> confrontation of the [performance ledger](performance_ledger.md)'s predictions with reality.
> Raw data: [`experiments/baseline-fp16/`](../experiments/baseline-fp16/),
> [`…-sharegpt/`](../experiments/baseline-fp16-sharegpt/), [`…-validation/`](../experiments/baseline-fp16-validation/).

## Environment

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 4090, 24 GB (RunPod secure cloud, $0.69/hr) |
| Driver / CUDA | 570.172.08 / 12.9 (cu129 wheels on a 12.8 driver — see note below) |
| vLLM / torch | 0.25.1+cu129 / 2.11.0+cu129, Python 3.12 |
| Server command | `vllm serve Qwen/Qwen2.5-7B-Instruct` — **all defaults** (the control) |
| Relevant defaults | bf16, prefix caching **ON**, chunked prefill ON, `gpu_memory_utilization=0.9`, `max_seq_len=32768` |

Install note: the default PyPI vLLM wheel bundles torch built for CUDA 13.0, which refuses to
start on this driver ("driver too old, found 12080"). The cu129 release wheel runs fine via CUDA
minor-version compatibility — `scripts/setup_pod.sh` encodes this (`VLLM_VARIANT=cu129`).

## Workloads

- **Synthetic sweep** (`baseline-fp16/`): ~512 input / 256 output tokens (`ignore_eos`),
  200-token shared system prefix, 128 seeded prompts, concurrency 1,2,4,8,16,32,64,
  closed-loop, 128 measured + 8 warmup requests per level. Measured server-side input:
  540 tokens/request (content + chat template).
- **ShareGPT** (`baseline-fp16-sharegpt/`): 256 real conversation openers at c=16
  (measured: mean 219 in / 257 out, median input only 60 — real traffic is short-prompt-heavy).
- **Validation** (`baseline-fp16-validation/`): 512/256, **no** shared prefix, c=16, run
  head-to-head against vLLM's own `vllm bench serve` under the matching condition.

## Sweep results

| c | throughput (window avg, tok/s) | steady-state est.¹ | TTFT p50 (ms) | TTFT p99 (ms) | TPOT p50 (ms) | latency p50 (s) | errors |
|---|---|---|---|---|---|---|---|
| 1 | 62.2 | 62 | 56 | 59 | 15.87 | 4.10 | 0 |
| 2 | 122.7 | 123 | 39 | 54 | 16.15 | 4.16 | 0 |
| 4 | 243.1 | 244 | 48 | 71 | 16.25 | 4.19 | 0 |
| 8 | 475.3 | 478 | 74 | 118 | 16.43 | 4.29 | 0 |
| 16 | 814.1 | 914 | 115 | 182 | 16.93 | 4.48 | 0 |
| 32 | 1303.3 | 1586 | 193 | 357 | 19.10 | 5.17 | 0 |
| 64 | 1982.9 | 2795 | 210 | 854 | 21.32 | 5.86 | 0 |

¹ `concurrency × 256 / median latency`. The window average is dragged down at high c by a
closed-loop artifact: 136 requests at c=64 is only ~2.1 "waves", so the measurement window ends
with a long partial-concurrency drain (and 8 warmup requests consume server capacity inside the
window while being excluded from the numerator). At c≤8 (≥17 waves) the two numbers agree; the
steady-state estimate is the honest number at c≥16. **M5/M6 rule: `num_requests ≥ 4×max
concurrency` per level.**

Plots: [throughput](../experiments/baseline-fp16/throughput_vs_concurrency.png) ·
[TTFT](../experiments/baseline-fp16/ttft_vs_concurrency.png) ·
[latency vs throughput](../experiments/baseline-fp16/latency_vs_throughput.png)

**Prefix-caching caveat (affects TTFT rows only):** vLLM now enables automatic prefix caching by
default, and all 7 levels reuse the same 128 prompts — server metrics show a **90% cache hit
rate** across the sweep. Level c=1 ran first (mostly cold: only the 200-token shared prefix was
cached, ~332 fresh tokens per prefill); levels 2–7 prefilled almost nothing. Throughput/TPOT
conclusions are unaffected (decode dominates: 256 output vs ≤540 input tokens), but the TTFT
column at c≥2 measures *cached* prefills. This is fine for a default-settings control — and it
is exactly what a real agent/RAG workload with repeated prefixes experiences — but M5/M6 A/B
runs that touch prefill must either use fresh seeds per run or pin caching explicitly.

## Predicted vs measured (the ledger's day in court)

Single-GPU session (4090 only), so the A10 rows (P1) and cross-GPU ratios (P3, P5) stay open;
P10–P12 are deferred by design.

| # | Prediction (4090, FP16, 512/256) | Predicted | Measured | Verdict |
|---|---|---|---|---|
| P2 | Decode speed, c=1 | 46–56 tok/s (ceiling 66) | **63.0 tok/s** (1/TPOT) | ✅ but **above** the band — 95% of the naive ceiling, 88% of the embed-corrected one (14.1 GB/step → 71.5 tok/s). GDDR6X + vLLM kernels are simply better than the assumed 70–85% bandwidth efficiency. |
| P4′ | TTFT, c=1 | 75–150 ms (512 fresh tokens) | **56 ms** at ~332 fresh tokens (prefix cached) | ✅ cache-adjusted: scaling the band by 332/512 gives 49–98 ms; 56 ms sits there at ~53% MFU, matching the assumed 40–60%. |
| P6′ | Throughput @ c=64 (A10 band ×1.68 bandwidth ratio) | 2,520–3,190 tok/s | window avg **1,983**; steady-state **2,795** | ✅ steady-state in band; the window number is a measurement-protocol artifact (see ¹), not a server property. |
| P7 | Scaling 1→8 | ≥6× | **7.64×** | ✅ near-linear, batching amortizes weight reads as predicted. |
| P8 | No knee ≤64; KV wall ~120 | no preemption | **0 preemptions**, no knee, errors 0/952 | ✅ `vllm:num_preemptions_total = 0`. |
| P9 | TPOT degradation 1→64 | 10–20% | **+34%** (15.87→21.32 ms) | ❌ missed — root-cause below. |

**P9 root cause (+34% vs predicted 10–20%).** The ledger modeled decode slowdown as pure KV
read growth (~+15% at c=64: ~2.3 GB KV per step vs 15.2 GB weights). Two real effects it
ignored: (a) per-step scheduler/sampler overhead that grows with batch size, and (b) at c≥32
each step's fixed costs are amortized over a *changing* mix as chunked prefill interleaves new
requests' prefill work into decode steps — even with 90% cache hits, ~10% of prefills ran fresh
mid-stream. The shape is right (gradual, not cliff-shaped — the "gradual degradation" half of
the prediction held), the magnitude was optimistic. Ledger updated with a ~0.85 batch-efficiency
factor (see addendum in `performance_ledger.md`).

**P2 lesson (measured > predicted).** Being *above* the predicted band is still a miss in the
epistemics: the 70–85% "bandwidth efficiency" assumption was calibrated on older-generation
memcpy folklore. Updated to 85–95% for GDDR6X + current vLLM kernels.

## Harness validation vs `vllm bench serve`

Matching condition: 512 in (no shared prefix) / 256 out, `ignore_eos`, c=16, 128 requests.
Reproduced 4× (ours) and 3× (bench), alternating order — every number below is stable to ±2%
across reps ([`reps/`](../experiments/baseline-fp16-validation/reps/)).

| Metric | harness | vllm bench | Δ | Verdict |
|---|---|---|---|---|
| TPOT p50 | 17.7–18.0 ms | 18.0 ms | **≤2%** | ✅ agrees |
| TTFT p99 | 718–764 ms | 732–733 ms | **≤4%** | ✅ agrees |
| TTFT p50 | 460–509 ms | 410–412 ms | +12–23% | ⚠️ **definitional** |
| Output throughput | 722–727 tok/s | 818–819 tok/s | −12% | ⚠️ **accounting** |

Both ⚠️ rows are understood mechanically, not hand-waved:

- **TTFT**: vLLM's bench (`vllm/benchmarks/lib/endpoint_request_func.py`, openai-chat backend)
  stamps TTFT on the first SSE chunk containing a `choices` entry — *including the role-only,
  empty-content chunk* vLLM opens every stream with. Our harness deliberately waits for the
  first non-empty content token (the M2 design decision, now vindicated on the real server).
  Ours is the stricter, user-visible definition; the gap (~80–95 ms) is the role-chunk lead time.
- **Throughput**: fully decomposed — warmup requests consume server capacity inside our
  measurement window but are excluded from the numerator (~6%, deliberate conservative
  convention), TTFT definition (~2.5%), closed-loop worker turnaround (measured: 9 ms median
  per request, ~2.5%), TPOT (~1%). Bench's `output_throughput` divides all tokens by full wall
  time instead. (Bench's "Peak concurrent requests: 28" at `--max-concurrency 16` is a
  per-second bucketing artifact in its own report — true concurrency was 16.)

**Conclusion: harness validated.** Per-token pace matches within 2%; every remaining difference
is a documented convention that cancels in A/B comparisons (both sides of every experiment use
the same harness and the same convention).

## ShareGPT (realistic traffic)

791.9 tok/s output at c=16 with mean 219 in / 257 out; TTFT p50 47 ms, p99 300 ms; TPOT p50
17.4 ms; 0 errors. Short real-world prompts make prefill cheap: TTFT is ~2.4× lower than the
synthetic 540-token workload at the same concurrency.

## Quality reference — GSM8K

**92.7%** (278/300, 0 errors), seeded 300-question subset, 5-shot, temperature 0
([`experiments/baseline-fp16/eval/`](../experiments/baseline-fp16/eval/)). This is the
reference score every M5/M6 variant is compared against with the flip-list comparator.

## Cost

At $0.69/hr: **$0.24 per 1M output tokens** at c=16, **$0.10** at c=64 (window avg), **$0.069**
at c=64 steady-state. No knee was reached by c=64, so the true peak-throughput cost floor is
lower still — to be found in M6's batching sweep. (Commercial-API comparison lands in M7.)

Session: ~4.0 GPU-hours ≈ **$2.76** including all setup, one failed launch, the full sweep,
validation reps, and the eval (recorded in every run's `meta.json`).

## Methodology lessons carried into M5/M6

1. **Automatic prefix caching is default-on**: identical-seed reruns against a live server
   measure the cache, not prefill. Fresh seeds per run (or explicit cache pinning) for anything
   prefill-sensitive.
2. **Closed-loop wave quantization**: `num_requests ≥ 4× concurrency` or report steady-state.
3. **cu129 wheels on 12.8 drivers** (and: venv on pod-local NVMe, not the network volume —
   the network-volume install cost 55 minutes).
4. TTFT definitions differ across tools; ours = first non-empty content token, always say so.
