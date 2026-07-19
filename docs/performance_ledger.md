# Performance Ledger — Qwen2.5-7B-Instruct on A10 24GB & RTX 4090 24GB

> **Phase 0 deliverable.** Theoretical performance predictions derived from first principles,
> written *before* touching a GPU. Every measurement in M4+ gets judged against this document:
> hitting ~70–85% of a ceiling here means the deployment is healthy; missing a prediction by >30%
> means something is wrong with either the deployment or the prediction — both are findings.
>
> All numbers are sourced (see [Sources](#sources)) or derived with the work shown. GB = 10⁹ bytes.

## The cast

**Model — Qwen2.5-7B-Instruct** (from its [config.json] and [model card]):

| Property | Value |
|---|---|
| Parameters (total / non-embedding) | 7.61B / 6.53B |
| Layers | 28 |
| Hidden size | 3584 |
| Attention heads (query) | 28 (head_dim = 3584/28 = **128**) |
| KV heads (GQA) | **4** → 7:1 grouped-query ratio |
| Native dtype | bfloat16 (2 bytes/param) |
| Max context | 32,768 tokens |
| Vocab | 152,064, embeddings untied (embed_tokens + lm_head each ≈ 0.545B params) |

**GPUs** (from NVIDIA's [A10 datasheet] and [Ada whitepaper]):

| Spec | A10 | RTX 4090 | 4090 ÷ A10 |
|---|---|---|---|
| Memory | 24 GB GDDR6 | 24 GB GDDR6X | 1.0× |
| Memory bandwidth | 600 GB/s | 1008 GB/s | **1.68×** |
| FP16 Tensor Core, dense | 125 TFLOPS | 165.2 TFLOPS¹ | **1.32×** |
| TDP | 150 W | 450 W |

¹ FP16 with FP32 accumulate, which is what inference frameworks use. Marketing materials
sometimes quote 330 TFLOPS — that's either 2:4 sparsity or FP16 accumulate; neither applies here.

Those two ratios are the punchline of the whole GPU comparison: the 4090 buys **1.68× more
bandwidth** and **1.32× more compute** but **zero extra memory**. Which ratio shows up in a
measurement tells you which resource that measurement was limited by.

---

## 1. Memory footprint

### 1a. Weights

Weights are just `parameter_count × bytes_per_parameter`:

| Precision | Bytes/param | Weight memory | Left for KV on 24 GB @ 90% util² |
|---|---|---|---|
| FP16/BF16 | 2 | 7.61B × 2 = **15.2 GB** | ~5.4 GB |
| INT8 / FP8 | 1 | **7.6 GB** | ~13.0 GB |
| 4-bit ideal | 0.5 | 3.8 GB | — |
| 4-bit real (AWQ/GPTQ) | ~0.59 avg | **~5.6 GB** | ~15.0 GB |

² vLLM's `gpu-memory-utilization` defaults to 0.90 → 21.6 GB usable; I also reserve ~1 GB for
activations/CUDA overhead. So KV budget ≈ 21.6 − weights − 1.

The "4-bit real" row deserves its arithmetic, because naive 3.8 GB is off by nearly 50%:
AWQ/GPTQ quantize only the transformer blocks (6.53B params → 3.27 GB) plus per-group scale/zero
metadata (group size 128, ~3% → 3.4 GB), and keep both embedding matrices in FP16
(2 × 0.545B × 2 = 2.2 GB). Total ≈ **5.6 GB** — matching published AWQ checkpoint sizes.

### 1b. KV cache per token — where GQA earns its keep

Each token in each sequence stores a key and a value vector per layer, but only for the **4 KV
heads**, not all 28 query heads:

```
KV bytes/token = 2 (K and V) × layers × kv_heads × head_dim × bytes/elem
              = 2 × 28 × 4 × 128 × 2  =  57,344 B  ≈ 57 KB/token (FP16)
```

Without GQA (28 KV heads) it would be 401 KB/token — **7× worse**. This single architecture
choice is why the concurrency numbers in §4 are as high as they are.

Per request: a 1k-token context holds ~59 MB of KV; 4k → 235 MB; 8k → 470 MB; a full 32k
context → 1.9 GB, an eighth of the GPU, for *one* request.

---

## 2. Prefill vs decode — the mental model everything hangs on

A request has two phases with opposite performance characters:

- **Prefill**: all N prompt tokens are processed in one pass. Each weight matrix is read from
  memory **once** and multiplied against **N token vectors** — big matrix-matrix multiplies that
  keep tensor cores busy. → **compute-bound**. Determines **TTFT**.
- **Decode**: tokens generate one at a time. Every step reads **all** the weights (plus the
  sequence's KV cache) from memory to do one matrix-*vector* multiply per matrix. → **memory-bandwidth-bound**. Determines **TPOT**.

The formal tool is *arithmetic intensity* (FLOPs performed per byte read) vs the GPU's *ridge
point* (FLOPs it can do per byte it can fetch):

```
A10 ridge point = 125 TFLOPS / 600 GB/s ≈ 208 FLOPs/byte   (4090: ≈ 164)
```

- Decode, batch 1: ~2 FLOPs per parameter per token, 2 bytes read per parameter (FP16)
  → intensity ≈ **1 FLOP/byte**. 208× below the ridge → tensor cores idle ~99.5% of a decode
  step; bandwidth is everything.
- Prefill, 512 tokens: the same weight read serves 512 tokens → intensity ≈ **512× higher**,
  above the ridge → compute-bound.

Almost every optimization in this project is an attack on one side of this divide:
quantization shrinks the bytes decode must read; batching raises decode's arithmetic intensity
by sharing each weight read across B sequences; prefix caching skips prefill compute entirely.

---

## 3. Theoretical ceilings

### 3a. Single-request decode speed = bandwidth ÷ bytes read per token

At batch 1, a decode step can't finish faster than the time to stream the weights through the
memory bus (KV read is negligible at short contexts: a 768-token sequence adds only ~44 MB
against 15.2 GB of weights):

| Precision | Bytes/token | A10 (600 GB/s) | RTX 4090 (1008 GB/s) |
|---|---|---|---|
| FP16 | 15.2 GB | **39 tok/s** | **66 tok/s** |
| INT8/FP8 | 7.6 GB | 79 tok/s | 132 tok/s |
| 4-bit (~5.6 GB) | ~5.1 GB³ | ~118 tok/s | ~198 tok/s |

³ Decode reads the quantized blocks + FP16 lm_head (~1.1 GB) but not embed_tokens (a lookup
fetches one row). Same correction applied consistently would make FP16 ≈ 14.1 GB → 42 tok/s;
I keep the uncorrected FP16 number as the headline since the difference (~7%) is within
measurement noise.

These are *ceilings*: real kernels don't achieve 100% of peak bandwidth. Prediction: **70–85%
of these numbers** in M4 (i.e. A10 FP16 ≈ 28–33 tok/s). Note the expected 4090/A10 ratio is
the **bandwidth ratio 1.68×**, not the compute ratio — a directly testable signature of
decode being memory-bound.

### 3b. TTFT for a 512-token prompt = prefill FLOPs ÷ compute

Forward-pass cost ≈ 2 FLOPs per parameter per token (one multiply + one add per weight):

```
Prefill FLOPs = 2 × 7.61e9 × 512 ≈ 7.8 TFLOP
```

| GPU | Ideal (100% MFU) | Realistic (~50% MFU⁴) |
|---|---|---|
| A10 | 7.8 / 125 = **62 ms** | ~125 ms |
| RTX 4090 | 7.8 / 165.2 = **47 ms** | ~95 ms |

⁴ MFU = model FLOPs utilization. 512 tokens is a smallish matmul; 40–60% of peak tensor
throughput is typical. Measured TTFT also includes tokenization, scheduling, and one decode
step, so predict **100–200 ms (A10)** / **75–150 ms (4090)** at concurrency 1. Here the
expected 4090/A10 ratio is the **compute ratio 1.32×** — the mirror-image signature of §3a.

### 3c. Sanity crosscheck

At batch 1 the GPU decodes 39 tok/s while capable of 125 TFLOPS — it is doing
`39 × 15.2 GFLOP ≈ 0.6 TFLOP/s`, i.e. **0.5% of peak compute**. That idle 99.5% is exactly the
headroom continuous batching exists to harvest, and why throughput in §5 can scale ~linearly
for a long time.

---

## 4. Concurrency ceiling — how many requests fit in the KV pool

Max concurrent sequences = KV budget ÷ (57 KB × context length). Using the KV budgets from §1a
(both GPUs have 24 GB, so **this table is identical for both** — the 4090 buys speed, not
capacity):

| Context length | FP16 wts (5.4 GB KV) | INT8/FP8 wts (13 GB) | 4-bit wts (15 GB) |
|---|---|---|---|
| 1k tokens | ~92 | ~221 | ~255 |
| 4k tokens | ~23 | ~55 | ~63 |
| 8k tokens | ~11 | ~27 | ~31 |

The real product of quantization at scale is visible here: 4-bit doesn't just speed up batch-1
decode — it nearly **triples the concurrency ceiling**, because every GB freed from weights
becomes KV pool. (An orthogonal lever: FP8 KV cache halves the 57 KB/token itself — worth an
M6 experiment.)

Above these limits vLLM doesn't crash — PagedAttention allocates KV in small blocks on demand,
and when the pool runs out the scheduler **preempts/queues** sequences ([PagedAttention paper]).
Symptom to watch for in M4/M6: P99 latency spikes and throughput plateaus while preemption
counters climb.

For the M4 baseline workload (512 in + 256 out → ≤768 tokens/req, ~44 MB): FP16 KV pool holds
**~120 concurrent requests** — comfortably above the planned max sweep level of 64, so the
baseline sweep should *not* hit the KV wall.

---

## 5. Predicted curve shapes

**Throughput vs concurrency** should look like: `throughput ≈ min(N × batch-1 rate, ceiling)`
— linear at first, then a knee, then flat (or slightly down under preemption thrash).

Why linear: decode at batch N reads the weights *once* per step for all N sequences, so N
requests cost barely more than one. Each user still gets ~39 tok/s while aggregate throughput
multiplies — batching is nearly free until one of three walls:

1. **Compute wall**: per-step FLOPs grow with N until decode itself goes compute-bound at
   `N* ≈ ridge_point × bytes_per_param ÷ 2` ≈ **208** on A10 for FP16. Quantization moves this
   wall *left* (4-bit: N* ≈ 60) — quantized decode goes compute-bound at much smaller batches,
   so **4-bit helps less at high concurrency than its batch-1 speedup suggests**.
2. **KV-capacity wall**: ~120 concurrent for our workload at FP16 (§4).
3. **KV-bandwidth drag** (before either wall): the KV cache read per step grows with N —
   at N=64, 64 × ~640 avg tokens × 57 KB ≈ 2.3 GB/step vs 15.2 GB of weights → steps ~15%
   slower; per-user TPOT degrades gradually even in the "linear" region.

**Peak throughput estimate (A10, FP16, 512/256 workload)**: near the KV wall (N≈120), step time
≈ (15.2 GB weights + 4.4 GB KV) / 600 GB/s ≈ 33 ms → ideal ≈ 3,700 tok/s; at 60–70% efficiency
**≈ 2,200–2,600 tok/s** output. At the M4 sweep max of N=64: ideal ≈ 2,190 tok/s, predict
**≈ 1,500–1,900 tok/s**. 4090 scales by ~1.68× until its (earlier, ridge ≈ 164) compute wall.

**Latency-vs-throughput** should be L-shaped: flat latency while throughput climbs (the free
region), then latency rising steeply for little extra throughput past the knee. The knee is
the operating point a latency-sensitive service should sit at.

**Under 4-bit weights**: batch-1 decode ~3× faster, knee arrives at *lower* N (compute wall
moved left), KV ceiling ~2.8× higher — so the curve starts higher and steeper but flattens
earlier; peak throughput gain should be well under 3×.

---

## 6. Predictions table

| # | Metric (workload: 512 in / 256 out) | Prediction | Rests on |
|---|---|---|---|
| P1 | A10 FP16 decode, concurrency 1 | 28–33 tok/s (ceiling 39) | 600 GB/s BW, 15.2 GB weights, 70–85% BW efficiency |
| P2 | 4090 FP16 decode, concurrency 1 | 46–56 tok/s (ceiling 66) | same, 1008 GB/s |
| P3 | 4090/A10 decode ratio | ≈ 1.68× (bandwidth ratio) | decode is memory-bound |
| P4 | A10 TTFT @ 512-token prompt, conc. 1 | 100–200 ms (ideal 62) | 7.8 TFLOP prefill, ~40–60% MFU |
| P5 | 4090/A10 TTFT ratio | ≈ 1.32× faster (compute ratio) | prefill is compute-bound |
| P6 | A10 FP16 throughput @ conc. 64 | 1,500–1,900 tok/s | step-time model incl. KV read traffic |
| P7 | Throughput conc. 1→8 scaling | ≥ 6× (near-linear) | batching amortizes weight reads |
| P8 | Knee location, A10 FP16, this workload | none before 64; KV wall ~120 | 5.4 GB KV pool ÷ 44 MB/req |
| P9 | Per-user TPOT degradation 1→64 | ~10–20% slower | KV read grows to ~2.3 GB/step at N=64 |
| P10 | 4-bit batch-1 decode speedup | ~2.5–3× vs FP16 | bytes/token 15.2 → ~5.1 GB |
| P11 | 4-bit peak-throughput speedup | < 2× (much less than P10) | compute wall moves to N* ≈ 60 |
| P12 | Max concurrency @ 4k ctx, FP16 | ~23 before preemption | 57 KB/token KV, 5.4 GB pool |

### Predictions to verify in M4

- [ ] P1/P2: batch-1 decode speed within predicted band; compute % of theoretical ceiling
- [ ] P3/P5: if both GPUs tested — decode ratio tracks bandwidth, TTFT ratio tracks compute
- [ ] P4: TTFT at concurrency 1 in band; if far off, profile where the extra time goes
- [ ] P6: throughput at concurrency 64 in band
- [ ] P7: near-linear scaling region exists (1→8)
- [ ] P8: no preemption events during the ≤64 sweep (check vLLM logs/metrics)
- [ ] P9: TPOT degradation is gradual, not cliff-shaped
- [ ] P12 (stretch): long-context run triggers preemption near predicted concurrency
- [ ] P10/P11: deferred to M5 (quantization experiments)

Any prediction missed by >30% gets a root-cause paragraph in `docs/baseline_results.md`.

---

## Sources

- Qwen2.5-7B-Instruct [model card] and [config.json] — parameter counts, architecture, GQA config. Accessed 2026-07-18.
- NVIDIA [A10 datasheet] — 24 GB GDDR6, 600 GB/s, 125 TFLOPS FP16 Tensor (dense).
- NVIDIA [Ada whitepaper] (RTX 4090) — 24 GB GDDR6X, 1008 GB/s, 165.2 TFLOPS FP16 Tensor w/ FP32 accumulate (dense).
- Kwon et al., [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180), SOSP 2023 — block-based KV allocation, preemption behavior.
- kipply, [Transformer Inference Arithmetic](https://kipp.ly/transformer-inference-arithmetic/) — the 2·params FLOPs/token rule, KV cache formulas, bandwidth-vs-compute framing.
- Williams et al., [Roofline: An Insightful Visual Performance Model](https://dl.acm.org/doi/10.1145/1498765.1498785), CACM 2009 — arithmetic intensity / ridge point.

[config.json]: https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/blob/main/config.json
[model card]: https://huggingface.co/Qwen/Qwen2.5-7B-Instruct
[A10 datasheet]: https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a10/pdf/a10-datasheet.pdf
[Ada whitepaper]: https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf
