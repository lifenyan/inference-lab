#!/usr/bin/env bash
# M6 caching & batching experiments — run ON the pod (RTX 4090, same class as
# M4/M5), after setup_pod.sh with MODEL_ID=Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4:
#
#   bash $REPO_DIR/scripts/run_m6.sh
#
# Base config (settled by M5): GPTQ-Int4, served with defaults except the flags
# each experiment arm varies. Three experiments:
#
#   A  — prefix caching on vs off x 3 workload shapes x c={16,64}
#        -> experiments/prefix-cache-{on,off}-{chat,rag,none}/c{16,64}/
#   B2 — KV-pressure / preemption probe at max-num-seqs=256:
#        unique ~2k-token contexts driven across the computed KV wall (~142 @
#        util 0.90), plus a shared-prefix contrast cell and a util=0.80 cell
#        -> experiments/kv-pressure/<cell>/
#   B1 — batching sweet-spot grid: max-num-seqs x {64,128,160} on the standard
#        512/256 shape at util 0.90, then gpu-memory-utilization {0.80,0.95}
#        at max-num-seqs=256 -> experiments/batching-grid/<cell>/
#
# Order is A -> B2 -> B1 so budget pressure lands on B1 (per M6 plan: trim B1,
# never B2). Every cell is a separate harness invocation with its OWN seed
# (fresh seeds per run: an already-seen seeded workload measures the prefix
# cache, not prefill) and num_prompts = warmup + num_requests (no round-robin
# replay inside a level). num_requests = 4 x concurrency per the M4 rule.
#
# WHY B2 uses unique contexts, not the shared 1500-token prefix: with prefix
# caching ON (production default, kept here), vLLM shares the prefix blocks
# across concurrent sequences, so a shared-prefix sequence only holds ~500
# unique KV tokens and the preemption wall sits at ~560 concurrent — above
# max-num-seqs, unreachable. The wall arithmetic (~1,996 tok/seq -> ~142) only
# holds when each sequence carries its full context, i.e. unique long contexts
# (realistic RAG: different retrieved docs per request). The shared-prefix
# contrast cell at c=160 measures exactly this gap: same offered load, no
# preemption expected.
#
# Each step skips if its output exists, so the script resumes after any crash.
set -euo pipefail

WORKSPACE=${WORKSPACE:-/workspace}
REPO_DIR=${REPO_DIR:-$WORKSPACE/inference_lab}
VENV=${VENV:-/root/venv}
ENDPOINT=${ENDPOINT:-http://localhost:8000/v1}
LOGS=$WORKSPACE/logs
EXP=$REPO_DIR/experiments
SPECS=$WORKSPACE/m6_specs
MODEL=${MODEL:-Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4}
TOKENIZER=Qwen/Qwen2.5-7B-Instruct   # same tokenizer as all prior milestones
export HF_HOME=${HF_HOME:-$WORKSPACE/hf}

STAGES=${STAGES:-a b2 b1 b1util}

mkdir -p "$LOGS" "$SPECS"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
cd "$REPO_DIR"

log() { echo "[m6 $(date -u +%H:%M:%S)] $*"; }

server_up() { curl -sf "$ENDPOINT/models" > /dev/null 2>&1; }

stop_server() {
    if [ -f "$LOGS/vllm_serve.pid" ]; then
        kill "$(cat "$LOGS/vllm_serve.pid")" 2>/dev/null || true
        rm -f "$LOGS/vllm_serve.pid"
    fi
    # bracket trick: the pattern must not match this script's own command line
    pkill -f "[v]llm serve" 2>/dev/null || true
    for _ in $(seq 1 24); do
        used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
        [ "$used" -lt 2000 ] && { log "GPU drained (${used} MiB used)"; return 0; }
        sleep 5
    done
    log "WARNING: GPU still holds ${used} MiB after 2 min; continuing"
}

CURRENT_CONFIG=""
# start_server <config-name> [extra vllm flags...]
start_server() {
    local config=$1; shift
    if server_up && [ "$CURRENT_CONFIG" = "$config" ]; then
        log "server already up with config $config"
        return 0
    fi
    stop_server
    log "starting vllm serve $MODEL $* -> $LOGS/vllm_serve_$config.log"
    nohup vllm serve "$MODEL" "$@" > "$LOGS/vllm_serve_$config.log" 2>&1 < /dev/null &
    echo $! > "$LOGS/vllm_serve.pid"
    for i in $(seq 1 180); do
        server_up && break
        kill -0 "$(cat "$LOGS/vllm_serve.pid")" 2>/dev/null \
            || { echo "ERROR: server process died"; tail -40 "$LOGS/vllm_serve_$config.log"; exit 1; }
        [ "$i" -eq 180 ] && { echo "ERROR: server not up after 15 min"; tail -40 "$LOGS/vllm_serve_$config.log"; exit 1; }
        sleep 5
    done
    CURRENT_CONFIG=$config
    log "server is up ($config)"
}

# capture_serve_evidence <config-name> <out-dir>  — KV-pool / max-concurrency /
# prefix-caching lines from the serve log: the preemption-regime evidence.
capture_serve_evidence() {
    local config=$1 out=$2
    mkdir -p "$out"
    grep -iE "quantiz|marlin|gptq|model weights|Loading model|KV cache|Maximum concurrency|gpu_memory|prefix_caching|max_num_seqs|preempt" \
        "$LOGS/vllm_serve_$config.log" | head -80 > "$out/serve_log_extract.txt" || true
    nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv > "$out/gpu_snapshot.csv"
}

# assert_prefix_caching <on|off> <config-name> — smoke-check the flag took effect
assert_prefix_caching() {
    local want=$1 config=$2 expect
    case "$want" in
        on)  expect="enable_prefix_caching=True" ;;
        off) expect="enable_prefix_caching=False" ;;
    esac
    if grep -q "$expect" "$LOGS/vllm_serve_$config.log"; then
        log "prefix caching verified: $expect"
    else
        echo "ERROR: expected '$expect' in serve log for $config"; exit 1
    fi
}

smoke_test() {
    # (no `curl | head`: with pipefail, head's early exit SIGPIPEs curl and kills the script)
    log "smoke test (streamed)"
    curl -sN --max-time 60 "$ENDPOINT/chat/completions" -H 'Content-Type: application/json' -d "{
        \"model\": \"$MODEL\",
        \"messages\": [{\"role\": \"user\", \"content\": \"Count from 1 to 5.\"}],
        \"max_tokens\": 32, \"stream\": true}" -o "$LOGS/smoke.out"
    grep -q '"content"' "$LOGS/smoke.out" || { echo "ERROR: smoke test got no content tokens"; exit 1; }
    log "smoke test OK"
}

# gen_spec <path> <input_tokens> <prefix_tokens> <num_prompts> <seed>
gen_spec() {
    cat > "$1" <<EOF
{
  "mode": "synthetic",
  "tokenizer": "$TOKENIZER",
  "input_tokens": $2,
  "output_tokens": 256,
  "shared_prefix_tokens": $3,
  "num_prompts": $4,
  "seed": $5,
  "ignore_eos": true
}
EOF
}

# run_cell <out-dir> <input_tokens> <prefix_tokens> <concurrency> <seed>
# num_requests = 4 x concurrency, num_prompts = num_requests + 8 warmup.
run_cell() {
    local out=$1 input=$2 prefix=$3 c=$4 seed=$5
    local n=$((4 * c))
    if [ -f "$out/summary.json" ]; then
        log "$out already done, skipping"
        return 0
    fi
    rm -rf "$out"   # partial run (no summary.json): redo the whole cell
    mkdir -p "$out"
    local spec="$SPECS/$(echo "$out" | tr '/' '_' | sed "s/.*experiments_//").json"
    gen_spec "$spec" "$input" "$prefix" $((n + 8)) "$seed"
    curl -s localhost:8000/metrics > "$out/metrics_before.prom" || true
    log "cell $out: c=$c n=$n seed=$seed (input=$input prefix=$prefix)"
    python -m inference_lab.loadtest \
        --endpoint "$ENDPOINT" --model "$MODEL" \
        --workload "$spec" \
        --concurrency "$c" --num-requests "$n" --warmup 8 \
        --out "$out"
    curl -s localhost:8000/metrics > "$out/metrics_after.prom" || true
}

# ---------------------------------------------------------------- Experiment A
# Prefix caching on/off x shapes chat(200+512) / rag(1500+240) / none(0+512),
# c in {16, 64}. One serve config per arm; every cell gets a fresh seed, and
# the on/off arms REUSE the same seed per cell (identical workload both sides;
# safe because each server lifetime sees the seed exactly once).
if [[ " $STAGES " == *" a "* ]]; then
    log "=== Experiment A: prefix caching on/off ==="
    for arm in on off; do
        if [ "$arm" = "on" ]; then
            start_server "prefix-$arm"
        else
            start_server "prefix-$arm" --no-enable-prefix-caching
        fi
        assert_prefix_caching "$arm" "prefix-$arm"
        smoke_test
        # shape: name input prefix ; seeds fixed per (shape, c), shared across arms
        for shape_def in "chat 512 200" "rag 240 1500" "none 512 0"; do
            read -r shape input prefix <<< "$shape_def"
            case "$shape" in
                chat) s16=6101; s64=6102 ;;
                rag)  s16=6111; s64=6112 ;;
                none) s16=6121; s64=6122 ;;
            esac
            run_cell "$EXP/prefix-cache-$arm-$shape/c16" "$input" "$prefix" 16 "$s16"
            run_cell "$EXP/prefix-cache-$arm-$shape/c64" "$input" "$prefix" 64 "$s64"
            capture_serve_evidence "prefix-$arm" "$EXP/prefix-cache-$arm-$shape"
        done
    done
fi

# ---------------------------------------------------------------- Experiment B2
# Preemption probe: max-num-seqs=256 so admission cannot mask the KV wall.
# Unique ~2k-token contexts (input 1740, no prefix -> ~2,008 tok/seq with
# template + 256 out). Wall = pool / 2008; straddled by c={96,128,160,192}.
if [[ " $STAGES " == *" b2 "* ]]; then
    log "=== Experiment B2: KV-pressure / preemption probe ==="
    start_server "b2-util0.90" --max-num-seqs 256 --gpu-memory-utilization 0.90
    smoke_test
    capture_serve_evidence "b2-util0.90" "$EXP/kv-pressure"
    pool=$(grep -oE "GPU KV cache size: [0-9,]+" "$LOGS/vllm_serve_b2-util0.90.log" | head -1 | tr -dc '0-9')
    if [ -n "$pool" ]; then
        wall=$((pool / 2008))
        log "KV pool $pool tokens @ util 0.90 -> unique-context wall ~= $wall concurrent"
        echo "{\"kv_pool_tokens\": $pool, \"tokens_per_seq\": 2008, \"wall_concurrency\": $wall}" \
            > "$EXP/kv-pressure/wall_util0.90.json"
    else
        log "WARNING: could not parse KV pool size from serve log"
    fi
    for c in 96 128 160 192; do
        case "$c" in 96) s=6301;; 128) s=6302;; 160) s=6303;; 192) s=6304;; esac
        run_cell "$EXP/kv-pressure/unique-c$c" 1740 0 "$c" "$s"
    done
    # contrast cell: SAME offered load shape/size but shared 1500-token prefix —
    # APC shares those blocks across sequences, so no preemption is expected.
    run_cell "$EXP/kv-pressure/shared-prefix-c160" 240 1500 160 6305

    # smaller pool, same workload: wall ~= pool(0.80)/2008 (~120); c=128 sits
    # below the 0.90 wall but above the 0.80 wall — util alone flips the regime.
    start_server "b2-util0.80" --max-num-seqs 256 --gpu-memory-utilization 0.80
    smoke_test
    capture_serve_evidence "b2-util0.80" "$EXP/kv-pressure/util0.80-c128"
    pool=$(grep -oE "GPU KV cache size: [0-9,]+" "$LOGS/vllm_serve_b2-util0.80.log" | head -1 | tr -dc '0-9')
    [ -n "$pool" ] && log "KV pool $pool tokens @ util 0.80 -> wall ~= $((pool / 2008))" \
        && echo "{\"kv_pool_tokens\": $pool, \"tokens_per_seq\": 2008, \"wall_concurrency\": $((pool / 2008))}" \
            > "$EXP/kv-pressure/wall_util0.80.json"
    run_cell "$EXP/kv-pressure/util0.80-c128" 1740 0 128 6306
fi

# ---------------------------------------------------------------- Experiment B1
# Sweet-spot grid on the standard 512/256 chat shape (no preemption possible:
# wall ~358 > max-num-seqs). max-num-seqs is an admission knob: above the cap,
# queueing raises TTFT/P99 while throughput flattens.
if [[ " $STAGES " == *" b1 "* ]]; then
    log "=== Experiment B1: batching sweet-spot grid (util 0.90) ==="
    seed=6201
    for mns in 32 64 128 256; do
        start_server "b1-mns$mns" --max-num-seqs "$mns" --gpu-memory-utilization 0.90
        smoke_test
        for c in 64 128 160; do
            run_cell "$EXP/batching-grid/mns$mns-util0.90/c$c" 512 200 "$c" "$seed"
            seed=$((seed + 1))
        done
        capture_serve_evidence "b1-mns$mns" "$EXP/batching-grid/mns$mns-util0.90"
    done
fi

# gpu-memory-utilization at fixed max-num-seqs=256, c=160: far from the KV wall
# the pool size should barely matter on this shape — measured, not assumed.
if [[ " $STAGES " == *" b1util "* ]]; then
    log "=== Experiment B1: gpu-memory-utilization sweep (mns=256) ==="
    seed=6221
    for util in 0.80 0.95; do
        start_server "b1-util$util" --max-num-seqs 256 --gpu-memory-utilization "$util"
        smoke_test
        run_cell "$EXP/batching-grid/mns256-util$util/c160" 512 200 160 "$seed"
        seed=$((seed + 1))
        capture_serve_evidence "b1-util$util" "$EXP/batching-grid/mns256-util$util"
    done
fi

stop_server

log "merging pod_env.json into run meta.json files"
python - <<PY
import json, pathlib
env = json.loads(pathlib.Path("$WORKSPACE/pod_env.json").read_text())
n = 0
for pattern in ("prefix-cache-*/**/meta.json", "batching-grid/**/meta.json", "kv-pressure/**/meta.json"):
    for meta in sorted(pathlib.Path("$EXP").glob(pattern)):
        data = json.loads(meta.read_text())
        data["environment"] = env
        meta.write_text(json.dumps(data, indent=2) + "\n")
        n += 1
print(f"updated {n} meta.json files")
PY

log "ALL DONE. rsync experiments/ back to the dev machine, then TERMINATE THE POD."
