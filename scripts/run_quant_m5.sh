#!/usr/bin/env bash
# M5 quantization experiments — run ON the pod (RTX 4090, same class as M4),
# inside tmux, after setup_pod.sh:
#
#   bash $REPO_DIR/scripts/run_quant_m5.sh
#
# For each variant (awq, gptq, fp8), all serve flags identical to the M4 control
# (bare `vllm serve <checkpoint>`, everything else default):
#   1. pre-pull the prequantized checkpoint
#   2. serve; wait healthy; capture quantization/memory evidence from the serve log
#   3. smoke-test with one streamed completion
#   4. the IDENTICAL M4 synthetic sweep (verbatim experiments/baseline-fp16/workload.json,
#      same seeds, c=1..64, 128 measured + 8 warmup)  -> experiments/quant-<variant>/
#   5. GSM8K eval (same 300-question seeded subset)   -> experiments/quant-<variant>/eval/
#   6. tear the server down and wait for VRAM to drain before the next variant
# Then merge pod_env.json into every run's meta.json.
#
# Each step skips if its output exists, so the script resumes after any crash.
set -euo pipefail

WORKSPACE=${WORKSPACE:-/workspace}
REPO_DIR=${REPO_DIR:-$WORKSPACE/inference_lab}
VENV=${VENV:-/root/venv}
ENDPOINT=${ENDPOINT:-http://localhost:8000/v1}
LOGS=$WORKSPACE/logs
EXP=$REPO_DIR/experiments
BASELINE_WORKLOAD=$EXP/baseline-fp16/workload.json
export HF_HOME=${HF_HOME:-$WORKSPACE/hf}

VARIANTS=${VARIANTS:-awq gptq fp8}
model_for() {
    case "$1" in
        awq)  echo "Qwen/Qwen2.5-7B-Instruct-AWQ" ;;
        gptq) echo "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4" ;;
        fp8)  echo "RedHatAI/Qwen2.5-7B-Instruct-FP8-dynamic" ;;
        *)    echo "ERROR: unknown variant $1" >&2; return 1 ;;
    esac
}

mkdir -p "$LOGS"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
cd "$REPO_DIR"
[ -f "$BASELINE_WORKLOAD" ] || { echo "ERROR: $BASELINE_WORKLOAD missing (M4 baseline is the control)"; exit 1; }

log() { echo "[m5 $(date -u +%H:%M:%S)] $*"; }

server_up() { curl -sf "$ENDPOINT/models" > /dev/null 2>&1; }

served_model() { curl -sf "$ENDPOINT/models" 2>/dev/null | python -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])' 2>/dev/null || true; }

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

start_server() {
    local model=$1 variant=$2
    log "starting vllm serve $model (defaults) -> $LOGS/vllm_serve_$variant.log"
    nohup vllm serve "$model" > "$LOGS/vllm_serve_$variant.log" 2>&1 < /dev/null &
    echo $! > "$LOGS/vllm_serve.pid"
    for i in $(seq 1 180); do
        server_up && { log "server is up"; return 0; }
        kill -0 "$(cat "$LOGS/vllm_serve.pid")" 2>/dev/null || { echo "ERROR: server process died"; tail -40 "$LOGS/vllm_serve_$variant.log"; exit 1; }
        [ "$i" -eq 180 ] && { echo "ERROR: server not up after 15 min"; tail -40 "$LOGS/vllm_serve_$variant.log"; exit 1; }
        sleep 5
    done
}

capture_serve_evidence() {
    local variant=$1
    local out=$EXP/quant-$variant
    mkdir -p "$out"
    # what quantization path vLLM actually took (kernel choice, weight/KV memory, concurrency)
    grep -iE "quantiz|marlin|fp8|awq|gptq|compressed|scaled_mm|model weights|Loading model|KV cache|Maximum concurrency|gpu_memory" \
        "$LOGS/vllm_serve_$variant.log" | head -60 > "$out/serve_log_extract.txt" || true
    nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv > "$out/gpu_snapshot.csv"
}

smoke_test() {
    local model=$1
    # (no `curl | head`: with pipefail, head's early exit SIGPIPEs curl and kills the script)
    log "smoke test (streamed)"
    curl -sN --max-time 60 "$ENDPOINT/chat/completions" -H 'Content-Type: application/json' -d "{
        \"model\": \"$model\",
        \"messages\": [{\"role\": \"user\", \"content\": \"Count from 1 to 5.\"}],
        \"max_tokens\": 32, \"stream\": true}" -o "$LOGS/smoke.out"
    grep -q '"content"' "$LOGS/smoke.out" || { echo "ERROR: smoke test got no content tokens"; exit 1; }
    head -c 300 "$LOGS/smoke.out"; echo
    log "smoke test OK"
}

for variant in $VARIANTS; do
    model=$(model_for "$variant")
    out=$EXP/quant-$variant

    if [ -f "$out/summary.json" ] && [ -f "$out/eval/score.json" ]; then
        log "quant-$variant fully done, skipping"
        continue
    fi

    log "=== variant $variant ($model) ==="

    log "pre-pulling $model"
    python - <<PY
from huggingface_hub import snapshot_download
print(snapshot_download("$model"))
PY

    if server_up && [ "$(served_model)" = "$model" ]; then
        log "server already serving $model"
    else
        stop_server
        start_server "$model" "$variant"
    fi
    capture_serve_evidence "$variant"
    smoke_test "$model"

    if [ -f "$out/summary.json" ]; then
        log "quant-$variant sweep already done, skipping"
    else
        log "sweep c=1..64 with the verbatim M4 workload spec"
        python -m inference_lab.loadtest \
            --endpoint "$ENDPOINT" --model "$model" \
            --workload "$BASELINE_WORKLOAD" \
            --concurrency 1,2,4,8,16,32,64 --num-requests 128 --warmup 8 \
            --out "$out"
        curl -s localhost:8000/metrics > "$out/server_metrics.prom" || true
    fi

    if [ -f "$out/eval/score.json" ]; then
        log "quant-$variant eval already done, skipping"
    else
        log "GSM8K eval (300 questions, c=8, temperature 0)"
        python -m inference_lab.evals \
            --endpoint "$ENDPOINT" --model "$model" \
            --task gsm8k --num-questions 300 --concurrency 8 \
            --out "$out"
    fi

    stop_server
done

log "merging pod_env.json into run meta.json files"
python - <<PY
import json, pathlib
env = json.loads(pathlib.Path("$WORKSPACE/pod_env.json").read_text())
for meta in sorted(pathlib.Path("$EXP").glob("quant-*/**/meta.json")) + sorted(pathlib.Path("$EXP").glob("quant-*/meta.json")):
    data = json.loads(meta.read_text())
    data["environment"] = env
    meta.write_text(json.dumps(data, indent=2) + "\n")
    print("updated", meta)
PY

log "ALL DONE. rsync experiments/ back to the dev machine, then TERMINATE THE POD."
