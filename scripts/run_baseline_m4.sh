#!/usr/bin/env bash
# M4 baseline measurement session — run ON the pod, inside tmux, after setup_pod.sh:
#
#   bash $REPO_DIR/scripts/run_baseline_m4.sh
#
# Steps (each skipped if its output already exists, so the script is resumable):
#   1. serve Qwen2.5-7B-Instruct FP16 with DEFAULT vLLM settings (the control)
#   2. smoke-test with one streamed completion
#   3. synthetic sweep  -> experiments/baseline-fp16/         (512/256, 200-tok prefix, c=1..64)
#   4. ShareGPT run     -> experiments/baseline-fp16-sharegpt/ (c=16)
#   5. validation run   -> experiments/baseline-fp16-validation/ (no prefix, c=16)
#      + vLLM's own `vllm bench serve` under the matching condition
#   6. GSM8K eval       -> experiments/baseline-fp16/eval/
#   7. merge pod_env.json + session cost into every run's meta.json
set -euo pipefail

WORKSPACE=${WORKSPACE:-/workspace}
REPO_DIR=${REPO_DIR:-$WORKSPACE/inference_lab}
VENV=${VENV:-/root/venv}
MODEL_ID=${MODEL_ID:-Qwen/Qwen2.5-7B-Instruct}
ENDPOINT=${ENDPOINT:-http://localhost:8000/v1}
LOGS=$WORKSPACE/logs
EXP=$REPO_DIR/experiments
export HF_HOME=${HF_HOME:-$WORKSPACE/hf}

mkdir -p "$LOGS"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
cd "$REPO_DIR"

log() { echo "[m4 $(date -u +%H:%M:%S)] $*"; }

# ---- 1. serve (default settings — resist tuning; this run is the control) ----
if curl -sf "$ENDPOINT/models" > /dev/null 2>&1; then
    log "vLLM already serving"
else
    log "starting vllm serve $MODEL_ID (defaults) -> $LOGS/vllm_serve.log"
    nohup vllm serve "$MODEL_ID" > "$LOGS/vllm_serve.log" 2>&1 &
    echo $! > "$LOGS/vllm_serve.pid"
    for i in $(seq 1 120); do
        curl -sf "$ENDPOINT/models" > /dev/null 2>&1 && break
        [ "$i" -eq 120 ] && { echo "ERROR: server not up after 10 min"; tail -30 "$LOGS/vllm_serve.log"; exit 1; }
        sleep 5
    done
    log "server is up"
fi

# ---- 2. smoke test: one streamed completion ----
# (no `| head` here: with pipefail, head's early exit SIGPIPEs curl and kills the script)
log "smoke test (streamed)"
curl -sN --max-time 60 "$ENDPOINT/chat/completions" -H 'Content-Type: application/json' -d "{
    \"model\": \"$MODEL_ID\",
    \"messages\": [{\"role\": \"user\", \"content\": \"Count from 1 to 5.\"}],
    \"max_tokens\": 32, \"stream\": true}" -o "$LOGS/smoke.out"
grep -q '"content"' "$LOGS/smoke.out" || { echo "ERROR: smoke test got no content tokens"; exit 1; }
head -c 400 "$LOGS/smoke.out"; echo
log "smoke test OK"

# ---- 3. synthetic baseline sweep ----
if [ -f "$EXP/baseline-fp16/summary.json" ]; then
    log "baseline-fp16 sweep already done, skipping"
else
    cat > "$LOGS/wl_baseline.json" <<'EOF'
{"mode": "synthetic", "input_tokens": 512, "output_tokens": 256,
 "shared_prefix_tokens": 200, "num_prompts": 128, "seed": 0, "ignore_eos": true}
EOF
    log "baseline sweep c=1,2,4,8,16,32,64 (128 measured + 8 warmup per level)"
    python -m inference_lab.loadtest \
        --endpoint "$ENDPOINT" --model "$MODEL_ID" \
        --workload "$LOGS/wl_baseline.json" \
        --concurrency 1,2,4,8,16,32,64 --num-requests 128 --warmup 8 \
        --out "$EXP/baseline-fp16"
    curl -s localhost:8000/metrics > "$EXP/baseline-fp16/server_metrics.prom" || true
fi

# ---- 4. ShareGPT run (realistic traffic, moderate concurrency) ----
if [ -f "$EXP/baseline-fp16-sharegpt/summary.json" ]; then
    log "sharegpt run already done, skipping"
else
    cat > "$LOGS/wl_sharegpt.json" <<'EOF'
{"mode": "sharegpt", "num_prompts": 256, "seed": 0}
EOF
    log "ShareGPT run at c=16"
    python -m inference_lab.loadtest \
        --endpoint "$ENDPOINT" --model "$MODEL_ID" \
        --workload "$LOGS/wl_sharegpt.json" \
        --concurrency 16 --num-requests 256 --warmup 8 \
        --out "$EXP/baseline-fp16-sharegpt"
fi

# ---- 5. harness validation vs vLLM's own benchmark (matching condition) ----
if [ -f "$EXP/baseline-fp16-validation/summary.json" ]; then
    log "validation harness run already done, skipping"
else
    cat > "$LOGS/wl_validation.json" <<'EOF'
{"mode": "synthetic", "input_tokens": 512, "output_tokens": 256,
 "shared_prefix_tokens": 0, "num_prompts": 128, "seed": 0, "ignore_eos": true}
EOF
    log "validation run: no prefix, c=16 only"
    python -m inference_lab.loadtest \
        --endpoint "$ENDPOINT" --model "$MODEL_ID" \
        --workload "$LOGS/wl_validation.json" \
        --concurrency 16 --num-requests 128 --warmup 8 \
        --out "$EXP/baseline-fp16-validation"
fi

if ls "$EXP/baseline-fp16-validation/vllm_bench/"*.json > /dev/null 2>&1; then
    log "vllm bench already done, skipping"
else
    mkdir -p "$EXP/baseline-fp16-validation/vllm_bench"
    log "vllm bench serve under the matching condition (may need flag fixes per vllm version)"
    vllm bench serve \
        --model "$MODEL_ID" \
        --backend openai-chat --base-url http://localhost:8000 \
        --endpoint /v1/chat/completions \
        --dataset-name random --random-input-len 512 --random-output-len 256 \
        --num-prompts 128 --max-concurrency 16 --ignore-eos --seed 0 \
        --save-result --result-dir "$EXP/baseline-fp16-validation/vllm_bench" \
        2>&1 | tee "$LOGS/vllm_bench.log" || log "WARNING: vllm bench failed — fix flags and rerun"
fi

# ---- 6. GSM8K quality reference ----
if [ -f "$EXP/baseline-fp16/eval/score.json" ]; then
    log "eval already done, skipping"
else
    log "GSM8K eval (300 questions, c=8, temperature 0)"
    python -m inference_lab.evals \
        --endpoint "$ENDPOINT" --model "$MODEL_ID" \
        --task gsm8k --num-questions 300 --concurrency 8 \
        --out "$EXP/baseline-fp16"
fi

# ---- 7. merge environment into every run's meta.json ----
log "merging pod_env.json into run meta.json files"
python - <<PY
import json, pathlib
env = json.loads(pathlib.Path("$WORKSPACE/pod_env.json").read_text())
for meta in pathlib.Path("$EXP").glob("baseline-fp16*/**/meta.json"):
    data = json.loads(meta.read_text())
    data["environment"] = env
    meta.write_text(json.dumps(data, indent=2) + "\n")
    print("updated", meta)
for meta in pathlib.Path("$EXP").glob("baseline-fp16*/meta.json"):
    data = json.loads(meta.read_text())
    data["environment"] = env
    meta.write_text(json.dumps(data, indent=2) + "\n")
    print("updated", meta)
PY

curl -s localhost:8000/metrics > "$LOGS/server_metrics_final.prom" || true
grep -ci "preempt" "$LOGS/vllm_serve.log" > "$LOGS/preemption_grep_count.txt" || true

log "ALL DONE. rsync experiments/ back to the dev machine, then STOP THE POD."
