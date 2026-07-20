#!/usr/bin/env bash
# M8 demo serve — run ON the pod (RTX 4090) after setup_pod.sh with
#   MODEL_ID=Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4
#
#   bash $REPO_DIR/scripts/run_m8_serve.sh
#
# Serves the report's recommended chat/batch configuration (report §8):
# GPTQ-Int4, --max-num-seqs 128, --gpu-memory-utilization 0.90, prefix caching
# ON (the default). Never 0.95: M6 measured it OOMing at CUDA-graph capture on
# this 24 GB card (experiments/batching-grid/mns256-util0.95-FAILED/).
#
# --host 0.0.0.0 so the pod's exposed TCP port is reachable from the dev
# machine (the gateway runs there for the e2e demo). Expose the port in
# RunPod's UI and smoke-test from the Mac before starting the demo.
#
# Serve-config evidence is captured into the repo's demo dir so the demo run
# records its environment like every other experiment (rsync back afterwards).
set -euo pipefail

WORKSPACE=${WORKSPACE:-/workspace}
REPO_DIR=${REPO_DIR:-$WORKSPACE/inference_lab}
VENV=${VENV:-/root/venv}
MODEL=${MODEL:-Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4}
PORT=${PORT:-8000}
LOGS=$WORKSPACE/logs
EVIDENCE=$REPO_DIR/experiments/demo-gateway-e2e/serve-evidence
export HF_HOME=${HF_HOME:-$WORKSPACE/hf}

mkdir -p "$LOGS" "$EVIDENCE"
# shellcheck disable=SC1091
source "$VENV/bin/activate"

log() { echo "[m8 $(date -u +%H:%M:%S)] $*"; }

server_up() { curl -sf "http://localhost:$PORT/v1/models" > /dev/null 2>&1; }

if server_up; then
    log "server already up"
else
    # bracket trick: pattern must not match this script's own command line
    pkill -f "[v]llm serve" 2>/dev/null || true
    log "starting vllm serve $MODEL (mns=128, util=0.90, APC default-on)"
    nohup vllm serve "$MODEL" \
        --host 0.0.0.0 --port "$PORT" \
        --max-num-seqs 128 \
        --gpu-memory-utilization 0.90 \
        > "$LOGS/vllm_serve_m8.log" 2>&1 < /dev/null &
    echo $! > "$LOGS/vllm_serve.pid"
    for i in $(seq 1 180); do
        server_up && break
        kill -0 "$(cat "$LOGS/vllm_serve.pid")" 2>/dev/null \
            || { echo "ERROR: server process died"; tail -40 "$LOGS/vllm_serve_m8.log"; exit 1; }
        [ "$i" -eq 180 ] && { echo "ERROR: server not up after 15 min"; tail -40 "$LOGS/vllm_serve_m8.log"; exit 1; }
        sleep 5
    done
    log "server is up"
fi

# (no `curl | head`: with pipefail, head's early exit SIGPIPEs curl and kills the script)
log "smoke test (streamed)"
curl -sN --max-time 60 "http://localhost:$PORT/v1/chat/completions" \
    -H 'Content-Type: application/json' -d "{
    \"model\": \"$MODEL\",
    \"messages\": [{\"role\": \"user\", \"content\": \"Count from 1 to 5.\"}],
    \"max_tokens\": 32, \"stream\": true}" -o "$LOGS/smoke.out"
grep -q '"content"' "$LOGS/smoke.out" || { echo "ERROR: smoke test got no content tokens"; exit 1; }
log "smoke test OK"

log "capturing serve evidence -> $EVIDENCE"
grep -iE "quantiz|marlin|gptq|model weights|Loading model|KV cache|Maximum concurrency|gpu_memory|prefix_caching|max_num_seqs" \
    "$LOGS/vllm_serve_m8.log" | head -80 > "$EVIDENCE/serve_log_extract.txt" || true
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv > "$EVIDENCE/gpu_snapshot.csv"
curl -s "http://localhost:$PORT/metrics" > "$EVIDENCE/metrics_at_start.prom" || true

log "READY. From the Mac: curl http://<pod-ip>:<exposed-port>/v1/models"
log "When the demo is done: rsync experiments/demo-gateway-e2e/ back, then TERMINATE THE POD."
