#!/usr/bin/env bash
# Idempotent GPU-pod setup for inference_lab experiments (M4+).
#
# The repo is rsync'd from the dev machine to $REPO_DIR first (the GitHub repo
# is private; rsync keeps credentials off a disposable pod), then:
#
#   POD_USD_PER_HR=0.69 POD_TIER=secure bash $REPO_DIR/scripts/setup_pod.sh
#
# Everything lives under /workspace (survives pod stop; the container disk
# does not). Re-running skips completed work: venv creation, pip installs and
# the model pull are all no-ops when already done.
set -euo pipefail

WORKSPACE=${WORKSPACE:-/workspace}
REPO_DIR=${REPO_DIR:-$WORKSPACE/inference_lab}
# venv on the pod's LOCAL disk, not the network volume: extracting the ~18 GB of
# torch/CUDA wheels onto MooseFS cost ~55 min in M4 vs ~5 min locally. The venv
# dies with the pod, but the disposable-pod policy re-runs this script anyway.
VENV=${VENV:-/root/venv}
VLLM_VERSION=${VLLM_VERSION:-0.25.1}
# CUDA variant of the vLLM/torch wheels. RunPod hosts commonly run driver 570.x
# (CUDA 12.8): the cu129 build works there via CUDA minor-version compatibility,
# while the default PyPI wheel (cu130, a major-version jump) refuses to start.
# Set VLLM_VARIANT=default to install the plain PyPI wheel instead.
VLLM_VARIANT=${VLLM_VARIANT:-cu129}
MODEL_ID=${MODEL_ID:-Qwen/Qwen2.5-7B-Instruct}
POD_USD_PER_HR=${POD_USD_PER_HR:-unknown}
POD_TIER=${POD_TIER:-unknown}
export HF_HOME=${HF_HOME:-$WORKSPACE/hf}
export TMPDIR=$WORKSPACE/tmp   # wheel downloads are large; keep them off the small root disk
mkdir -p "$TMPDIR"

log() { echo "[setup_pod $(date -u +%H:%M:%S)] $*"; }

[ -d "$REPO_DIR" ] || { echo "ERROR: repo not found at $REPO_DIR (rsync it first)"; exit 1; }

log "venv at $VENV"
[ -d "$VENV" ] || python3 -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install -q --upgrade pip

log "installing vllm==$VLLM_VERSION ($VLLM_VARIANT) + inference_lab (no-op if already installed)"
if [ "$VLLM_VARIANT" = "default" ]; then
    pip install -q "vllm==$VLLM_VERSION"
else
    pip install -q \
        "https://github.com/vllm-project/vllm/releases/download/v${VLLM_VERSION}/vllm-${VLLM_VERSION}+${VLLM_VARIANT}-cp38-abi3-manylinux_2_28_$(uname -m).whl" \
        --extra-index-url "https://download.pytorch.org/whl/${VLLM_VARIANT}"
fi
pip install -q -e "$REPO_DIR"

log "pre-pulling $MODEL_ID into HF_HOME=$HF_HOME"
python - <<PY
from huggingface_hub import snapshot_download
path = snapshot_download("$MODEL_ID")
print(f"model at {path}")
PY

log "recording environment -> $WORKSPACE/pod_env.json"
python - <<PY
import json, subprocess, importlib.metadata as md
gpu_name, driver, vram_mib = subprocess.run(
    ["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
     "--format=csv,noheader,nounits"],
    check=True, capture_output=True, text=True,
).stdout.strip().split(", ")
import torch
env = {
    "gpu": gpu_name,
    "vram_mib": int(vram_mib),
    "driver": driver,
    "cuda": torch.version.cuda,
    "torch": md.version("torch"),
    "vllm": md.version("vllm"),
    "model": "$MODEL_ID",
    "pod_usd_per_hr": "$POD_USD_PER_HR",
    "pod_tier": "$POD_TIER",
}
with open("$WORKSPACE/pod_env.json", "w") as f:
    json.dump(env, f, indent=2)
print(json.dumps(env, indent=2))
PY

log "done. Activate with: source $VENV/bin/activate  (HF_HOME=$HF_HOME)"
