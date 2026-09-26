#!/usr/bin/env bash
set -euo pipefail

# Kaggle T4 serving entrypoint. Every benchmark must save the matching startup
# log so backend selection and KV-cache capacity remain auditable.
MODEL="${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
TP="${TP:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.70}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
ENFORCE_EAGER="${ENFORCE_EAGER:-1}"

export USE_TF=0
export USE_TORCH=1
export VLLM_ATTENTION_BACKEND="${ATTENTION_BACKEND:-TRITON_ATTN}"

engine_args=()
if [[ "$ENFORCE_EAGER" == "1" ]]; then
  engine_args+=(--enforce-eager)
fi

vllm serve "$MODEL" \
  --dtype half \
  --tensor-parallel-size "$TP" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --max-model-len "$MAX_MODEL_LEN" \
  --host "$HOST" \
  --port "$PORT" \
  "${engine_args[@]}"
