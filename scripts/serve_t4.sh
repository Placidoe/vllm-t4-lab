#!/usr/bin/env bash
set -euo pipefail

# Kaggle T4 baseline: verify backend selection in startup logs every time.
export USE_TF=0
export USE_TORCH=1
export VLLM_ATTENTION_BACKEND=TRITON_ATTN

vllm serve Qwen/Qwen2.5-1.5B-Instruct \
  --dtype half \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.70 \
  --max-model-len 2048 \
  --enforce-eager \
  --host 0.0.0.0 \
  --port 8000
