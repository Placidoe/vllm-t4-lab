#!/usr/bin/env bash
# Collect GPU evidence alongside a benchmark without scraping private request data.
# Stop with Ctrl-C, or set DURATION_SECONDS to end automatically.
set -euo pipefail

INTERVAL_SECONDS="${INTERVAL_SECONDS:-1}"
DURATION_SECONDS="${DURATION_SECONDS:-0}"
GPU_METRICS_OUTPUT="${GPU_METRICS_OUTPUT:-/kaggle/working/vllm_t4_results/gpu_metrics.csv}"

mkdir -p "$(dirname "$GPU_METRICS_OUTPUT")"
echo "timestamp,index,name,utilization_gpu_pct,utilization_memory_pct,memory_used_mib,memory_total_mib,power_draw_w,temperature_c" > "$GPU_METRICS_OUTPUT"

started_at="$(date +%s)"
while true; do
  nvidia-smi \
    --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu \
    --format=csv,noheader,nounits >> "$GPU_METRICS_OUTPUT"
  if [[ "$DURATION_SECONDS" != "0" ]] && (( $(date +%s) - started_at >= DURATION_SECONDS )); then
    break
  fi
  sleep "$INTERVAL_SECONDS"
done
