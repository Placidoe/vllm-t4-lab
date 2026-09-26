#!/usr/bin/env bash
# Run one online serving point against the public ShareGPT workload.
#
# Start `scripts/serve_t4.sh` in a separate terminal first. The raw dataset is
# downloaded into a mktemp directory and deleted on exit; only vLLM's aggregate
# JSON result is written beneath BENCHMARK_OUTPUT_DIR.
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-1}"
NUM_PROMPTS="${NUM_PROMPTS:-100}"
NUM_WARMUPS="${NUM_WARMUPS:-10}"
SEED="${SEED:-20260927}"
OUTPUT_LEN="${OUTPUT_LEN:-64}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
BENCHMARK_OUTPUT_DIR="${BENCHMARK_OUTPUT_DIR:-/kaggle/working/vllm_t4_results/online_sharegpt}"
DATASET_URL="${DATASET_URL:-https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json}"

work_dir="$(mktemp -d)"
dataset_path="$work_dir/ShareGPT_V3_unfiltered_cleaned_split.json"
trap 'rm -rf "$work_dir"' EXIT
mkdir -p "$BENCHMARK_OUTPUT_DIR"

curl --fail --location --retry 3 --output "$dataset_path" "$DATASET_URL"
if command -v sha256sum >/dev/null 2>&1; then
  dataset_sha256="$(sha256sum "$dataset_path" | awk '{print $1}')"
else
  dataset_sha256="$(shasum -a 256 "$dataset_path" | awk '{print $1}')"
fi

vllm bench serve \
  --backend vllm \
  --host "$HOST" \
  --port "$PORT" \
  --model "$MODEL" \
  --endpoint /v1/completions \
  --dataset-name sharegpt \
  --dataset-path "$dataset_path" \
  --num-prompts "$NUM_PROMPTS" \
  --num-warmups "$NUM_WARMUPS" \
  --max-concurrency "$MAX_CONCURRENCY" \
  --request-rate inf \
  --seed "$SEED" \
  --output-len "$OUTPUT_LEN" \
  --percentile-metrics ttft,tpot,e2el \
  --save-result \
  --result-dir "$BENCHMARK_OUTPUT_DIR" \
  --metadata \
    workload=public_sharegpt \
    dataset_sha256="$dataset_sha256" \
    seed="$SEED" \
    max_concurrency="$MAX_CONCURRENCY" \
    output_len="$OUTPUT_LEN"
