#!/usr/bin/env python3
"""Reproducible vLLM batch-matrix benchmark for Kaggle T4 GPUs.

Run this script in a fresh process for each tensor-parallel configuration:

  python benchmark_t4_batch_matrix.py --tp 1 --output-dir /kaggle/working/vllm_t4_results
  python benchmark_t4_batch_matrix.py --tp 2 --output-dir /kaggle/working/vllm_t4_results

The benchmark intentionally measures offline *batch* throughput.  It exercises
vLLM's scheduler but is not a substitute for a client-arrival, OpenAI-server
benchmark; record it separately from TTFT/P95 serving results.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
BATCH_SIZES = (1, 4, 8, 16)
REPEATS = 2
MAX_TOKENS = 64
VLLM_VERSION = "0.18.0"
PROMPT = (
    "Explain, in exactly two concise sentences, why an LLM inference engine "
    "needs a KV cache and how batching affects throughput."
)


def gpu_snapshot() -> list[dict[str, float | int | str]]:
    import torch

    snapshots = []
    for device in range(torch.cuda.device_count()):
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        snapshots.append(
            {
                "device": device,
                "name": torch.cuda.get_device_name(device),
                "free_gib": round(free_bytes / 2**30, 3),
                "total_gib": round(total_bytes / 2**30, 3),
                "allocated_gib": round(torch.cuda.memory_allocated(device) / 2**30, 3),
                "reserved_gib": round(torch.cuda.memory_reserved(device) / 2**30, 3),
            }
        )
    return snapshots


def ensure_vllm() -> None:
    """Install the pinned engine in a fresh Kaggle image when necessary."""
    try:
        import vllm  # noqa: F401

        if vllm.__version__ == VLLM_VERSION:
            return
    except Exception:
        pass
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--quiet", f"vllm=={VLLM_VERSION}"]
    )


def generated_tokens(outputs) -> int:
    return sum(len(request.outputs[0].token_ids) for request in outputs)


def run(args: argparse.Namespace) -> dict:
    ensure_vllm()
    import torch
    import vllm
    from vllm import LLM, SamplingParams

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable: enable Kaggle T4 x2 before running.")
    if torch.cuda.device_count() < args.tp:
        raise RuntimeError(
            f"TP={args.tp} requires {args.tp} GPUs, but only {torch.cuda.device_count()} are visible."
        )

    started_at = datetime.now(timezone.utc).isoformat()
    sampling = SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS)
    # A bounded context makes the KV-cache budget explicit and keeps the two TP
    # configurations comparable on a 16 GiB T4.
    llm = LLM(
        model=MODEL,
        tensor_parallel_size=args.tp,
        dtype="half",
        max_model_len=2048,
        gpu_memory_utilization=0.80,
        seed=7,
        enforce_eager=args.enforce_eager,
    )
    after_load = gpu_snapshot()

    # Warm-up separates compilation/model-load effects from measured generation.
    llm.generate([PROMPT], sampling)

    rows = []
    for batch_size in BATCH_SIZES:
        samples = []
        for _ in range(REPEATS):
            prompts = [PROMPT] * batch_size
            start = time.perf_counter()
            outputs = llm.generate(prompts, sampling)
            seconds = time.perf_counter() - start
            tokens = generated_tokens(outputs)
            samples.append({"seconds": seconds, "output_tokens": tokens})

        elapsed = [item["seconds"] for item in samples]
        total_tokens = [item["output_tokens"] for item in samples]
        rows.append(
            {
                "batch_size": batch_size,
                "repeats": REPEATS,
                "mean_seconds": round(statistics.mean(elapsed), 4),
                "p50_seconds": round(statistics.median(elapsed), 4),
                "mean_output_tokens": round(statistics.mean(total_tokens), 1),
                "mean_output_tokens_per_second": round(
                    statistics.mean(tokens / seconds for tokens, seconds in zip(total_tokens, elapsed)), 3
                ),
                "request_per_second": round(batch_size / statistics.mean(elapsed), 3),
                "samples": [
                    {"seconds": round(item["seconds"], 4), "output_tokens": item["output_tokens"]}
                    for item in samples
                ],
            }
        )

    return {
        "status": "ok",
        "started_at_utc": started_at,
        "benchmark_type": "offline_fixed_batch_generation",
        "model": MODEL,
        "vllm_version": vllm.__version__,
        "tensor_parallel_size": args.tp,
        "gpu_count_visible": torch.cuda.device_count(),
        "gpu_after_model_load": after_load,
        "max_model_len": 2048,
        "gpu_memory_utilization": 0.80,
        "sampling": {"temperature": 0.0, "max_tokens": MAX_TOKENS},
        "prompt_characters": len(PROMPT),
        "enforce_eager": args.enforce_eager,
        "rows": rows,
        "interpretation": (
            "Use this only to compare fixed-batch decoder throughput across TP settings. "
            "Do not label it a serving P95/TTFT result."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tp", type=int, choices=(1, 2))
    parser.add_argument("--output-dir", type=Path, default=Path("/kaggle/working/vllm_t4_results"))
    parser.add_argument("--enforce-eager", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.tp is None:
        # vLLM owns CUDA workers.  Isolating TP=1 and TP=2 in child processes
        # prevents residual workers/KV allocations from contaminating the second
        # measurement.
        for tp in (1, 2):
            subprocess.run(
                [
                    sys.executable,
                    __file__,
                    "--tp",
                    str(tp),
                    "--output-dir",
                    str(args.output_dir),
                    *( ["--enforce-eager"] if args.enforce_eager else [] ),
                ],
                check=True,
            )
        combined = {
            "benchmark_type": "offline_fixed_batch_generation",
            "children": [
                json.loads((args.output_dir / f"batch_matrix_tp{tp}.json").read_text(encoding="utf-8"))
                for tp in (1, 2)
            ],
        }
        (args.output_dir / "batch_matrix_summary.json").write_text(
            json.dumps(combined, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(combined, ensure_ascii=False, indent=2))
        return

    destination = args.output_dir / f"batch_matrix_tp{args.tp}.json"
    try:
        result = run(args)
    except Exception as exc:  # Keep TP=1 evidence even if the TP=2 trial fails.
        result = {
            "status": "failed",
            "tensor_parallel_size": args.tp,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    # Preserve the error evidence as a Kaggle output instead of discarding the
    # entire version.  The notebook can then report an honest TP=2 failure while
    # retaining the already-finished TP=1 JSON artifact.


if __name__ == "__main__":
    main()
