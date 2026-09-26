"""Minimal vLLM inference baseline for a Kaggle NVIDIA T4.

This mirrors the successful single-GPU experiment. It intentionally keeps
``enforce_eager=True`` so CUDA Graph capture does not obscure first-run
debugging. Do not treat its single-request speed as a serving throughput score.
"""

from __future__ import annotations

import os

# Set before importing vLLM/Transformers. This avoids the TensorFlow path that
# conflicted with the Kaggle image used for the original experiment.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")


def main() -> None:
    import torch
    from vllm import LLM, SamplingParams
    from vllm.config import AttentionConfig
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable. Enable a Kaggle GPU accelerator and restart the kernel.")

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Torch: {torch.__version__}; CUDA: {torch.version.cuda}")

    attention = AttentionConfig(backend=AttentionBackendEnum.TRITON_ATTN)
    llm = LLM(
        model="Qwen/Qwen2.5-1.5B-Instruct",
        dtype="half",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.70,
        max_model_len=2048,
        enforce_eager=True,
        attention_config=attention,
    )
    prompts = [
        "用三句话解释 PagedAttention 为什么能提高推理服务吞吐。",
        "Explain the difference between prefill and decode in LLM inference.",
    ]
    params = SamplingParams(temperature=0.2, max_tokens=96)

    for item in llm.generate(prompts, params):
        print(f"\nPROMPT: {item.prompt}")
        print(f"OUTPUT: {item.outputs[0].text.strip()}")


if __name__ == "__main__":
    main()
