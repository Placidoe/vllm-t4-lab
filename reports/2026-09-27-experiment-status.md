# vLLM T4 Lab 实验状态（2026-09-27）

## 已完成：双 T4 离线 batch / Tensor Parallel 基线

![TP=1 与 TP=2 的离线吞吐图](../assets/diagrams/t4-tp-throughput-2026-09-27.svg)

| 项目 | 固定值 |
| --- | --- |
| 平台 | Kaggle interactive notebook，Tesla T4 × 2 |
| 模型 | `Qwen/Qwen2.5-1.5B-Instruct` |
| 引擎 | `vllm==0.18.0`，FP16，typed `TRITON_ATTN` |
| 上下文 / KV 配置 | `max_model_len=2048`，`gpu_memory_utilization=0.80` |
| 工作负载 | 自编固定 prompt、temperature 0、每请求最多 64 output tokens、每点 2 次重复 |

| batch | TP=1 output tok/s | TP=2 output tok/s | TP=2 / TP=1 |
| ---: | ---: | ---: | ---: |
| 1 | 68.084 | 117.291 | 1.723× |
| 4 | 260.912 | 420.756 | 1.613× |
| 8 | 495.084 | 805.488 | 1.627× |
| 16 | 948.685 | **1,283.935** | **1.353×** |

TP=2 在所有四个固定 batch 点均更快，最强绝对工作点为 batch=16 的 1,283.935 output tok/s。
但加速比随 batch 增大而下降，符合小模型上通信与并行协调固定成本逐渐显露的预期。详见
[完整报告](2026-09-27-kaggle-t4x2-batch-matrix.md)与
[聚合 measurements](measurements/2026-09-27-kaggle-t4x2-batch-matrix.json)。

## 未完成：在线服务矩阵

| 实验 | 状态 | 不可替代的指标 |
| --- | --- | --- |
| TP=1，ShareGPT 并发 1/4/16 | 待运行 | TTFT、TPOT、P50/P95、aggregate output tok/s、GPU 指标 |
| TP=2，ShareGPT 并发 1/4/16 | 待运行 | 同上；与 TP=1 保持同一模型、请求集、采样和 KV 配置 |
| 并发 32 饱和点 | 待运行 | 排队、KV pressure、错误/预emptions 与尾延迟折中 |

在线实验使用公开 ShareGPT 的临时下载、固定 hash/seed 与聚合指标；原始对话、提示词、模型回复
和逐请求明细不会归档到 Git。完整流程见 [压测计划](../docs/benchmark-plan.md)。

## 已取消版本

最近一个 Kaggle 保存版本处于已取消状态，输出为 0 B。它不构成失败样本，也没有可解读的性能
数据；不会与上面的已完成离线基线混合或用于任何比较。

## 下一个判定门槛

只有在 TP=1/TP=2 的在线工作点均成功、配置可比、并且同一时间窗的 GPU 指标已保存时，才比较
在线吞吐与尾延迟。若 TP=2 只增加吞吐但使 P95 明显恶化，应把结论写成吞吐/体验权衡，而非单纯
“更快”。
