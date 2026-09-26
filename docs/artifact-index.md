# 文档、图、代码与实验产物索引

此索引覆盖仓库内的自有 Markdown、原创 SVG、可执行脚本和实验记录；不依赖外部协作文档。

| 类别 | 文件 | 内容 |
| --- | --- | --- |
| 学习入口 | [README](../README.md) | T4 环境、已验证基线与学习路线 |
| 深入原理 | [vllm-principles.md](vllm-principles.md) | Prefill/Decode、PagedAttention、连续批处理与指标 |
| 模块设计 | [vllm-system-design.md](vllm-system-design.md) | Scheduler、KV manager、backend、并行、量化与投机解码 |
| 实验计划 | [benchmark-plan.md](benchmark-plan.md) | 并发矩阵、控制变量和记录模板 |
| workload 协议 | [public-workloads.md](public-workloads.md) | 公开数据来源、可复现抽样、数据不入库与证据要求 |
| 已完成离线基线 | [`benchmark_t4_batch_matrix.py`](../scripts/benchmark_t4_batch_matrix.py) | Kaggle T4 × 2 的固定 batch 与 TP=1/2 对照；结果以 JSON 产物为准 |
| 结果分析 | [`analyze_batch_matrix.py`](../scripts/analyze_batch_matrix.py) | 强制配置可比性检查后，生成离线吞吐 Markdown 与自绘 SVG |
| 在线压测 | [`run_public_sharegpt_serve_benchmark.sh`](../scripts/run_public_sharegpt_serve_benchmark.sh) | 公共 ShareGPT 的一次在线并发工作点；原始数据临时下载并删除 |
| 硬件证据 | [`collect_gpu_metrics.sh`](../scripts/collect_gpu_metrics.sh) | 同时间窗的 `nvidia-smi` 指标，不记录请求内容 |
| Kaggle 配置 | [`kernel-metadata.json`](../kernel-metadata.json) | 私有 GPU 内核的可复现入口与运行设置 |
| 实验报告 | [single T4 baseline](../reports/2026-09-25-single-t4-baseline.md) | 真实单卡 T4 基线、排障链路和结果边界 |
| 双卡基线 | [T4 × 2 batch matrix](../reports/2026-09-27-kaggle-t4x2-batch-matrix.md) | TP=1/2 的成功离线吞吐对照与可复核聚合 JSON |
| 实验状态 | [2026-09-27 status](../reports/2026-09-27-experiment-status.md) | 已完成、已取消版本与在线服务矩阵的明确边界 |
| 兼容性报告 | [Kaggle backend compatibility](../reports/2026-09-26-kaggle-backend-compatibility.md) | FlashInfer JIT 链接失败的原始证据、归因与修复边界 |
| 可运行代码 | [`scripts/`](../scripts) | 最小生成和 OpenAI-compatible serving |
| Kaggle Notebook | [`notebooks/`](../notebooks) | 干净的单卡 T4 baseline |
| 论文材料 | [`papers/`](../papers) | 论文索引、阅读笔记模板与自有技术写作目录 |
| 原创图 | [`assets/diagrams/`](../assets/diagrams) | 服务路径、Paged KV、连续 batching、T4 基线 |

## 图与文档对应关系

| 图 | 解释的关系 | 嵌入位置 |
| --- | --- | --- |
| `serving-path.svg` | 请求、调度、KV、worker、attention 与 sampler 的职责 | README、原理文档、系统设计 |
| `paged-kv.svg` | 逻辑 token page 与物理 KV block 的映射 | 原理文档、系统设计 |
| `continuous-batching.svg` | 结束/进入请求如何让 batch 动态更新 | 原理文档、压测计划、系统设计 |
| `t4-baseline.svg` | 单卡 T4 的显存构成、KV 容量与 token/s 观测 | 基线报告、系统设计 |
| `t4-tp-throughput-2026-09-27.svg` | 双 T4 TP=1/2 的固定 batch 吞吐及加速比 | 双卡基线报告、实验状态 |

各数字以对应实验报告的聚合 JSON 为准；图仅用于解释，不是额外实验结果。
