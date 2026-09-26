# Original diagrams

本目录是为 `vllm-t4-lab` 重新绘制的原创 SVG 图。它们只依据仓库中的脚本、实验报告和
Markdown 原理说明制作，和任何外部文档或附件无关。

| 图 | 对应内容 |
| --- | --- |
| `serving-path.svg` | 请求、调度器、KV manager、worker 与 sampler 的协作 |
| `paged-kv.svg` | 逻辑 token page 到物理 KV block 的映射 |
| `continuous-batching.svg` | 连续批处理的动态 batch 变化 |
| `t4-baseline.svg` | T4 单卡实验中的显存和速度观测 |
