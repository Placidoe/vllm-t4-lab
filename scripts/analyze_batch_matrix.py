#!/usr/bin/env python3
"""Turn vLLM offline batch JSON into an auditable Markdown report and SVG.

The input is the ``batch_matrix_summary.json`` emitted by
``benchmark_t4_batch_matrix.py``.  This deliberately refuses to calculate a
TP speedup when the two children did not complete with identical engine
settings.  It therefore cannot accidentally turn a backend/config change into
a performance claim.
"""

from __future__ import annotations

import argparse
import json
from html import escape
from pathlib import Path


COMPARABILITY_FIELDS = (
    "model",
    "vllm_version",
    "attention_backend_requested",
    "max_model_len",
    "gpu_memory_utilization",
    "sampling",
    "enforce_eager",
)


def load_children(path: Path) -> dict[int, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    children = payload.get("children", [])
    result = {
        int(child["tensor_parallel_size"]): child
        for child in children
        if "tensor_parallel_size" in child
    }
    if set(result) != {1, 2}:
        raise ValueError("expected exactly TP=1 and TP=2 children")
    return result


def comparable(children: dict[int, dict]) -> tuple[bool, list[str]]:
    problems: list[str] = []
    for tp, child in children.items():
        if child.get("status") != "ok":
            problems.append(f"TP={tp} status is {child.get('status', 'missing')!r}")
    if problems:
        return False, problems

    baseline = children[1]
    for field in COMPARABILITY_FIELDS:
        if children[2].get(field) != baseline.get(field):
            problems.append(
                f"{field}: TP=1={baseline.get(field)!r}, TP=2={children[2].get(field)!r}"
            )
    return not problems, problems


def rows_by_batch(child: dict) -> dict[int, dict]:
    rows = {int(row["batch_size"]): row for row in child.get("rows", [])}
    if not rows:
        raise ValueError(f"TP={child.get('tensor_parallel_size')} has no rows")
    return rows


def write_svg(series: dict[int, dict[int, dict]], destination: Path) -> None:
    width, height = 900, 440
    left, right, top, bottom = 86, 32, 44, 72
    batches = sorted(set().union(*(rows.keys() for rows in series.values())))
    max_value = max(
        row["mean_output_tokens_per_second"]
        for rows in series.values()
        for row in rows.values()
    )
    max_value = max_value * 1.15 if max_value else 1
    chart_width, chart_height = width - left - right, height - top - bottom

    def x(batch: int) -> float:
        index = batches.index(batch)
        return left + chart_width * index / max(1, len(batches) - 1)

    def y(value: float) -> float:
        return top + chart_height * (1 - value / max_value)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#0d1117"/>',
        '<style>text { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; fill: #c9d1d9; font-size: 14px; } .grid { stroke: #30363d; stroke-width: 1; } .axis { stroke: #8b949e; stroke-width: 1.5; }</style>',
        f'<text x="{left}" y="25" font-size="18">Offline decoder throughput · higher is better</text>',
        f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top + chart_height}"/>',
        f'<line class="axis" x1="{left}" y1="{top + chart_height}" x2="{width - right}" y2="{top + chart_height}"/>',
    ]
    for tick in range(5):
        value = max_value * tick / 4
        y_pos = y(value)
        parts += [
            f'<line class="grid" x1="{left}" y1="{y_pos:.1f}" x2="{width - right}" y2="{y_pos:.1f}"/>',
            f'<text x="10" y="{y_pos + 5:.1f}">{value:.1f}</text>',
        ]
    for batch in batches:
        x_pos = x(batch)
        parts += [
            f'<line class="grid" x1="{x_pos:.1f}" y1="{top}" x2="{x_pos:.1f}" y2="{top + chart_height}"/>',
            f'<text x="{x_pos - 10:.1f}" y="{height - 38}">{batch}</text>',
        ]
    parts.append(f'<text x="{width / 2 - 88:.1f}" y="{height - 12}">batch size (requests)</text>')
    palette = {1: "#58a6ff", 2: "#f2cc60"}
    for tp, rows in sorted(series.items()):
        points = " ".join(
            f"{x(batch):.1f},{y(rows[batch]['mean_output_tokens_per_second']):.1f}"
            for batch in batches
            if batch in rows
        )
        color = palette.get(tp, "#f778ba")
        parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="3"/>')
        for batch in batches:
            if batch not in rows:
                continue
            value = rows[batch]["mean_output_tokens_per_second"]
            parts.append(f'<circle cx="{x(batch):.1f}" cy="{y(value):.1f}" r="5" fill="{color}"/>')
        legend_y = 45 + (tp - 1) * 22
        parts += [
            f'<line x1="{width - 182}" y1="{legend_y}" x2="{width - 156}" y2="{legend_y}" stroke="{color}" stroke-width="3"/>',
            f'<text x="{width - 148}" y="{legend_y + 5}">TP={tp}</text>',
        ]
    parts.append('</svg>')
    destination.write_text("\n".join(parts) + "\n", encoding="utf-8")


def write_report(children: dict[int, dict], is_comparable: bool, problems: list[str], destination: Path) -> None:
    lines = [
        "# Kaggle T4 × 2：离线 batch 吞吐结果",
        "",
        "> 自动生成；只回答固定 batch 的 decoder 吞吐，不可代替在线 serving 的 TTFT、TPOT 或 P95 结论。",
        "",
        "## 可比性门槛",
        "",
    ]
    if is_comparable:
        lines.append("TP=1 与 TP=2 的状态和引擎配置一致，可比较同一 batch size 的输出吞吐。")
    else:
        lines.append("**不可得出 TP speedup。** 原因：")
        lines.extend(f"- {problem}" for problem in problems)
    lines += ["", "## 配置", "", "| 字段 | 值 |", "| --- | --- |"]
    for field in COMPARABILITY_FIELDS:
        lines.append(f"| {field} | `{json.dumps(children[1].get(field), ensure_ascii=False)}` |")
    lines += ["", "## 吞吐矩阵", ""]
    if is_comparable:
        lines += ["| batch | TP=1 tok/s | TP=2 tok/s | TP2 / TP1 |", "| ---: | ---: | ---: | ---:"]
        tp1, tp2 = rows_by_batch(children[1]), rows_by_batch(children[2])
        for batch in sorted(set(tp1) | set(tp2)):
            left = tp1.get(batch, {}).get("mean_output_tokens_per_second")
            right = tp2.get(batch, {}).get("mean_output_tokens_per_second")
            ratio = f"{right / left:.3f}×" if left and right else "—"
            lines.append(f"| {batch} | {left if left is not None else '—'} | {right if right is not None else '—'} | {ratio} |")
    else:
        lines.append("没有生成有效吞吐表；保留失败 JSON 作为排障证据。")
    lines += [
        "",
        "## 解释边界",
        "",
        "- 这张表是同一 prompt 的离线固定 batch 测试；它衡量的是 batch decode 的工作点，而不是有到达过程的服务 SLO。",
        "- 每个工作点只有脚本中定义的 repeats；观察到的差异要用独立复跑和在线 ShareGPT 压测验证。",
        "- 报告旁必须保留原始 JSON、server startup log 与 GPU CSV，才能审计 backend、KV 容量和 GPU 利用率。",
        "",
    ]
    destination.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", type=Path, help="batch_matrix_summary.json")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--chart", type=Path, required=True)
    args = parser.parse_args()

    children = load_children(args.summary)
    is_comparable, problems = comparable(children)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.chart.parent.mkdir(parents=True, exist_ok=True)
    write_report(children, is_comparable, problems, args.report)
    if is_comparable:
        write_svg({tp: rows_by_batch(child) for tp, child in children.items()}, args.chart)
    else:
        args.chart.write_text(
            '<svg xmlns="http://www.w3.org/2000/svg" width="900" height="160"><rect width="100%" height="100%" fill="#0d1117"/><text x="30" y="80" fill="#f85149" font-family="monospace" font-size="20">Results are not comparable; no speedup chart generated.</text></svg>\n',
            encoding="utf-8",
        )
    print(json.dumps({"comparable": is_comparable, "problems": problems}, ensure_ascii=False))


if __name__ == "__main__":
    main()
