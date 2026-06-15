from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

def _safe(x: Any) -> str:
    if x is None:
        return "-"
    return f"{x:.3f}"


def load_benchmark_results(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    methods = data.get("methods", {})
    if not methods:
        raise ValueError("No methods found in benchmark json.")
    return data


def render_method_table(methods: dict[str, dict[str, Any]]) -> str:
    lines: list[str] = []
    lines.append("| Method | target_accuracy | wrong_object_rate | task_success_rate | attribute_accuracy | state_accuracy | relation_accuracy | group_accuracy | history_accuracy |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name in sorted(methods):
        stats = methods[name]
        row = "| ".join(
            [
                name,
                _safe(stats.get("target_accuracy")),
                _safe(stats.get("wrong_object_rate")),
                _safe(stats.get("task_success_rate")),
                _safe(stats.get("attribute_accuracy")),
                _safe(stats.get("state_accuracy")),
                _safe(stats.get("relation_accuracy")),
                _safe(stats.get("group_accuracy")),
                _safe(stats.get("history_accuracy")),
            ]
        )
        lines.append("| " + row + " |")
    return "\n".join(lines)


def render_by_task_tables(methods: dict[str, dict[str, Any]]) -> list[str]:
    task_names = sorted(
        {
            task
            for method in methods.values()
            for task in method.get("by_task", {}).keys()
        }
    )
    if not task_names:
        return []

    lines: list[str] = []
    lines.append("### 按任务类型拆分")
    for method_name in sorted(methods):
        lines.append(f"\n#### {method_name}")
        lines.append("")
        lines.append("| Task | n | accuracy |")
        lines.append("|---|---:|---:|")
        by_task = methods[method_name].get("by_task", {})
        for task in task_names:
            stats = by_task.get(task, {"n": 0, "accuracy": 0.0})
            lines.append(f"| {task} | {stats.get('n', 0)} | {_safe(stats.get('accuracy'))} |")
        lines.append("")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Render benchmark results into markdown tables for paper writing.")
    parser.add_argument("--input", type=Path, required=True, help="Path to benchmark_results.json")
    parser.add_argument("--out-md", type=Path, default=None, help="Optional output markdown path")
    parser.add_argument("--stdout", action="store_true", help="Print markdown to stdout as well")
    args = parser.parse_args()

    results = load_benchmark_results(args.input)
    methods = results.get("methods", {})

    lines = ["# Benchmark Tables", ""]
    lines.append("## Overall Metrics")
    lines.append("")
    lines.append(render_method_table(methods))
    lines.append("")
    lines.extend(render_by_task_tables(methods))

    markdown = "\n".join(lines).strip() + "\n"

    if args.out_md is not None:
        args.out_md.write_text(markdown, encoding="utf-8")
        print(f"Wrote table markdown: {args.out_md}")

    if args.stdout:
        print(markdown)

    if args.out_md is None and not args.stdout:
        print(markdown)


if __name__ == "__main__":
    main()
