from __future__ import annotations

import json
from pathlib import Path
from typing import Any

try:
    import yaml
except ModuleNotFoundError:
    yaml = None

from .dedup import deduplicate_records
from .filters import QualityFilter
from .manifest import read_manifest, write_jsonl, write_manifest
from .pseudo_labeler import PseudoLabeler
from .schemas import ImageAnnotationBundle, TaskAnnotation, WebImageRecord
from .sources import make_source


def load_query_plan(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        text = f.read()
    if yaml is not None:
        return yaml.safe_load(text) or {}
    return _load_simple_query_yaml(text)


def flatten_queries(plan: dict[str, Any]) -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []
    for task_type, task_items in plan.items():
        if not isinstance(task_items, dict):
            continue
        for intent, query_groups in task_items.items():
            if not isinstance(query_groups, dict):
                continue
            for key in ("positive_queries", "negative_queries", "queries"):
                for query in query_groups.get(key, []) or []:
                    rows.append((task_type, intent, query))
    return rows


def build_web_dataset(
    *,
    source_name: str,
    input_dir: str | Path | None,
    queries_path: str | Path,
    max_per_query: int,
    output_dir: str | Path,
    mode: str,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    root_dir = output_dir.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir = root_dir / "annotations"
    annotations_dir.mkdir(parents=True, exist_ok=True)
    plan = load_query_plan(queries_path)
    queries = flatten_queries(plan)
    source = make_source(source_name, input_dir=input_dir)
    raw_records: list[WebImageRecord] = []
    source_errors: list[str] = []
    for _, _, query in queries:
        try:
            results = source.search(query, max_per_query)
            for result in results:
                try:
                    raw_records.append(source.download(result, output_dir))
                except Exception as exc:
                    source_errors.append(f"{query}: {exc}")
        except Exception as exc:
            source_errors.append(f"{query}: {exc}")
    records, duplicates = deduplicate_records(raw_records)
    labeler = PseudoLabeler()
    quality = QualityFilter()
    bundles: list[ImageAnnotationBundle] = []
    tasks: list[dict[str, Any]] = []
    sft_rows: list[dict[str, Any]] = []
    pref_rows: list[dict[str, Any]] = []
    for record in records:
        bundle = labeler.label(record, mode=mode)
        bundle.quality_score = quality.score(record, bundle)
        bundles.append(bundle)
        (annotations_dir / f"{record.image_id}.json").write_text(
            json.dumps(bundle.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        for task in bundle.candidate_tasks:
            task_row = web_task_row(record, task)
            tasks.append(task_row)
            sft_rows.append(task_to_sft(record, task))
            pref_rows.append(task_to_preference(record, task))

    manifest_path = root_dir / "web_manifest.jsonl"
    tasks_path = root_dir / "web_tasks.jsonl"
    sft_path = root_dir / "oarlvla_web_sft.jsonl"
    pref_path = root_dir / "oarlvla_web_preferences.jsonl"
    report_path = Path("outputs") / "web_dataset_report.json"
    write_manifest(manifest_path, records)
    write_jsonl(tasks_path, tasks)
    write_jsonl(sft_path, sft_rows)
    write_jsonl(pref_path, pref_rows)
    report = {
        "source": source_name,
        "mode": mode,
        "queries": len(queries),
        "downloaded_or_imported": len(raw_records),
        "records": len(records),
        "duplicates": len(duplicates),
        "weak_grounding_tasks": len(tasks),
        "sft_samples": len(sft_rows),
        "preference_samples": len(pref_rows),
        "manifest_path": str(manifest_path),
        "tasks_path": str(tasks_path),
        "annotations_dir": str(annotations_dir),
        "sft_path": str(sft_path),
        "preference_path": str(pref_path),
        "errors": source_errors,
        "average_quality_score": sum(b.quality_score for b in bundles) / len(bundles) if bundles else 0.0,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def web_task_row(record: WebImageRecord, task: TaskAnnotation) -> dict[str, Any]:
    return {
        "image_id": record.image_id,
        "image_path": record.local_path,
        "instruction": task.instruction,
        "program": task.program,
        "target_type": task.target_type,
        "target_id": task.target_id,
        "task_type": task.task_type,
        "target_description": task.target_description,
        "label_quality": "weak",
        "requires_manual_verification": task.requires_manual_verification,
        "confidence": task.confidence,
        "source": "web",
        "metadata": {
            "source_name": record.source_name,
            "source_url": record.source_url,
            "license": record.license,
            "author": record.author,
            "query": record.query,
            "sha256": record.sha256,
        },
    }


def task_to_sft(record: WebImageRecord, task: TaskAnnotation) -> dict[str, Any]:
    return {
        "image": record.local_path,
        "messages": [
            {
                "role": "user",
                "content": f"Which object should the robot pick? Instruction: {task.instruction}",
            },
            {
                "role": "assistant",
                "content": json.dumps(
                    {
                        "program": task.program,
                        "target_description": task.target_description,
                        "confidence": task.confidence,
                        "label_quality": "weak",
                    },
                    ensure_ascii=False,
                ),
            },
        ],
        "metadata": {
            "source": "web",
            "license": record.license,
            "query": record.query,
            "task_type": task.task_type,
            "requires_manual_verification": task.requires_manual_verification,
        },
    }


def task_to_preference(record: WebImageRecord, task: TaskAnnotation) -> dict[str, Any]:
    rejected = [
        {
            "program": "filter(category='banana')->filter_state(key='is_blackened', value=True)",
            "target_description": "blackened banana",
        },
        {"program": "filter(category='apple')->select_best", "target_description": "apple"},
    ]
    if "banana" not in task.target_description:
        rejected[0] = {"program": "filter(category='banana')->select_best", "target_description": "banana"}
    return {
        "image": record.local_path,
        "instruction": task.instruction,
        "chosen": {
            "program": task.program,
            "target_description": task.target_description,
        },
        "rejected": rejected,
        "preference_source": "rule_generated",
        "label_quality": "weak",
        "metadata": {"source_url": record.source_url, "license": record.license, "query": record.query},
    }


def summarize_manifest(manifest_path: str | Path, annotations_dir: str | Path | None = None) -> dict[str, Any]:
    records = read_manifest(manifest_path)
    by_source: dict[str, int] = {}
    for record in records:
        by_source[record.source_name] = by_source.get(record.source_name, 0) + 1
    scores = []
    by_task: dict[str, int] = {}
    annotations_dir = Path(annotations_dir) if annotations_dir else Path(manifest_path).parent / "annotations"
    for record in records:
        ann_path = annotations_dir / f"{record.image_id}.json"
        if ann_path.exists():
            ann = json.loads(ann_path.read_text(encoding="utf-8"))
            scores.append(float(ann.get("quality_score", 0.0)))
            for task in ann.get("candidate_tasks", []):
                by_task[task["task_type"]] = by_task.get(task["task_type"], 0) + 1
    valid = sum(1 for score in scores if score >= 0.3)
    return {
        "total_images": len(records),
        "valid_images": valid,
        "rejected_images": max(0, len(records) - valid),
        "average_quality_score": sum(scores) / len(scores) if scores else 0.0,
        "by_source": by_source,
        "by_task_type": by_task,
    }


def export_review_html(manifest_path: str | Path, output_html: str | Path, annotations_dir: str | Path | None = None) -> Path:
    records = read_manifest(manifest_path)
    annotations_dir = Path(annotations_dir) if annotations_dir else Path(manifest_path).parent / "annotations"
    cards = []
    for record in records:
        ann_path = annotations_dir / f"{record.image_id}.json"
        ann = json.loads(ann_path.read_text(encoding="utf-8")) if ann_path.exists() else {}
        tasks = ann.get("candidate_tasks", [])
        task_html = "".join(
            f"<li><b>{task.get('task_type')}</b>: {task.get('instruction')}<br><code>{task.get('program')}</code></li>"
            for task in tasks
        )
        cards.append(
            f"""
            <section class="card">
              <img src="{Path(record.local_path).resolve().as_uri()}" alt="{record.image_id}">
              <div>
                <h2>{record.image_id}</h2>
                <p><b>Source:</b> <a href="{record.source_url}">{record.source_name}</a></p>
                <p><b>License:</b> {record.license or 'unknown'} | <b>Author:</b> {record.author or 'unknown'}</p>
                <p><b>Query:</b> {record.query}</p>
                <p><b>Quality:</b> {ann.get('quality_score', 0.0):.3f}</p>
                <p><b>Manual review:</b> required for weak labels</p>
                <ul>{task_html}</ul>
                <pre>{json.dumps(ann.get('pseudo_labels', []), ensure_ascii=False, indent=2)}</pre>
              </div>
            </section>
            """
        )
    html = f"""
    <!doctype html>
    <html>
    <head>
      <meta charset="utf-8">
      <title>OARL-VLA Web Dataset Review</title>
      <style>
        body {{ font-family: system-ui, sans-serif; margin: 24px; background: #f8f8f6; color: #202124; }}
        .card {{ display: grid; grid-template-columns: 240px 1fr; gap: 18px; padding: 16px; margin-bottom: 16px; border: 1px solid #ddd; background: white; border-radius: 8px; }}
        img {{ max-width: 240px; max-height: 220px; object-fit: contain; border: 1px solid #ccc; }}
        code, pre {{ background: #f2f2f2; padding: 2px 4px; border-radius: 4px; }}
        pre {{ overflow: auto; padding: 12px; }}
      </style>
    </head>
    <body>
      <h1>OARL-VLA Web Dataset Review</h1>
      <p>Weak labels are review candidates, not final evaluation ground truth.</p>
      {''.join(cards) if cards else '<p>No records found.</p>'}
    </body>
    </html>
    """
    output_html = Path(output_html)
    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(html, encoding="utf-8")
    return output_html


def _load_simple_query_yaml(text: str) -> dict[str, Any]:
    data: dict[str, Any] = {}
    current_task: str | None = None
    current_intent: str | None = None
    current_key: str | None = None
    for raw_line in text.splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        line = raw_line.strip()
        if indent == 0 and line.endswith(":"):
            current_task = line[:-1]
            data[current_task] = {}
            current_intent = None
            current_key = None
        elif indent == 2 and line.endswith(":") and current_task:
            current_intent = line[:-1]
            data[current_task][current_intent] = {}
            current_key = None
        elif indent == 4 and line.endswith(":") and current_task and current_intent:
            current_key = line[:-1]
            data[current_task][current_intent][current_key] = []
        elif indent >= 6 and line.startswith("- ") and current_task and current_intent and current_key:
            value = line[2:].strip().strip('"').strip("'")
            data[current_task][current_intent][current_key].append(value)
    return data
