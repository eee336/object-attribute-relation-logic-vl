from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oarlvla.webdata.manifest import read_jsonl, write_jsonl


def run_command(name: str, cmd: list[str], *, dry_run: bool = False) -> dict[str, Any]:
    print(f"\n[{name}] {' '.join(cmd)}", flush=True)
    record: dict[str, Any] = {"name": name, "command": cmd, "returncode": None}
    if dry_run:
        record["dry_run"] = True
        record["returncode"] = 0
        return record
    proc = subprocess.run(cmd, cwd=ROOT, text=True)
    record["returncode"] = proc.returncode
    if proc.returncode != 0:
        raise SystemExit(f"{name} failed with exit code {proc.returncode}")
    return record


def py_cmd(script: str, *args: str | Path | int | float) -> list[str]:
    return [sys.executable, script, *[str(arg) for arg in args]]


def compute_cmd(script: str, *args: str | Path | int | float) -> list[str]:
    return ["python3", script, *[str(arg) for arg in args]]


def file_sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def split_rows(rows: list[dict[str, Any]], train_ratio: float, val_ratio: float, test_ratio: float) -> dict[str, list[dict[str, Any]]]:
    total = train_ratio + val_ratio + test_ratio
    if total <= 0:
        raise ValueError("At least one split ratio must be positive.")
    train_ratio, val_ratio, test_ratio = train_ratio / total, val_ratio / total, test_ratio / total
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row.get("task_type", "unknown")].append(row)

    splits = {"train": [], "val": [], "test": []}
    for _, bucket in sorted(grouped.items()):
        n = len(bucket)
        train_n = int(round(n * train_ratio))
        val_n = int(round(n * val_ratio))
        if train_n + val_n > n:
            val_n = max(0, n - train_n)
        test_n = n - train_n - val_n
        if n >= 3 and test_ratio > 0 and test_n == 0:
            test_n = 1
            if train_n > val_n and train_n > 1:
                train_n -= 1
            elif val_n > 0:
                val_n -= 1
        splits["train"].extend(bucket[:train_n])
        splits["val"].extend(bucket[train_n : train_n + val_n])
        splits["test"].extend(bucket[train_n + val_n :])
    return splits


def write_splits(source_path: Path, output_dir: Path, name: str, ratios: tuple[float, float, float]) -> dict[str, Any]:
    rows = read_jsonl(source_path)
    if not rows:
        raise ValueError(f"No rows to split in {source_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    splits = split_rows(rows, *ratios)
    split_paths: dict[str, str] = {}
    split_stats: dict[str, Any] = {}
    for split_name, split_rows_ in splits.items():
        path = output_dir / f"{name}_{split_name}.jsonl"
        write_jsonl(path, split_rows_)
        split_paths[split_name] = str(path)
        split_stats[split_name] = dataset_stats(path)
    return {"source": str(source_path), "paths": split_paths, "stats": split_stats}


def dataset_stats(path: Path) -> dict[str, Any]:
    rows = read_jsonl(path)
    task_counts = Counter(row.get("task_type", "unknown") for row in rows)
    source_counts = Counter(row.get("source", "unknown") for row in rows)
    label_counts = Counter(row.get("label_quality", "unknown") for row in rows)
    missing_images = 0
    image_rows = 0
    missing_targets = 0
    target_rows = 0
    for row in rows:
        if row.get("image_path"):
            image_rows += 1
            if not Path(row["image_path"]).exists():
                missing_images += 1
        if row.get("label_quality") == "gold":
            target_rows += 1
            if not row.get("target_id"):
                missing_targets += 1
    return {
        "path": str(path),
        "sha256": file_sha256(path),
        "num_rows": len(rows),
        "task_counts": dict(sorted(task_counts.items())),
        "source_counts": dict(sorted(source_counts.items())),
        "label_quality_counts": dict(sorted(label_counts.items())),
        "image_rows": image_rows,
        "missing_images": missing_images,
        "gold_rows": target_rows,
        "missing_gold_targets": missing_targets,
    }


def build_train_commands(args, manifest: dict[str, Any]) -> dict[str, list[str]]:
    synthetic = manifest["splits"]["synthetic"]["paths"]
    grid = manifest["splits"]["grid"]["paths"]
    web_tasks = manifest.get("web_weak", {}).get("tasks_path")
    commands: dict[str, list[str]] = {
        "stage0_benchmark": compute_cmd(
            "scripts/run_benchmark.py",
            "--num-scenes",
            args.benchmark_scenes,
            "--objects-per-scene",
            args.objects_per_scene,
            "--seed",
            args.seed,
            "--output-dir",
            args.output_dir,
        ),
        "stage1_train_grid": compute_cmd(
            "scripts/train_vla.py",
            "--dataset",
            grid["train"],
            "--eval-dataset",
            grid["val"],
            "--epochs",
            args.stage1_epochs,
            "--batch-size",
            args.batch_size,
            "--hidden-dim",
            args.hidden_dim,
            "--output",
            args.stage1_checkpoint,
        ),
        "stage1_eval_grid_test": compute_cmd(
            "scripts/eval_vla.py",
            "--dataset",
            grid["test"],
            "--checkpoint",
            args.stage1_checkpoint,
            "--batch-size",
            args.batch_size,
        ),
        "stage2_train_synthetic_web": compute_cmd(
            "scripts/train_vla.py",
            "--dataset",
            synthetic["train"],
            "--eval-dataset",
            synthetic["val"],
            "--epochs",
            args.stage2_epochs,
            "--batch-size",
            args.batch_size,
            "--hidden-dim",
            args.hidden_dim,
            "--init-checkpoint",
            args.stage1_checkpoint,
            "--extend-tokenizer",
            "--freeze-modules",
            "object_encoder",
            "graph_encoder",
            "action_head",
            "--output",
            args.stage2_checkpoint,
        ),
        "stage2_eval_synthetic_test": compute_cmd(
            "scripts/eval_vla.py",
            "--dataset",
            synthetic["test"],
            "--checkpoint",
            args.stage2_checkpoint,
            "--batch-size",
            args.batch_size,
        ),
        "aaai_ablation_suite": compute_cmd(
            "scripts/run_aaai_ablation_suite.py",
            "--dataset",
            grid["train"],
            "--eval-dataset",
            grid["val"],
            "--epochs",
            args.ablation_epochs,
            "--batch-size",
            args.batch_size,
            "--hidden-dim",
            args.hidden_dim,
        ),
    }
    if web_tasks:
        commands["stage2_train_synthetic_web"].extend(["--web-weak-dataset", web_tasks, "--web-repeat", args.web_repeat])
    return {key: [str(item) for item in value] for key, value in commands.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare all train-ready OARL-VLA datasets, splits, manifests, and compute-machine commands.")
    parser.add_argument("--bundle-dir", type=Path, default=Path("data/training_bundle"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--synthetic-scenes", type=int, default=2000)
    parser.add_argument("--grid-scenes", type=int, default=2000)
    parser.add_argument("--benchmark-scenes", type=int, default=500)
    parser.add_argument("--objects-per-scene", type=int, default=12)
    parser.add_argument("--grid-size", type=int, default=8)
    parser.add_argument("--cell-size", type=int, default=64)
    parser.add_argument("--split", default="0.8,0.1,0.1", help="train,val,test ratios")
    parser.add_argument("--build-web", action="store_true", help="Build weak web tasks from the configured source.")
    parser.add_argument("--web-source", choices=["local", "wikimedia"], default="local")
    parser.add_argument("--web-input-dir", type=Path, default=Path("tests/fixtures/images"))
    parser.add_argument("--web-queries", type=Path, default=Path("configs/web_queries.yaml"))
    parser.add_argument("--max-per-query", type=int, default=2)
    parser.add_argument("--web-mode", choices=["metadata_only", "heuristic", "model_assisted"], default="metadata_only")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--stage1-epochs", type=int, default=20)
    parser.add_argument("--stage2-epochs", type=int, default=5)
    parser.add_argument("--ablation-epochs", type=int, default=5)
    parser.add_argument("--web-repeat", type=int, default=1)
    parser.add_argument("--stage1-checkpoint", type=Path, default=Path("checkpoints/oarlvla_grid_stage1.pt"))
    parser.add_argument("--stage2-checkpoint", type=Path, default=Path("checkpoints/oarlvla_stage2_web_weak.pt"))
    args = parser.parse_args()

    ratios = tuple(float(part) for part in args.split.split(","))
    if len(ratios) != 3:
        raise ValueError("--split must contain train,val,test ratios")

    args.bundle_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = args.bundle_dir / "raw"
    split_dir = args.bundle_dir / "splits"
    image_dir = args.bundle_dir / "grid_images"
    asset_dir = args.bundle_dir / "grid_assets"
    web_dir = args.bundle_dir / "web_dataset"
    synthetic_full = raw_dir / "oarlvla_synthetic_full.jsonl"
    grid_full = raw_dir / "oarlvla_grid_full.jsonl"

    commands: list[dict[str, Any]] = []
    if args.force or not synthetic_full.exists():
        commands.append(
            run_command(
                "generate_synthetic_full",
                py_cmd(
                    "scripts/generate_dataset.py",
                    "--num-scenes",
                    args.synthetic_scenes,
                    "--objects-per-scene",
                    args.objects_per_scene,
                    "--seed",
                    args.seed,
                    "--output",
                    synthetic_full,
                ),
                dry_run=args.dry_run,
            )
        )
    if args.force or not grid_full.exists():
        commands.append(
            run_command(
                "generate_grid_full",
                py_cmd(
                    "scripts/generate_grid_dataset.py",
                    "--num-scenes",
                    args.grid_scenes,
                    "--grid-size",
                    args.grid_size,
                    "--cell-size",
                    args.cell_size,
                    "--seed",
                    args.seed,
                    "--output",
                    grid_full,
                    "--image-dir",
                    image_dir,
                    "--asset-dir",
                    asset_dir,
                ),
                dry_run=args.dry_run,
            )
        )

    if args.dry_run:
        print("Dry run complete; no split manifest written.")
        return

    manifest: dict[str, Any] = {
        "bundle_dir": str(args.bundle_dir),
        "seed": args.seed,
        "split_ratios": {"train": ratios[0], "val": ratios[1], "test": ratios[2]},
        "raw": {
            "synthetic": dataset_stats(synthetic_full),
            "grid": dataset_stats(grid_full),
        },
        "splits": {
            "synthetic": write_splits(synthetic_full, split_dir, "synthetic", ratios),
            "grid": write_splits(grid_full, split_dir, "grid", ratios),
        },
        "commands_run": commands,
        "external_benchmark_note": {
            "libero": "Use OARL target grounding as a target selector before LIBERO policy execution.",
            "maniskill": "Use OARL target ids/bboxes to condition ManiSkill task policies.",
            "robomimic": "Export observation-action rollouts later through robomimic dataset schema; current bundle prepares grounding/action supervision.",
        },
    }

    if args.build_web:
        commands.append(
            run_command(
                "build_web_weak",
                py_cmd(
                    "scripts/build_web_dataset.py",
                    "--source",
                    args.web_source,
                    "--input-dir",
                    args.web_input_dir,
                    "--queries",
                    args.web_queries,
                    "--max-per-query",
                    args.max_per_query,
                    "--output-dir",
                    web_dir,
                    "--mode",
                    args.web_mode,
                ),
                dry_run=False,
            )
        )
        web_root = web_dir.parent
        web_tasks = web_root / "web_tasks.jsonl"
        manifest["web_weak"] = {
            "tasks_path": str(web_tasks),
            "manifest_path": str(web_root / "web_manifest.jsonl"),
            "sft_path": str(web_root / "oarlvla_web_sft.jsonl"),
            "preference_path": str(web_root / "oarlvla_web_preferences.jsonl"),
            "stats": dataset_stats(web_tasks),
        }

    manifest["training_commands"] = build_train_commands(args, manifest)
    manifest_path = args.bundle_dir / "training_manifest.json"
    commands_path = args.bundle_dir / "compute_training_commands.sh"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    command_lines = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
    for name, cmd in manifest["training_commands"].items():
        command_lines.append(f"echo '[{name}]'")
        command_lines.append(" ".join(cmd))
        command_lines.append("")
    commands_path.write_text("\n".join(command_lines), encoding="utf-8")
    print(f"Wrote training manifest: {manifest_path}")
    print(f"Wrote compute commands: {commands_path}")


if __name__ == "__main__":
    main()
