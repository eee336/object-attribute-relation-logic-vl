from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def run_command(name: str, cmd: list[str], *, dry_run: bool = False) -> dict[str, Any]:
    print(f"\n[{name}] {' '.join(cmd)}", flush=True)
    record: dict[str, Any] = {"name": name, "command": cmd, "returncode": None}
    if dry_run:
        record["returncode"] = 0
        record["dry_run"] = True
        return record
    proc = subprocess.run(cmd, cwd=ROOT, text=True)
    record["returncode"] = proc.returncode
    if proc.returncode != 0:
        raise SystemExit(f"{name} failed with exit code {proc.returncode}")
    return record


def py_cmd(script: str, *args: str | Path | int | float) -> list[str]:
    return [sys.executable, script, *[str(arg) for arg in args]]


def ensure_synthetic(args, report: dict[str, Any]) -> None:
    if args.synthetic_dataset.exists() and not args.force:
        report["skipped"].append(f"synthetic exists: {args.synthetic_dataset}")
        return
    report["commands"].append(
        run_command(
            "stage0.generate_synthetic",
            py_cmd(
                "scripts/generate_dataset.py",
                "--num-scenes",
                args.synthetic_scenes,
                "--objects-per-scene",
                args.objects_per_scene,
                "--seed",
                args.seed,
                "--output",
                args.synthetic_dataset,
            ),
            dry_run=args.dry_run,
        )
    )


def run_stage0(args, report: dict[str, Any]) -> None:
    ensure_synthetic(args, report)
    report["commands"].append(
        run_command(
            "stage0.benchmark",
            py_cmd(
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
            dry_run=args.dry_run,
        )
    )
    report["commands"].append(
        run_command(
            "stage0.paper_tables",
            py_cmd(
                "scripts/make_paper_tables.py",
                "--input",
                args.output_dir / "benchmark_results.json",
                "--out-md",
                args.output_dir / "benchmark_paper_tables.md",
            ),
            dry_run=args.dry_run,
        )
    )


def run_stage1(args, report: dict[str, Any]) -> None:
    if args.grid_dataset.exists() and not args.force:
        report["skipped"].append(f"grid dataset exists: {args.grid_dataset}")
    else:
        report["commands"].append(
            run_command(
                "stage1.generate_grid",
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
                    args.grid_dataset,
                    "--image-dir",
                    args.grid_image_dir,
                    "--asset-dir",
                    args.grid_asset_dir,
                ),
                dry_run=args.dry_run,
            )
        )
    if args.skip_training:
        report["skipped"].append("stage1 training skipped")
        return
    report["commands"].append(
        run_command(
            "stage1.train_grid",
            py_cmd(
                "scripts/train_vla.py",
                "--dataset",
                args.grid_dataset,
                "--epochs",
                args.stage1_epochs,
                "--batch-size",
                args.batch_size,
                "--hidden-dim",
                args.hidden_dim,
                "--val-ratio",
                args.val_ratio,
                "--output",
                args.stage1_checkpoint,
            ),
            dry_run=args.dry_run,
        )
    )
    report["commands"].append(
        run_command(
            "stage1.eval_grid",
            py_cmd(
                "scripts/eval_vla.py",
                "--dataset",
                args.grid_dataset,
                "--checkpoint",
                args.stage1_checkpoint,
                "--batch-size",
                args.batch_size,
            ),
            dry_run=args.dry_run,
        )
    )


def run_stage2(args, report: dict[str, Any]) -> None:
    ensure_synthetic(args, report)
    web_tasks_path = args.web_tasks or (args.web_output_dir.parent / "web_tasks.jsonl")
    if args.web_source != "none":
        report["commands"].append(
            run_command(
                "stage2.build_web_weak",
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
                    args.web_output_dir,
                    "--mode",
                    args.web_mode,
                ),
                dry_run=args.dry_run,
            )
        )
        web_tasks_path = args.web_output_dir.parent / "web_tasks.jsonl"
    if args.skip_training:
        report["skipped"].append("stage2 training skipped")
        return

    train_cmd = py_cmd(
        "scripts/train_vla.py",
        "--dataset",
        args.synthetic_dataset,
        "--eval-dataset",
        args.synthetic_dataset,
        "--epochs",
        args.stage2_epochs,
        "--batch-size",
        args.batch_size,
        "--hidden-dim",
        args.hidden_dim,
        "--val-ratio",
        0.0,
        "--vlm-backbone",
        args.vlm_backbone,
        "--output",
        args.stage2_checkpoint,
    )
    if web_tasks_path.exists() or args.dry_run:
        train_cmd.extend(["--web-weak-dataset", str(web_tasks_path), "--web-repeat", str(args.web_repeat)])
    else:
        report["skipped"].append(f"stage2 weak web training rows missing: {web_tasks_path}")
    if args.stage2_init_checkpoint.exists() and args.vlm_backbone == "tiny":
        train_cmd.extend(["--init-checkpoint", str(args.stage2_init_checkpoint), "--extend-tokenizer"])
    report["commands"].append(run_command("stage2.train_web_weak", train_cmd, dry_run=args.dry_run))
    report["commands"].append(
        run_command(
            "stage2.eval_gold",
            py_cmd(
                "scripts/eval_vla.py",
                "--dataset",
                args.synthetic_dataset,
                "--checkpoint",
                args.stage2_checkpoint,
                "--batch-size",
                args.batch_size,
            ),
            dry_run=args.dry_run,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run staged OARL-VLA data/training/evaluation pipeline.")
    parser.add_argument("--stage", choices=["stage0", "stage1", "stage2", "all"], default="all")
    parser.add_argument("--quick", action="store_true", help="Use small counts and one epoch per stage.")
    parser.add_argument("--force", action="store_true", help="Regenerate datasets even if files already exist.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--objects-per-scene", type=int, default=12)
    parser.add_argument("--synthetic-scenes", type=int, default=400)
    parser.add_argument("--benchmark-scenes", type=int, default=100)
    parser.add_argument("--grid-scenes", type=int, default=400)
    parser.add_argument("--grid-size", type=int, default=8)
    parser.add_argument("--cell-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--stage1-epochs", type=int, default=2)
    parser.add_argument("--stage2-epochs", type=int, default=1)
    parser.add_argument("--vlm-backbone", choices=["tiny", "qwen_vl"], default="tiny")
    parser.add_argument("--web-source", choices=["local", "wikimedia", "none"], default="local")
    parser.add_argument("--web-input-dir", type=Path, default=Path("tests/fixtures/images"))
    parser.add_argument("--web-queries", type=Path, default=Path("configs/web_queries.yaml"))
    parser.add_argument("--web-mode", choices=["metadata_only", "heuristic", "model_assisted"], default="metadata_only")
    parser.add_argument("--max-per-query", type=int, default=2)
    parser.add_argument("--web-repeat", type=int, default=1)
    parser.add_argument("--synthetic-dataset", type=Path, default=Path("data/oarlvla_synthetic.jsonl"))
    parser.add_argument("--grid-dataset", type=Path, default=Path("data/oarlvla_grid_sprites.jsonl"))
    parser.add_argument("--grid-image-dir", type=Path, default=Path("data/grid_images"))
    parser.add_argument("--grid-asset-dir", type=Path, default=Path("data/grid_assets"))
    parser.add_argument("--web-output-dir", type=Path, default=Path("data/web_dataset"))
    parser.add_argument("--web-tasks", type=Path, default=None, help="Optional prebuilt web_tasks.jsonl. Build output is used when web-source is active.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--stage1-checkpoint", type=Path, default=Path("checkpoints/oarlvla_grid_stage1.pt"))
    parser.add_argument("--stage2-checkpoint", type=Path, default=Path("checkpoints/oarlvla_stage2_web_weak.pt"))
    parser.add_argument("--stage2-init-checkpoint", type=Path, default=Path("checkpoints/oarlvla_grid_stage1.pt"))
    parser.add_argument("--report", type=Path, default=Path("outputs/stage_pipeline_report.json"))
    args = parser.parse_args()

    if args.quick:
        args.synthetic_scenes = min(args.synthetic_scenes, 40)
        args.benchmark_scenes = min(args.benchmark_scenes, 20)
        args.grid_scenes = min(args.grid_scenes, 40)
        args.stage1_epochs = 1
        args.stage2_epochs = 1
        args.batch_size = min(args.batch_size, 8)

    report: dict[str, Any] = {"stage": args.stage, "commands": [], "skipped": []}
    if args.stage in {"stage0", "all"}:
        run_stage0(args, report)
    if args.stage in {"stage1", "all"}:
        run_stage1(args, report)
    if args.stage in {"stage2", "all"}:
        run_stage2(args, report)

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote stage pipeline report: {args.report}")


if __name__ == "__main__":
    main()
