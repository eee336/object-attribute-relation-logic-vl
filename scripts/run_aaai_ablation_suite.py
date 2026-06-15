from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


ABLATIONS: dict[str, list[str]] = {
    "full": [],
    "no_relation_graph": ["--no-relation-graph"],
    "no_attribute_state": ["--ablate-attribute-state"],
    "no_group_candidates": ["--no-groups"],
    "no_program_supervision": ["--program-loss-weight", "0.0"],
}


def run(cmd: list[str], dry_run: bool = False) -> int:
    print(" ".join(cmd), flush=True)
    if dry_run:
        return 0
    proc = subprocess.run(cmd, cwd=ROOT, text=True)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)
    return proc.returncode


def main() -> None:
    parser = argparse.ArgumentParser(description="Run learned OARL-VLA ablations for AAAI tables.")
    parser.add_argument("--dataset", type=Path, default=Path("data/oarlvla_grid_sprites.jsonl"))
    parser.add_argument("--eval-dataset", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/aaai_ablations"))
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--only", nargs="*", choices=sorted(ABLATIONS), default=None)
    parser.add_argument("--report", type=Path, default=Path("outputs/aaai_ablation_commands.json"))
    args = parser.parse_args()

    selected = args.only or list(ABLATIONS)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []

    for name in selected:
        checkpoint = args.output_dir / f"oarlvla_{name}.pt"
        train_cmd = [
            sys.executable,
            "scripts/train_vla.py",
            "--dataset",
            str(args.dataset),
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--hidden-dim",
            str(args.hidden_dim),
            "--val-ratio",
            str(args.val_ratio),
            "--seed",
            str(args.seed),
            "--output",
            str(checkpoint),
            *ABLATIONS[name],
        ]
        if args.eval_dataset is not None:
            train_cmd.extend(["--eval-dataset", str(args.eval_dataset)])
        run(train_cmd, dry_run=args.dry_run)

        eval_dataset = args.eval_dataset or args.dataset
        eval_cmd = [
            sys.executable,
            "scripts/eval_vla.py",
            "--dataset",
            str(eval_dataset),
            "--checkpoint",
            str(checkpoint),
            "--batch-size",
            str(args.batch_size),
        ]
        run(eval_cmd, dry_run=args.dry_run)
        records.append({"name": name, "checkpoint": str(checkpoint), "train_cmd": train_cmd, "eval_cmd": eval_cmd})

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote ablation command report: {args.report}")


if __name__ == "__main__":
    main()

