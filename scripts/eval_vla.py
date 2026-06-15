from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEPS = ROOT / ".deps"
if DEPS.exists():
    sys.path.insert(0, str(DEPS))
sys.path.insert(0, str(ROOT / "src"))

try:
    from oarlvla.baselines import all_baselines
    from oarlvla.models.checkpoints import load_checkpoint
    from oarlvla.models.collate import vla_collate_fn
    from oarlvla.models.datasets import SyntheticVLADataset
    from oarlvla.models.qwen_vl import QwenVLProcessorAdapter
    from oarlvla.models.trainer import TrainConfig, evaluate_model
    from oarlvla.models.vla_model import require_torch
    from oarlvla.reasoning import LogicAwareReasoner
    from oarlvla.scene import scene_from_dict
    from oarlvla.webdata.manifest import read_jsonl
except RuntimeError as exc:
    print(exc, file=sys.stderr)
    sys.exit(2)


torch, _ = require_torch()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    model, tokenizer, feature_metadata, extra = load_checkpoint(args.checkpoint, map_location=args.device)
    model.to(args.device)
    dataset = SyntheticVLADataset(args.dataset, tokenizer=tokenizer)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=vla_collate_fn)
    qwen_processor = (
        QwenVLProcessorAdapter(model.config.qwen_model_name)
        if model.config.vlm_backbone == "qwen_vl"
        else None
    )
    metrics = evaluate_model(
        model,
        dataloader,
        TrainConfig(batch_size=args.batch_size, device=args.device, qwen_processor=qwen_processor),
    )
    print(f"Target Accuracy: {metrics.get('target_accuracy', 0.0):.3f}")
    print(f"Program Accuracy: {metrics.get('program_accuracy', 0.0):.3f}")
    print(f"Action MSE: {metrics.get('action_mse', 0.0):.6f}")
    print("By task type:")
    for task_type, stats in metrics.get("by_task", {}).items():
        print(
            f"  {task_type}: target={stats['target_accuracy']:.3f} "
            f"program={stats['program_accuracy']:.3f} action_mse={stats['action_mse']:.6f} ({stats['n']})"
        )
    print_rule_and_baseline_comparison(args.dataset)


def print_rule_and_baseline_comparison(dataset_path: Path) -> None:
    rows = read_jsonl(dataset_path)
    methods = {"OARL Rule Reasoner": LogicAwareReasoner(), **{baseline.name: baseline for baseline in all_baselines(13)}}
    print("Rule/baseline comparison:")
    for name, method in methods.items():
        correct = 0
        total = 0
        for row in rows:
            scene = scene_from_dict(
                {
                    "id": row["scene_id"],
                    "width": row.get("width", 640),
                    "height": row.get("height", 480),
                    "objects": row["objects"],
                    "groups": row["groups"],
                    "history": row.get("history", []),
                }
            )
            if name == "OARL Rule Reasoner":
                result = method.reason(scene, row["instruction"])
                pred = result.target_id
            else:
                pred = method.predict(scene, row["instruction"]).target_id
            correct += int(pred == row["target_id"])
            total += 1
        print(f"  {name}: target_accuracy={correct / max(total, 1):.3f}")


if __name__ == "__main__":
    main()
