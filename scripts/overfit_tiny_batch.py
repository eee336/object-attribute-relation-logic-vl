from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEPS = ROOT / ".deps"
if DEPS.exists():
    sys.path.insert(0, str(DEPS))
sys.path.insert(0, str(ROOT / "src"))

try:
    from oarlvla.evaluation import sample_to_jsonl_row
    from oarlvla.instruction import InstructionExample
    from oarlvla.reasoning import LogicAwareReasoner
    from oarlvla.models.checkpoints import save_checkpoint
    from oarlvla.models.collate import vla_collate_fn
    from oarlvla.models.datasets import SyntheticVLADataset
    from oarlvla.models.trainer import TrainConfig, evaluate_model, make_optimizer, model_inputs
    from oarlvla.models.vla_model import OARLVLAConfig, OARLVLAModel, require_torch
    from oarlvla.scene import generate_scene
except RuntimeError as exc:
    print(exc, file=sys.stderr)
    sys.exit(2)


torch, _ = require_torch()


def build_tiny_dataset(path: Path, num_samples: int = 12, seed: int = 123) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    reasoner = LogicAwareReasoner()
    instruction = "Pick the banana that has not turned black."
    with path.open("w", encoding="utf-8") as f:
        for idx in range(num_samples):
            scene = generate_scene(seed + idx, 12, scene_id=f"tiny_{idx:04d}")
            result = reasoner.reason(scene, instruction)
            example = InstructionExample(
                instruction=instruction,
                program=result.program,
                target_id=result.target_id,
                target_type=result.target_type,
                task_type="state_filtering",
                reasoning_steps=result.steps,
            )
            f.write(json.dumps(sample_to_jsonl_row(scene, example), ensure_ascii=False) + "\n")
    return path


def run_overfit(
    steps: int = 120,
    num_samples: int = 12,
    hidden_dim: int = 96,
    lr: float = 3e-3,
    output: Path = Path("checkpoints/oarlvla_tiny_overfit.pt"),
) -> dict:
    dataset_path = build_tiny_dataset(Path("data/oarlvla_tiny_overfit.jsonl"), num_samples=num_samples)
    dataset = SyntheticVLADataset(dataset_path)
    loader = torch.utils.data.DataLoader(dataset, batch_size=num_samples, shuffle=False, collate_fn=vla_collate_fn)
    batch = next(iter(loader))
    config = OARLVLAConfig(
        vocab_size=len(dataset.tokenizer),
        object_feature_dim=dataset.feature_metadata["feature_dim"],
        hidden_dim=hidden_dim,
        num_relation_types=len(dataset.feature_metadata["relation_types"]),
        num_program_types=len(dataset.feature_metadata["task_types"]),
        dropout=0.0,
    )
    model = OARLVLAModel(config)
    optimizer = make_optimizer(model, lr)
    train_config = TrainConfig(batch_size=num_samples, lr=lr, device="cpu")
    from oarlvla.models.losses import VLALossWeights, compute_vla_loss

    loss_weights = VLALossWeights(target_loss_weight=1.0, action_loss_weight=0.05, program_loss_weight=0.1)

    for step in range(steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        outputs = model(**model_inputs(batch))
        loss, metrics = compute_vla_loss(outputs, batch, loss_weights)
        if step in {0, steps // 2, steps}:
            print(f"Step {step} loss={metrics['loss']:.4f} target_accuracy={metrics['target_accuracy']:.3f}")
        if step < steps:
            loss.backward()
            optimizer.step()
    eval_metrics = evaluate_model(model, loader, train_config)
    save_checkpoint(output, model, dataset.tokenizer, dataset.feature_metadata, extra={"overfit_metrics": eval_metrics})
    print(f"Final tiny-batch target accuracy: {eval_metrics.get('target_accuracy', 0.0):.2f}")
    print(f"Saved checkpoint: {output}")
    return eval_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--num-samples", type=int, default=12)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--output", type=Path, default=Path("checkpoints/oarlvla_tiny_overfit.pt"))
    args = parser.parse_args()
    run_overfit(args.steps, args.num_samples, args.hidden_dim, args.lr, args.output)


if __name__ == "__main__":
    main()
