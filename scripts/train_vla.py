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
    from oarlvla.models.checkpoints import save_checkpoint
    from oarlvla.models.collate import vla_collate_fn
    from oarlvla.models.datasets import SyntheticVLADataset
    from oarlvla.models.qwen_vl import QwenVLProcessorAdapter
    from oarlvla.models.trainer import TrainConfig, evaluate_model, make_optimizer, train_epoch
    from oarlvla.models.vla_model import OARLVLAConfig, OARLVLAModel, require_torch
except RuntimeError as exc:
    print(exc, file=sys.stderr)
    sys.exit(2)


torch, _ = require_torch()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--vlm-backbone", choices=["tiny", "qwen_vl"], default="tiny")
    parser.add_argument("--qwen-model-name", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--unfreeze-qwen-vl", action="store_true")
    parser.add_argument("--qwen-device-map", default=None)
    parser.add_argument("--output", type=Path, default=Path("checkpoints/oarlvla_tiny.pt"))
    args = parser.parse_args()

    dataset = SyntheticVLADataset(args.dataset)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=vla_collate_fn)
    eval_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=vla_collate_fn)
    config = OARLVLAConfig(
        vocab_size=len(dataset.tokenizer),
        object_feature_dim=dataset.feature_metadata["feature_dim"],
        hidden_dim=args.hidden_dim,
        num_relation_types=len(dataset.feature_metadata["relation_types"]),
        num_program_types=len(dataset.feature_metadata["task_types"]),
        vlm_backbone=args.vlm_backbone,
        qwen_model_name=args.qwen_model_name,
        freeze_qwen_vl=not args.unfreeze_qwen_vl,
        qwen_device_map=args.qwen_device_map,
    )
    model = OARLVLAModel(config).to(args.device)
    optimizer = make_optimizer(model, args.lr)
    qwen_processor = (
        QwenVLProcessorAdapter(args.qwen_model_name)
        if args.vlm_backbone == "qwen_vl"
        else None
    )
    train_config = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        qwen_processor=qwen_processor,
    )
    history = []
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_epoch(model, dataloader, optimizer, train_config)
        eval_metrics = evaluate_model(model, eval_loader, train_config)
        history.append({"epoch": epoch, "train": train_metrics, "eval": eval_metrics})
        print(
            "Epoch {epoch}/{total} loss={loss:.4f} target_accuracy={target:.3f} "
            "program_accuracy={program:.3f} action_mse={action:.5f}".format(
                epoch=epoch,
                total=args.epochs,
                loss=train_metrics.get("loss", 0.0),
                target=eval_metrics.get("target_accuracy", 0.0),
                program=eval_metrics.get("program_accuracy", 0.0),
                action=eval_metrics.get("action_mse", 0.0),
            )
        )
    save_checkpoint(
        args.output,
        model,
        dataset.tokenizer,
        dataset.feature_metadata,
        extra={"history": history, "dataset": str(args.dataset), "train_args": vars(args)},
    )
    print(f"Saved checkpoint: {args.output}")
    print(json.dumps({"final_eval": history[-1]["eval"] if history else {}}, indent=2))


if __name__ == "__main__":
    main()
