from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEPS = ROOT / ".deps"
if DEPS.exists():
    sys.path.insert(0, str(DEPS))
sys.path.insert(0, str(ROOT / "src"))

try:
    from oarlvla.models.checkpoints import load_checkpoint
    from oarlvla.models.checkpoints import save_checkpoint
    from oarlvla.models.collate import vla_collate_fn
    from oarlvla.models.datasets import MixedVLADataset, SyntheticVLADataset
    from oarlvla.models.encoders import SimpleTokenizer
    from oarlvla.models.qwen_vl import QwenVLProcessorAdapter
    from oarlvla.models.trainer import TrainConfig, evaluate_model, make_optimizer, train_epoch
    from oarlvla.models.vla_model import OARLVLAConfig, OARLVLAModel, require_torch
    from oarlvla.webdata.manifest import read_jsonl
except RuntimeError as exc:
    print(exc, file=sys.stderr)
    sys.exit(2)


torch, _ = require_torch()


def build_datasets(dataset: SyntheticVLADataset, val_ratio: float, seed: int):
    if val_ratio <= 0.0:
        return dataset, None

    n = len(dataset)
    if n < 2:
        return dataset, None

    indices = list(range(n))
    random.Random(seed).shuffle(indices)
    val_size = max(1, int(n * val_ratio))
    val_size = min(val_size, max(1, n - 1))
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]
    return torch.utils.data.Subset(dataset, train_idx), torch.utils.data.Subset(dataset, val_idx)


def build_train_dataset(args, tokenizer: SimpleTokenizer | None = None):
    ablate_feature_groups = []
    if args.ablate_attribute_state:
        ablate_feature_groups.append("attribute_state")
    if args.web_weak_dataset is not None:
        return MixedVLADataset(
            args.dataset,
            web_weak_jsonl_path=args.web_weak_dataset,
            tokenizer=tokenizer,
            web_repeat=args.web_repeat,
            ablate_feature_groups=ablate_feature_groups,
            include_groups=not args.no_groups,
        )
    return SyntheticVLADataset(
        args.dataset,
        tokenizer=tokenizer,
        ablate_feature_groups=ablate_feature_groups,
        include_groups=not args.no_groups,
    )


def extend_tokenizer_from_paths(tokenizer: SimpleTokenizer, paths: list[Path | None]) -> None:
    instructions: list[str] = []
    for path in paths:
        if path is None or not path.exists():
            continue
        instructions.extend(row["instruction"] for row in read_jsonl(path) if "instruction" in row)
    tokenizer.build_vocab(instructions)


def resize_text_embeddings(model: OARLVLAModel, vocab_size: int) -> None:
    old_embedding = model.text_encoder.embedding
    if old_embedding.num_embeddings == vocab_size:
        return
    new_embedding = torch.nn.Embedding(vocab_size, old_embedding.embedding_dim, padding_idx=old_embedding.padding_idx)
    copy_n = min(old_embedding.num_embeddings, vocab_size)
    with torch.no_grad():
        new_embedding.weight[:copy_n].copy_(old_embedding.weight[:copy_n])
    model.text_encoder.embedding = new_embedding
    model.config.vocab_size = vocab_size


def _matches_prefix(name: str, prefixes: list[str]) -> bool:
    return any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes)


def _expand_module_prefixes(prefixes: list[str] | None) -> list[str] | None:
    if prefixes is None:
        return None
    aliases = {
        "object_encoder": "oarl_core.object_encoder",
        "graph_encoder": "oarl_core.graph_encoder",
        "fusion": "oarl_core.fusion",
        "target_head": "oarl_core.target_head",
        "global_norm": "oarl_core.global_norm",
        "region_encoder": "oarl_core.region_encoder",
        "oarl_adapter": "oarl_core",
    }
    expanded: list[str] = []
    for prefix in prefixes:
        expanded.append(prefix)
        if prefix in aliases:
            expanded.append(aliases[prefix])
    return expanded


def apply_freezing(model: OARLVLAModel, train_modules: list[str] | None, freeze_modules: list[str]) -> dict:
    train_modules = _expand_module_prefixes(train_modules)
    freeze_modules = _expand_module_prefixes(freeze_modules) or []
    if train_modules:
        for name, param in model.named_parameters():
            param.requires_grad = _matches_prefix(name, train_modules)
    for name, param in model.named_parameters():
        if _matches_prefix(name, freeze_modules):
            param.requires_grad = False
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    by_module: dict[str, dict[str, int]] = {}
    for name, param in model.named_parameters():
        module = name.split(".", 1)[0]
        stats = by_module.setdefault(module, {"total": 0, "trainable": 0})
        stats["total"] += param.numel()
        if param.requires_grad:
            stats["trainable"] += param.numel()
    return {"total": total, "trainable": trainable, "frozen": total - trainable, "by_module": by_module}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--web-weak-dataset", type=Path, default=None, help="Optional weak web task JSONL for Stage-2 program warm-up.")
    parser.add_argument("--web-repeat", type=int, default=1, help="Repeat weak web rows in the mixed training dataset.")
    parser.add_argument("--ablate-attribute-state", action="store_true", help="Zero attribute/state object feature channels.")
    parser.add_argument("--no-groups", action="store_true", help="Remove group candidates from model inputs.")
    parser.add_argument("--no-relation-graph", action="store_true", help="Disable relation graph message passing.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--target-loss-weight", type=float, default=1.0)
    parser.add_argument("--action-loss-weight", type=float, default=0.5)
    parser.add_argument("--program-loss-weight", type=float, default=0.2)
    parser.add_argument("--action-head-type", choices=["mlp", "flow_matching"], default=None)
    parser.add_argument("--action-chunk-size", type=int, default=None)
    parser.add_argument("--action-denoise-steps", type=int, default=None)
    parser.add_argument("--action-head-layers", type=int, default=None)
    parser.add_argument("--action-head-heads", type=int, default=None)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-dataset", type=Path, default=None)
    parser.add_argument("--vlm-backbone", choices=["tiny", "qwen_vl"], default="tiny")
    parser.add_argument("--qwen-model-name", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--unfreeze-qwen-vl", action="store_true")
    parser.add_argument("--qwen-device-map", default=None)
    parser.add_argument("--init-checkpoint", type=Path, default=None, help="Optional checkpoint to continue from.")
    parser.add_argument("--extend-tokenizer", action="store_true", help="Extend a loaded checkpoint tokenizer with new dataset instructions.")
    parser.add_argument(
        "--train-modules",
        nargs="*",
        default=None,
        help="If set, only parameters whose names start with these prefixes are trainable.",
    )
    parser.add_argument(
        "--freeze-modules",
        nargs="*",
        default=[],
        help="Freeze parameters whose names start with these prefixes, e.g. object_encoder graph_encoder action_head.",
    )
    parser.add_argument("--output", type=Path, default=Path("checkpoints/oarlvla_tiny.pt"))
    args = parser.parse_args()

    model = None
    tokenizer = None
    checkpoint_extra = {}
    if args.init_checkpoint is not None:
        model, tokenizer, checkpoint_feature_metadata, checkpoint_extra = load_checkpoint(
            args.init_checkpoint,
            map_location=args.device,
        )
        if args.extend_tokenizer:
            extend_tokenizer_from_paths(tokenizer, [args.dataset, args.web_weak_dataset, args.eval_dataset])
            resize_text_embeddings(model, len(tokenizer))
        model.config.use_relation_graph = not args.no_relation_graph
        if (
            args.action_head_type is not None
            or args.action_chunk_size is not None
            or args.action_denoise_steps is not None
            or args.action_head_layers is not None
            or args.action_head_heads is not None
        ):
            model.replace_action_head(
                args.action_head_type or model.config.action_head_type,
                action_chunk_size=args.action_chunk_size,
                action_denoise_steps=args.action_denoise_steps,
                action_head_layers=args.action_head_layers,
                action_head_heads=args.action_head_heads,
            )
        print(f"Loaded initial checkpoint: {args.init_checkpoint}")

    dataset = build_train_dataset(args, tokenizer=tokenizer)
    train_set, val_set = build_datasets(dataset, args.val_ratio, args.seed)
    if args.eval_dataset is not None:
        eval_dataset = SyntheticVLADataset(
            args.eval_dataset,
            tokenizer=dataset.tokenizer,
            ablate_feature_groups=["attribute_state"] if args.ablate_attribute_state else [],
            include_groups=not args.no_groups,
        )
        train_loader = torch.utils.data.DataLoader(
            train_set if isinstance(train_set, torch.utils.data.Dataset) else dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=vla_collate_fn,
        )
        eval_loader = torch.utils.data.DataLoader(
            eval_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=vla_collate_fn
        )
    else:
        train_data = train_set if val_set is not None else dataset
        eval_data = val_set if val_set is not None else dataset
        if val_set is None:
            print(f"Using all {len(dataset)} samples for train/eval (val ratio={args.val_ratio})")
        else:
            print(f"Training split: {len(train_set)} | Validation split: {len(val_set)}")
        train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=args.batch_size, shuffle=True, collate_fn=vla_collate_fn
        )
        eval_loader = torch.utils.data.DataLoader(
            eval_data, batch_size=args.batch_size, shuffle=False, collate_fn=vla_collate_fn
        )
    if model is None:
        config = OARLVLAConfig(
            vocab_size=len(dataset.tokenizer),
            object_feature_dim=dataset.feature_metadata["feature_dim"],
            hidden_dim=args.hidden_dim,
            num_relation_types=len(dataset.feature_metadata["relation_types"]),
            num_program_types=len(dataset.feature_metadata["task_types"]),
            action_head_type=args.action_head_type or "flow_matching",
            action_chunk_size=args.action_chunk_size or 8,
            action_denoise_steps=args.action_denoise_steps or 10,
            action_head_layers=args.action_head_layers or 2,
            action_head_heads=args.action_head_heads or 4,
            vlm_backbone=args.vlm_backbone,
            qwen_model_name=args.qwen_model_name,
            freeze_qwen_vl=not args.unfreeze_qwen_vl,
            qwen_device_map=args.qwen_device_map,
            use_relation_graph=not args.no_relation_graph,
        )
        model = OARLVLAModel(config)
    freeze_summary = apply_freezing(model, args.train_modules, args.freeze_modules)
    print("Trainable parameters: {trainable}/{total} ({pct:.2f}%)".format(
        trainable=freeze_summary["trainable"],
        total=freeze_summary["total"],
        pct=100.0 * freeze_summary["trainable"] / max(freeze_summary["total"], 1),
    ))
    for module, stats in sorted(freeze_summary["by_module"].items()):
        print(f"  {module}: trainable={stats['trainable']} total={stats['total']}")
    model.to(args.device)
    optimizer = make_optimizer(model, args.lr)
    qwen_processor = (
        QwenVLProcessorAdapter(model.config.qwen_model_name)
        if model.config.vlm_backbone == "qwen_vl"
        else None
    )
    train_config = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        target_loss_weight=args.target_loss_weight,
        action_loss_weight=args.action_loss_weight,
        program_loss_weight=args.program_loss_weight,
        qwen_processor=qwen_processor,
    )
    history = []
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_epoch(model, train_loader, optimizer, train_config)
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
        extra={
            "history": history,
            "dataset": str(args.dataset),
            "web_weak_dataset": str(args.web_weak_dataset) if args.web_weak_dataset else None,
            "init_checkpoint": str(args.init_checkpoint) if args.init_checkpoint else None,
            "init_checkpoint_extra": checkpoint_extra,
            "freeze_summary": freeze_summary,
            "train_args": vars(args),
        },
    )
    print(f"Saved checkpoint: {args.output}")
    print(json.dumps({"final_eval": history[-1]["eval"] if history else {}}, indent=2))


if __name__ == "__main__":
    main()
