from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from .collate import batch_to_device
from .datasets import ID_TO_TASK_TYPE
from .losses import VLALossWeights, compute_vla_loss
from .torch_utils import require_torch


torch, _ = require_torch()


@dataclass
class TrainConfig:
    epochs: int = 5
    batch_size: int = 16
    lr: float = 1e-3
    device: str = "cpu"
    target_loss_weight: float = 1.0
    action_loss_weight: float = 0.5
    program_loss_weight: float = 0.2


def model_inputs(batch: dict[str, Any]) -> dict[str, Any]:
    return {
        "tokenized_instruction": batch["instruction_ids"],
        "object_features": batch["object_features"],
        "relation_features": {"edge_index": batch["relation_edges"], "edge_type": batch["relation_types"]},
        "object_mask": batch["object_mask"],
        "relation_mask": batch["relation_mask"],
    }


def train_epoch(model, dataloader, optimizer, config: TrainConfig) -> dict[str, float]:
    model.train()
    weights = VLALossWeights(config.target_loss_weight, config.action_loss_weight, config.program_loss_weight)
    totals = defaultdict(float)
    n = 0
    for batch in dataloader:
        batch = batch_to_device(batch, config.device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(**model_inputs(batch))
        loss, metrics = compute_vla_loss(outputs, batch, weights)
        loss.backward()
        optimizer.step()
        batch_n = int(batch["instruction_ids"].shape[0])
        n += batch_n
        for key, value in metrics.items():
            totals[key] += value * batch_n
    return {key: value / max(n, 1) for key, value in totals.items()}


@torch.no_grad()
def evaluate_model(model, dataloader, config: TrainConfig) -> dict[str, Any]:
    model.eval()
    totals = defaultdict(float)
    by_task = defaultdict(lambda: {"n": 0, "target_correct": 0, "program_correct": 0, "action_mse_sum": 0.0})
    n = 0
    for batch in dataloader:
        batch = batch_to_device(batch, config.device)
        outputs = model(**model_inputs(batch))
        loss, metrics = compute_vla_loss(outputs, batch)
        batch_n = int(batch["instruction_ids"].shape[0])
        n += batch_n
        for key, value in metrics.items():
            totals[key] += value * batch_n
        target_pred = outputs["target_logits"].argmax(dim=-1)
        program_pred = outputs["program_logits"].argmax(dim=-1)
        action_sq = (outputs["action_pred"][:, :3] - batch["target_center"][:, :3]).pow(2).mean(dim=-1)
        for idx in range(batch_n):
            task_type = batch["task_type"][idx]
            valid = int(batch["target_index"][idx].item()) >= 0
            by_task[task_type]["n"] += 1
            if valid and int(target_pred[idx].item()) == int(batch["target_index"][idx].item()):
                by_task[task_type]["target_correct"] += 1
            if int(program_pred[idx].item()) == int(batch["task_type_id"][idx].item()):
                by_task[task_type]["program_correct"] += 1
            by_task[task_type]["action_mse_sum"] += float(action_sq[idx].detach().cpu())
    result = {key: value / max(n, 1) for key, value in totals.items()}
    result["by_task"] = {
        task: {
            "n": stats["n"],
            "target_accuracy": stats["target_correct"] / stats["n"] if stats["n"] else 0.0,
            "program_accuracy": stats["program_correct"] / stats["n"] if stats["n"] else 0.0,
            "action_mse": stats["action_mse_sum"] / stats["n"] if stats["n"] else 0.0,
        }
        for task, stats in sorted(by_task.items())
    }
    return result


def make_optimizer(model, lr: float = 1e-3):
    return torch.optim.AdamW(model.parameters(), lr=lr)

