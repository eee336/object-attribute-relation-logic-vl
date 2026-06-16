from __future__ import annotations

from dataclasses import dataclass

from .torch_utils import require_torch


torch, nn = require_torch()
F = torch.nn.functional


@dataclass
class VLALossWeights:
    target_loss_weight: float = 1.0
    action_loss_weight: float = 0.5
    program_loss_weight: float = 0.2


def compute_vla_loss(outputs: dict, batch: dict, weights: VLALossWeights | None = None) -> tuple[torch.Tensor, dict]:
    weights = weights or VLALossWeights()
    device = outputs["target_logits"].device
    total = torch.zeros((), device=device)
    metrics: dict[str, float] = {}

    target_index = batch.get("target_index")
    valid_target = target_index is not None and (target_index >= 0).any()
    if valid_target:
        valid_mask = target_index >= 0
        target_loss = F.cross_entropy(outputs["target_logits"][valid_mask], target_index[valid_mask].long())
        target_pred = outputs["target_logits"].argmax(dim=-1)
        target_accuracy = (target_pred[valid_mask] == target_index[valid_mask]).float().mean()
        total = total + weights.target_loss_weight * target_loss
        metrics["target_loss"] = float(target_loss.detach().cpu())
        metrics["target_accuracy"] = float(target_accuracy.detach().cpu())
    else:
        metrics["target_loss"] = 0.0
        metrics["target_accuracy"] = 0.0

    if "target_center" in batch and valid_target:
        valid_mask = target_index >= 0
        action_label = batch["target_center"][valid_mask].to(device)
        action_pred = outputs["action_pred"][valid_mask]
        action_dims = min(action_pred.shape[-1], action_label.shape[-1])
        action_mse = F.mse_loss(action_pred[:, :action_dims], action_label[:, :action_dims])
        action_flow_losses = outputs.get("action_flow_losses")
        if action_flow_losses is not None:
            action_loss = action_flow_losses[valid_mask].mean()
            metrics["action_flow_loss"] = float(action_loss.detach().cpu())
        else:
            action_loss = action_mse
            metrics["action_flow_loss"] = 0.0
        total = total + weights.action_loss_weight * action_loss
        metrics["action_mse"] = float(action_mse.detach().cpu())
    else:
        metrics["action_mse"] = 0.0
        metrics["action_flow_loss"] = 0.0

    if outputs.get("program_logits") is not None and "task_type_id" in batch:
        program_loss = F.cross_entropy(outputs["program_logits"], batch["task_type_id"].long().to(device))
        program_pred = outputs["program_logits"].argmax(dim=-1)
        program_accuracy = (program_pred == batch["task_type_id"].to(device)).float().mean()
        total = total + weights.program_loss_weight * program_loss
        metrics["program_loss"] = float(program_loss.detach().cpu())
        metrics["program_accuracy"] = float(program_accuracy.detach().cpu())
    else:
        metrics["program_loss"] = 0.0
        metrics["program_accuracy"] = 0.0

    metrics["loss"] = float(total.detach().cpu())
    return total, metrics
