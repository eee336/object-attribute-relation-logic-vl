from __future__ import annotations

from typing import Any

from .torch_utils import require_torch


torch, _ = require_torch()


def vla_collate_fn(samples: list[dict[str, Any]]) -> dict[str, Any]:
    batch_size = len(samples)
    max_objects = max(sample["object_features"].shape[0] for sample in samples)
    feature_dim = samples[0]["object_features"].shape[-1]
    max_edges = max(sample["relation_edges"].shape[0] for sample in samples)
    object_features = torch.zeros(batch_size, max_objects, feature_dim, dtype=torch.float32)
    object_region_features = None
    object_mask = torch.zeros(batch_size, max_objects, dtype=torch.bool)
    relation_edges = torch.zeros(batch_size, max_edges, 2, dtype=torch.long)
    relation_types = torch.zeros(batch_size, max_edges, dtype=torch.long)
    relation_mask = torch.zeros(batch_size, max_edges, dtype=torch.bool)
    instruction_ids = torch.stack([sample["instruction_ids"] for sample in samples], dim=0)
    target_index = torch.stack([sample["target_index"] for sample in samples], dim=0)
    target_center = torch.stack([sample["target_center"] for sample in samples], dim=0)
    task_type_id = torch.stack([sample["task_type_id"] for sample in samples], dim=0)
    has_gold_target = torch.tensor([bool(sample.get("has_gold_target", sample.get("target_id") is not None)) for sample in samples], dtype=torch.bool)
    action_chunk = None
    if any("action_chunk" in sample for sample in samples):
        action_rows = []
        for sample in samples:
            if "action_chunk" in sample:
                action_rows.append(sample["action_chunk"])
            else:
                action_rows.append(sample["target_center"].unsqueeze(0))
        max_action_steps = max(row.shape[0] for row in action_rows)
        action_dim = max(row.shape[-1] for row in action_rows)
        action_chunk = torch.zeros(batch_size, max_action_steps, action_dim, dtype=torch.float32)
        for idx, row in enumerate(action_rows):
            action_chunk[idx, : row.shape[0], : row.shape[-1]] = row.float()
    if any("object_region_features" in sample for sample in samples):
        region_dim = max(
            sample["object_region_features"].shape[-1]
            for sample in samples
            if "object_region_features" in sample
        )
        object_region_features = torch.zeros(batch_size, max_objects, region_dim, dtype=torch.float32)

    for idx, sample in enumerate(samples):
        num_objects = sample["object_features"].shape[0]
        num_edges = sample["relation_edges"].shape[0]
        object_features[idx, :num_objects] = sample["object_features"]
        if object_region_features is not None and "object_region_features" in sample:
            region_features = sample["object_region_features"].float()
            object_region_features[idx, : region_features.shape[0], : region_features.shape[-1]] = region_features
        object_mask[idx, :num_objects] = True
        if num_edges:
            relation_edges[idx, :num_edges] = sample["relation_edges"]
            relation_types[idx, :num_edges] = sample["relation_types"]
            relation_mask[idx, :num_edges] = True

    return {
        "instruction_ids": instruction_ids,
        "object_features": object_features,
        **({"object_region_features": object_region_features} if object_region_features is not None else {}),
        "relation_edges": relation_edges,
        "relation_types": relation_types,
        "object_mask": object_mask,
        "relation_mask": relation_mask,
        "target_index": target_index,
        "target_center": target_center,
        "task_type_id": task_type_id,
        "has_gold_target": has_gold_target,
        **({"action_chunk": action_chunk} if action_chunk is not None else {}),
        "sample_id": [sample["sample_id"] for sample in samples],
        "instruction": [sample["instruction"] for sample in samples],
        "task_type": [sample["task_type"] for sample in samples],
        "candidate_ids": [sample["candidate_ids"] for sample in samples],
        "target_id": [sample["target_id"] for sample in samples],
        "image_path": [sample.get("image_path") for sample in samples],
        "label_quality": [sample["label_quality"] for sample in samples],
    }


def batch_to_device(batch: dict[str, Any], device: str):
    moved = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if hasattr(value, "to") else value
    return moved
