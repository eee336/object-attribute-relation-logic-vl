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
    object_mask = torch.zeros(batch_size, max_objects, dtype=torch.bool)
    relation_edges = torch.zeros(batch_size, max_edges, 2, dtype=torch.long)
    relation_types = torch.zeros(batch_size, max_edges, dtype=torch.long)
    relation_mask = torch.zeros(batch_size, max_edges, dtype=torch.bool)
    instruction_ids = torch.stack([sample["instruction_ids"] for sample in samples], dim=0)
    target_index = torch.stack([sample["target_index"] for sample in samples], dim=0)
    target_center = torch.stack([sample["target_center"] for sample in samples], dim=0)
    task_type_id = torch.stack([sample["task_type_id"] for sample in samples], dim=0)
    has_gold_target = torch.tensor([bool(sample.get("has_gold_target", sample.get("target_id") is not None)) for sample in samples], dtype=torch.bool)

    for idx, sample in enumerate(samples):
        num_objects = sample["object_features"].shape[0]
        num_edges = sample["relation_edges"].shape[0]
        object_features[idx, :num_objects] = sample["object_features"]
        object_mask[idx, :num_objects] = True
        if num_edges:
            relation_edges[idx, :num_edges] = sample["relation_edges"]
            relation_types[idx, :num_edges] = sample["relation_types"]
            relation_mask[idx, :num_edges] = True

    return {
        "instruction_ids": instruction_ids,
        "object_features": object_features,
        "relation_edges": relation_edges,
        "relation_types": relation_types,
        "object_mask": object_mask,
        "relation_mask": relation_mask,
        "target_index": target_index,
        "target_center": target_center,
        "task_type_id": task_type_id,
        "has_gold_target": has_gold_target,
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
