from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from oarlvla.instruction import TASK_TYPES
from oarlvla.webdata.manifest import read_jsonl

from .encoders import SimpleTokenizer
from .torch_utils import require_torch


torch, _ = require_torch()


SUPER_CATEGORIES = [
    "fruit",
    "food",
    "drink",
    "container",
    "footwear",
    "drinkware",
    "utensil",
    "waste_related",
    "readable_object",
    "electronics",
]
CATEGORIES = [
    "unknown",
    "apple",
    "banana",
    "orange",
    "bottle",
    "water_bottle",
    "can",
    "soda_can",
    "juice_box",
    "cup",
    "mug",
    "bowl",
    "shoe",
    "spoon",
    "trash_bin",
    "book",
    "remote",
    "shoe_pair",
    "book_stack",
]
COLORS = ["unknown", "red", "yellow", "orange", "green", "blue", "white", "gray", "black", "clear", "silver", "brown"]
SHAPES = ["unknown", "round", "curved", "cylinder", "box", "shoe", "bin", "rectangle", "slender"]
MATERIALS = ["unknown", "fruit_skin", "plastic", "metal", "paper", "ceramic", "leather"]
RELATION_TYPES = ["left_of", "right_of", "above", "below", "near", "far", "same_group", "member_of"]
TASK_TYPE_TO_ID = {task: idx for idx, task in enumerate(TASK_TYPES)}
ID_TO_TASK_TYPE = {idx: task for task, idx in TASK_TYPE_TO_ID.items()}

FEATURE_NAMES = [
    "category_id_norm",
    *[f"super_{name}" for name in SUPER_CATEGORIES],
    "color_id_norm",
    "shape_id_norm",
    "material_id_norm",
    "bbox_x1",
    "bbox_y1",
    "bbox_x2",
    "bbox_y2",
    "center_x",
    "center_y",
    "size",
    "black_spot_ratio",
    "ripeness",
    "is_blackened",
    "is_rotten",
    "is_edible",
    "volume_ml",
    "fill_level",
    "is_opened",
    "is_empty",
    "cleanliness",
    "is_broken",
    "is_wearable",
    "group_flag",
    "member_count",
]

ATTRIBUTE_STATE_FEATURES = [
    "black_spot_ratio",
    "ripeness",
    "is_blackened",
    "is_rotten",
    "is_edible",
    "volume_ml",
    "fill_level",
    "is_opened",
    "is_empty",
    "cleanliness",
    "is_broken",
    "is_wearable",
]


class SyntheticVLADataset(torch.utils.data.Dataset):
    def __init__(
        self,
        jsonl_path: str | Path,
        tokenizer: SimpleTokenizer | None = None,
        max_length: int = 32,
        ablate_feature_groups: list[str] | None = None,
        include_groups: bool = True,
    ):
        self.path = Path(jsonl_path)
        self.rows = read_jsonl(self.path)
        if not self.rows:
            raise ValueError(f"No rows found in {self.path}")
        self.tokenizer = tokenizer or SimpleTokenizer(max_length=max_length)
        if tokenizer is None:
            self.tokenizer.build_vocab([row["instruction"] for row in self.rows])
        self.max_length = max_length
        self.include_groups = include_groups
        self.feature_ablation_indices = feature_ablation_indices(ablate_feature_groups or [])
        self.feature_metadata = feature_metadata()

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        candidates = candidate_entities(row, include_groups=self.include_groups)
        candidate_ids = [entity["id"] for entity in candidates]
        target_id = row.get("target_id")
        target_index = candidate_ids.index(target_id) if target_id in candidate_ids else -1
        width = float(row.get("width") or 640)
        height = float(row.get("height") or 480)
        object_features = [
            apply_feature_ablation(entity_to_features(entity, width, height), self.feature_ablation_indices)
            for entity in candidates
        ]
        edge_index, edge_type = build_relation_edges(candidates, width, height)
        if target_index >= 0:
            center = candidates[target_index].get("center", [0.0, 0.0])
            target_center = [float(center[0]) / width, float(center[1]) / height, 1.0]
        else:
            target_center = [0.0, 0.0, 0.0]
        task_type = row.get("task_type", "state_filtering")
        return {
            "sample_id": row.get("scene_id", f"sample_{idx}"),
            "instruction": row["instruction"],
            "instruction_ids": torch.tensor(self.tokenizer.encode(row["instruction"], self.max_length), dtype=torch.long),
            "object_features": torch.tensor(object_features, dtype=torch.float32),
            "relation_edges": torch.tensor(edge_index, dtype=torch.long),
            "relation_types": torch.tensor(edge_type, dtype=torch.long),
            "target_index": torch.tensor(target_index, dtype=torch.long),
            "target_center": torch.tensor(target_center, dtype=torch.float32),
            "task_type_id": torch.tensor(TASK_TYPE_TO_ID.get(task_type, 0), dtype=torch.long),
            "task_type": task_type,
            "candidate_ids": candidate_ids,
            "target_id": target_id,
            "has_gold_target": target_id is not None,
            "image_path": row.get("image_path"),
            "label_quality": row.get("label_quality", "gold"),
            "source": row.get("source", "synthetic"),
        }


class WebWeakVLADataset(torch.utils.data.Dataset):
    """Weak web tasks. Samples without verified target_id use target_index=-1."""

    def __init__(
        self,
        jsonl_path: str | Path,
        tokenizer: SimpleTokenizer | None = None,
        max_length: int = 32,
        ablate_feature_groups: list[str] | None = None,
    ):
        self.path = Path(jsonl_path)
        self.rows = read_jsonl(self.path)
        self.tokenizer = tokenizer or SimpleTokenizer(max_length=max_length)
        if tokenizer is None and self.rows:
            self.tokenizer.build_vocab([row["instruction"] for row in self.rows])
        self.max_length = max_length
        self.feature_ablation_indices = feature_ablation_indices(ablate_feature_groups or [])
        self.feature_metadata = feature_metadata()

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        task_type = row.get("task_type", "open_vocab")
        return {
            "sample_id": row.get("image_id", f"web_{idx}"),
            "instruction": row["instruction"],
            "instruction_ids": torch.tensor(self.tokenizer.encode(row["instruction"], self.max_length), dtype=torch.long),
            "object_features": torch.tensor(
                [apply_feature_ablation([0.0] * len(FEATURE_NAMES), self.feature_ablation_indices)],
                dtype=torch.float32,
            ),
            "relation_edges": torch.zeros(0, 2, dtype=torch.long),
            "relation_types": torch.zeros(0, dtype=torch.long),
            "target_index": torch.tensor(-1, dtype=torch.long),
            "target_center": torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32),
            "task_type_id": torch.tensor(TASK_TYPE_TO_ID.get(task_type, 0), dtype=torch.long),
            "task_type": task_type,
            "candidate_ids": ["unknown_candidate"],
            "target_id": row.get("target_id"),
            "has_gold_target": False,
            "image_path": row.get("image_path"),
            "label_quality": row.get("label_quality", "weak"),
            "source": "web",
        }


class MixedVLADataset(torch.utils.data.Dataset):
    """Gold synthetic/grid data plus weak web tasks under one tokenizer.

    Weak web samples usually have no verified target id. The training loss
    already masks target/action supervision for target_index=-1, so these rows
    contribute program/task supervision without corrupting grounding labels.
    """

    def __init__(
        self,
        synthetic_jsonl_path: str | Path,
        web_weak_jsonl_path: str | Path | None = None,
        tokenizer: SimpleTokenizer | None = None,
        max_length: int = 32,
        web_repeat: int = 1,
        ablate_feature_groups: list[str] | None = None,
        include_groups: bool = True,
    ):
        self.synthetic_path = Path(synthetic_jsonl_path)
        self.web_weak_path = Path(web_weak_jsonl_path) if web_weak_jsonl_path else None
        self.max_length = max_length
        self.tokenizer = tokenizer or SimpleTokenizer(max_length=max_length)
        if tokenizer is None:
            instructions = [row["instruction"] for row in read_jsonl(self.synthetic_path)]
            if self.web_weak_path is not None and self.web_weak_path.exists():
                instructions.extend(row["instruction"] for row in read_jsonl(self.web_weak_path))
            self.tokenizer.build_vocab(instructions)

        self.datasets: list[torch.utils.data.Dataset] = [
            SyntheticVLADataset(
                self.synthetic_path,
                tokenizer=self.tokenizer,
                max_length=max_length,
                ablate_feature_groups=ablate_feature_groups,
                include_groups=include_groups,
            )
        ]
        if self.web_weak_path is not None and self.web_weak_path.exists():
            web_dataset = WebWeakVLADataset(
                self.web_weak_path,
                tokenizer=self.tokenizer,
                max_length=max_length,
                ablate_feature_groups=ablate_feature_groups,
            )
            if len(web_dataset) > 0:
                self.datasets.extend(web_dataset for _ in range(max(1, int(web_repeat))))
        self.feature_metadata = feature_metadata()
        self._offsets: list[tuple[int, torch.utils.data.Dataset]] = []
        offset = 0
        for dataset in self.datasets:
            offset += len(dataset)
            self._offsets.append((offset, dataset))

    def __len__(self) -> int:
        return self._offsets[-1][0] if self._offsets else 0

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if idx < 0:
            idx = len(self) + idx
        previous = 0
        for end, dataset in self._offsets:
            if idx < end:
                return dataset[idx - previous]
            previous = end
        raise IndexError(idx)


def feature_metadata() -> dict[str, Any]:
    return {
        "feature_names": FEATURE_NAMES,
        "feature_dim": len(FEATURE_NAMES),
        "categories": CATEGORIES,
        "super_categories": SUPER_CATEGORIES,
        "colors": COLORS,
        "shapes": SHAPES,
        "materials": MATERIALS,
        "relation_types": RELATION_TYPES,
        "task_types": TASK_TYPES,
    }


def candidate_entities(row: dict[str, Any], include_groups: bool = True) -> list[dict[str, Any]]:
    objects = [dict(obj, is_group=False) for obj in row.get("objects", [])]
    groups = [dict(group, is_group=True) for group in row.get("groups", [])] if include_groups else []
    return objects + groups


def feature_ablation_indices(groups: list[str]) -> set[int]:
    names: set[str] = set()
    for group in groups:
        if group == "attribute_state":
            names.update(ATTRIBUTE_STATE_FEATURES)
        elif group == "geometry":
            names.update({"bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2", "center_x", "center_y", "size"})
        elif group in FEATURE_NAMES:
            names.add(group)
        elif group:
            raise ValueError(f"Unsupported feature ablation group: {group}")
    return {FEATURE_NAMES.index(name) for name in names}


def apply_feature_ablation(values: list[float], indices: set[int]) -> list[float]:
    if not indices:
        return values
    values = list(values)
    for idx in indices:
        values[idx] = 0.0
    return values


def entity_to_features(entity: dict[str, Any], width: float, height: float) -> list[float]:
    category = entity.get("category", "unknown")
    supers = set(entity.get("super_categories", []))
    color = entity.get("color", "unknown")
    shape = entity.get("shape", "unknown") or "unknown"
    material = entity.get("material", "unknown") or "unknown"
    bbox = entity.get("bbox") or _bbox_from_center(entity.get("center", [0.0, 0.0]))
    center = entity.get("center", [0.0, 0.0])
    attrs = entity.get("attributes", {}) or {}
    states = entity.get("states", {}) or {}
    size = float(entity.get("size", attrs.get("size", 0.0)) or 0.0)
    values = [
        _norm_id(category, CATEGORIES),
        *[1.0 if name in supers else 0.0 for name in SUPER_CATEGORIES],
        _norm_id(color, COLORS),
        _norm_id(shape, SHAPES),
        _norm_id(material, MATERIALS),
        float(bbox[0]) / width,
        float(bbox[1]) / height,
        float(bbox[2]) / width,
        float(bbox[3]) / height,
        float(center[0]) / width,
        float(center[1]) / height,
        size,
        _num(attrs, "black_spot_ratio"),
        _num(attrs, "ripeness"),
        _bool(states, "is_blackened"),
        _bool(states, "is_rotten"),
        _bool(states, "is_edible"),
        _num(attrs, "volume_ml") / 1000.0,
        _num(states, "fill_level"),
        _bool(states, "is_opened"),
        _bool(states, "is_empty"),
        _num(attrs, "cleanliness"),
        _bool(states, "is_broken"),
        _bool(states, "is_wearable"),
        1.0 if entity.get("is_group") else 0.0,
        float(attrs.get("member_count", len(entity.get("member_ids", [])) if entity.get("is_group") else 1)) / 10.0,
    ]
    return values


def build_relation_edges(candidates: list[dict[str, Any]], width: float, height: float) -> tuple[list[list[int]], list[int]]:
    edges: list[list[int]] = []
    types: list[int] = []
    centers = [entity.get("center", [0.0, 0.0]) for entity in candidates]
    near_threshold = 105.0
    for i, src in enumerate(candidates):
        for j, dst in enumerate(candidates):
            if i == j:
                continue
            sx, sy = centers[i]
            dx, dy = centers[j]
            if sx < dx:
                _add_edge(edges, types, i, j, "left_of")
            if sx > dx:
                _add_edge(edges, types, i, j, "right_of")
            if sy < dy:
                _add_edge(edges, types, i, j, "above")
            if sy > dy:
                _add_edge(edges, types, i, j, "below")
            distance = ((sx - dx) ** 2 + (sy - dy) ** 2) ** 0.5
            if distance <= near_threshold:
                _add_edge(edges, types, i, j, "near")
            elif distance >= near_threshold * 2:
                _add_edge(edges, types, i, j, "far")
            if src.get("group_id") and src.get("group_id") == dst.get("group_id"):
                _add_edge(edges, types, i, j, "same_group")
    id_to_idx = {entity["id"]: idx for idx, entity in enumerate(candidates)}
    for idx, entity in enumerate(candidates):
        for member_id in entity.get("member_ids", []) or []:
            if member_id in id_to_idx:
                _add_edge(edges, types, id_to_idx[member_id], idx, "member_of")
    return edges, types


def _add_edge(edges: list[list[int]], types: list[int], src: int, dst: int, relation: str) -> None:
    edges.append([src, dst])
    types.append(RELATION_TYPES.index(relation))


def _norm_id(value: str, vocab: list[str]) -> float:
    idx = vocab.index(value) if value in vocab else 0
    return idx / max(1, len(vocab) - 1)


def _num(mapping: dict[str, Any], key: str) -> float:
    value = mapping.get(key, 0.0)
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _bool(mapping: dict[str, Any], key: str) -> float:
    return 1.0 if mapping.get(key, False) else 0.0


def _bbox_from_center(center: list[float]) -> list[float]:
    x, y = center
    return [x - 16, y - 16, x + 16, y + 16]
