from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from oarlvla.parser import parse_instruction
from oarlvla.taxonomy import get_super_categories

from .image_utils import mean_rgb
from .schemas import ImageAnnotationBundle, ObjectAnnotation, TaskAnnotation, WebImageRecord


@dataclass
class PseudoTaskSpec:
    task_type: str
    instruction: str
    category: str | None
    target_description: str
    states: dict[str, Any]
    attributes: dict[str, Any]


class PseudoLabeler:
    def label(self, record: WebImageRecord, mode: str = "metadata_only") -> ImageAnnotationBundle:
        if mode not in {"metadata_only", "heuristic", "model_assisted"}:
            raise ValueError(f"Unsupported pseudo-label mode: {mode}")
        spec = infer_task_from_query(record.query)
        confidence = 0.45
        source = "metadata"
        attributes = dict(spec.attributes)
        states = dict(spec.states)
        if mode in {"heuristic", "model_assisted"}:
            heuristic = image_heuristics(record.local_path)
            attributes.update(heuristic.get("attributes", {}))
            states.update(heuristic.get("states", {}))
            confidence = max(confidence, heuristic.get("confidence", 0.0))
            source = "heuristic"
        if mode == "model_assisted":
            # Extension point: keep running without model weights.
            confidence = min(confidence, 0.6)
        task_id = "task_" + hashlib.sha1(f"{record.image_id}:{spec.instruction}".encode("utf-8")).hexdigest()[:12]
        program = parse_instruction(spec.instruction).to_string()
        task = TaskAnnotation(
            task_id=task_id,
            task_type=spec.task_type,
            instruction=spec.instruction,
            program=program,
            target_id=None,
            target_type="unknown",
            confidence=confidence,
            source=source,  # type: ignore[arg-type]
            target_description=spec.target_description,
            requires_manual_verification=True,
        )
        objects = []
        if spec.category:
            objects.append(
                ObjectAnnotation(
                    id=f"weak_{spec.category}_candidate",
                    category=spec.category,
                    super_categories=get_super_categories(spec.category),
                    bbox=None,
                    mask_path=None,
                    attributes=attributes,
                    states=states,
                    confidence=confidence,
                    source=source,  # type: ignore[arg-type]
                )
            )
        return ImageAnnotationBundle(
            image_id=record.image_id,
            objects=objects,
            groups=[],
            candidate_tasks=[task],
            pseudo_labels=[
                {
                    "label_quality": "weak",
                    "mode": mode,
                    "query": record.query,
                    "weak_label": {
                        "category": spec.category,
                        "attributes": attributes,
                        "states": states,
                    },
                    "confidence": confidence,
                    "requires_manual_verification": True,
                }
            ],
        )


def infer_task_from_query(query: str) -> PseudoTaskSpec:
    q = query.lower()
    if "banana" in q and ("no black" in q or "fresh" in q or "still good" in q):
        return PseudoTaskSpec(
            "state_filtering",
            "Pick the banana that has not turned black.",
            "banana",
            "the banana without black spots",
            {"is_blackened": False},
            {},
        )
    if "banana" in q and ("black" in q or "rotten" in q or "overripe" in q):
        return PseudoTaskSpec(
            "state_filtering",
            "Pick the blackened banana.",
            "banana",
            "the blackened banana",
            {"is_blackened": True},
            {},
        )
    if any(term in q for term in ["unopened", "sealed"]) and any(term in q for term in ["drink", "bottle", "can", "juice"]):
        return PseudoTaskSpec(
            "state_filtering",
            "Pick the drink that is unopened.",
            "bottle" if "bottle" in q else "can" if "can" in q else "juice_box" if "juice" in q else None,
            "the unopened drink container",
            {"is_opened": False},
            {},
        )
    if "empty" in q and "bottle" in q:
        return PseudoTaskSpec(
            "negation",
            "Pick the bottle that is not empty.",
            "bottle",
            "the bottle that is not empty",
            {"is_empty": False},
            {},
        )
    if "pair" in q and "shoe" in q:
        return PseudoTaskSpec(
            "group_grounding",
            "Pick the farthest pair of shoes.",
            "shoe",
            "a pair of shoes",
            {},
            {"group_type": "pair_of_shoes"},
        )
    if "coffee" in q and ("mug" in q or "cup" in q):
        return PseudoTaskSpec(
            "affordance",
            "Pick the object suitable for drinking coffee.",
            "mug" if "mug" in q else "cup",
            "a coffee-suitable cup or mug",
            {"is_broken": False},
            {},
        )
    if "clean" in q and ("cup" in q or "mug" in q):
        return PseudoTaskSpec(
            "attribute_comparison",
            "Pick the cleanest cup.",
            "cup",
            "the cleanest cup",
            {},
            {"cleanliness": 0.7},
        )
    if "drink" in q or "beverage" in q or "bottle" in q:
        return PseudoTaskSpec(
            "attribute_comparison",
            "Pick the largest drink.",
            "bottle",
            "the largest drink",
            {},
            {},
        )
    if "fruit" in q or any(term in q for term in ["apple", "orange", "banana"]):
        return PseudoTaskSpec(
            "category_taxonomy",
            "Pick the edible fruit.",
            "apple" if "apple" in q else "orange" if "orange" in q else "banana" if "banana" in q else None,
            "an edible fruit",
            {"is_edible": True},
            {},
        )
    return PseudoTaskSpec(
        "open_vocab",
        "Pick the object described by the query.",
        None,
        query,
        {},
        {},
    )


def image_heuristics(path: str) -> dict[str, Any]:
    mean = mean_rgb(path)
    if mean is None:
        return {"confidence": 0.0, "attributes": {}, "states": {}}
    r, g, b = mean
    yellow_score = max(0.0, min(1.0, ((r + g) / 2 - b) / 255))
    dark_score = max(0.0, min(1.0, (255 - (r + g + b) / 3) / 255))
    attributes: dict[str, Any] = {}
    states: dict[str, Any] = {}
    confidence = 0.48
    if yellow_score > 0.25:
        attributes["yellow_region_score"] = yellow_score
        states.setdefault("is_blackened", dark_score > 0.45)
        confidence = 0.55
    attributes["brightness"] = (r + g + b) / (3 * 255)
    return {"confidence": confidence, "attributes": attributes, "states": states}
