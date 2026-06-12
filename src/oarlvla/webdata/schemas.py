from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from oarlvla.scene import Scene


LabelSource = Literal["manual", "model", "heuristic", "metadata"]


@dataclass
class WebImageRecord:
    image_id: str
    local_path: str
    source_name: str
    source_url: str
    license: str | None
    author: str | None
    query: str
    downloaded_at: str
    width: int
    height: int
    sha256: str
    split: str
    raw_metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ObjectAnnotation:
    id: str
    category: str
    super_categories: list[str]
    bbox: tuple[float, float, float, float] | None
    mask_path: str | None
    attributes: dict[str, Any]
    states: dict[str, Any]
    confidence: float
    source: LabelSource

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GroupAnnotation:
    id: str
    group_type: str
    member_ids: list[str]
    category: str
    bbox: tuple[float, float, float, float] | None
    confidence: float
    source: LabelSource

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TaskAnnotation:
    task_id: str
    task_type: str
    instruction: str
    program: str
    target_id: str | None
    target_type: Literal["object", "group", "unknown"]
    confidence: float
    source: LabelSource
    target_description: str = ""
    requires_manual_verification: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ImageAnnotationBundle:
    image_id: str
    objects: list[ObjectAnnotation] = field(default_factory=list)
    groups: list[GroupAnnotation] = field(default_factory=list)
    candidate_tasks: list[TaskAnnotation] = field(default_factory=list)
    pseudo_labels: list[dict[str, Any]] = field(default_factory=list)
    quality_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_id": self.image_id,
            "objects": [obj.to_dict() for obj in self.objects],
            "groups": [grp.to_dict() for grp in self.groups],
            "candidate_tasks": [task.to_dict() for task in self.candidate_tasks],
            "pseudo_labels": self.pseudo_labels,
            "quality_score": self.quality_score,
        }


@dataclass
class GroundingSample:
    sample_id: str
    image_path: str | None
    scene: Scene | None
    instruction: str
    program: str
    target_id: str | None
    target_type: str
    task_type: str
    label_quality: Literal["gold", "silver", "weak", "unknown"]
    source: Literal["synthetic", "web", "manual"]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["scene"] = self.scene.to_dict() if self.scene else None
        return data

