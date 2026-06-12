from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


BBox = tuple[float, float, float, float]
Point = tuple[float, float]


@dataclass
class ObjectInstance:
    id: str
    category: str
    super_categories: list[str]
    color: str
    shape: str | None
    material: str | None
    bbox: BBox
    center: Point
    size: float
    attributes: dict[str, Any]
    states: dict[str, Any]
    group_id: str | None = None
    container_id: str | None = None
    history_tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def object_from_dict(data: dict[str, Any]) -> ObjectInstance:
    return ObjectInstance(
        id=data["id"],
        category=data["category"],
        super_categories=list(data.get("super_categories", [])),
        color=data.get("color", "unknown"),
        shape=data.get("shape"),
        material=data.get("material"),
        bbox=tuple(data["bbox"]),  # type: ignore[arg-type]
        center=tuple(data["center"]),  # type: ignore[arg-type]
        size=float(data.get("size", 0.0)),
        attributes=dict(data.get("attributes", {})),
        states=dict(data.get("states", {})),
        group_id=data.get("group_id"),
        container_id=data.get("container_id"),
        history_tags=list(data.get("history_tags", [])),
    )

