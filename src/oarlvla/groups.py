from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .objects import BBox, Point


@dataclass
class ObjectGroup:
    id: str
    group_type: str
    member_ids: list[str]
    category: str
    super_categories: list[str]
    center: Point
    attributes: dict[str, Any]
    states: dict[str, Any]
    bbox: BBox | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def group_from_dict(data: dict[str, Any]) -> ObjectGroup:
    bbox = data.get("bbox")
    return ObjectGroup(
        id=data["id"],
        group_type=data["group_type"],
        member_ids=list(data.get("member_ids", [])),
        category=data.get("category", data["group_type"]),
        super_categories=list(data.get("super_categories", [])),
        center=tuple(data["center"]),  # type: ignore[arg-type]
        attributes=dict(data.get("attributes", {})),
        states=dict(data.get("states", {})),
        bbox=tuple(bbox) if bbox else None,  # type: ignore[arg-type]
    )


def build_group_bbox(member_bboxes: list[BBox]) -> BBox:
    x1 = min(b[0] for b in member_bboxes)
    y1 = min(b[1] for b in member_bboxes)
    x2 = max(b[2] for b in member_bboxes)
    y2 = max(b[3] for b in member_bboxes)
    return (x1, y1, x2, y2)


def build_group_center(member_centers: list[Point]) -> Point:
    return (
        sum(p[0] for p in member_centers) / len(member_centers),
        sum(p[1] for p in member_centers) / len(member_centers),
    )

