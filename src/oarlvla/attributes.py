from __future__ import annotations

from typing import Any, Union

from .groups import ObjectGroup
from .objects import ObjectInstance


Entity = Union[ObjectInstance, ObjectGroup]


def get_value(entity: Entity, key: str, fallback: str | None = None) -> float | int | str | bool | None:
    if key == "size" and hasattr(entity, "size"):
        return getattr(entity, "size")
    if key == "distance_to_origin":
        x, y = entity.center
        return (x**2 + y**2) ** 0.5
    if key in entity.attributes:
        return entity.attributes[key]
    if key in entity.states:
        return entity.states[key]
    if fallback:
        return get_value(entity, fallback)
    return None


def numeric_value(entity: Entity, key: str, fallback: str | None = None) -> float:
    value = get_value(entity, key, fallback)
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    return float("-inf")


def cleanliness(entity: Entity) -> float:
    return numeric_value(entity, "cleanliness")


def black_spot_ratio(entity: Entity) -> float:
    return numeric_value(entity, "black_spot_ratio")


def fill_level(entity: Entity) -> float:
    return numeric_value(entity, "fill_level")


def merged_attributes(*parts: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for part in parts:
        merged.update(part)
    return merged
