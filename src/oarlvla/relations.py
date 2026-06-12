from __future__ import annotations

from dataclasses import dataclass, field
from math import dist
from typing import Any, Union

from .groups import ObjectGroup
from .objects import ObjectInstance
from .scene import Scene
from .taxonomy import normalize_term


Entity = Union[ObjectInstance, ObjectGroup]


@dataclass
class RelationGraph:
    relations: dict[str, dict[str, list[str]]] = field(default_factory=dict)

    def add(self, source_id: str, relation: str, target_id: str) -> None:
        self.relations.setdefault(source_id, {}).setdefault(relation, []).append(target_id)

    def targets(self, source_id: str, relation: str) -> list[str]:
        return self.relations.get(source_id, {}).get(relation, [])

    def to_dict(self) -> dict[str, Any]:
        return self.relations


def entity_distance(a: Entity, b: Entity) -> float:
    return dist(a.center, b.center)


def compute_spatial_relations(scene: Scene, near_threshold: float = 105.0) -> RelationGraph:
    graph = RelationGraph()
    entities: list[Entity] = [*scene.objects, *scene.groups]
    for a in entities:
        for b in entities:
            if a.id == b.id:
                continue
            if a.center[0] < b.center[0]:
                graph.add(a.id, "left_of", b.id)
            if a.center[0] > b.center[0]:
                graph.add(a.id, "right_of", b.id)
            if a.center[1] < b.center[1]:
                graph.add(a.id, "above", b.id)
            if a.center[1] > b.center[1]:
                graph.add(a.id, "below", b.id)
            d = entity_distance(a, b)
            if d <= near_threshold:
                graph.add(a.id, "near", b.id)
            if d >= near_threshold * 2:
                graph.add(a.id, "far", b.id)
    for obj in scene.objects:
        if obj.container_id:
            graph.add(obj.id, "inside", obj.container_id)
            graph.add(obj.container_id, "contains", obj.id)
    return graph


def sort_objects(objects: list[ObjectInstance], direction: str) -> list[ObjectInstance]:
    reverse = direction in {"right_to_left", "right", "rightmost"}
    return sorted(objects, key=lambda obj: obj.center[0], reverse=reverse)


def filter_objects(
    scene: Scene,
    category: str | None = None,
    super_category: str | None = None,
    color: str | None = None,
) -> list[ObjectInstance]:
    category_norm = normalize_term(category) if category else None
    super_norm = normalize_term(super_category) if super_category else None
    out: list[ObjectInstance] = []
    for obj in scene.objects:
        if category_norm and obj.category != category_norm:
            if category_norm not in obj.super_categories:
                continue
        if super_norm and super_norm not in obj.super_categories and obj.category != super_norm:
            continue
        if color and obj.color != color:
            continue
        out.append(obj)
    return out


def filter_groups(scene: Scene, group_type: str | None = None, super_category: str | None = None) -> list[ObjectGroup]:
    super_norm = normalize_term(super_category) if super_category else None
    return [
        group
        for group in scene.groups
        if (group_type is None or group.group_type == group_type)
        and (super_norm is None or super_norm in group.super_categories or group.category == super_norm)
    ]


def select_nth(objects: list[ObjectInstance], n: int, direction: str) -> ObjectInstance | None:
    ordered = sort_objects(objects, direction)
    if 1 <= n <= len(ordered):
        return ordered[n - 1]
    return None


def select_nearest(source: Entity, candidates: list[Entity]) -> Entity | None:
    if not candidates:
        return None
    return min(candidates, key=lambda candidate: entity_distance(source, candidate))


def select_farthest(source: Entity, candidates: list[Entity]) -> Entity | None:
    if not candidates:
        return None
    return max(candidates, key=lambda candidate: entity_distance(source, candidate))


def entities_left_of(candidates: list[ObjectInstance], reference: Entity) -> list[ObjectInstance]:
    return [obj for obj in candidates if obj.center[0] < reference.center[0]]


def entities_right_of(candidates: list[ObjectInstance], reference: Entity) -> list[ObjectInstance]:
    return [obj for obj in candidates if obj.center[0] > reference.center[0]]


def not_near(candidates: list[ObjectInstance], references: list[Entity], threshold: float = 105.0) -> list[ObjectInstance]:
    return [
        candidate
        for candidate in candidates
        if all(entity_distance(candidate, reference) > threshold for reference in references)
    ]


def between(candidates: list[ObjectInstance], a: Entity, b: Entity) -> list[ObjectInstance]:
    min_x, max_x = sorted([a.center[0], b.center[0]])
    min_y, max_y = sorted([a.center[1], b.center[1]])
    return [
        obj
        for obj in candidates
        if min_x <= obj.center[0] <= max_x and min_y - 60 <= obj.center[1] <= max_y + 60
    ]
