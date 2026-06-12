from __future__ import annotations

import random
from dataclasses import asdict, dataclass, field
from typing import Any

from .groups import ObjectGroup, build_group_bbox, build_group_center, group_from_dict
from .objects import BBox, ObjectInstance, Point, object_from_dict
from .states import banana_states, cup_states, drink_states, shoe_states
from .taxonomy import get_super_categories


@dataclass
class SceneEvent:
    event_type: str
    object_id: str
    timestamp: int
    description: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Scene:
    id: str
    width: int
    height: int
    objects: list[ObjectInstance]
    groups: list[ObjectGroup]
    history: list[SceneEvent] = field(default_factory=list)

    def object_by_id(self, object_id: str) -> ObjectInstance | None:
        return next((obj for obj in self.objects if obj.id == object_id), None)

    def group_by_id(self, group_id: str) -> ObjectGroup | None:
        return next((grp for grp in self.groups if grp.id == group_id), None)

    def entity_by_id(self, entity_id: str) -> ObjectInstance | ObjectGroup | None:
        return self.object_by_id(entity_id) or self.group_by_id(entity_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "width": self.width,
            "height": self.height,
            "objects": [obj.to_dict() for obj in self.objects],
            "groups": [grp.to_dict() for grp in self.groups],
            "history": [event.to_dict() for event in self.history],
        }


def scene_from_dict(data: dict[str, Any]) -> Scene:
    return Scene(
        id=data["id"],
        width=int(data.get("width", 640)),
        height=int(data.get("height", 480)),
        objects=[object_from_dict(obj) for obj in data.get("objects", [])],
        groups=[group_from_dict(grp) for grp in data.get("groups", [])],
        history=[SceneEvent(**event) for event in data.get("history", [])],
    )


def _bbox(center: Point, w: float = 42, h: float = 32) -> BBox:
    x, y = center
    return (x - w / 2, y - h / 2, x + w / 2, y + h / 2)


def _obj(
    object_id: str,
    category: str,
    color: str,
    center: Point,
    size: float,
    *,
    shape: str | None = None,
    material: str | None = None,
    attributes: dict[str, Any] | None = None,
    states: dict[str, Any] | None = None,
    group_id: str | None = None,
    history_tags: list[str] | None = None,
) -> ObjectInstance:
    attributes = attributes or {}
    states = states or {}
    return ObjectInstance(
        id=object_id,
        category=category,
        super_categories=get_super_categories(category),
        color=color,
        shape=shape,
        material=material,
        bbox=_bbox(center, 48 + size * 40, 34 + size * 30),
        center=center,
        size=size,
        attributes=attributes,
        states=states,
        group_id=group_id,
        history_tags=history_tags or [],
    )


def _make_shoe_pair(pair_id: str, x: float, y: float, cleanliness: float) -> tuple[list[ObjectInstance], ObjectGroup]:
    left = _obj(
        f"{pair_id}_left",
        "shoe",
        "black",
        (x - 22, y),
        0.18,
        shape="shoe",
        material="leather",
        attributes={"side": "left", "cleanliness": cleanliness},
        states=shoe_states(True),
        group_id=pair_id,
    )
    right = _obj(
        f"{pair_id}_right",
        "shoe",
        "black",
        (x + 22, y + 4),
        0.18,
        shape="shoe",
        material="leather",
        attributes={"side": "right", "cleanliness": cleanliness + 0.03},
        states=shoe_states(True),
        group_id=pair_id,
    )
    bbox = build_group_bbox([left.bbox, right.bbox])
    center = build_group_center([left.center, right.center])
    group = ObjectGroup(
        id=pair_id,
        group_type="pair_of_shoes",
        member_ids=[left.id, right.id],
        category="shoe_pair",
        super_categories=["footwear"],
        center=center,
        attributes={"cleanliness": cleanliness, "member_count": 2},
        states={"is_wearable": True},
        bbox=bbox,
    )
    return [left, right], group


def generate_scene(seed: int = 0, objects_per_scene: int = 12, scene_id: str | None = None) -> Scene:
    rng = random.Random(seed)
    jitter = lambda scale=8: rng.uniform(-scale, scale)
    objects: list[ObjectInstance] = []
    groups: list[ObjectGroup] = []

    banana_specs = [
        ("banana_1", (90 + jitter(), 250 + jitter()), 0.64, 0.06),
        ("banana_2", (175 + jitter(), 235 + jitter()), 0.72, 0.18),
        ("banana_3", (270 + jitter(), 250 + jitter()), 0.98, 0.78),
    ]
    for object_id, center, ripeness, spots in banana_specs:
        objects.append(
            _obj(
                object_id,
                "banana",
                "yellow" if spots < 0.35 else "brown",
                center,
                0.24,
                shape="curved",
                material="fruit_skin",
                attributes={"ripeness": ripeness, "black_spot_ratio": spots},
                states=banana_states(ripeness, spots),
            )
        )

    objects.extend(
        [
            _obj(
                "apple_1",
                "apple",
                "red",
                (355 + jitter(), 235 + jitter()),
                0.2,
                shape="round",
                material="fruit_skin",
                attributes={"ripeness": 0.76},
                states={"is_rotten": False, "is_edible": True},
            ),
            _obj(
                "orange_1",
                "orange",
                "orange",
                (515 + jitter(), 110 + jitter()),
                0.22,
                shape="round",
                material="fruit_skin",
                attributes={"ripeness": 0.82},
                states={"is_rotten": False, "is_edible": True},
            ),
            _obj(
                "trash_bin_1",
                "trash_bin",
                "gray",
                (555 + jitter(), 395 + jitter()),
                0.55,
                shape="bin",
                material="plastic",
                attributes={"odor_level": 0.5},
                states={"is_opened": True},
            ),
        ]
    )

    drink_specs = [
        ("bottle_1", "bottle", "green", (90 + jitter(), 95 + jitter()), 0.31, 750, "tea", False, 0.80),
        ("soda_can_1", "soda_can", "red", (175 + jitter(), 95 + jitter()), 0.16, 330, "soda", False, 0.95),
        ("juice_box_1", "juice_box", "blue", (260 + jitter(), 100 + jitter()), 0.26, 1000, "juice", True, 0.35),
        ("water_bottle_1", "water_bottle", "clear", (342 + jitter(), 95 + jitter()), 0.18, 500, "water", False, 0.02),
    ]
    for object_id, category, color, center, size, volume, liquid, opened, fill in drink_specs:
        objects.append(
            _obj(
                object_id,
                category,
                color,
                center,
                size,
                shape="cylinder" if "box" not in category else "box",
                material="plastic" if "bottle" in category else "metal" if "can" in category else "paper",
                attributes={"volume_ml": volume, "brand": "generic", "liquid_type": liquid},
                states=drink_states(opened, fill),
            )
        )

    cup_specs = [
        ("cup_1", "cup", "white", (430 + jitter(), 250 + jitter()), 0.19, 0.92, False),
        ("cup_2", "cup", "gray", (500 + jitter(), 255 + jitter()), 0.18, 0.28, False),
        ("mug_1", "mug", "red", (430 + jitter(), 330 + jitter()), 0.25, 0.78, False),
    ]
    for object_id, category, color, center, size, clean, broken in cup_specs:
        objects.append(
            _obj(
                object_id,
                category,
                color,
                center,
                size,
                shape="cylinder",
                material="ceramic" if category == "mug" else "plastic",
                attributes={"cleanliness": clean, "capacity_ml": 250 if category == "cup" else 350},
                states=cup_states(broken),
            )
        )

    for pair_id, x, y, clean in [
        ("shoe_pair_1", 565 + jitter(), 62 + jitter(), 0.78),
        ("shoe_pair_2", 112 + jitter(), 410 + jitter(), 0.42),
    ]:
        pair_objects, group = _make_shoe_pair(pair_id, x, y, clean)
        objects.extend(pair_objects)
        groups.append(group)

    stack_books = [
        _obj("book_1", "book", "blue", (300 + jitter(), 385 + jitter()), 0.2, shape="rectangle", material="paper"),
        _obj("book_2", "book", "green", (305 + jitter(), 365 + jitter()), 0.18, shape="rectangle", material="paper"),
    ]
    objects.extend(stack_books)
    groups.append(
        ObjectGroup(
            id="book_stack_1",
            group_type="stack_of_books",
            member_ids=[book.id for book in stack_books],
            category="book_stack",
            super_categories=["readable_object"],
            center=build_group_center([book.center for book in stack_books]),
            attributes={"member_count": 2},
            states={},
            bbox=build_group_bbox([book.bbox for book in stack_books]),
        )
    )

    objects.extend(
        [
            _obj("spoon_1", "spoon", "silver", (385 + jitter(), 252 + jitter()), 0.1, shape="slender", material="metal"),
            _obj("bowl_1", "bowl", "white", (365 + jitter(), 325 + jitter()), 0.28, shape="round", material="ceramic"),
            _obj(
                "remote_1",
                "remote",
                "black",
                (230 + jitter(), 365 + jitter()),
                0.16,
                shape="rectangle",
                material="plastic",
                history_tags=["just_put_down"],
            ),
        ]
    )

    # Keep the requested count as a soft minimum so every benchmark task remains feasible.
    if objects_per_scene > len(objects):
        categories = ["apple", "orange", "can", "cup"]
        for idx in range(objects_per_scene - len(objects)):
            category = categories[idx % len(categories)]
            objects.append(
                _obj(
                    f"{category}_extra_{idx}",
                    category,
                    rng.choice(["red", "yellow", "green", "blue", "white"]),
                    (rng.uniform(60, 580), rng.uniform(60, 430)),
                    rng.uniform(0.12, 0.32),
                    attributes={"cleanliness": rng.random()} if category == "cup" else {},
                    states=cup_states(False) if category == "cup" else {"is_edible": category in {"apple", "orange"}},
                )
            )

    history = [
        SceneEvent(
            event_type="put_down",
            object_id="remote_1",
            timestamp=100,
            description="The user just put down remote_1.",
        )
    ]
    return Scene(
        id=scene_id or f"scene_{seed:06d}",
        width=640,
        height=480,
        objects=objects,
        groups=groups,
        history=history,
    )

