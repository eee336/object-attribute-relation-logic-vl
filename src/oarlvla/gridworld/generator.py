from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from oarlvla.evaluation import sample_to_jsonl_row
from oarlvla.groups import ObjectGroup, build_group_bbox, build_group_center
from oarlvla.instruction import TASK_TYPES, InstructionExample, generate_instruction
from oarlvla.objects import ObjectInstance
from oarlvla.reasoning import LogicAwareReasoner
from oarlvla.scene import Scene, SceneEvent
from oarlvla.states import banana_states, cup_states, drink_states, shoe_states
from oarlvla.taxonomy import get_super_categories

from .renderer import render_grid_scene


def generate_grid_scene(seed: int = 0, grid_size: int = 8, cell_size: int = 64, scene_id: str | None = None) -> Scene:
    rng = random.Random(seed)
    width = grid_size * cell_size
    height = grid_size * cell_size
    objects: list[ObjectInstance] = []
    groups: list[ObjectGroup] = []

    def cell(col: int, row: int) -> tuple[float, float]:
        jitter = cell_size * 0.07
        return ((col + 0.5) * cell_size + rng.uniform(-jitter, jitter), (row + 0.5) * cell_size + rng.uniform(-jitter, jitter))

    def add(
        object_id: str,
        category: str,
        col: int,
        row: int,
        *,
        color: str,
        size: float = 0.22,
        shape: str | None = None,
        material: str | None = None,
        attributes: dict[str, Any] | None = None,
        states: dict[str, Any] | None = None,
        group_id: str | None = None,
        history_tags: list[str] | None = None,
    ) -> ObjectInstance:
        center = cell(col, row)
        sprite = cell_size * 0.58
        bbox = (center[0] - sprite / 2, center[1] - sprite / 2, center[0] + sprite / 2, center[1] + sprite / 2)
        obj = ObjectInstance(
            id=object_id,
            category=category,
            super_categories=get_super_categories(category),
            color=color,
            shape=shape,
            material=material,
            bbox=bbox,
            center=center,
            size=size,
            attributes=attributes or {},
            states=states or {},
            group_id=group_id,
            history_tags=history_tags or [],
        )
        objects.append(obj)
        return obj

    banana_specs = [
        ("banana_1", 1, 4, 0.60, 0.05),
        ("banana_2", 2, 4, 0.78, 0.18),
        ("banana_3", 3, 4, 0.98, 0.76),
    ]
    for object_id, col, row, ripeness, spots in banana_specs:
        add(
            object_id,
            "banana",
            col,
            row,
            color="yellow" if spots < 0.35 else "brown",
            shape="curved",
            material="fruit_skin",
            attributes={"ripeness": ripeness, "black_spot_ratio": spots},
            states=banana_states(ripeness, spots),
        )

    add("apple_1", "apple", 5, 4, color="red", shape="round", material="fruit_skin", states={"is_rotten": False, "is_edible": True})
    add("orange_1", "orange", 6, 1, color="orange", shape="round", material="fruit_skin", states={"is_rotten": False, "is_edible": True})
    add("trash_bin_1", "trash_bin", 7, 7, color="gray", shape="bin", material="plastic", size=0.5, states={"is_opened": True})

    add("bottle_1", "bottle", 1, 1, color="green", shape="cylinder", material="plastic", size=0.31, attributes={"volume_ml": 750, "liquid_type": "tea"}, states=drink_states(False, 0.80))
    add("soda_can_1", "soda_can", 2, 1, color="red", shape="cylinder", material="metal", size=0.16, attributes={"volume_ml": 330, "liquid_type": "soda"}, states=drink_states(False, 0.95))
    add("juice_box_1", "juice_box", 3, 1, color="blue", shape="box", material="paper", size=0.26, attributes={"volume_ml": 1000, "liquid_type": "juice"}, states=drink_states(True, 0.35))
    add("water_bottle_1", "water_bottle", 4, 1, color="clear", shape="cylinder", material="plastic", size=0.18, attributes={"volume_ml": 500, "liquid_type": "water"}, states=drink_states(False, 0.02))

    add("spoon_1", "spoon", 4, 3, color="silver", shape="slender", material="metal", size=0.10)
    add("cup_1", "cup", 5, 3, color="white", shape="cylinder", material="plastic", size=0.19, attributes={"cleanliness": 0.92, "capacity_ml": 250}, states=cup_states(False))
    add("cup_2", "cup", 6, 3, color="gray", shape="cylinder", material="plastic", size=0.18, attributes={"cleanliness": 0.25, "capacity_ml": 250}, states=cup_states(False))
    add("mug_1", "mug", 5, 5, color="red", shape="cylinder", material="ceramic", size=0.25, attributes={"cleanliness": 0.78, "capacity_ml": 350}, states=cup_states(False))
    add("bowl_1", "bowl", 4, 5, color="white", shape="round", material="ceramic", size=0.28)

    pair1 = [
        add("shoe_pair_1_left", "shoe", 6, 6, color="black", shape="shoe", material="leather", size=0.18, attributes={"side": "left", "cleanliness": 0.75}, states=shoe_states(True), group_id="shoe_pair_1"),
        add("shoe_pair_1_right", "shoe", 7, 6, color="black", shape="shoe", material="leather", size=0.18, attributes={"side": "right", "cleanliness": 0.78}, states=shoe_states(True), group_id="shoe_pair_1"),
    ]
    pair2 = [
        add("shoe_pair_2_left", "shoe", 0, 0, color="black", shape="shoe", material="leather", size=0.18, attributes={"side": "left", "cleanliness": 0.42}, states=shoe_states(True), group_id="shoe_pair_2"),
        add("shoe_pair_2_right", "shoe", 1, 0, color="black", shape="shoe", material="leather", size=0.18, attributes={"side": "right", "cleanliness": 0.45}, states=shoe_states(True), group_id="shoe_pair_2"),
    ]
    for pair_id, members, clean in [("shoe_pair_1", pair1, 0.76), ("shoe_pair_2", pair2, 0.43)]:
        groups.append(
            ObjectGroup(
                id=pair_id,
                group_type="pair_of_shoes",
                member_ids=[obj.id for obj in members],
                category="shoe_pair",
                super_categories=["footwear"],
                center=build_group_center([obj.center for obj in members]),
                attributes={"cleanliness": clean, "member_count": 2},
                states={"is_wearable": True},
                bbox=build_group_bbox([obj.bbox for obj in members]),
            )
        )

    books = [
        add("book_1", "book", 2, 6, color="blue", shape="rectangle", material="paper", size=0.20),
        add("book_2", "book", 2, 7, color="green", shape="rectangle", material="paper", size=0.18),
    ]
    groups.append(
        ObjectGroup(
            id="book_stack_1",
            group_type="stack_of_books",
            member_ids=[book.id for book in books],
            category="book_stack",
            super_categories=["readable_object"],
            center=build_group_center([book.center for book in books]),
            attributes={"member_count": 2},
            states={},
            bbox=build_group_bbox([book.bbox for book in books]),
        )
    )
    add("remote_1", "remote", 0, 5, color="black", shape="rectangle", material="plastic", size=0.16, history_tags=["just_put_down"])

    history = [SceneEvent("put_down", "remote_1", 100, "The user just put down remote_1 on the grid.")]
    return Scene(scene_id or f"grid_{seed:06d}", width, height, objects, groups, history)


def generate_grid_sample(scene: Scene, task_type: str, seed: int, image_path: str | Path) -> dict[str, Any]:
    example = generate_instruction(scene, task_type, seed)
    if example.target_id is None:
        result = LogicAwareReasoner().reason(scene, "Pick the banana that has not turned black.")
        example = InstructionExample(
            instruction="Pick the banana that has not turned black.",
            program=result.program,
            target_id=result.target_id,
            target_type=result.target_type,
            task_type="state_filtering",
            reasoning_steps=result.steps,
        )
    row = sample_to_jsonl_row(scene, example)
    row["sample_id"] = scene.id
    row["image_path"] = str(image_path)
    row["source"] = "synthetic_grid"
    row["label_quality"] = "gold"
    target = scene.entity_by_id(example.target_id) if example.target_id else None
    row["target_bbox"] = list(target.bbox) if target and getattr(target, "bbox", None) else None
    row["target_center"] = list(target.center) if target else None
    return row


def generate_grid_dataset(
    *,
    num_scenes: int,
    grid_size: int,
    cell_size: int,
    seed: int,
    output: str | Path,
    image_dir: str | Path,
) -> dict[str, Any]:
    output = Path(output)
    image_dir = Path(image_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)
    task_counts = {task: 0 for task in TASK_TYPES}
    with output.open("w", encoding="utf-8") as f:
        for idx in range(num_scenes):
            scene = generate_grid_scene(seed + idx, grid_size=grid_size, cell_size=cell_size, scene_id=f"grid_{idx:06d}")
            image_path = image_dir / f"{scene.id}.png"
            render_grid_scene(scene, image_path, grid_size, cell_size)
            task_type = TASK_TYPES[idx % len(TASK_TYPES)]
            row = generate_grid_sample(scene, task_type, seed + idx, image_path)
            row["grid_size"] = grid_size
            row["cell_size"] = cell_size
            task_counts[row["task_type"]] = task_counts.get(row["task_type"], 0) + 1
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {
        "output": str(output),
        "image_dir": str(image_dir),
        "num_samples": num_scenes,
        "grid_size": grid_size,
        "cell_size": cell_size,
        "task_counts": {key: value for key, value in task_counts.items() if value},
    }
