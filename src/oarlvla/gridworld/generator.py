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
    occupied: set[tuple[int, int]] = set()
    category_counts: dict[str, int] = {}

    def free_cells() -> list[tuple[int, int]]:
        cells = [(col, row) for row in range(grid_size) for col in range(grid_size) if (col, row) not in occupied]
        rng.shuffle(cells)
        return cells

    def claim(cell: tuple[int, int]) -> tuple[int, int]:
        occupied.add(cell)
        return cell

    def reserve_cell(candidates: list[tuple[int, int]] | None = None) -> tuple[int, int]:
        pool = candidates if candidates is not None else free_cells()
        pool = [cell for cell in pool if 0 <= cell[0] < grid_size and 0 <= cell[1] < grid_size and cell not in occupied]
        if not pool:
            raise RuntimeError("No free grid cells left while generating a scene.")
        rng.shuffle(pool)
        return claim(pool[0])

    def reserve_ordered_pair() -> tuple[tuple[int, int], tuple[int, int]]:
        candidates: list[tuple[tuple[int, int], tuple[int, int]]] = []
        available = [cell for cell in free_cells()]
        for left in available:
            for right in available:
                if left == right:
                    continue
                if right[0] > left[0]:
                    candidates.append((left, right))
        if not candidates:
            raise RuntimeError("Could not reserve ordered grid cells.")
        rng.shuffle(candidates)
        left, right = candidates[0]
        claim(left)
        claim(right)
        return left, right

    def reserve_adjacent_pair() -> tuple[tuple[int, int], tuple[int, int]]:
        candidates: list[tuple[tuple[int, int], tuple[int, int]]] = []
        for row in range(grid_size):
            for col in range(grid_size):
                for dx, dy in ((1, 0), (0, 1)):
                    a = (col, row)
                    b = (col + dx, row + dy)
                    if b[0] < grid_size and b[1] < grid_size and a not in occupied and b not in occupied:
                        candidates.append((a, b))
        if not candidates:
            raise RuntimeError("Could not reserve adjacent grid cells.")
        rng.shuffle(candidates)
        a, b = candidates[0]
        claim(a)
        claim(b)
        return a, b

    def reserve_far_from(ref: tuple[int, int], min_distance_cells: float = 2.0) -> tuple[int, int]:
        candidates = []
        for cell_candidate in free_cells():
            dx = cell_candidate[0] - ref[0]
            dy = cell_candidate[1] - ref[1]
            if (dx * dx + dy * dy) ** 0.5 >= min_distance_cells:
                candidates.append(cell_candidate)
        return reserve_cell(candidates or None)

    def cell(col: int, row: int) -> tuple[float, float]:
        jitter = cell_size * 0.16
        return ((col + 0.5) * cell_size + rng.uniform(-jitter, jitter), (row + 0.5) * cell_size + rng.uniform(-jitter, jitter))

    def add(
        category: str,
        grid_cell: tuple[int, int],
        *,
        object_id: str | None = None,
        color: str,
        size: float = 0.22,
        shape: str | None = None,
        material: str | None = None,
        attributes: dict[str, Any] | None = None,
        states: dict[str, Any] | None = None,
        group_id: str | None = None,
        history_tags: list[str] | None = None,
    ) -> ObjectInstance:
        category_counts[category] = category_counts.get(category, 0) + 1
        if object_id is None:
            object_id = f"{category}_{category_counts[category]}"
        col, row = grid_cell
        center = cell(col, row)
        sprite = cell_size * rng.uniform(0.50, 0.70)
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

    trash_cell = reserve_cell()
    add("trash_bin", trash_cell, object_id="trash_bin_1", color="gray", shape="bin", material="plastic", size=0.5, states={"is_opened": True})

    bottle_cell, orange_cell = reserve_ordered_pair()
    bottle_fill = rng.uniform(0.35, 0.95)
    add(
        "bottle",
        bottle_cell,
        object_id="bottle_1",
        color="green",
        shape="cylinder",
        material="plastic",
        size=0.31,
        attributes={"volume_ml": 750, "liquid_type": "tea"},
        states=drink_states(rng.choice([False, False, True]), bottle_fill),
    )
    add("orange", orange_cell, object_id="orange_1", color="orange", shape="round", material="fruit_skin", states={"is_rotten": False, "is_edible": True})

    spoon_cell, cup_cell = reserve_ordered_pair()
    add("spoon", spoon_cell, object_id="spoon_1", color="silver", shape="slender", material="metal", size=0.10)
    add(
        "cup",
        cup_cell,
        object_id="cup_1",
        color="white",
        shape="cylinder",
        material="plastic",
        size=0.19,
        attributes={"cleanliness": rng.uniform(0.72, 0.98), "capacity_ml": 250},
        states=cup_states(False),
    )

    banana_specs = [
        ("banana_1", rng.uniform(0.52, 0.75), rng.uniform(0.02, 0.16)),
        ("banana_2", rng.uniform(0.72, 0.88), rng.uniform(0.16, 0.32)),
        ("banana_3", rng.uniform(0.96, 1.00), rng.uniform(0.70, 0.92)),
    ]
    rng.shuffle(banana_specs)
    for object_id, ripeness, spots in banana_specs:
        add(
            "banana",
            reserve_cell(),
            object_id=object_id,
            color="yellow" if spots < 0.35 else "brown",
            shape="curved",
            material="fruit_skin",
            attributes={"ripeness": ripeness, "black_spot_ratio": spots},
            states=banana_states(ripeness, spots),
        )

    add("apple", reserve_far_from(trash_cell), object_id="apple_1", color="red", shape="round", material="fruit_skin", states={"is_rotten": False, "is_edible": True})
    add("soda_can", reserve_cell(), object_id="soda_can_1", color="red", shape="cylinder", material="metal", size=0.16, attributes={"volume_ml": 330, "liquid_type": "soda"}, states=drink_states(rng.choice([False, True]), rng.uniform(0.25, 1.0)))
    add("juice_box", reserve_cell(), object_id="juice_box_1", color="blue", shape="box", material="paper", size=0.26, attributes={"volume_ml": 1000, "liquid_type": "juice"}, states=drink_states(rng.choice([False, True]), rng.uniform(0.15, 0.8)))
    add("water_bottle", reserve_cell(), object_id="water_bottle_1", color="clear", shape="cylinder", material="plastic", size=0.18, attributes={"volume_ml": 500, "liquid_type": "water"}, states=drink_states(False, rng.uniform(0.02, 0.55)))

    add(
        "cup",
        reserve_cell(),
        object_id="cup_2",
        color="gray",
        shape="cylinder",
        material="plastic",
        size=0.18,
        attributes={"cleanliness": rng.uniform(0.04, 0.35), "capacity_ml": 250},
        states=cup_states(False),
    )
    add(
        "mug",
        reserve_cell(),
        object_id="mug_1",
        color="red",
        shape="cylinder",
        material="ceramic",
        size=0.25,
        attributes={"cleanliness": rng.uniform(0.45, 0.9), "capacity_ml": 350},
        states=cup_states(False),
    )
    add("bowl", reserve_cell(), object_id="bowl_1", color="white", shape="round", material="ceramic", size=0.28)

    pair1_cells = reserve_adjacent_pair()
    pair2_cells = reserve_adjacent_pair()
    pair1_clean = rng.uniform(0.65, 0.96)
    pair2_clean = rng.uniform(0.18, 0.58)
    pair1 = [
        add("shoe", pair1_cells[0], object_id="shoe_pair_1_left", color="black", shape="shoe", material="leather", size=0.18, attributes={"side": "left", "cleanliness": pair1_clean}, states=shoe_states(True), group_id="shoe_pair_1"),
        add("shoe", pair1_cells[1], object_id="shoe_pair_1_right", color="black", shape="shoe", material="leather", size=0.18, attributes={"side": "right", "cleanliness": min(0.99, pair1_clean + rng.uniform(-0.04, 0.04))}, states=shoe_states(True), group_id="shoe_pair_1"),
    ]
    pair2 = [
        add("shoe", pair2_cells[0], object_id="shoe_pair_2_left", color="black", shape="shoe", material="leather", size=0.18, attributes={"side": "left", "cleanliness": pair2_clean}, states=shoe_states(True), group_id="shoe_pair_2"),
        add("shoe", pair2_cells[1], object_id="shoe_pair_2_right", color="black", shape="shoe", material="leather", size=0.18, attributes={"side": "right", "cleanliness": min(0.99, pair2_clean + rng.uniform(-0.04, 0.04))}, states=shoe_states(True), group_id="shoe_pair_2"),
    ]
    for pair_id, members, clean in [("shoe_pair_1", pair1, pair1_clean), ("shoe_pair_2", pair2, pair2_clean)]:
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

    book_cells = reserve_adjacent_pair()
    books = [
        add("book", book_cells[0], object_id="book_1", color="blue", shape="rectangle", material="paper", size=0.20),
        add("book", book_cells[1], object_id="book_2", color="green", shape="rectangle", material="paper", size=0.18),
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

    add("remote", reserve_cell(), object_id="remote_1", color="black", shape="rectangle", material="plastic", size=0.16, history_tags=["just_put_down"])

    optional_specs = [
        ("apple", {"color": "red", "shape": "round", "material": "fruit_skin", "states": {"is_rotten": False, "is_edible": True}}),
        ("banana", {"color": "yellow", "shape": "curved", "material": "fruit_skin", "attributes": {"ripeness": 0.70, "black_spot_ratio": 0.12}, "states": banana_states(0.70, 0.12)}),
        ("cup", {"color": "white", "shape": "cylinder", "material": "plastic", "attributes": {"cleanliness": rng.uniform(0.2, 0.95), "capacity_ml": 250}, "states": cup_states(False)}),
        ("book", {"color": "blue", "shape": "rectangle", "material": "paper"}),
    ]
    for category, kwargs in rng.sample(optional_specs, k=rng.randint(0, min(3, len(optional_specs)))):
        if free_cells():
            add(category, reserve_cell(), **kwargs)

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
    asset_dir: str | Path | None = None,
    render_group_boxes: bool = False,
) -> dict[str, Any]:
    output = Path(output)
    image_dir = Path(image_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)
    asset_dir = Path(asset_dir) if asset_dir else image_dir.parent / "grid_assets"
    task_counts = {task: 0 for task in TASK_TYPES}
    with output.open("w", encoding="utf-8") as f:
        for idx in range(num_scenes):
            scene = generate_grid_scene(seed + idx, grid_size=grid_size, cell_size=cell_size, scene_id=f"grid_{idx:06d}")
            image_path = image_dir / f"{scene.id}.png"
            render_grid_scene(scene, image_path, grid_size, cell_size, asset_dir=asset_dir, render_group_boxes=render_group_boxes)
            task_type = TASK_TYPES[idx % len(TASK_TYPES)]
            row = generate_grid_sample(scene, task_type, seed + idx, image_path)
            row["grid_size"] = grid_size
            row["cell_size"] = cell_size
            task_counts[row["task_type"]] = task_counts.get(row["task_type"], 0) + 1
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {
        "output": str(output),
        "image_dir": str(image_dir),
        "asset_dir": str(asset_dir),
        "render_group_boxes": render_group_boxes,
        "num_samples": num_scenes,
        "grid_size": grid_size,
        "cell_size": cell_size,
        "task_counts": {key: value for key, value in task_counts.items() if value},
    }
