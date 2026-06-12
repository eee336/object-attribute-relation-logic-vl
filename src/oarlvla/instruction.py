from __future__ import annotations

import random
from dataclasses import dataclass

from .reasoning import LogicAwareReasoner
from .scene import Scene


TASK_TYPES = [
    "spatial_relation",
    "ordinal_relation",
    "attribute_comparison",
    "state_filtering",
    "category_taxonomy",
    "group_grounding",
    "negation",
    "history_reference",
    "affordance",
]


TEMPLATES: dict[str, list[str]] = {
    "spatial_relation": [
        "Pick the banana nearest to the trash_bin.",
        "Pick the banana farthest to the trash_bin.",
        "Pick the spoon left of the cup.",
        "Pick the orange right of the bottle.",
    ],
    "ordinal_relation": [
        "Pick the 1st banana from left to right.",
        "Pick the 2nd banana from left to right.",
        "Pick the 1st banana from right to left.",
    ],
    "attribute_comparison": [
        "Pick the largest drink.",
        "Pick the smallest drink.",
        "Pick the cleanest cup.",
        "Pick the dirtiest cup.",
    ],
    "state_filtering": [
        "Pick the banana that has not turned black.",
        "Pick the blackened banana.",
        "Pick the banana that is not rotten.",
        "Pick the drink that is unopened.",
        "Pick the bottle that is not empty.",
    ],
    "category_taxonomy": [
        "Pick the largest drink.",
        "Pick the edible fruit.",
        "Pick the cleanest drinkware.",
    ],
    "group_grounding": [
        "Pick the farthest pair of shoes.",
        "Pick the nearest pair of shoes.",
        "Pick the cleanest pair of shoes.",
    ],
    "negation": [
        "Pick the fruit that is not near the trash bin.",
        "Pick the banana that is not blackened.",
        "Pick the bottle that is not empty.",
        "Pick the drink that is not opened.",
    ],
    "history_reference": [
        "Pick the object I just put down.",
        "Pick the object that was moved most recently.",
    ],
    "affordance": [
        "Pick the object suitable for drinking coffee.",
    ],
}


@dataclass
class InstructionExample:
    instruction: str
    program: str
    target_id: str | None
    target_type: str
    task_type: str
    reasoning_steps: list[str]


def generate_instruction(scene: Scene, instruction_type: str | None = None, seed: int = 0) -> InstructionExample:
    rng = random.Random(seed)
    task_type = instruction_type or rng.choice(TASK_TYPES)
    if task_type not in TEMPLATES:
        raise ValueError(f"Unsupported instruction type: {task_type}")
    reasoner = LogicAwareReasoner()
    templates = list(TEMPLATES[task_type])
    rng.shuffle(templates)
    last_result = None
    for instruction in templates:
        result = reasoner.reason(scene, instruction)
        last_result = result
        if result.target_id is not None:
            return InstructionExample(
                instruction=instruction,
                program=result.program,
                target_id=result.target_id,
                target_type=result.target_type,
                task_type=task_type,
                reasoning_steps=result.steps,
            )
    assert last_result is not None
    return InstructionExample(
        instruction=templates[0],
        program=last_result.program,
        target_id=last_result.target_id,
        target_type=last_result.target_type,
        task_type=task_type,
        reasoning_steps=last_result.steps,
    )

