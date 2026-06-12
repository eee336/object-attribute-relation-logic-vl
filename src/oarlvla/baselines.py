from __future__ import annotations

import random
from dataclasses import dataclass

from .objects import ObjectInstance
from .parser import Program, parse_instruction
from .reasoning import LogicAwareReasoner, ProgramExecutor
from .scene import Scene


@dataclass
class BaselineResult:
    target_id: str | None
    target_type: str
    program: str
    steps: list[str]
    confidence: float = 0.25


class BaseBaseline:
    name = "Base"

    def __init__(self, seed: int = 0) -> None:
        self.rng = random.Random(seed)

    def predict(self, scene: Scene, instruction: str) -> BaselineResult:
        raise NotImplementedError

    def _choose(self, candidates: list[ObjectInstance], groups: bool = False) -> BaselineResult:
        if not candidates:
            return BaselineResult(None, "none", "", ["No candidates"], confidence=0.0)
        target = self.rng.choice(candidates)
        return BaselineResult(target.id, "group" if groups else "object", "", [f"Randomly selected {target.id}"])


class RandomObjectBaseline(BaseBaseline):
    name = "Random Object"

    def predict(self, scene: Scene, instruction: str) -> BaselineResult:
        entities = [*scene.objects, *scene.groups]
        if not entities:
            return BaselineResult(None, "none", "", ["No entities"], confidence=0.0)
        target = self.rng.choice(entities)
        target_type = "group" if target in scene.groups else "object"
        return BaselineResult(target.id, target_type, "random_entity()", [f"Randomly selected {target.id}"])


class RandomSameCategoryBaseline(BaseBaseline):
    name = "Random Same Category"

    def predict(self, scene: Scene, instruction: str) -> BaselineResult:
        candidates, target_type = _coarse_candidates(scene, instruction)
        if not candidates:
            return RandomObjectBaseline(self.rng.randint(0, 9999)).predict(scene, instruction)
        target = self.rng.choice(candidates)
        return BaselineResult(target.id, target_type, "coarse_category()->random()", [f"Coarse candidates: {[c.id for c in candidates]}"])


class AttributeIgnorantBaseline(BaseBaseline):
    name = "Attribute-Ignorant"

    def predict(self, scene: Scene, instruction: str) -> BaselineResult:
        program = parse_instruction(instruction)
        stripped = Program(
            [
                step
                for step in program.steps
                if step.op
                not in {
                    "filter_state",
                    "filter_threshold",
                    "filter_affordance",
                    "argmax",
                    "argmin",
                }
            ],
            task_type=program.task_type,
        )
        result = ProgramExecutor().execute(scene, stripped)
        candidates = result.candidates_after_filtering
        if candidates:
            chosen = self.rng.choice(candidates)
            entity = scene.entity_by_id(chosen)
            return BaselineResult(
                chosen,
                "group" if entity in scene.groups else "object",
                stripped.to_string(),
                [*result.trace, f"Ignored attributes/states and chose {chosen}"],
            )
        return RandomSameCategoryBaseline(self.rng.randint(0, 9999)).predict(scene, instruction)


class RelationIgnorantBaseline(BaseBaseline):
    name = "Relation-Ignorant"

    def predict(self, scene: Scene, instruction: str) -> BaselineResult:
        program = parse_instruction(instruction)
        stripped = Program(
            [
                step
                for step in program.steps
                if step.op
                not in {
                    "relation",
                    "between",
                    "nearest_to",
                    "farthest_from",
                    "exclude_near",
                    "nth",
                }
                and not (step.op in {"argmax", "argmin"} and step.args.get("attribute") == "distance_to_origin")
            ],
            task_type=program.task_type,
        )
        result = ProgramExecutor().execute(scene, stripped)
        candidates = result.candidates_after_filtering
        if candidates:
            chosen = self.rng.choice(candidates)
            entity = scene.entity_by_id(chosen)
            return BaselineResult(
                chosen,
                "group" if entity in scene.groups else "object",
                stripped.to_string(),
                [*result.trace, f"Ignored relation and chose {chosen}"],
            )
        return RandomSameCategoryBaseline(self.rng.randint(0, 9999)).predict(scene, instruction)


def all_baselines(seed: int = 0) -> list[BaseBaseline]:
    return [
        RandomObjectBaseline(seed),
        RandomSameCategoryBaseline(seed + 1),
        AttributeIgnorantBaseline(seed + 2),
        RelationIgnorantBaseline(seed + 3),
    ]


def _coarse_candidates(scene: Scene, instruction: str):
    text = instruction.lower()
    if "pair of shoes" in text:
        return scene.groups, "group"
    if "banana" in text:
        return [obj for obj in scene.objects if obj.category == "banana"], "object"
    if "drink" in text or "bottle" in text or "water" in text:
        if "bottle" in text and "drink" not in text:
            return [obj for obj in scene.objects if obj.category == "bottle"], "object"
        return [obj for obj in scene.objects if "drink" in obj.super_categories], "object"
    if "fruit" in text:
        return [obj for obj in scene.objects if "fruit" in obj.super_categories], "object"
    if "cup" in text or "coffee" in text or "drinkware" in text:
        return [obj for obj in scene.objects if "drinkware" in obj.super_categories], "object"
    if "shoe" in text:
        return [obj for obj in scene.objects if obj.category == "shoe"], "object"
    if "object" in text:
        return scene.objects, "object"
    return [], "none"


def logic_baseline_result(scene: Scene, instruction: str) -> BaselineResult:
    result = LogicAwareReasoner().reason(scene, instruction)
    return BaselineResult(result.target_id, result.target_type, result.program, result.steps, result.confidence)

