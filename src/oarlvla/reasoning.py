from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Union

from .attributes import numeric_value
from .groups import ObjectGroup
from .objects import ObjectInstance
from .parser import Program, ProgramStep, parse_instruction
from .relations import (
    between,
    entities_left_of,
    entities_right_of,
    entity_distance,
    filter_groups,
    filter_objects,
    not_near,
    select_farthest,
    select_nearest,
    select_nth,
)
from .scene import Scene
from .states import coffee_suitable


Entity = Union[ObjectInstance, ObjectGroup]


@dataclass
class ReasoningResult:
    target_id: str | None
    target_type: Literal["object", "group", "none"]
    program: str
    steps: list[str]
    confidence: float
    candidates_before_filtering: list[str]
    candidates_after_filtering: list[str]
    failure_reason: str | None = None
    task_type: str = "unknown"


@dataclass
class ExecutionResult:
    target: Entity | None
    target_type: Literal["object", "group", "none"]
    trace: list[str]
    candidates_before_filtering: list[str]
    candidates_after_filtering: list[str]
    failure_reason: str | None = None


class ProgramExecutor:
    def execute(self, scene: Scene, program: Program) -> ExecutionResult:
        current: list[Entity] = list(scene.objects)
        target: Entity | None = None
        target_type: Literal["object", "group", "none"] = "none"
        trace: list[str] = []
        before: list[str] = [obj.id for obj in scene.objects]

        for step in program.steps:
            if step.op == "filter":
                current = self._filter_objects(scene, current, step)
                before = before if before else [entity.id for entity in current]
                trace.append(f"Filtered candidates with {step.args}: {[e.id for e in current]}")
            elif step.op == "filter_group":
                current = filter_groups(scene, group_type=step.args.get("group_type"), super_category=step.args.get("super_category"))
                before = [group.id for group in current] if not before else before
                trace.append(f"Found groups {step.args}: {[e.id for e in current]}")
            elif step.op == "filter_state":
                key = step.args["key"]
                value = step.args["value"]
                current = [entity for entity in current if getattr(entity, "states", {}).get(key) == value]
                trace.append(f"Filtered state {key}={value}: {[e.id for e in current]}")
            elif step.op == "filter_threshold":
                attr = step.args["attribute"]
                op = step.args["op"]
                value = float(step.args["value"])
                current = [entity for entity in current if _compare(numeric_value(entity, attr), op, value)]
                trace.append(f"Filtered {attr} {op} {value}: {[e.id for e in current]}")
            elif step.op == "filter_affordance":
                if step.args.get("name") == "coffee_suitable":
                    current = [entity for entity in current if isinstance(entity, ObjectInstance) and coffee_suitable(entity)]
                    trace.append(f"Applied coffee affordance: {[e.id for e in current]}")
            elif step.op == "exclude_near":
                refs = filter_objects(scene, category=step.args.get("ref_category"), super_category=step.args.get("ref_super_category"))
                current = not_near([e for e in current if isinstance(e, ObjectInstance)], refs)
                trace.append(f"Excluded candidates near {[r.id for r in refs]}: {[e.id for e in current]}")
            elif step.op == "relation":
                current = self._apply_relation(scene, current, step, trace)
            elif step.op == "between":
                refs_a = filter_objects(scene, category=step.args.get("ref_category_a"))
                refs_b = filter_objects(scene, category=step.args.get("ref_category_b"))
                if refs_a and refs_b:
                    current = between([e for e in current if isinstance(e, ObjectInstance)], refs_a[0], refs_b[0])
                else:
                    current = []
                trace.append(f"Filtered between references: {[e.id for e in current]}")
            elif step.op == "nearest_to":
                refs = filter_objects(scene, category=step.args.get("ref_category"), super_category=step.args.get("ref_super_category"))
                target = self._nearest_or_farthest_to_refs(current, refs, nearest=True)
                current = [target] if target else []
                trace.append(f"Selected nearest to {[r.id for r in refs]}: {[e.id for e in current]}")
            elif step.op == "farthest_from":
                refs = filter_objects(scene, category=step.args.get("ref_category"), super_category=step.args.get("ref_super_category"))
                target = self._nearest_or_farthest_to_refs(current, refs, nearest=False)
                current = [target] if target else []
                trace.append(f"Selected farthest from {[r.id for r in refs]}: {[e.id for e in current]}")
            elif step.op == "nth":
                objects = [e for e in current if isinstance(e, ObjectInstance)]
                selected = select_nth(objects, int(step.args["n"]), step.args.get("direction", "left_to_right"))
                current = [selected] if selected else []
                trace.append(f"Selected nth object: {[e.id for e in current]}")
            elif step.op in {"argmax", "argmin"}:
                current = self._arg_extreme(current, step)
                trace.append(f"Applied {step.op} on {step.args}: {[e.id for e in current]}")
            elif step.op == "select_best":
                current = sorted(current, key=lambda entity: entity.id)[:1]
                trace.append(f"Selected best stable candidate: {[e.id for e in current]}")
            elif step.op == "select_from_history":
                target = self._select_from_history(scene, step)
                current = [target] if target else []
                trace.append(f"Selected from history {step.args}: {[e.id for e in current]}")
            else:
                return ExecutionResult(None, "none", trace, before, [e.id for e in current], f"Unsupported step: {step.op}")

            if not current:
                return ExecutionResult(None, "none", trace, before, [], f"No candidates after {step.op}")

        target = current[0] if current else None
        if isinstance(target, ObjectGroup):
            target_type = "group"
        elif isinstance(target, ObjectInstance):
            target_type = "object"
        return ExecutionResult(target, target_type, trace, before, [e.id for e in current], None if target else "No target")

    def _filter_objects(self, scene: Scene, current: list[Entity], step: ProgramStep) -> list[Entity]:
        category = step.args.get("category")
        super_category = step.args.get("super_category")
        color = step.args.get("color")
        if category == "object":
            candidates: list[ObjectInstance] = list(scene.objects)
        else:
            candidates = filter_objects(scene, category=category, super_category=super_category, color=color)
        current_ids = {entity.id for entity in current if isinstance(entity, ObjectInstance)}
        if current_ids and len(current) != len(scene.objects):
            return [obj for obj in candidates if obj.id in current_ids]
        return candidates

    def _apply_relation(self, scene: Scene, current: list[Entity], step: ProgramStep, trace: list[str]) -> list[Entity]:
        op = step.args["op"]
        refs = filter_objects(scene, category=step.args.get("ref_category"), super_category=step.args.get("ref_super_category"))
        objects = [entity for entity in current if isinstance(entity, ObjectInstance)]
        if not refs:
            return []
        if op == "left_of":
            out = entities_left_of(objects, refs[0])
        elif op == "right_of":
            out = entities_right_of(objects, refs[0])
        else:
            out = []
        trace.append(f"Applied relation {op} to {refs[0].id}: {[obj.id for obj in out]}")
        return out

    def _nearest_or_farthest_to_refs(self, current: list[Entity], refs: list[Entity], nearest: bool) -> Entity | None:
        if not current or not refs:
            return None

        def score(entity: Entity) -> float:
            return min(entity_distance(entity, ref) for ref in refs)

        return min(current, key=score) if nearest else max(current, key=score)

    def _arg_extreme(self, current: list[Entity], step: ProgramStep) -> list[Entity]:
        if not current:
            return []
        attr = step.args["attribute"]
        fallback = step.args.get("fallback")
        selector = max if step.op == "argmax" else min
        selected = selector(current, key=lambda entity: numeric_value(entity, attr, fallback))
        return [selected]

    def _select_from_history(self, scene: Scene, step: ProgramStep) -> Entity | None:
        events = scene.history
        event_type = step.args.get("event_type")
        if event_type:
            events = [event for event in events if event.event_type == event_type]
        if not events:
            return None
        event = max(events, key=lambda e: e.timestamp) if step.args.get("most_recent", True) else events[0]
        return scene.entity_by_id(event.object_id)


class LogicAwareReasoner:
    def __init__(self) -> None:
        self.executor = ProgramExecutor()

    def reason(self, scene: Scene, instruction: str) -> ReasoningResult:
        program = parse_instruction(instruction)
        execution = self.executor.execute(scene, program)
        target_id = execution.target.id if execution.target else None
        confidence = 0.95 if execution.target and execution.failure_reason is None else 0.0
        return ReasoningResult(
            target_id=target_id,
            target_type=execution.target_type,
            program=program.to_string(),
            steps=execution.trace,
            confidence=confidence,
            candidates_before_filtering=execution.candidates_before_filtering,
            candidates_after_filtering=execution.candidates_after_filtering,
            failure_reason=execution.failure_reason,
            task_type=program.task_type,
        )


def _compare(left: float, op: str, right: float) -> bool:
    if op == ">":
        return left > right
    if op == ">=":
        return left >= right
    if op == "<":
        return left < right
    if op == "<=":
        return left <= right
    if op == "==":
        return left == right
    raise ValueError(f"Unsupported comparison: {op}")
