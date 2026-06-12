from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .policy import Action
from .scene import Scene


@dataclass
class RewardBreakdown:
    total: float
    grounding: float
    attribute: float
    relation: float
    action: float
    success: float
    wrong_object_penalty: float

    def to_dict(self) -> dict[str, float]:
        return self.__dict__.copy()


class RewardModel:
    def compute_reward(
        self,
        scene: Scene,
        instruction: str,
        predicted_target: str | None,
        action: Action | None,
        ground_truth: str | None,
        task_type: str = "unknown",
    ) -> RewardBreakdown:
        correct = predicted_target is not None and predicted_target == ground_truth
        action_ok = action is not None and action.target_id == predicted_target and scene.entity_by_id(action.target_id) is not None
        grounding = 1.0 if correct else 0.0
        wrong = 0.0 if correct else 1.0
        attribute = 1.0 if correct and task_type in {"attribute_comparison", "state_filtering", "affordance"} else 0.0
        relation = 1.0 if correct and task_type in {"spatial_relation", "ordinal_relation", "group_grounding", "negation"} else 0.0
        action_reward = 1.0 if action_ok else 0.0
        success = 1.0 if correct and action_ok else 0.0
        total = grounding + attribute + relation + action_reward + success - wrong
        return RewardBreakdown(total, grounding, attribute, relation, action_reward, success, wrong)


def success_metrics(predicted_target: str | None, ground_truth: str | None) -> dict[str, Any]:
    correct = predicted_target is not None and predicted_target == ground_truth
    return {"correct": correct, "wrong_object": not correct, "success": correct}

