from __future__ import annotations

from dataclasses import dataclass

from .scene import Scene


@dataclass
class Action:
    target_id: str
    target_type: str
    grasp_point: tuple[float, float]
    gripper: str = "close"


class TargetConditionedPolicy:
    def predict_action(self, scene: Scene, target_id: str | None, target_type: str) -> Action | None:
        if target_id is None:
            return None
        entity = scene.group_by_id(target_id) if target_type == "group" else scene.object_by_id(target_id)
        if entity is None:
            return None
        return Action(target_id=target_id, target_type=target_type, grasp_point=entity.center)

