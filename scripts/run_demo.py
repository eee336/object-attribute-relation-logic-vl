from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oarlvla.instruction import TASK_TYPES, generate_instruction
from oarlvla.policy import TargetConditionedPolicy
from oarlvla.reasoning import LogicAwareReasoner
from oarlvla.reward import RewardModel
from oarlvla.scene import generate_scene
from oarlvla.visualization import visualize_scene


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--instruction-type", choices=TASK_TYPES, default="state_filtering")
    parser.add_argument("--objects-per-scene", type=int, default=12)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    scene = generate_scene(args.seed, args.objects_per_scene)
    example = generate_instruction(scene, args.instruction_type, args.seed)
    reasoner = LogicAwareReasoner()
    result = reasoner.reason(scene, example.instruction)
    action = TargetConditionedPolicy().predict_action(scene, result.target_id, result.target_type)
    reward = RewardModel().compute_reward(scene, example.instruction, result.target_id, action, example.target_id, example.task_type)
    output = args.output or Path("outputs") / f"demo_{args.instruction_type}_seed{args.seed}.png"
    visualize_scene(
        scene,
        output,
        ground_truth_id=example.target_id,
        predicted_id=result.target_id,
        title=f"{example.task_type}: {example.instruction}",
    )
    print(
        json.dumps(
            {
                "scene_id": scene.id,
                "instruction": example.instruction,
                "program": result.program,
                "ground_truth": example.target_id,
                "predicted": result.target_id,
                "target_type": result.target_type,
                "action": action.__dict__ if action else None,
                "reward": reward.to_dict(),
                "visualization": str(output),
                "reasoning_steps": result.steps,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

