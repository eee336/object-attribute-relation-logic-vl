from __future__ import annotations

import json
from pathlib import Path

from oarlvla.evaluation import sample_to_jsonl_row
from oarlvla.instruction import InstructionExample
from oarlvla.reasoning import LogicAwareReasoner
from oarlvla.scene import generate_scene


def write_tiny_model_dataset(path: Path, num_samples: int = 4) -> Path:
    reasoner = LogicAwareReasoner()
    instructions = [
        ("Pick the banana that has not turned black.", "state_filtering"),
        ("Pick the largest drink.", "attribute_comparison"),
        ("Pick the farthest pair of shoes.", "group_grounding"),
        ("Pick the object I just put down.", "history_reference"),
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for idx in range(num_samples):
            scene = generate_scene(100 + idx, 12, scene_id=f"model_test_{idx:04d}")
            instruction, task_type = instructions[idx % len(instructions)]
            result = reasoner.reason(scene, instruction)
            example = InstructionExample(
                instruction=instruction,
                program=result.program,
                target_id=result.target_id,
                target_type=result.target_type,
                task_type=task_type,
                reasoning_steps=result.steps,
            )
            f.write(json.dumps(sample_to_jsonl_row(scene, example), ensure_ascii=False) + "\n")
    return path

