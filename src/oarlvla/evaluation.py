from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .baselines import all_baselines
from .instruction import TASK_TYPES, generate_instruction
from .policy import TargetConditionedPolicy
from .reasoning import LogicAwareReasoner
from .reward import RewardModel
from .scene import generate_scene
from .visualization import visualize_scene


def run_benchmark(
    num_scenes: int = 100,
    objects_per_scene: int = 12,
    seed: int = 42,
    output_dir: str | Path = "outputs",
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    methods = {"OARL-VLA Logic Reasoner": LogicAwareReasoner(), **{b.name: b for b in all_baselines(seed)}}
    totals: dict[str, dict[str, Any]] = {
        name: {"n": 0, "correct": 0, "wrong": 0, "success": 0, "by_task": defaultdict(lambda: {"n": 0, "correct": 0})}
        for name in methods
    }
    examples: dict[str, Any] = {}
    policy = TargetConditionedPolicy()
    reward_model = RewardModel()

    for idx in range(num_scenes):
        scene = generate_scene(seed + idx, objects_per_scene, scene_id=f"scene_{idx:06d}")
        task_type = TASK_TYPES[idx % len(TASK_TYPES)]
        example = generate_instruction(scene, task_type, seed + idx)
        ground_truth = example.target_id
        for name, method in methods.items():
            if isinstance(method, LogicAwareReasoner):
                pred = method.reason(scene, example.instruction)
                target_id, target_type, program, steps = pred.target_id, pred.target_type, pred.program, pred.steps
            else:
                pred = method.predict(scene, example.instruction)
                target_id, target_type, program, steps = pred.target_id, pred.target_type, pred.program, pred.steps
            action = policy.predict_action(scene, target_id, target_type)
            reward = reward_model.compute_reward(scene, example.instruction, target_id, action, ground_truth, task_type)
            correct = target_id == ground_truth and target_id is not None
            bucket = totals[name]
            bucket["n"] += 1
            bucket["correct"] += int(correct)
            bucket["wrong"] += int(not correct)
            bucket["success"] += int(reward.success > 0)
            bucket["by_task"][task_type]["n"] += 1
            bucket["by_task"][task_type]["correct"] += int(correct)
            if name == "OARL-VLA Logic Reasoner":
                if correct and "success" not in examples:
                    examples["success"] = (scene, example, target_id)
                if task_type == "attribute_comparison":
                    examples["attribute"] = (scene, example, target_id)
                if task_type == "group_grounding":
                    examples["group"] = (scene, example, target_id)
            elif name == "Random Object" and not correct and "failure" not in examples:
                examples["failure"] = (scene, example, target_id)

    results = {"num_scenes": num_scenes, "objects_per_scene": objects_per_scene, "seed": seed, "methods": {}}
    for name, bucket in totals.items():
        n = max(bucket["n"], 1)
        by_task = {
            task: {
                "n": stats["n"],
                "accuracy": stats["correct"] / stats["n"] if stats["n"] else 0.0,
            }
            for task, stats in sorted(bucket["by_task"].items())
        }
        results["methods"][name] = {
            "target_accuracy": bucket["correct"] / n,
            "wrong_object_rate": bucket["wrong"] / n,
            "task_success_rate": bucket["success"] / n,
            "attribute_accuracy": _task_group_accuracy(by_task, ["attribute_comparison", "affordance"]),
            "state_accuracy": _task_group_accuracy(by_task, ["state_filtering", "negation"]),
            "relation_accuracy": _task_group_accuracy(by_task, ["spatial_relation", "ordinal_relation"]),
            "group_accuracy": _task_group_accuracy(by_task, ["group_grounding"]),
            "history_accuracy": _task_group_accuracy(by_task, ["history_reference"]),
            "by_task": by_task,
        }

    (output_dir / "benchmark_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    _write_csv(output_dir / "benchmark_results.csv", results)
    _save_examples(output_dir, examples)
    return results


def _task_group_accuracy(by_task: dict[str, dict[str, Any]], tasks: list[str]) -> float:
    counts = [by_task.get(task, {"n": 0, "accuracy": 0.0}) for task in tasks]
    total_n = sum(item["n"] for item in counts)
    if total_n == 0:
        return 0.0
    return sum(item["accuracy"] * item["n"] for item in counts) / total_n


def _write_csv(path: Path, results: dict[str, Any]) -> None:
    rows = []
    for method, stats in results["methods"].items():
        row = {
            "method": method,
            "target_accuracy": stats["target_accuracy"],
            "wrong_object_rate": stats["wrong_object_rate"],
            "task_success_rate": stats["task_success_rate"],
            "attribute_accuracy": stats["attribute_accuracy"],
            "state_accuracy": stats["state_accuracy"],
            "relation_accuracy": stats["relation_accuracy"],
            "group_accuracy": stats["group_accuracy"],
            "history_accuracy": stats["history_accuracy"],
        }
        rows.append(row)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _save_examples(output_dir: Path, examples: dict[str, Any]) -> None:
    mapping = {
        "success": "example_success.png",
        "failure": "example_failure.png",
        "attribute": "example_attribute_task.png",
        "group": "example_group_task.png",
    }
    for key, filename in mapping.items():
        if key in examples:
            scene, example, predicted = examples[key]
            visualize_scene(
                scene,
                output_dir / filename,
                ground_truth_id=example.target_id,
                predicted_id=predicted,
                title=f"{example.task_type}: {example.instruction}",
            )


def print_benchmark_report(results: dict[str, Any]) -> str:
    lines: list[str] = []
    for method, stats in results["methods"].items():
        lines.append(f"Method: {method}")
        lines.append(f"Target Accuracy: {stats['target_accuracy']:.3f}")
        lines.append(f"Wrong Object Rate: {stats['wrong_object_rate']:.3f}")
        lines.append(f"Task Success Rate: {stats['task_success_rate']:.3f}")
        lines.append(f"Attribute Accuracy: {stats['attribute_accuracy']:.3f}")
        lines.append(f"State Accuracy: {stats['state_accuracy']:.3f}")
        lines.append(f"Relation Accuracy: {stats['relation_accuracy']:.3f}")
        lines.append(f"Group Accuracy: {stats['group_accuracy']:.3f}")
        lines.append(f"History Accuracy: {stats['history_accuracy']:.3f}")
        lines.append("By task:")
        for task, task_stats in stats["by_task"].items():
            lines.append(f"  {task}: {task_stats['accuracy']:.3f} ({task_stats['n']})")
        lines.append("")
    return "\n".join(lines)


def sample_to_jsonl_row(scene, example) -> dict[str, Any]:
    return {
        "scene_id": scene.id,
        "instruction": example.instruction,
        "program": example.program,
        "target_id": example.target_id,
        "target_type": example.target_type,
        "task_type": example.task_type,
        "objects": [obj.to_dict() for obj in scene.objects],
        "groups": [grp.to_dict() for grp in scene.groups],
        "reasoning_steps": example.reasoning_steps,
    }
