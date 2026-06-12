from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oarlvla.evaluation import sample_to_jsonl_row
from oarlvla.instruction import TASK_TYPES, generate_instruction
from oarlvla.scene import generate_scene


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-scenes", type=int, default=1000)
    parser.add_argument("--objects-per-scene", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("data/oarlvla_synthetic.jsonl"))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for idx in range(args.num_scenes):
            scene = generate_scene(args.seed + idx, args.objects_per_scene, scene_id=f"scene_{idx:06d}")
            task_type = TASK_TYPES[idx % len(TASK_TYPES)]
            example = generate_instruction(scene, task_type, args.seed + idx)
            f.write(json.dumps(sample_to_jsonl_row(scene, example), ensure_ascii=False) + "\n")
    print(f"Wrote {args.num_scenes} synthetic gold samples to {args.output}")


if __name__ == "__main__":
    main()

