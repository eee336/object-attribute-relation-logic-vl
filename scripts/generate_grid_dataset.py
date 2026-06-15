from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEPS = ROOT / ".deps"
if DEPS.exists():
    sys.path.insert(0, str(DEPS))
sys.path.insert(0, str(ROOT / "src"))

from oarlvla.gridworld import generate_grid_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate stage-1 grid/cutout gold VLA data.")
    parser.add_argument("--num-scenes", type=int, default=1000)
    parser.add_argument("--grid-size", type=int, default=8)
    parser.add_argument("--cell-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("data/oarlvla_grid_sprites.jsonl"))
    parser.add_argument("--image-dir", type=Path, default=Path("data/grid_images"))
    parser.add_argument("--asset-dir", type=Path, default=Path("data/grid_assets"))
    parser.add_argument("--debug-group-boxes", action="store_true", help="Render group bounding boxes for visual debugging.")
    args = parser.parse_args()
    report = generate_grid_dataset(
        num_scenes=args.num_scenes,
        grid_size=args.grid_size,
        cell_size=args.cell_size,
        seed=args.seed,
        output=args.output,
        image_dir=args.image_dir,
        asset_dir=args.asset_dir,
        render_group_boxes=args.debug_group_boxes,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
