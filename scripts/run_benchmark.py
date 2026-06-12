from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oarlvla.evaluation import print_benchmark_report, run_benchmark


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-scenes", type=int, default=100)
    parser.add_argument("--objects-per-scene", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    results = run_benchmark(args.num_scenes, args.objects_per_scene, args.seed, args.output_dir)
    print(print_benchmark_report(results))


if __name__ == "__main__":
    main()

