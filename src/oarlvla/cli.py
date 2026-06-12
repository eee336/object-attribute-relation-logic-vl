from __future__ import annotations

from .evaluation import print_benchmark_report, run_benchmark


def benchmark_main(num_scenes: int = 100, objects_per_scene: int = 12, seed: int = 42) -> str:
    return print_benchmark_report(run_benchmark(num_scenes, objects_per_scene, seed))

