from pathlib import Path

from oarlvla.evaluation import run_benchmark
from oarlvla.instruction import generate_instruction
from oarlvla.scene import generate_scene
from oarlvla.visualization import visualize_scene


def test_benchmark_runs_and_writes_outputs(tmp_path: Path):
    results = run_benchmark(num_scenes=9, objects_per_scene=12, seed=9, output_dir=tmp_path)
    assert results["methods"]["OARL-VLA Logic Reasoner"]["target_accuracy"] == 1.0
    assert (tmp_path / "benchmark_results.json").exists()
    assert (tmp_path / "benchmark_results.csv").exists()


def test_visualization_saves_file(tmp_path: Path):
    scene = generate_scene(seed=10)
    example = generate_instruction(scene, "group_grounding", seed=10)
    path = visualize_scene(scene, tmp_path / "scene.png", ground_truth_id=example.target_id, predicted_id=example.target_id)
    assert path.exists()
    assert path.stat().st_size > 0

