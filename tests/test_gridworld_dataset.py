from pathlib import Path

from oarlvla.gridworld import generate_grid_dataset
from oarlvla.models.collate import vla_collate_fn
from oarlvla.models.datasets import SyntheticVLADataset


def test_gridworld_dataset_generation_and_loader(tmp_path: Path):
    output = tmp_path / "grid.jsonl"
    image_dir = tmp_path / "images"
    report = generate_grid_dataset(
        num_scenes=9,
        grid_size=8,
        cell_size=48,
        seed=5,
        output=output,
        image_dir=image_dir,
    )
    assert report["num_samples"] == 9
    assert output.exists()
    assert len(list(image_dir.glob("*.png"))) == 9
    dataset = SyntheticVLADataset(output)
    sample = dataset[0]
    assert sample["target_index"].item() >= 0
    assert sample["image_path"].endswith(".png")
    batch = vla_collate_fn([dataset[0], dataset[1]])
    assert batch["object_features"].shape[0] == 2
    assert batch["object_mask"].any()

