from pathlib import Path

from oarlvla.gridworld import generate_grid_dataset
from oarlvla.gridworld.generator import generate_grid_scene
from oarlvla.gridworld.renderer import render_grid_scene
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
        asset_dir=tmp_path / "assets",
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


def test_group_boxes_are_debug_only(tmp_path: Path):
    from PIL import Image

    scene = generate_grid_scene(seed=7, grid_size=8, cell_size=48)
    default_path = render_grid_scene(
        scene,
        tmp_path / "default.png",
        grid_size=8,
        cell_size=48,
        asset_dir=tmp_path / "assets",
    )
    debug_path = render_grid_scene(
        scene,
        tmp_path / "debug.png",
        grid_size=8,
        cell_size=48,
        asset_dir=tmp_path / "assets",
        render_group_boxes=True,
    )

    purple = (123, 44, 191)
    with Image.open(default_path).convert("RGB") as image:
        assert purple not in image.getdata()
    with Image.open(debug_path).convert("RGB") as image:
        assert purple in image.getdata()
