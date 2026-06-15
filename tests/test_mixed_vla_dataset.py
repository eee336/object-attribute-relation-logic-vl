import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from oarlvla.models.datasets import MixedVLADataset

from model_test_utils import write_tiny_model_dataset


def test_mixed_vla_dataset_masks_unverified_web_targets(tmp_path: Path):
    synthetic_path = write_tiny_model_dataset(tmp_path / "synthetic.jsonl", 4)
    web_path = tmp_path / "web_tasks.jsonl"
    web_row = {
        "image_id": "web_0001",
        "image_path": "tests/fixtures/images/fresh_banana.ppm",
        "instruction": "Pick the banana that has not turned black.",
        "program": "filter(category='banana')->filter_state(key='is_blackened', value=False)",
        "target_type": "unknown",
        "target_id": None,
        "task_type": "state_filtering",
        "label_quality": "weak",
        "source": "web",
    }
    web_path.write_text(json.dumps(web_row) + "\n", encoding="utf-8")

    dataset = MixedVLADataset(synthetic_path, web_path)
    assert len(dataset) == 5
    sample = dataset[-1]
    assert sample["source"] == "web"
    assert sample["target_index"].item() == -1
    assert sample["image_path"] == "tests/fixtures/images/fresh_banana.ppm"

