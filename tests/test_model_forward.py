import pytest

torch = pytest.importorskip("torch")

from oarlvla.models.collate import vla_collate_fn
from oarlvla.models.datasets import SyntheticVLADataset
from oarlvla.models.trainer import model_inputs
from oarlvla.models.vla_model import OARLVLAConfig, OARLVLAModel

from model_test_utils import write_tiny_model_dataset


def test_model_forward_shapes(tmp_path):
    dataset = SyntheticVLADataset(write_tiny_model_dataset(tmp_path / "tiny.jsonl", 3))
    batch = vla_collate_fn([dataset[0], dataset[1], dataset[2]])
    config = OARLVLAConfig(
        vocab_size=len(dataset.tokenizer),
        object_feature_dim=dataset.feature_metadata["feature_dim"],
        hidden_dim=32,
        num_relation_types=len(dataset.feature_metadata["relation_types"]),
        num_program_types=len(dataset.feature_metadata["task_types"]),
    )
    model = OARLVLAModel(config)
    outputs = model(**model_inputs(batch))
    assert outputs["target_logits"].shape == batch["object_mask"].shape
    assert outputs["action_pred"].shape == (3, 3)
    assert outputs["action_chunk_pred"].shape == (3, config.action_chunk_size, 3)
    assert outputs["action_flow_losses"].shape == (3, config.action_chunk_size, 3)
    assert outputs["program_logits"].shape == (3, len(dataset.feature_metadata["task_types"]))


def test_model_forward_accepts_object_region_features(tmp_path):
    dataset = SyntheticVLADataset(write_tiny_model_dataset(tmp_path / "tiny.jsonl", 2))
    batch = vla_collate_fn([dataset[0], dataset[1]])
    config = OARLVLAConfig(
        vocab_size=len(dataset.tokenizer),
        object_feature_dim=dataset.feature_metadata["feature_dim"],
        region_feature_dim=5,
        hidden_dim=32,
        num_relation_types=len(dataset.feature_metadata["relation_types"]),
        num_program_types=len(dataset.feature_metadata["task_types"]),
    )
    model = OARLVLAModel(config)
    inputs = model_inputs(batch)
    inputs["object_region_features"] = torch.randn(batch["object_features"].shape[0], batch["object_features"].shape[1], 5)
    outputs = model(**inputs)
    assert outputs["object_tokens"].shape == outputs["fused_object_tokens"].shape
    assert outputs["target_logits"].shape == batch["object_mask"].shape
