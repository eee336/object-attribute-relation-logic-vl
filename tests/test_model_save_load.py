import pytest

torch = pytest.importorskip("torch")

from oarlvla.models.checkpoints import load_checkpoint, save_checkpoint
from oarlvla.models.collate import vla_collate_fn
from oarlvla.models.datasets import SyntheticVLADataset
from oarlvla.models.trainer import model_inputs
from oarlvla.models.vla_model import OARLVLAConfig, OARLVLAModel

from model_test_utils import write_tiny_model_dataset


def test_model_checkpoint_save_load_restores_outputs(tmp_path):
    dataset = SyntheticVLADataset(write_tiny_model_dataset(tmp_path / "tiny.jsonl", 2))
    batch = vla_collate_fn([dataset[0], dataset[1]])
    config = OARLVLAConfig(
        vocab_size=len(dataset.tokenizer),
        object_feature_dim=dataset.feature_metadata["feature_dim"],
        hidden_dim=32,
        num_relation_types=len(dataset.feature_metadata["relation_types"]),
        num_program_types=len(dataset.feature_metadata["task_types"]),
        dropout=0.0,
    )
    model = OARLVLAModel(config)
    model.eval()
    with torch.no_grad():
        before = model(**model_inputs(batch))["target_logits"]
    ckpt = save_checkpoint(tmp_path / "model.pt", model, dataset.tokenizer, dataset.feature_metadata, extra={"note": "test"})
    loaded, tokenizer, metadata, extra = load_checkpoint(ckpt)
    loaded.eval()
    with torch.no_grad():
        after = loaded(**model_inputs(batch))["target_logits"]
    assert torch.allclose(before, after)
    assert len(tokenizer) == len(dataset.tokenizer)
    assert metadata["feature_dim"] == dataset.feature_metadata["feature_dim"]
    assert extra["note"] == "test"

