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


def test_legacy_checkpoint_keys_load_after_oarl_core_rename(tmp_path):
    dataset = SyntheticVLADataset(write_tiny_model_dataset(tmp_path / "tiny.jsonl", 2))
    config = OARLVLAConfig(
        vocab_size=len(dataset.tokenizer),
        object_feature_dim=dataset.feature_metadata["feature_dim"],
        hidden_dim=32,
        num_relation_types=len(dataset.feature_metadata["relation_types"]),
        num_program_types=len(dataset.feature_metadata["task_types"]),
        action_head_type="mlp",
        dropout=0.0,
    )
    model = OARLVLAModel(config)
    legacy_state = {}
    reverse_prefixes = {
        "oarl_core.object_encoder.": "object_encoder.",
        "oarl_core.graph_encoder.": "graph_encoder.",
        "oarl_core.fusion.": "fusion.",
        "oarl_core.target_head.": "target_head.",
        "oarl_core.global_norm.": "global_norm.",
    }
    for key, value in model.state_dict().items():
        if key.startswith("oarl_core.object_region_norm."):
            continue
        legacy_key = key
        for new_prefix, old_prefix in reverse_prefixes.items():
            if key.startswith(new_prefix):
                legacy_key = old_prefix + key[len(new_prefix) :]
                break
        legacy_state[legacy_key] = value
    legacy_config = config.to_dict()
    legacy_config.pop("region_feature_dim")
    legacy_config.pop("action_head_type")
    ckpt = tmp_path / "legacy.pt"
    torch.save(
        {
            "model_state": legacy_state,
            "config": legacy_config,
            "tokenizer": dataset.tokenizer.to_dict(),
            "feature_metadata": dataset.feature_metadata,
            "extra": {},
        },
        ckpt,
    )
    loaded, _, _, _ = load_checkpoint(ckpt)
    batch = vla_collate_fn([dataset[0], dataset[1]])
    with torch.no_grad():
        outputs = loaded(**model_inputs(batch))
    assert outputs["target_logits"].shape == batch["object_mask"].shape


def test_oarl_adapter_checkpoint_keys_remap_to_oarl_core(tmp_path):
    dataset = SyntheticVLADataset(write_tiny_model_dataset(tmp_path / "tiny.jsonl", 2))
    config = OARLVLAConfig(
        vocab_size=len(dataset.tokenizer),
        object_feature_dim=dataset.feature_metadata["feature_dim"],
        hidden_dim=32,
        num_relation_types=len(dataset.feature_metadata["relation_types"]),
        num_program_types=len(dataset.feature_metadata["task_types"]),
        action_head_type="mlp",
        dropout=0.0,
    )
    model = OARLVLAModel(config)
    adapter_state = {
        ("oarl_adapter." + key[len("oarl_core.") :]) if key.startswith("oarl_core.") else key: value
        for key, value in model.state_dict().items()
    }
    ckpt = tmp_path / "adapter_named.pt"
    torch.save(
        {
            "model_state": adapter_state,
            "config": config.to_dict(),
            "tokenizer": dataset.tokenizer.to_dict(),
            "feature_metadata": dataset.feature_metadata,
            "extra": {},
        },
        ckpt,
    )
    loaded, _, _, _ = load_checkpoint(ckpt)
    assert any(name.startswith("oarl_core.") for name, _ in loaded.named_parameters())
