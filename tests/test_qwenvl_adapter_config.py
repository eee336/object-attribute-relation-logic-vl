from oarlvla.models.qwen_vl import QwenVLProcessorAdapter
from oarlvla.models.vla_model import OARLVLAConfig


def test_qwenvl_config_roundtrip_without_loading_weights():
    config = OARLVLAConfig(
        vocab_size=8,
        object_feature_dim=35,
        hidden_dim=64,
        vlm_backbone="qwen_vl",
        qwen_model_name="Qwen/Qwen2.5-VL-3B-Instruct",
        freeze_qwen_vl=True,
    )
    restored = OARLVLAConfig.from_dict(config.to_dict())
    assert restored.vlm_backbone == "qwen_vl"
    assert restored.qwen_model_name == "Qwen/Qwen2.5-VL-3B-Instruct"
    assert restored.freeze_qwen_vl is True


def test_qwenvl_message_builder_without_transformers():
    messages = QwenVLProcessorAdapter.build_messages(
        "Pick the banana that has not turned black.",
        image_path="tests/fixtures/images/fresh_banana.ppm",
    )
    assert messages[0]["role"] == "user"
    assert messages[0]["content"][0]["type"] == "image"
    assert messages[0]["content"][1]["type"] == "text"

