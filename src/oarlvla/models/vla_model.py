from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .action_head import MLPActionHead, SmolStyleFlowActionHead
from .encoders import SimpleCNNImageEncoder, TextEncoder
from .oarl_core import OARLReasoningCore
from .program_head import ProgramHead
from .qwen_vl import QwenVLBackbone
from .torch_utils import require_torch


torch, nn = require_torch()


@dataclass
class OARLVLAConfig:
    vocab_size: int
    object_feature_dim: int
    region_feature_dim: int = 0
    hidden_dim: int = 128
    text_embed_dim: int = 64
    num_relation_types: int = 8
    num_program_types: int = 9
    action_dim: int = 3
    action_head_type: str = "flow_matching"
    action_chunk_size: int = 8
    action_denoise_steps: int = 10
    action_head_layers: int = 2
    action_head_heads: int = 4
    vlm_backbone: str = "tiny"
    qwen_model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    freeze_qwen_vl: bool = True
    qwen_trust_remote_code: bool = True
    qwen_torch_dtype: str = "auto"
    qwen_device_map: str | None = None
    image_mode: str = "symbolic"
    image_channels: int = 3
    use_relation_graph: bool = True
    dropout: float = 0.1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OARLVLAConfig":
        data = dict(data)
        data.setdefault("region_feature_dim", 0)
        if "action_head_type" not in data:
            data["action_head_type"] = "mlp"
        return cls(**data)


class OARLVLAModel(nn.Module):
    """Target-grounded object-centric VLA with an explicit OARL reasoning core."""

    def __init__(self, config: OARLVLAConfig):
        super().__init__()
        self.config = config
        if config.vlm_backbone not in {"tiny", "qwen_vl"}:
            raise ValueError(f"Unsupported vlm_backbone: {config.vlm_backbone}")
        self.text_encoder = TextEncoder(config.vocab_size, config.text_embed_dim, config.hidden_dim, config.dropout)
        self.qwen_vl = (
            QwenVLBackbone(
                config.qwen_model_name,
                config.hidden_dim,
                freeze=config.freeze_qwen_vl,
                trust_remote_code=config.qwen_trust_remote_code,
                torch_dtype=config.qwen_torch_dtype,
                device_map=config.qwen_device_map,
            )
            if config.vlm_backbone == "qwen_vl"
            else None
        )
        self.oarl_core = OARLReasoningCore(
            config.object_feature_dim,
            config.hidden_dim,
            config.num_relation_types,
            dropout=config.dropout,
            use_relation_graph=config.use_relation_graph,
            region_feature_dim=config.region_feature_dim,
        )
        self.image_encoder = (
            SimpleCNNImageEncoder(config.image_channels, config.hidden_dim)
            if config.image_mode == "cnn_stub"
            else None
        )
        self.program_head = ProgramHead(config.hidden_dim, config.num_program_types)
        self.action_head = self._build_action_head(config)

    def _build_action_head(self, config: OARLVLAConfig):
        if config.action_head_type == "mlp":
            return MLPActionHead(config.hidden_dim, config.action_dim)
        if config.action_head_type == "flow_matching":
            return SmolStyleFlowActionHead(
                config.hidden_dim,
                config.action_dim,
                chunk_size=config.action_chunk_size,
                num_steps=config.action_denoise_steps,
                num_layers=config.action_head_layers,
                num_heads=config.action_head_heads,
                dropout=config.dropout,
            )
        raise ValueError(f"Unsupported action_head_type: {config.action_head_type}")

    def replace_action_head(
        self,
        action_head_type: str,
        *,
        action_chunk_size: int | None = None,
        action_denoise_steps: int | None = None,
        action_head_layers: int | None = None,
        action_head_heads: int | None = None,
    ) -> None:
        self.config.action_head_type = action_head_type
        if action_chunk_size is not None:
            self.config.action_chunk_size = action_chunk_size
        if action_denoise_steps is not None:
            self.config.action_denoise_steps = action_denoise_steps
        if action_head_layers is not None:
            self.config.action_head_layers = action_head_layers
        if action_head_heads is not None:
            self.config.action_head_heads = action_head_heads
        self.action_head = self._build_action_head(self.config)

    def forward(
        self,
        image_features=None,
        tokenized_instruction=None,
        object_features=None,
        relation_features=None,
        qwen_inputs=None,
        object_region_features=None,
        object_mask=None,
        relation_mask=None,
        action_labels=None,
    ) -> dict[str, Any]:
        if object_features is None:
            raise ValueError("object_features are required")
        if self.qwen_vl is None and tokenized_instruction is None:
            raise ValueError("tokenized_instruction is required when vlm_backbone='tiny'")
        if self.qwen_vl is not None and qwen_inputs is None and tokenized_instruction is None:
            raise ValueError("qwen_inputs or tokenized_instruction is required when vlm_backbone='qwen_vl'")

        qwen_embedding = self.qwen_vl(qwen_inputs) if self.qwen_vl is not None and qwen_inputs is not None else None
        text_embedding = qwen_embedding if qwen_embedding is not None else self.text_encoder(tokenized_instruction)

        image_embedding = None
        if qwen_embedding is not None:
            image_embedding = qwen_embedding
        elif self.image_encoder is not None and image_features is not None:
            image_embedding = self.image_encoder(image_features)
        elif image_features is not None and image_features.dim() == 2:
            image_embedding = image_features

        core_outputs = self.oarl_core(
            text_embedding=text_embedding,
            object_features=object_features,
            relation_features=relation_features,
            image_embedding=image_embedding,
            object_region_features=object_region_features,
            object_mask=object_mask,
            relation_mask=relation_mask,
            use_relation_graph=self.config.use_relation_graph,
        )
        action_outputs = self._predict_actions(
            core_outputs["action_context_tokens"],
            core_outputs["action_context_mask"],
            core_outputs["selected_target_token"],
            action_labels,
        )
        program_logits = self.program_head(core_outputs["global_embedding"])
        outputs = {
            "target_logits": core_outputs["target_logits"],
            "program_logits": program_logits,
            "state_logits": None,
            "attribute_logits": None,
            "object_tokens": core_outputs["object_tokens"],
            "fused_object_tokens": core_outputs["fused_object_tokens"],
            "global_embedding": core_outputs["global_embedding"],
            "selected_target_token": core_outputs["selected_target_token"],
            "action_context_tokens": core_outputs["action_context_tokens"],
        }
        outputs.update(action_outputs)
        return outputs

    def _predict_actions(self, action_context, action_context_mask, selected_target_token, action_labels=None):
        if self.config.action_head_type == "mlp":
            action_pred = self.action_head(selected_target_token)
            return {"action_pred": action_pred, "action_chunk_pred": action_pred.unsqueeze(1)}
        if action_labels is not None:
            flow_outputs = self.action_head(action_context, action_labels, context_mask=action_context_mask)
            action_chunk = flow_outputs["action_chunk_pred"]
            return {
                **flow_outputs,
                "action_pred": action_chunk[:, 0, :],
            }
        action_chunk = self.action_head.sample(action_context, context_mask=action_context_mask)
        return {
            "action_chunk_pred": action_chunk,
            "action_pred": action_chunk[:, 0, :],
        }
