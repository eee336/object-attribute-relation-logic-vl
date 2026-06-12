from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .action_head import ActionHead
from .encoders import ObjectEncoder, SimpleCNNImageEncoder, TextEncoder
from .fusion import CrossAttentionFusion
from .graph_encoder import SimpleRelationGraphEncoder
from .grounding_head import TargetGroundingHead
from .program_head import ProgramHead
from .torch_utils import require_torch


torch, nn = require_torch()


@dataclass
class OARLVLAConfig:
    vocab_size: int
    object_feature_dim: int
    hidden_dim: int = 128
    text_embed_dim: int = 64
    num_relation_types: int = 8
    num_program_types: int = 9
    action_dim: int = 3
    image_mode: str = "symbolic"
    image_channels: int = 3
    dropout: float = 0.1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OARLVLAConfig":
        return cls(**data)


class OARLVLAModel(nn.Module):
    """Tiny symbolic VLA prototype with object tokens, relation graph, grounding, and action heads."""

    def __init__(self, config: OARLVLAConfig):
        super().__init__()
        self.config = config
        self.text_encoder = TextEncoder(config.vocab_size, config.text_embed_dim, config.hidden_dim, config.dropout)
        self.object_encoder = ObjectEncoder(config.object_feature_dim, config.hidden_dim, config.dropout)
        self.graph_encoder = SimpleRelationGraphEncoder(config.hidden_dim, config.num_relation_types, config.dropout)
        self.image_encoder = (
            SimpleCNNImageEncoder(config.image_channels, config.hidden_dim)
            if config.image_mode == "cnn_stub"
            else None
        )
        self.fusion = CrossAttentionFusion(config.hidden_dim, config.dropout)
        self.target_head = TargetGroundingHead(config.hidden_dim)
        self.program_head = ProgramHead(config.hidden_dim, config.num_program_types)
        self.action_head = ActionHead(config.hidden_dim, config.action_dim)
        self.global_norm = nn.LayerNorm(config.hidden_dim)

    def forward(
        self,
        image_features=None,
        tokenized_instruction=None,
        object_features=None,
        relation_features=None,
        object_mask=None,
        relation_mask=None,
    ) -> dict[str, Any]:
        if tokenized_instruction is None or object_features is None:
            raise ValueError("tokenized_instruction and object_features are required")

        text_embedding = self.text_encoder(tokenized_instruction)
        object_tokens = self.object_encoder(object_features)

        edge_index = None
        edge_type = None
        if isinstance(relation_features, dict):
            edge_index = relation_features.get("edge_index")
            edge_type = relation_features.get("edge_type")
        elif isinstance(relation_features, (tuple, list)) and len(relation_features) >= 2:
            edge_index, edge_type = relation_features[0], relation_features[1]

        if edge_index is not None and edge_type is not None:
            object_tokens = self.graph_encoder(object_tokens, edge_index, edge_type, relation_mask)

        image_embedding = None
        if self.image_encoder is not None and image_features is not None:
            image_embedding = self.image_encoder(image_features)
        elif image_features is not None and image_features.dim() == 2:
            image_embedding = image_features

        fused_tokens = self.fusion(text_embedding, object_tokens, image_embedding=image_embedding, object_mask=object_mask)
        target_logits = self.target_head(fused_tokens, object_mask)
        target_probs = torch.softmax(target_logits, dim=-1).unsqueeze(-1)
        selected_target_token = (fused_tokens * target_probs).sum(dim=1)
        action_pred = self.action_head(selected_target_token)
        global_embedding = self._global_embedding(fused_tokens, text_embedding, object_mask, image_embedding)
        program_logits = self.program_head(global_embedding)
        return {
            "target_logits": target_logits,
            "action_pred": action_pred,
            "program_logits": program_logits,
            "state_logits": None,
            "attribute_logits": None,
            "fused_object_tokens": fused_tokens,
            "global_embedding": global_embedding,
        }

    def _global_embedding(self, fused_tokens, text_embedding, object_mask=None, image_embedding=None):
        if object_mask is None:
            pooled = fused_tokens.mean(dim=1)
        else:
            mask = object_mask.float().unsqueeze(-1)
            pooled = (fused_tokens * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        global_embedding = pooled + text_embedding
        if image_embedding is not None and image_embedding.shape == global_embedding.shape:
            global_embedding = global_embedding + image_embedding
        return self.global_norm(global_embedding)
