from __future__ import annotations

from typing import Any

from .encoders import ObjectEncoder
from .fusion import CrossAttentionFusion
from .graph_encoder import SimpleRelationGraphEncoder
from .grounding_head import TargetGroundingHead
from .torch_utils import require_torch


torch, nn = require_torch()


class OARLReasoningCore(nn.Module):
    """Object-Attribute-Relation-Logic reasoning core for OARL-VLA.

    This is the model's explicit target-grounding core, not a plug-in around a
    separate VLA. It turns object-centric scene tokens, optional region
    embeddings, relation edges, and language/VLM context into target-aware object
    tokens before the target-conditioned action policy runs.
    """

    def __init__(
        self,
        object_feature_dim: int,
        hidden_dim: int,
        num_relation_types: int,
        *,
        dropout: float = 0.1,
        use_relation_graph: bool = True,
        region_feature_dim: int = 0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_relation_graph = use_relation_graph
        self.object_encoder = ObjectEncoder(object_feature_dim, hidden_dim, dropout)
        self.region_encoder = (
            nn.Sequential(
                nn.Linear(region_feature_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
            )
            if region_feature_dim > 0
            else None
        )
        self.object_region_norm = nn.LayerNorm(hidden_dim)
        self.graph_encoder = SimpleRelationGraphEncoder(hidden_dim, num_relation_types, dropout)
        self.fusion = CrossAttentionFusion(hidden_dim, dropout)
        self.target_head = TargetGroundingHead(hidden_dim)
        self.global_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        *,
        text_embedding,
        object_features,
        relation_features=None,
        image_embedding=None,
        object_region_features=None,
        object_mask=None,
        relation_mask=None,
        use_relation_graph: bool | None = None,
    ) -> dict[str, Any]:
        object_tokens = self.object_encoder(object_features)
        object_tokens = self._fuse_region_features(object_tokens, object_region_features)

        edge_index, edge_type = self._relation_tensors(relation_features)
        should_use_graph = self.use_relation_graph if use_relation_graph is None else use_relation_graph
        if should_use_graph and edge_index is not None and edge_type is not None:
            object_tokens = self.graph_encoder(object_tokens, edge_index, edge_type, relation_mask)

        fused_tokens = self.fusion(
            text_embedding,
            object_tokens,
            image_embedding=image_embedding,
            object_mask=object_mask,
        )
        target_logits = self.target_head(fused_tokens, object_mask)
        target_probs = torch.softmax(target_logits, dim=-1).unsqueeze(-1)
        selected_target_token = (fused_tokens * target_probs).sum(dim=1)
        global_embedding = self._global_embedding(fused_tokens, text_embedding, object_mask, image_embedding)
        action_context_tokens, action_context_mask = self._action_context(
            fused_tokens,
            selected_target_token,
            global_embedding,
            text_embedding,
            object_mask,
        )
        return {
            "object_tokens": object_tokens,
            "fused_object_tokens": fused_tokens,
            "target_logits": target_logits,
            "target_probs": target_probs,
            "selected_target_token": selected_target_token,
            "global_embedding": global_embedding,
            "action_context_tokens": action_context_tokens,
            "action_context_mask": action_context_mask,
        }

    def _fuse_region_features(self, object_tokens, object_region_features=None):
        if object_region_features is None:
            return object_tokens
        if self.region_encoder is not None:
            region_tokens = self.region_encoder(object_region_features.float())
        elif object_region_features.shape[-1] == self.hidden_dim:
            region_tokens = object_region_features.float()
        else:
            raise ValueError(
                "object_region_features require OARLVLAConfig.region_feature_dim "
                f"or a final dimension equal to hidden_dim={self.hidden_dim}"
            )
        return self.object_region_norm(object_tokens + region_tokens)

    @staticmethod
    def _relation_tensors(relation_features):
        if isinstance(relation_features, dict):
            return relation_features.get("edge_index"), relation_features.get("edge_type")
        if isinstance(relation_features, (tuple, list)) and len(relation_features) >= 2:
            return relation_features[0], relation_features[1]
        return None, None

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

    @staticmethod
    def _action_context(fused_tokens, selected_target_token, global_embedding, text_embedding, object_mask=None):
        prefix = torch.stack([global_embedding, selected_target_token, text_embedding], dim=1)
        context = torch.cat([prefix, fused_tokens], dim=1)
        if object_mask is None:
            return context, None
        prefix_mask = torch.ones(object_mask.shape[0], prefix.shape[1], dtype=torch.bool, device=object_mask.device)
        return context, torch.cat([prefix_mask, object_mask.bool()], dim=1)
