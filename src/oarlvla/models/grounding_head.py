from __future__ import annotations

from .torch_utils import require_torch


torch, nn = require_torch()


class TargetGroundingHead(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.scorer = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))

    def forward(self, fused_object_tokens, object_mask=None):
        logits = self.scorer(fused_object_tokens).squeeze(-1)
        if object_mask is not None:
            logits = logits.masked_fill(~object_mask.bool(), -1e9)
        return logits
