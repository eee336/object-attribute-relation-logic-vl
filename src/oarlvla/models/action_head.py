from __future__ import annotations

from .torch_utils import require_torch


torch, nn = require_torch()


class ActionHead(nn.Module):
    def __init__(self, hidden_dim: int, action_dim: int = 3):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, action_dim))

    def forward(self, selected_or_fused_target_token):
        return self.net(selected_or_fused_target_token)
