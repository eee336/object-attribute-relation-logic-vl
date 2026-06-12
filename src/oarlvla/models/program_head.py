from __future__ import annotations

from .torch_utils import require_torch


torch, nn = require_torch()


class ProgramHead(nn.Module):
    def __init__(self, hidden_dim: int, num_program_types: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, num_program_types))

    def forward(self, global_embedding):
        return self.net(global_embedding)
