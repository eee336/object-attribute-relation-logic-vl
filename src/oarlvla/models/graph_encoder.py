from __future__ import annotations

from .torch_utils import require_torch


torch, nn = require_torch()


class SimpleRelationGraphEncoder(nn.Module):
    """Batched message passing without PyG."""

    def __init__(self, hidden_dim: int, num_relation_types: int, dropout: float = 0.1):
        super().__init__()
        self.relation_embedding = nn.Embedding(num_relation_types, hidden_dim)
        self.message_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, object_tokens, edge_index, edge_type, edge_mask=None):
        batch_size, num_nodes, hidden_dim = object_tokens.shape
        updated = []
        for batch_idx in range(batch_size):
            nodes = object_tokens[batch_idx]
            edges = edge_index[batch_idx].long()
            types = edge_type[batch_idx].long()
            if edge_mask is not None:
                mask = edge_mask[batch_idx].bool()
                edges = edges[mask]
                types = types[mask]
            if edges.numel() == 0:
                updated.append(nodes)
                continue
            src = edges[:, 0].clamp(0, num_nodes - 1)
            dst = edges[:, 1].clamp(0, num_nodes - 1)
            rel = self.relation_embedding(types.clamp_min(0))
            messages = self.message_mlp(nodes[src] + rel)
            agg = torch.zeros_like(nodes)
            agg.index_add_(0, dst, messages)
            degree = torch.zeros(num_nodes, device=nodes.device, dtype=nodes.dtype)
            degree.index_add_(0, dst, torch.ones_like(dst, dtype=nodes.dtype))
            agg = agg / degree.clamp_min(1.0).unsqueeze(-1)
            updated.append(self.norm(nodes + agg))
        return torch.stack(updated, dim=0)
