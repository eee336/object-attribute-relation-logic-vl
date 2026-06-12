from __future__ import annotations

from .torch_utils import require_torch


torch, nn = require_torch()


class CrossAttentionFusion(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.text_projection = nn.Linear(hidden_dim, hidden_dim)
        self.image_projection = nn.Linear(hidden_dim, hidden_dim)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads=4, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(nn.Linear(hidden_dim, hidden_dim * 2), nn.GELU(), nn.Linear(hidden_dim * 2, hidden_dim))

    def forward(self, text_embedding, object_tokens, image_embedding=None, object_mask=None):
        context = [self.text_projection(text_embedding).unsqueeze(1)]
        if image_embedding is not None and image_embedding.shape[-1] == object_tokens.shape[-1]:
            context.append(self.image_projection(image_embedding).unsqueeze(1))
        context_tokens = torch.cat(context, dim=1)
        key_padding_mask = None
        attn_out, _ = self.cross_attn(query=object_tokens, key=context_tokens, value=context_tokens, key_padding_mask=key_padding_mask)
        fused = self.norm(object_tokens + attn_out)
        return self.norm(fused + self.ffn(fused))
