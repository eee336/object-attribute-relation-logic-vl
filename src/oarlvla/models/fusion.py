from __future__ import annotations

from .torch_utils import require_torch


torch, nn = require_torch()


class CrossAttentionFusion(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.text_projection = nn.Linear(hidden_dim, hidden_dim)
        self.image_projection = nn.Linear(hidden_dim, hidden_dim)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads=4, dropout=dropout, batch_first=True)
        self.pair_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(nn.Linear(hidden_dim, hidden_dim * 2), nn.GELU(), nn.Linear(hidden_dim * 2, hidden_dim))

    def forward(self, text_embedding, object_tokens, image_embedding=None, object_mask=None):
        text_context = self.text_projection(text_embedding)
        context = [text_context.unsqueeze(1)]
        if image_embedding is not None and image_embedding.shape[-1] == object_tokens.shape[-1]:
            image_context = self.image_projection(image_embedding)
            text_context = text_context + image_context
            context.append(image_context.unsqueeze(1))
        context_tokens = torch.cat(context, dim=1)
        key_padding_mask = None
        attn_out, _ = self.cross_attn(query=object_tokens, key=context_tokens, value=context_tokens, key_padding_mask=key_padding_mask)
        broadcast_context = text_context.unsqueeze(1).expand_as(object_tokens)
        pair_out = self.pair_fusion(torch.cat([object_tokens, broadcast_context, object_tokens * broadcast_context], dim=-1))
        fused = self.norm(object_tokens + attn_out + pair_out)
        return self.norm(fused + self.ffn(fused))
