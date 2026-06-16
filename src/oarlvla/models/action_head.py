from __future__ import annotations

import math

from .torch_utils import require_torch


torch, nn = require_torch()
F = torch.nn.functional


def sinusoidal_time_embedding(timestep, dim: int, min_period: float = 4e-3, max_period: float = 4.0):
    """Sinusoidal timestep embedding in the same spirit as SmolVLA's flow head."""
    if dim <= 0:
        raise ValueError("dim must be positive")
    half_dim = dim // 2
    if half_dim == 0:
        return timestep[:, None]
    device = timestep.device
    dtype = timestep.dtype
    periods = torch.exp(
        torch.linspace(math.log(min_period), math.log(max_period), half_dim, device=device, dtype=dtype)
    )
    angles = timestep[:, None] / periods[None, :]
    emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


def _valid_num_heads(hidden_dim: int, preferred_heads: int) -> int:
    for heads in range(max(1, preferred_heads), 0, -1):
        if hidden_dim % heads == 0:
            return heads
    return 1


class MLPActionHead(nn.Module):
    def __init__(self, hidden_dim: int, action_dim: int = 3):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, action_dim))

    def forward(self, selected_or_fused_target_token):
        return self.net(selected_or_fused_target_token)


class SmolStyleFlowActionHead(nn.Module):
    """Lightweight SmolVLA-style flow-matching action chunk head.

    SmolVLA trains an action expert to predict the velocity field between
    Gaussian noise and an action chunk. This module keeps that training signal
    while using a compact Transformer decoder instead of the full SmolVLM expert,
    so it remains usable in the current research prototype.
    """

    def __init__(
        self,
        hidden_dim: int,
        action_dim: int = 3,
        chunk_size: int = 8,
        num_steps: int = 10,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        min_period: float = 4e-3,
        max_period: float = 4.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.num_steps = num_steps
        self.min_period = min_period
        self.max_period = max_period
        heads = _valid_num_heads(hidden_dim, num_heads)
        self.action_in_proj = nn.Linear(action_dim, hidden_dim)
        self.action_time_mlp_in = nn.Linear(hidden_dim * 2, hidden_dim)
        self.action_time_mlp_out = nn.Linear(hidden_dim, hidden_dim)
        self.position_embedding = nn.Parameter(torch.zeros(1, chunk_size, hidden_dim))
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.action_expert = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.action_out_proj = nn.Linear(hidden_dim, action_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def sample_noise(self, shape, device):
        return torch.normal(mean=0.0, std=1.0, size=shape, device=device)

    def sample_time(self, batch_size: int, device):
        beta_dist = torch.distributions.Beta(concentration1=1.5, concentration0=1.0)
        time = beta_dist.sample((batch_size,)).to(device=device, dtype=torch.float32)
        return time * 0.999 + 0.001

    def format_actions(self, actions):
        if actions.dim() == 2:
            actions = actions[:, None, :]
        if actions.shape[-1] < self.action_dim:
            actions = F.pad(actions, (0, self.action_dim - actions.shape[-1]))
        elif actions.shape[-1] > self.action_dim:
            actions = actions[..., : self.action_dim]
        if actions.shape[1] == self.chunk_size:
            return actions
        if actions.shape[1] == 1:
            return actions.expand(-1, self.chunk_size, -1)
        if actions.shape[1] > self.chunk_size:
            return actions[:, : self.chunk_size]
        pad = actions[:, -1:, :].expand(-1, self.chunk_size - actions.shape[1], -1)
        return torch.cat([actions, pad], dim=1)

    def embed_suffix(self, noisy_actions, timestep):
        action_emb = self.action_in_proj(noisy_actions.float())
        time_emb = sinusoidal_time_embedding(
            timestep.float(),
            self.hidden_dim,
            min_period=self.min_period,
            max_period=self.max_period,
        ).type_as(action_emb)
        time_emb = time_emb[:, None, :].expand_as(action_emb)
        action_time_emb = torch.cat([action_emb, time_emb], dim=-1)
        action_time_emb = self.action_time_mlp_in(action_time_emb)
        action_time_emb = F.silu(action_time_emb)
        action_time_emb = self.action_time_mlp_out(action_time_emb)
        return self.norm(action_time_emb + self.position_embedding[:, : action_time_emb.shape[1]])

    def denoise_step(self, context_tokens, x_t, timestep, context_mask=None):
        suffix = self.embed_suffix(x_t, timestep)
        memory_key_padding_mask = None
        if context_mask is not None:
            memory_key_padding_mask = ~context_mask.bool()
        out = self.action_expert(
            tgt=suffix,
            memory=context_tokens,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        return self.action_out_proj(out.float())

    def forward(self, context_tokens, actions, context_mask=None, noise=None, time=None):
        actions = self.format_actions(actions).to(context_tokens.device)
        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)
        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)
        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1.0 - time_expanded) * actions
        velocity_target = noise - actions
        velocity_pred = self.denoise_step(context_tokens, x_t, time, context_mask=context_mask)
        losses = F.mse_loss(velocity_pred, velocity_target, reduction="none")
        clean_action_pred = x_t - time_expanded * velocity_pred
        return {
            "action_flow_losses": losses,
            "action_velocity_pred": velocity_pred,
            "action_velocity_target": velocity_target,
            "action_chunk_pred": clean_action_pred,
            "action_noise": noise,
            "action_time": time,
        }

    @torch.no_grad()
    def sample(self, context_tokens, context_mask=None, noise=None, num_steps: int | None = None):
        batch_size = context_tokens.shape[0]
        device = context_tokens.device
        if noise is None:
            noise = self.sample_noise((batch_size, self.chunk_size, self.action_dim), device)
        x_t = noise
        steps = num_steps or self.num_steps
        dt = -1.0 / max(steps, 1)
        for step in range(steps):
            time = 1.0 + step * dt
            timestep = torch.full((batch_size,), time, dtype=torch.float32, device=device)
            velocity = self.denoise_step(context_tokens, x_t, timestep, context_mask=context_mask)
            x_t = x_t + dt * velocity
        return x_t


ActionHead = MLPActionHead
