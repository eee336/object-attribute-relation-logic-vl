from __future__ import annotations

import re
from dataclasses import dataclass, field

from .torch_utils import require_torch


torch, nn = require_torch()


@dataclass
class SimpleTokenizer:
    token_to_id: dict[str, int] = field(default_factory=lambda: {"<pad>": 0, "<unk>": 1})
    max_length: int = 32

    @property
    def pad_id(self) -> int:
        return self.token_to_id["<pad>"]

    @property
    def unk_id(self) -> int:
        return self.token_to_id["<unk>"]

    def build_vocab(self, instructions: list[str], min_freq: int = 1) -> None:
        counts: dict[str, int] = {}
        for text in instructions:
            for token in self.tokenize(text):
                counts[token] = counts.get(token, 0) + 1
        for token in sorted(counts):
            if counts[token] >= min_freq and token not in self.token_to_id:
                self.token_to_id[token] = len(self.token_to_id)

    def tokenize(self, text: str) -> list[str]:
        return re.findall(r"[a-z0-9_]+", text.lower())

    def encode(self, text: str, max_length: int | None = None) -> list[int]:
        length = max_length or self.max_length
        ids = [self.token_to_id.get(token, self.unk_id) for token in self.tokenize(text)]
        ids = ids[:length]
        return ids + [self.pad_id] * (length - len(ids))

    def to_dict(self) -> dict:
        return {"token_to_id": self.token_to_id, "max_length": self.max_length}

    @classmethod
    def from_dict(cls, data: dict) -> "SimpleTokenizer":
        return cls(token_to_id=dict(data["token_to_id"]), max_length=int(data.get("max_length", 32)))

    def __len__(self) -> int:
        return len(self.token_to_id)


class TextEncoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids):
        embedded = self.dropout(self.embedding(input_ids.long()))
        _, hidden = self.gru(embedded)
        return hidden[-1]


class ObjectEncoder(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, object_features):
        return self.net(object_features.float())


class SimpleCNNImageEncoder(nn.Module):
    """Small CPU-friendly CNN stub for future RGB image experiments."""

    def __init__(self, in_channels: int = 3, hidden_dim: int = 128):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.proj = nn.Linear(32, hidden_dim)

    def forward(self, images):
        features = self.cnn(images.float()).flatten(1)
        return self.proj(features)
