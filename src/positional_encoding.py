"""Positional encodings: sinusoidal (required) and learned (optional experiment)."""
from __future__ import annotations

import math

import torch
import torch.nn as nn


def sinusoidal_table(max_len: int, d_model: int) -> torch.Tensor:
    """PE[pos, 2i]   = sin(pos / 10000^(2i/d_model))
       PE[pos, 2i+1] = cos(pos / 10000^(2i/d_model))      -> (max_len, d_model)"""
    pe = torch.zeros(max_len, d_model)
    pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div[: d_model // 2])
    return pe


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096, dropout: float = 0.0):
        super().__init__()
        self.register_buffer("pe", sinusoidal_table(max_len, d_model), persistent=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, T, d_model)
        return self.drop(x + self.pe[: x.size(1)].unsqueeze(0))


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096, dropout: float = 0.0):
        super().__init__()
        self.emb = nn.Embedding(max_len, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pos = torch.arange(x.size(1), device=x.device)
        return self.drop(x + self.emb(pos).unsqueeze(0))
