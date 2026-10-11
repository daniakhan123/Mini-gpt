"""Scaled dot-product attention and multi-head attention (written from scratch)."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def causal_mask(T: int, device=None) -> torch.Tensor:
    """Lower-triangular boolean matrix. True = position j MAY be attended by position i (j <= i)."""
    return torch.tril(torch.ones(T, T, dtype=torch.bool, device=device))


def scaled_dot_product_attention(q, k, v, causal: bool = False, mask=None, dropout: nn.Module | None = None):
    """q,k: (..., T, d_k)   v: (..., T, d_v)
    mask: optional bool tensor broadcastable to (..., T, T), True = allowed.
    Returns (output (..., T, d_v), weights (..., T, T))."""
    d_k = q.size(-1)
    scores = q @ k.transpose(-2, -1) / math.sqrt(d_k)  # (..., T_q, T_k)
    if causal:
        cm = causal_mask(scores.size(-1), device=scores.device)[-scores.size(-2):]
        scores = scores.masked_fill(~cm, float("-inf"))
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    weights = F.softmax(scores, dim=-1)
    if dropout is not None:
        weights = dropout(weights)
    return weights @ v, weights


class MultiHeadAttention(nn.Module):
    """d_model -> (n_heads x d_head) -> attention per head -> concat -> W_o -> d_model."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0, causal: bool = True):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model, self.n_heads, self.d_head = d_model, n_heads, d_model // n_heads
        self.causal = causal
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
        self.store_attn = False       # set True to keep the last attention weights
        self.last_attn = None         # (B, H, T, T)
        self.trace: list | None = None  # set to [] to record tensor shapes

    def _log(self, name, t):
        if self.trace is not None:
            self.trace.append((name, tuple(t.shape)))

    def split_heads(self, x):  # (B, T, D) -> (B, H, T, d_head)
        B, T, _ = x.shape
        return x.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

    def merge_heads(self, x):  # (B, H, T, d_head) -> (B, T, D)
        B, H, T, dh = x.shape
        return x.transpose(1, 2).contiguous().view(B, T, H * dh)

    def forward(self, x, mask=None):
        self._log("input x", x)
        q, k, v = self.w_q(x), self.w_k(x), self.w_v(x)
        self._log("Q/K/V projection", q)
        q, k, v = self.split_heads(q), self.split_heads(k), self.split_heads(v)
        self._log("split heads Q", q)
        out, w = scaled_dot_product_attention(q, k, v, causal=self.causal, mask=mask, dropout=self.attn_drop)
        self._log("attention weights", w)
        self._log("per-head output", out)
        if self.store_attn:
            self.last_attn = w.detach()
        out = self.merge_heads(out)
        self._log("concat heads", out)
        out = self.w_o(out)
        self._log("output projection W_o", out)
        return out
