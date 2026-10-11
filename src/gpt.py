"""MiniGPT: token embedding + position -> N decoder blocks -> LayerNorm -> LM head."""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .positional_encoding import LearnedPositionalEncoding, SinusoidalPositionalEncoding
from .transformer_block import TransformerBlock


@dataclass
class GPTConfig:
    vocab_size: int
    context_length: int = 128
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    ffn_mult: int = 4
    dropout: float = 0.1
    pos_encoding: str = "sinusoidal"  # sinusoidal | learned
    tie_weights: bool = False
    causal: bool = True

    @property
    def ffn_dim(self) -> int:
        return self.ffn_mult * self.d_model


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        pos_cls = {"sinusoidal": SinusoidalPositionalEncoding, "learned": LearnedPositionalEncoding}[cfg.pos_encoding]
        self.pos = pos_cls(cfg.d_model, max_len=cfg.context_length, dropout=cfg.dropout)
        self.blocks = nn.ModuleList(
            [TransformerBlock(cfg.d_model, cfg.n_heads, cfg.ffn_dim, cfg.dropout, cfg.causal) for _ in range(cfg.n_layers)]
        )
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_weights:
            self.lm_head.weight = self.tok_emb.weight
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        """idx: (B, T) token ids. targets: (B, T) ALREADY shifted (targets[:, t] = idx[:, t+1]).
        Returns logits (B, T, V) and, if targets given, mean token-level cross-entropy."""
        B, T = idx.shape
        assert T <= self.cfg.context_length, "sequence longer than context_length"
        x = self.tok_emb(idx) * math.sqrt(self.cfg.d_model)  # scale so PE doesn't dominate/vanish
        x = self.pos(x)
        for blk in self.blocks:
            x = blk(x)
        logits = self.lm_head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=100, temperature=0.8, top_k=None, top_p=None):
        """Autoregressive sampling. temperature<=0 means greedy decoding."""
        was_training = self.training
        self.eval()
        for _ in range(max_new_tokens):
            ctx = idx[:, -self.cfg.context_length :]
            logits, _ = self(ctx)
            logits = logits[:, -1, :]
            if temperature <= 0:
                nxt = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None:
                    kth = torch.topk(logits, min(top_k, logits.size(-1))).values[:, [-1]]
                    logits = logits.masked_fill(logits < kth, float("-inf"))
                if top_p is not None:
                    sl, si = torch.sort(logits, descending=True)
                    cum = F.softmax(sl, dim=-1).cumsum(-1)
                    remove = cum - F.softmax(sl, dim=-1) > top_p
                    sl = sl.masked_fill(remove, float("-inf"))
                    logits = torch.full_like(logits, float("-inf")).scatter(1, si, sl)
                nxt = torch.multinomial(F.softmax(logits, dim=-1), 1)
            idx = torch.cat([idx, nxt], dim=1)
        self.train(was_training)
        return idx

    def set_store_attn(self, flag: bool = True):
        for b in self.blocks:
            b.attn.store_attn = flag

    def summary(self) -> str:
        c = self.cfg
        total = sum(p.numel() for p in self.parameters())
        lines = [
            "MiniGPT summary",
            f"  vocab_size      : {c.vocab_size}",
            f"  context_length  : {c.context_length}",
            f"  d_model         : {c.d_model}",
            f"  n_heads         : {c.n_heads} (d_head={c.d_model // c.n_heads})",
            f"  n_layers        : {c.n_layers}",
            f"  ffn_dim         : {c.ffn_dim}",
            f"  dropout         : {c.dropout}",
            f"  pos_encoding    : {c.pos_encoding}",
            f"  tie_weights     : {c.tie_weights}",
            f"  causal          : {c.causal}",
            "  -- parameters per module --",
        ]
        for name, mod in self.named_children():
            lines.append(f"  {name:<14}: {sum(p.numel() for p in mod.parameters()):,}")
        lines.append(f"  TOTAL           : {total:,} ({total / 1e6:.2f}M)")
        return "\n".join(lines)
