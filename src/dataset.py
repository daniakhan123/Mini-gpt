"""Data pipeline: download, split, and next-token (shifted) batches."""
from __future__ import annotations

import urllib.request
from pathlib import Path

import torch
from torch.utils.data import Dataset

TINY_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)


def download_tiny_shakespeare(path: str = "data/tiny_shakespeare.txt") -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists():
        print(f"Downloading Tiny Shakespeare -> {p}")
        urllib.request.urlretrieve(TINY_SHAKESPEARE_URL, p)
    return str(p)


def load_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def train_val_split(ids: torch.Tensor, val_frac: float = 0.1):
    """Contiguous split (no shuffling) so validation text is never seen in training."""
    n = int(len(ids) * (1 - val_frac))
    return ids[:n], ids[n:]


def make_xy(chunk: torch.Tensor):
    """chunk has length T+1 -> x = chunk[:-1], y = chunk[1:].
    y[t] is the token that follows x[t]  (next-token / shifted targets)."""
    return chunk[:-1], chunk[1:]


def get_batch(data: torch.Tensor, batch_size: int, context_length: int, device, generator=None):
    """Random contiguous windows -> (x, y), both (B, T)."""
    hi = len(data) - context_length - 1
    ix = torch.randint(0, hi, (batch_size,), generator=generator)
    x = torch.stack([data[i : i + context_length] for i in ix])
    y = torch.stack([data[i + 1 : i + context_length + 1] for i in ix])
    return x.to(device), y.to(device)


class CausalLMDataset(Dataset):
    """Fixed-length contiguous windows with a configurable stride."""

    def __init__(self, ids: torch.Tensor, context_length: int, stride: int | None = None):
        self.ids = ids
        self.T = context_length
        self.stride = stride or context_length
        self.n = max(0, (len(ids) - context_length - 1) // self.stride + 1)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        s = i * self.stride
        return make_xy(self.ids[s : s + self.T + 1])
