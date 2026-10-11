"""Inference CLI.

    python -m src.generate --prompt "ROMEO:" --max-new-tokens 200 --temperature 0.8
"""
from __future__ import annotations

import argparse

import torch

from .utils import get_device, load_model_from_checkpoint


def generate_text(model, tok, prompt: str, max_new_tokens=100, temperature=0.8, top_k=None, top_p=None, device="cpu") -> str:
    ids = torch.tensor([tok.encode(prompt) or [tok.pad_id]], dtype=torch.long, device=device)
    out = model.generate(ids, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k, top_p=top_p)
    return tok.decode(out[0].tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/tiny_gpt.pt")
    ap.add_argument("--prompt", default="ROMEO:")
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.8, help="0 = greedy")
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--top-p", type=float, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()
    if a.seed is not None:
        torch.manual_seed(a.seed)
    device = get_device(a.device)
    model, tok, _ = load_model_from_checkpoint(a.checkpoint, device)
    text = generate_text(model, tok, a.prompt, a.max_new_tokens, a.temperature, a.top_k, a.top_p, device)
    n = sum(p.numel() for p in model.parameters())
    print(f"Prompt: {a.prompt}")
    print(f"Generated: {text}")
    print(f"model parameters: {n / 1e6:.2f}M")
    print(f"context length: {model.cfg.context_length}")
    print(f"temperature: {a.temperature}")


if __name__ == "__main__":
    main()
