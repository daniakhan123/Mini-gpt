"""Evaluate a checkpoint: validation loss / perplexity, fixed-prompt samples, decoding comparison.

    python -m experiments.evaluate --checkpoint checkpoints/tiny_gpt.pt
"""
import argparse
import math
from pathlib import Path

import torch

from src.dataset import get_batch, load_text, train_val_split
from src.generate import generate_text
from src.utils import get_device, load_model_from_checkpoint

PROMPTS = ["ROMEO:", "To be, or not to be", "KING HENRY:\nWhat", "First Citizen:\nBefore we proceed", "The future of"]


def distinct_n(text, n):
    grams = [text[i:i + n] for i in range(len(text) - n + 1)]
    return len(set(grams)) / max(1, len(grams))


def repeated_trigram_rate(text):
    return 1 - distinct_n(text, 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/tiny_gpt.pt")
    ap.add_argument("--data", default=None)
    ap.add_argument("--tokens", type=int, default=250)
    a = ap.parse_args()
    device = get_device("auto")
    model, tok, ckpt = load_model_from_checkpoint(a.checkpoint, device)
    cfg = ckpt["train_config"]
    torch.manual_seed(1234)

    data = a.data or cfg["data_path"]
    if Path(data).exists():
        ids = torch.tensor(tok.encode(load_text(data)), dtype=torch.long)
        _, val = train_val_split(ids, cfg["val_frac"])
        losses = []
        with torch.no_grad():
            for _ in range(100):
                x, y = get_batch(val, 32, model.cfg.context_length, device)
                losses.append(model(x, y)[1].item())
        vl = sum(losses) / len(losses)
        print(f"validation loss: {vl:.4f} | perplexity: {math.exp(vl):.3f} (character-level)")
    else:
        print(f"[skip] dataset {data} not found; skipping loss/perplexity")

    out = ["# Fixed-prompt samples (temperature 0.8)\n"]
    for p in PROMPTS:
        t = generate_text(model, tok, p, a.tokens, 0.8, device=device)
        out.append(f"## Prompt: {p!r}\n{t}\n")
        print(f"\n--- {p!r} ---\n{t}")

    print("\n=== Decoding comparison (same prompt) ===")
    out.append("# Decoding comparison\n")
    for name, kw in [("greedy", dict(temperature=0.0)), ("T=0.5", dict(temperature=0.5)), ("T=0.8", dict(temperature=0.8)),
                     ("T=1.2", dict(temperature=1.2)), ("T=0.8 top-k 20", dict(temperature=0.8, top_k=20))]:
        t = generate_text(model, tok, "ROMEO:", 400, device=device, **kw)
        line = f"{name:<16} distinct-3: {distinct_n(t, 3):.3f} | repeated-trigram rate: {repeated_trigram_rate(t):.3f}"
        print(line); out.append(line + "\n")
    Path("outputs/samples").mkdir(parents=True, exist_ok=True)
    Path("outputs/samples/evaluation.md").write_text("\n".join(out), encoding="utf-8")
    print("\nsaved outputs/samples/evaluation.md")


if __name__ == "__main__":
    main()
