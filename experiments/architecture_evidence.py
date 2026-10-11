"""Figures 3, 4, 5: multi-head shape trace, positional-encoding plot, model summary."""
import argparse
from pathlib import Path

import torch

from src.attention import MultiHeadAttention
from src.gpt import GPT, GPTConfig
from src.positional_encoding import sinusoidal_table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--context", type=int, default=128)
    ap.add_argument("--vocab", type=int, default=65)
    a = ap.parse_args()
    Path("outputs/figures").mkdir(parents=True, exist_ok=True)

    print("=== Figure 3: multi-head tensor-shape trace ===")
    mha = MultiHeadAttention(a.d_model, a.heads); mha.trace = []
    mha(torch.randn(2, 10, a.d_model))
    for name, shape in mha.trace:
        print(f"  {name:<24} {shape}")
    print(f"  (B=2, T=10, d_model={a.d_model}, H={a.heads}, d_head={a.d_model // a.heads})")

    print("\n=== Figure 5: model summary ===")
    model = GPT(GPTConfig(vocab_size=a.vocab, context_length=a.context, d_model=a.d_model, n_heads=a.heads, n_layers=a.layers))
    print(model.summary())

    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    pe = sinusoidal_table(a.context, a.d_model)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    im = ax[0].imshow(pe.numpy(), aspect="auto", cmap="RdBu")
    ax[0].set_title("Sinusoidal PE (position x dimension)"); ax[0].set_xlabel("dimension"); ax[0].set_ylabel("position")
    fig.colorbar(im, ax=ax[0])
    for d in [0, 1, 8, 9, 32]:
        ax[1].plot(pe[:, d].numpy(), label=f"dim {d}")
    ax[1].set_title("Selected dimensions: frequency falls with dim index"); ax[1].set_xlabel("position"); ax[1].legend()
    plt.tight_layout(); plt.savefig("outputs/figures/fig4_positional_encoding.png", dpi=150)
    print("saved outputs/figures/fig4_positional_encoding.png")


if __name__ == "__main__":
    main()
