"""Figure 7: attention heatmaps from a trained model (all heads of chosen layers).

    python -m experiments.attention_visualization --checkpoint checkpoints/tiny_gpt.pt --text "ROMEO:\nWhat light"
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from src.utils import load_model_from_checkpoint


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/tiny_gpt.pt")
    ap.add_argument("--text", default="ROMEO:\nWhat light through yonder window")
    ap.add_argument("--layers", type=int, nargs="*", default=None)
    a = ap.parse_args()

    model, tok, _ = load_model_from_checkpoint(a.checkpoint, "cpu")
    text = a.text.encode().decode("unicode_escape")[: model.cfg.context_length]
    idx = torch.tensor([tok.encode(text)])
    model.set_store_attn(True)
    with torch.no_grad():
        model(idx)
    layers = a.layers if a.layers is not None else [0, model.cfg.n_layers - 1]
    H = model.cfg.n_heads
    labels = [c if c != "\n" else "\\n" for c in text]
    fig, axes = plt.subplots(len(layers), H, figsize=(3.6 * H, 3.6 * len(layers)), squeeze=False)
    for r, L in enumerate(layers):
        w = model.blocks[L].attn.last_attn[0]  # (H, T, T)
        for h in range(H):
            ax = axes[r][h]
            ax.imshow(w[h].numpy(), cmap="magma", vmin=0)
            ax.set_title(f"layer {L} head {h}", fontsize=9)
            if len(labels) <= 40:
                ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, fontsize=5)
                ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=5)
            ax.set_xlabel("key (attended-to)", fontsize=7); ax.set_ylabel("query", fontsize=7)
    plt.tight_layout()
    Path("outputs/figures").mkdir(parents=True, exist_ok=True)
    plt.savefig("outputs/figures/fig7_attention.png", dpi=150)
    print("saved outputs/figures/fig7_attention.png (upper triangle is zero: causal mask)")


if __name__ == "__main__":
    main()
