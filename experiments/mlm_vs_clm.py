"""Lesson 06: bidirectional (BERT-style) vs causal (GPT) masks.

1. Draws both mask matrices.
2. Leakage experiment: train the same tiny model on the NEXT-TOKEN objective with a
   bidirectional mask. The model can read the answer from the future, so loss collapses
   (cheating) and the model is useless for generation. The causal model learns honestly.

    python -m experiments.mlm_vs_clm --steps 600
"""
import argparse
from dataclasses import replace
from pathlib import Path

import torch

from src.attention import causal_mask
from src.train import train
from src.utils import TrainConfig, load_config

DEMO_TEXT = ("To be, or not to be, that is the question: Whether 'tis nobler in the mind to suffer "
             "the slings and arrows of outrageous fortune, or to take arms against a sea of troubles. ") * 60


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/tiny_gpt.yaml")
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--data", default=None, help="optional text file; defaults to config data_path")
    a = ap.parse_args()

    T = 8
    print("Bidirectional (BERT) mask - every position sees every position:\n", torch.ones(T, T, dtype=torch.int))
    print("Causal (GPT) mask - position i sees j <= i:\n", causal_mask(T).int())

    base = load_config(a.config, {"max_steps": a.steps, "eval_interval": max(50, a.steps // 6), "n_layers": 2, "d_model": 64,
                                     "context_length": 64, "batch_size": 32, "sample_tokens": 0,
                                        "lr": 2e-3, "min_lr": 2e-4, "warmup_steps": 50})
    text = Path(a.data).read_text() if a.data else (Path(base.data_path).read_text() if Path(base.data_path).exists() else DEMO_TEXT)
    res = {}
    for name, causal in [("causal_gpt", True), ("bidirectional_leaky", False)]:
        cfg = replace(base, run_name=name, causal=causal)
        res[name] = train(cfg, verbose=False, text=text, save_ckpt=False)
        print(f"{name:<20} final train loss {res[name]['final_train_loss']:.4f} | val loss {res[name]['final_val_loss']:.4f}")
    print("\nObservation: the bidirectional model reaches a much lower loss on next-token prediction.")
    print("Interpretation: it is not a better language model - it reads the target token through attention."
          " That is why BERT uses a *masked-token* objective and GPT needs the causal mask.")

    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    ax[0].imshow(torch.ones(T, T), cmap="Greys", vmin=0, vmax=1.5); ax[0].set_title("BERT: bidirectional")
    ax[1].imshow(causal_mask(T).float(), cmap="Greys", vmin=0, vmax=1.5); ax[1].set_title("GPT: causal")
    for n, r in res.items():
        ax[2].plot([x["step"] for x in r["rows"]], [x["val_loss"] for x in r["rows"]], label=n)
    ax[2].set_title("Next-token val loss"); ax[2].set_xlabel("step"); ax[2].legend()
    Path("outputs/figures").mkdir(parents=True, exist_ok=True)
    plt.tight_layout(); plt.savefig("outputs/figures/mlm_vs_clm.png", dpi=150)
    print("saved outputs/figures/mlm_vs_clm.png")


if __name__ == "__main__":
    main()
