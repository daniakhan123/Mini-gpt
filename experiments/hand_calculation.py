"""Figures 1 & 2: worked 3-token attention example (d_k = d_v = 2) and causal mask."""
import math
from pathlib import Path

import torch

Q = torch.tensor([[1., 0.], [0., 1.], [1., 1.]])
K = torch.tensor([[1., 0.], [0., 1.], [1., 1.]])
V = torch.tensor([[1., 0.], [0., 2.], [1., 1.]])
torch.set_printoptions(precision=4, sci_mode=False)


def main():
    print("Q =\n", Q, "\nK =\n", K, "\nV =\n", V)
    s = Q @ K.T
    print("\n1) Q K^T =\n", s)
    ss = s / math.sqrt(2)
    print("\n2) / sqrt(d_k=2) =\n", ss)
    w = torch.softmax(ss, dim=-1)
    print("\n3) row-wise softmax =\n", w, "\n   row sums:", w.sum(-1))
    out = w @ V
    print("\n4) weights @ V =\n", out)
    mask = torch.tril(torch.ones(3, 3, dtype=torch.bool))
    ms = ss.masked_fill(~mask, float("-inf"))
    print("\n5) causal-masked scores =\n", ms)
    mw = torch.softmax(ms, dim=-1)
    print("\n6) masked weights =\n", mw)
    mo = mw @ V
    print("\n7) masked output =\n", mo)
    print("\nInterpretation: token 1 only sees itself (output = V[0]); token 2 averages V[0],V[1];"
          " token 3 sees all three. Without the mask every token mixes information from the future.")

    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    Path("outputs/figures").mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
    for a, m, t in zip(ax, [w, mask.float(), mw], ["softmax weights (no mask)", "causal mask (1=allowed)", "softmax weights (masked)"]):
        im = a.imshow(m.numpy(), cmap="viridis", vmin=0, vmax=1)
        a.set_title(t); a.set_xlabel("key position"); a.set_ylabel("query position")
        a.set_xticks(range(3)); a.set_yticks(range(3))
        for i in range(3):
            for j in range(3):
                a.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center", color="w")
    plt.tight_layout(); plt.savefig("outputs/figures/fig1_fig2_hand_calc_and_mask.png", dpi=150)
    print("saved outputs/figures/fig1_fig2_hand_calc_and_mask.png")


if __name__ == "__main__":
    main()
