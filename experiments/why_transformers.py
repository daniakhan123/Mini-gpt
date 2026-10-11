"""Lesson 01: sequential vs parallel depth.

An RNN needs T dependent steps (sequential depth O(T)); self-attention computes all
positions in one batched matmul (sequential depth O(1)) at O(T^2) memory.
This script measures forward time and attention-matrix memory vs. sequence length.
"""
import time

import torch
import torch.nn as nn

from src.attention import MultiHeadAttention


def time_fn(fn, reps=5):
    fn()
    t = time.time()
    for _ in range(reps):
        fn()
    return (time.time() - t) / reps


def main():
    d, B = 128, 8
    cell = nn.RNNCell(d, d)
    mha = MultiHeadAttention(d, 4)
    print(f"{'T':>6} {'RNN fwd (s)':>12} {'Attn fwd (s)':>13} {'attn matrix MB':>15}")
    with torch.no_grad():
        for T in [32, 64, 128, 256, 512]:
            x = torch.randn(B, T, d)

            def rnn():
                h = torch.zeros(B, d)
                for t in range(T):  # T sequential, dependent steps
                    h = cell(x[:, t], h)
                return h

            t_r, t_a = time_fn(rnn), time_fn(lambda: mha(x))
            mb = B * 4 * T * T * 4 / 2**20
            print(f"{T:>6} {t_r:>12.4f} {t_a:>13.4f} {mb:>15.2f}")
    print("\nSequential depth: RNN = O(T), attention layer = O(1). Memory: attention O(T^2) per head.")


if __name__ == "__main__":
    main()
