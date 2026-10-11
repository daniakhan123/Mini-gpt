import math

import torch
import torch.nn.functional as F

from src.attention import scaled_dot_product_attention

Q = torch.tensor([[1., 0.], [0., 1.], [1., 1.]])
K = torch.tensor([[1., 0.], [0., 1.], [1., 1.]])
V = torch.tensor([[1., 0.], [0., 2.], [1., 1.]])


def test_hand_calculation_matches_function():
    scores = Q @ K.T / math.sqrt(2)
    expected_w = torch.softmax(scores, dim=-1)
    out, w = scaled_dot_product_attention(Q, K, V)
    assert torch.allclose(w, expected_w, atol=1e-6)
    assert torch.allclose(out, expected_w @ V, atol=1e-6)


def test_rows_sum_to_one_and_shapes():
    q = torch.randn(2, 4, 5, 8)
    out, w = scaled_dot_product_attention(q, q, q)
    assert out.shape == (2, 4, 5, 8) and w.shape == (2, 4, 5, 5)
    assert torch.allclose(w.sum(-1), torch.ones(2, 4, 5), atol=1e-5)


def test_matches_pytorch_reference():
    q, k, v = (torch.randn(2, 3, 6, 4) for _ in range(3))
    ours, _ = scaled_dot_product_attention(q, k, v, causal=True)
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    assert torch.allclose(ours, ref, atol=1e-5)
