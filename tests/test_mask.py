import torch

from src.attention import MultiHeadAttention, causal_mask, scaled_dot_product_attention
from src.gpt import GPT, GPTConfig


def test_causal_mask_is_lower_triangular():
    m = causal_mask(4)
    assert torch.equal(m, torch.tril(torch.ones(4, 4, dtype=torch.bool)))


def test_future_positions_get_zero_attention():
    q = torch.randn(1, 2, 6, 8)
    _, w = scaled_dot_product_attention(q, q, q, causal=True)
    upper = torch.triu(torch.ones(6, 6, dtype=torch.bool), diagonal=1)
    assert torch.all(w[..., upper] == 0)
    assert torch.allclose(w.sum(-1), torch.ones(1, 2, 6), atol=1e-5)


def test_mha_output_ignores_future_tokens():
    torch.manual_seed(0)
    mha = MultiHeadAttention(16, 4).eval()
    x = torch.randn(1, 8, 16)
    y1 = mha(x)
    x2 = x.clone(); x2[:, 5:] = torch.randn(1, 3, 16)  # change tokens 5..7
    y2 = mha(x2)
    assert torch.allclose(y1[:, :5], y2[:, :5], atol=1e-6)
    assert not torch.allclose(y1[:, 5:], y2[:, 5:])


def test_gpt_never_uses_future_tokens():
    """Causal correctness: logits at position t depend only on tokens <= t."""
    torch.manual_seed(0)
    model = GPT(GPTConfig(vocab_size=30, context_length=16, d_model=32, n_heads=4, n_layers=2, dropout=0.0)).eval()
    a = torch.randint(0, 30, (2, 12))
    b = a.clone(); b[:, 7:] = torch.randint(0, 30, (2, 5))
    la, _ = model(a); lb, _ = model(b)
    assert torch.allclose(la[:, :7], lb[:, :7], atol=1e-5)


def test_non_causal_model_leaks_future():
    torch.manual_seed(0)
    model = GPT(GPTConfig(vocab_size=30, context_length=16, d_model=32, n_heads=4, n_layers=2, dropout=0.0, causal=False)).eval()
    a = torch.randint(0, 30, (1, 12)); b = a.clone(); b[:, 9:] = (b[:, 9:] + 1) % 30
    la, _ = model(a); lb, _ = model(b)
    assert not torch.allclose(la[:, :9], lb[:, :9], atol=1e-6)
