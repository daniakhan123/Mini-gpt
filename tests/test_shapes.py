import torch

from src.attention import MultiHeadAttention
from src.gpt import GPT, GPTConfig
from src.positional_encoding import LearnedPositionalEncoding, SinusoidalPositionalEncoding, sinusoidal_table
from src.transformer_block import TransformerBlock


def test_split_merge_roundtrip():
    mha = MultiHeadAttention(32, 4)
    x = torch.randn(3, 7, 32)
    h = mha.split_heads(x)
    assert h.shape == (3, 4, 7, 8)
    assert torch.equal(mha.merge_heads(h), x)


def test_mha_shape_trace():
    mha = MultiHeadAttention(32, 4); mha.trace = []
    y = mha(torch.randn(2, 10, 32))
    assert y.shape == (2, 10, 32)
    shapes = dict(mha.trace)
    assert shapes["split heads Q"] == (2, 4, 10, 8)
    assert shapes["attention weights"] == (2, 4, 10, 10)
    assert shapes["concat heads"] == (2, 10, 32)


def test_positional_encoding():
    pe = sinusoidal_table(50, 16)
    assert pe.shape == (50, 16)
    assert torch.allclose(pe[0, 0::2], torch.zeros(8)) and torch.allclose(pe[0, 1::2], torch.ones(8))
    x = torch.zeros(2, 10, 16)
    assert SinusoidalPositionalEncoding(16, 50)(x).shape == x.shape
    assert LearnedPositionalEncoding(16, 50)(x).shape == x.shape


def test_block_and_gpt_shapes():
    blk = TransformerBlock(32, 4, 128)
    assert blk(torch.randn(2, 9, 32)).shape == (2, 9, 32)
    cfg = GPTConfig(vocab_size=40, context_length=16, d_model=32, n_heads=4, n_layers=2)
    m = GPT(cfg)
    idx = torch.randint(0, 40, (3, 16)); tgt = torch.randint(0, 40, (3, 16))
    logits, loss = m(idx, tgt)
    assert logits.shape == (3, 16, 40) and loss.ndim == 0
    assert sum(p.numel() for p in m.parameters()) > 0
    out = m.generate(idx[:, :4], max_new_tokens=5)
    assert out.shape == (3, 9)


def test_initial_loss_near_log_vocab():
    import math
    torch.manual_seed(0)
    m = GPT(GPTConfig(vocab_size=65, context_length=16, d_model=32, n_heads=4, n_layers=2, dropout=0.0)).eval()
    idx = torch.randint(0, 65, (8, 16))
    _, loss = m(idx, torch.randint(0, 65, (8, 16)))
    assert abs(loss.item() - math.log(65)) < 0.5
