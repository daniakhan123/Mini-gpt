import torch

from src.dataset import make_xy
from src.gpt import GPT, GPTConfig


def test_target_shifting():
    chunk = torch.arange(10)
    x, y = make_xy(chunk)
    assert torch.equal(y[:-1], x[1:]) and y[-1] == 9 and x[0] == 0


def test_tiny_overfit():
    """A tiny repeated batch must be memorised: proves gradients + target shifting work."""
    torch.manual_seed(0)
    data = torch.randint(0, 20, (4, 33))
    x, y = data[:, :-1], data[:, 1:]
    model = GPT(GPTConfig(vocab_size=20, context_length=32, d_model=64, n_heads=4, n_layers=2, dropout=0.0))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    _, first = model(x, y)
    for _ in range(300):
        _, loss = model(x, y)
        opt.zero_grad(); loss.backward(); opt.step()
    assert loss.item() < 0.5 * first.item(), (first.item(), loss.item())
