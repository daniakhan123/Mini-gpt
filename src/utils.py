"""Config, seeding, device selection and checkpoint helpers."""
from __future__ import annotations

import random
from dataclasses import dataclass, asdict, fields
from pathlib import Path

import numpy as np
import torch
import yaml


@dataclass
class TrainConfig:
    run_name: str = "tiny_gpt"
    data_path: str = "data/tiny_shakespeare.txt"
    out_dir: str = "outputs"
    ckpt_dir: str = "checkpoints"
    # model
    context_length: int = 128
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    ffn_mult: int = 4
    dropout: float = 0.1
    pos_encoding: str = "sinusoidal"
    tie_weights: bool = False
    causal: bool = True  # False is only used by the MLM-vs-CLM leakage experiment
    # optimisation
    batch_size: int = 64
    lr: float = 3e-4
    min_lr: float = 3e-5
    weight_decay: float = 0.1
    max_steps: int = 5000
    warmup_steps: int = 200
    eval_interval: int = 250
    eval_iters: int = 50
    grad_clip: float = 1.0
    val_frac: float = 0.1
    seed: int = 1337
    device: str = "auto"
    # generation during training
    temperature: float = 0.8
    sample_prompt: str = "ROMEO:"
    sample_tokens: int = 200
    # debugging gate
    overfit: bool = False


def load_config(path: str | None, overrides: dict | None = None) -> TrainConfig:
    data = {}
    if path:
        with open(path) as f:
            data = yaml.safe_load(f) or {}
    data.update(overrides or {})
    valid = {f.name for f in fields(TrainConfig)}
    unknown = set(data) - valid
    if unknown:
        raise ValueError(f"Unknown config keys: {sorted(unknown)}")
    return TrainConfig(**data)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(path, model, tokenizer, cfg: TrainConfig, extra: dict | None = None) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_config": asdict(model.cfg),
            "tokenizer": tokenizer.to_dict(),
            "train_config": asdict(cfg),
            "extra": extra or {},
        },
        path,
    )


def load_model_from_checkpoint(path, device="cpu"):
    """Fresh-process loader: returns (model, tokenizer, ckpt_dict)."""
    from .gpt import GPT, GPTConfig
    from .tokenizer import CharTokenizer

    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = GPT(GPTConfig(**ckpt["model_config"]))
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    tok = CharTokenizer.from_dict(ckpt["tokenizer"])
    return model, tok, ckpt
