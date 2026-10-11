"""Training pipeline for MiniGPT.

Usage:
    python -m src.train --config configs/tiny_gpt.yaml
    python -m src.train --config configs/tiny_gpt.yaml --overfit          # debugging gate
    python -m src.train --config configs/tiny_gpt.yaml n_layers=2 max_steps=2000   # overrides
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import torch
import yaml

from .dataset import download_tiny_shakespeare, get_batch, load_text, train_val_split
from .gpt import GPT, GPTConfig
from .tokenizer import CharTokenizer
from .utils import TrainConfig, get_device, load_config, save_checkpoint, set_seed


def lr_at(step: int, cfg: TrainConfig) -> float:
    """Linear warmup then cosine decay to min_lr."""
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    if step >= cfg.max_steps:
        return cfg.min_lr
    progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    return cfg.min_lr + 0.5 * (1 + math.cos(math.pi * progress)) * (cfg.lr - cfg.min_lr)


@torch.no_grad()
def estimate_loss(model, splits: dict, cfg: TrainConfig, device, fixed_batch=None) -> dict:
    model.eval()
    out = {}
    for name, data in splits.items():
        losses = []
        for _ in range(1 if fixed_batch is not None else cfg.eval_iters):
            x, y = fixed_batch if fixed_batch is not None else get_batch(data, cfg.batch_size, cfg.context_length, device)
            _, loss = model(x, y)
            losses.append(loss.item())
        out[name] = sum(losses) / len(losses)
    model.train()
    return out


def sample_text(model, tok, prompt, n_tokens, temperature, device) -> str:
    ids = torch.tensor([tok.encode(prompt)], dtype=torch.long, device=device)
    out = model.generate(ids, max_new_tokens=n_tokens, temperature=temperature)
    return tok.decode(out[0].tolist())


def plot_curves(rows, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    steps = [r["step"] for r in rows]
    plt.figure(figsize=(6, 4))
    plt.plot(steps, [r["train_loss"] for r in rows], label="train")
    plt.plot(steps, [r["val_loss"] for r in rows], label="validation")
    plt.xlabel("step"); plt.ylabel("cross-entropy loss"); plt.legend(); plt.grid(alpha=0.3)
    plt.title("Training vs validation loss")
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


def train(cfg: TrainConfig, verbose: bool = True, text: str | None = None, save_ckpt: bool = True) -> dict:
    set_seed(cfg.seed)
    device = get_device(cfg.device)

    if text is None:
        if not Path(cfg.data_path).exists() and "shakespeare" in cfg.data_path:
            download_tiny_shakespeare(cfg.data_path)
        text = load_text(cfg.data_path)
    tok = CharTokenizer.from_text(text)
    ids = torch.tensor(tok.encode(text), dtype=torch.long)
    train_ids, val_ids = train_val_split(ids, cfg.val_frac)

    dropout = 0.0 if cfg.overfit else cfg.dropout
    model_cfg = GPTConfig(
        vocab_size=tok.vocab_size, context_length=cfg.context_length, d_model=cfg.d_model,
        n_heads=cfg.n_heads, n_layers=cfg.n_layers, ffn_mult=cfg.ffn_mult, dropout=dropout,
        pos_encoding=cfg.pos_encoding, tie_weights=cfg.tie_weights, causal=cfg.causal,
    )
    model = GPT(model_cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(model.summary())
        print(f"device={device}  train_tokens={len(train_ids):,}  val_tokens={len(val_ids):,}")

    decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.dim() < 2]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": cfg.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
        lr=cfg.lr, betas=(0.9, 0.95),
    )

    fixed_batch = None
    splits = {"train": train_ids, "val": val_ids}
    if cfg.overfit:  # tiny repeated batch: loss must fall substantially
        g = torch.Generator().manual_seed(cfg.seed)
        fixed_batch = get_batch(train_ids, min(cfg.batch_size, 8), cfg.context_length, device, g)

    use_amp = device.type == "cuda" and torch.cuda.is_bf16_supported()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    out_dir = Path(cfg.out_dir)
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    (out_dir / "samples").mkdir(parents=True, exist_ok=True)
    ckpt_path = Path(cfg.ckpt_dir) / f"{cfg.run_name}.pt"

    rows, samples, best_val = [], [], float("inf")
    tokens_seen, t0 = 0, time.time()
    model.train()
    for step in range(cfg.max_steps + 1):
        lr = lr_at(step, cfg)
        for g_ in opt.param_groups:
            g_["lr"] = lr

        if step % cfg.eval_interval == 0 or step == cfg.max_steps:
            losses = estimate_loss(model, splits, cfg, device, fixed_batch)
            elapsed = time.time() - t0
            row = {
                "step": step, "epoch": round(tokens_seen / max(1, len(train_ids)), 4),
                "train_loss": losses["train"], "val_loss": losses["val"], "lr": lr,
                "elapsed_s": round(elapsed, 2), "tokens_per_s": round(tokens_seen / max(elapsed, 1e-9), 1),
            }
            rows.append(row)
            if verbose:
                print(f"step {step:5d} | train {losses['train']:.4f} | val {losses['val']:.4f} | lr {lr:.2e} | {elapsed:.0f}s")
            if not cfg.overfit and step % (cfg.eval_interval * 2) == 0 and cfg.sample_tokens > 0:
                s = sample_text(model, tok, cfg.sample_prompt, min(cfg.sample_tokens, 120), cfg.temperature, device)
                samples.append({"step": step, "text": s})
                if verbose:
                    print("--- sample ---\n" + s + "\n--------------")
            if losses["val"] < best_val and save_ckpt:
                best_val = losses["val"]
                save_checkpoint(ckpt_path, model, tok, cfg, {"step": step, "val_loss": best_val})
            best_val = min(best_val, losses["val"])
        if step == cfg.max_steps:
            break

        x, y = fixed_batch if fixed_batch is not None else get_batch(train_ids, cfg.batch_size, cfg.context_length, device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
            _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        tokens_seen += x.numel()

    wall = time.time() - t0
    final = rows[-1]
    peak_mb = torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else None
    summary = {
        "run_name": cfg.run_name, "params": n_params, "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "wall_time_s": round(wall, 1), "tokens_per_s": round(tokens_seen / wall, 1),
        "peak_mem_mb": peak_mb, "final_train_loss": final["train_loss"], "final_val_loss": final["val_loss"],
        "best_val_loss": best_val, "val_perplexity": math.exp(min(final["val_loss"], 20)),
        "vocab_size": tok.vocab_size, "config": asdict(cfg), "checkpoint": str(ckpt_path) if save_ckpt else None,
    }
    with open(out_dir / "logs" / f"{cfg.run_name}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    with open(out_dir / "logs" / f"{cfg.run_name}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    if samples:
        (out_dir / "samples" / f"{cfg.run_name}_during_training.txt").write_text(
            "\n\n".join(f"=== step {s['step']} ===\n{s['text']}" for s in samples), encoding="utf-8")
    plot_curves(rows, out_dir / "figures" / f"fig6_loss_{cfg.run_name}.png")
    if verbose:
        print(json.dumps({k: v for k, v in summary.items() if k != "config"}, indent=2))
    summary["rows"] = rows
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/tiny_gpt.yaml")
    ap.add_argument("--overfit", action="store_true", help="train on one tiny repeated batch (debug gate)")
    ap.add_argument("overrides", nargs="*", help="key=value overrides, e.g. n_layers=2")
    a = ap.parse_args()
    ov = {}
    for item in a.overrides:
        k, v = item.split("=", 1)
        ov[k] = yaml.safe_load(v)
    if a.overfit:
        ov.setdefault("overfit", True)
        ov.setdefault("run_name", "overfit_test")
        ov.setdefault("max_steps", 300)
        ov.setdefault("eval_interval", 50)
        ov.setdefault("lr", 1e-3)
        ov.setdefault("warmup_steps", 10)
    train(load_config(a.config, ov), save_ckpt=not a.overfit)


if __name__ == "__main__":
    main()
