"""Controlled ablations: change ONE factor, keep everything else fixed.

    python -m experiments.ablation --experiment depth   --steps 2000
    python -m experiments.ablation --experiment heads   --steps 2000
    python -m experiments.ablation --experiment context --steps 2000
    python -m experiments.ablation --experiment position --steps 2000
    python -m experiments.ablation --experiment all
Outputs: outputs/ablation_<name>.csv/.md and outputs/figures/fig9_ablation_<name>.png
"""
import argparse
import csv
from dataclasses import replace
from pathlib import Path

from src.train import train
from src.utils import load_config

GRIDS = {
    "depth": ("n_layers", [2, 4, 6]),
    "heads": ("n_heads", [2, 4, 8]),
    "context": ("context_length", [64, 128, 256]),
    "position": ("pos_encoding", ["sinusoidal", "learned"]),
}


def run_experiment(name, base, steps):
    key, values = GRIDS[name]
    rows = []
    for v in values:
        cfg = replace(base, run_name=f"abl_{name}_{v}", max_steps=steps, eval_interval=max(100, steps // 10),
                      sample_tokens=0, **{key: v})
        print(f"\n##### {name}: {key}={v}")
        r = train(cfg, verbose=False, save_ckpt=False)
        rows.append({key: v, "params": r["params"], "train_loss": round(r["final_train_loss"], 4),
                     "val_loss": round(r["final_val_loss"], 4), "val_ppl": round(r["val_perplexity"], 3),
                     "time_s": r["wall_time_s"], "tokens_per_s": r["tokens_per_s"],
                     "peak_mem_mb": None if r["peak_mem_mb"] is None else round(r["peak_mem_mb"], 1)})
        print(rows[-1])
    out = Path("outputs"); (out / "figures").mkdir(parents=True, exist_ok=True)
    with open(out / f"ablation_{name}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    hdr = list(rows[0].keys())
    md = ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)] + ["| " + " | ".join(str(r[h]) for h in hdr) + " |" for r in rows]
    (out / f"ablation_{name}.md").write_text("\n".join(md))
    print("\n".join(md))
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(5, 3.6))
        plt.bar([str(r[key]) for r in rows], [r["val_loss"] for r in rows])
        plt.xlabel(key); plt.ylabel("final validation loss"); plt.title(f"Ablation: {name}")
        plt.tight_layout(); plt.savefig(out / "figures" / f"fig9_ablation_{name}.png", dpi=150); plt.close()
    except ImportError:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/tiny_gpt.yaml")
    ap.add_argument("--experiment", default="all", choices=list(GRIDS) + ["all"])
    ap.add_argument("--steps", type=int, default=2000)
    a = ap.parse_args()
    base = load_config(a.config)
    for name in (GRIDS if a.experiment == "all" else [a.experiment]):
        run_experiment(name, base, a.steps)


if __name__ == "__main__":
    main()
