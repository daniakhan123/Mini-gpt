# MiniGPT — a Transformer language model from scratch

A GPT-style causal language model, with every Transformer component (scaled dot-product attention,
multi-head attention, sinusoidal positions, pre-LN blocks, LM head, sampling) written by hand in PyTorch.
No pretrained weights and no `transformers` library are used.

* **Dataset:** [Tiny Shakespeare](https://huggingface.co/datasets/karpathy/tiny_shakespeare) (~1.1 MB, ~40k lines),
  downloaded automatically from the char-rnn mirror to `data/tiny_shakespeare.txt`.
* **Tokenizer:** character-level, `<pad>`=0, `<unk>`=1, then sorted unique characters (vocab ≈ 67).

## Where things live

| What | Where |
|---|---|
| Tokenizer / data pipeline / shifted targets | `src/tokenizer.py`, `src/dataset.py` |
| Scaled dot-product + multi-head attention, causal mask | `src/attention.py` |
| Sinusoidal (and learned) positions | `src/positional_encoding.py` |
| Transformer block (attn + FFN + residual + LayerNorm) | `src/transformer_block.py` |
| MiniGPT model, loss, `generate()` | `src/gpt.py` |
| Training loop, checkpointing, logging | `src/train.py` |
| Inference CLI / REST API | `src/generate.py`, `src/api.py` |
| Unit tests | `tests/` |
| Experiments and evidence figures | `experiments/` |
| Config | `configs/tiny_gpt.yaml` |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Acceptance checks (handout §16)

```bash
pytest -q                                              # unit tests incl. causal-correctness + tiny overfit
python -m src.train --overfit                          # debugging gate: loss must fall substantially
python -m src.train --config configs/tiny_gpt.yaml     # full training -> checkpoints/tiny_gpt.pt
python -m src.generate --prompt "ROMEO:" --max-new-tokens 200 --temperature 0.8   # reload + generate
```

## Evidence figures (handout §12)

```bash
python -m experiments.hand_calculation        # Fig 1 (hand-calc attention) + Fig 2 (causal mask)
python -m experiments.architecture_evidence   # Fig 3 shape trace, Fig 4 positional encoding, Fig 5 model summary
# Fig 6 loss curves:  outputs/figures/fig6_loss_tiny_gpt.png (written by training)
python -m experiments.attention_visualization --checkpoint checkpoints/tiny_gpt.pt        # Fig 7
python -m experiments.evaluate --checkpoint checkpoints/tiny_gpt.pt                        # Fig 8 samples + perplexity
python -m experiments.ablation --experiment all --steps 2000                               # Fig 9 ablations
python -m experiments.mlm_vs_clm                                                           # Lesson 06
python -m experiments.why_transformers                                                     # Lesson 01
```

Any config value can be overridden on the command line: `python -m src.train n_layers=2 max_steps=1000`.

## API (Fig 10) and deployment

```bash
CHECKPOINT_PATH=checkpoints/tiny_gpt.pt uvicorn src.api:app --port 8000
curl -X POST localhost:8000/generate -H 'Content-Type: application/json' \
     -d '{"prompt":"ROMEO:","max_new_tokens":120,"temperature":0.8}'
```

Endpoints: `GET /health`, `GET /info`, `POST /generate` (Swagger UI at `/docs`).

**Render:** push the repo to GitHub, host the checkpoint (GitHub Release or a shareable Drive direct link),
create a *Web Service* from the repo (it picks up `Dockerfile` / `render.yaml`), and set the
`CHECKPOINT_URL` environment variable to the checkpoint's direct-download link.

## Expected output

```
$ python -m src.generate --prompt "ROMEO:" --max-new-tokens 80
Prompt: ROMEO:
Generated: ROMEO: ...
model parameters: 0.80M
context length: 128
temperature: 0.8
```
(The baseline `d_model=128, 4 layers, 4 heads` model has roughly 0.8M parameters; exact numbers are printed by training.)

## Known limitations

* Character-level model: learns Shakespeare-like spelling/format/short phrases but not long-range coherence or meaning.
* Small corpus (~1 MB) → the larger models overfit (validation loss rises while training loss keeps falling).
* Context window is 128 characters, so only a short span of text is visible to the model.
* Generated text is an experimental artifact and is not factual.
* Peak memory is only recorded on CUDA.

## Sources

Attention formulation: Vaswani et al., *Attention Is All You Need* (2017). Pre-LN block / GPT layout: Radford et al. (GPT-2).
Dataset: Karpathy, char-rnn Tiny Shakespeare. Add any other tutorials you consult here.
