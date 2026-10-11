"""FastAPI service.

    CHECKPOINT_PATH=checkpoints/tiny_gpt.pt uvicorn src.api:app --host 0.0.0.0 --port 8000

If the checkpoint file is missing and CHECKPOINT_URL is set (GitHub release / shareable Drive
direct link), it is downloaded at startup.
"""
from __future__ import annotations

import os
import urllib.request
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .generate import generate_text
from .utils import load_model_from_checkpoint

CKPT = os.environ.get("CHECKPOINT_PATH", "checkpoints/tiny_gpt.pt")
CKPT_URL = os.environ.get("CHECKPOINT_URL")

app = FastAPI(title="MiniGPT API", version="1.0")
_state: dict = {}


class GenerateRequest(BaseModel):
    prompt: str = Field("ROMEO:", max_length=500)
    max_new_tokens: int = Field(100, ge=1, le=500)
    temperature: float = Field(0.8, ge=0.0, le=2.0)
    top_k: int | None = Field(None, ge=1)
    top_p: float | None = Field(None, gt=0.0, le=1.0)


@app.on_event("startup")
def load():
    p = Path(CKPT)
    if not p.exists() and CKPT_URL:
        p.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(CKPT_URL, p)
    if not p.exists():
        return  # /health reports model_loaded=false
    model, tok, _ = load_model_from_checkpoint(str(p), "cpu")
    _state.update(model=model, tok=tok)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "model" in _state}


@app.get("/info")
def info():
    if "model" not in _state:
        raise HTTPException(503, "model not loaded")
    m = _state["model"]
    return {"parameters": sum(p.numel() for p in m.parameters()), "context_length": m.cfg.context_length,
            "vocab_size": m.cfg.vocab_size, "n_layers": m.cfg.n_layers, "n_heads": m.cfg.n_heads}


@app.post("/generate")
def generate(req: GenerateRequest):
    if "model" not in _state:
        raise HTTPException(503, "model not loaded; set CHECKPOINT_PATH or CHECKPOINT_URL")
    text = generate_text(_state["model"], _state["tok"], req.prompt, req.max_new_tokens,
                         req.temperature, req.top_k, req.top_p, "cpu")
    return {"prompt": req.prompt, "generated": text, "temperature": req.temperature,
            "context_length": _state["model"].cfg.context_length}
