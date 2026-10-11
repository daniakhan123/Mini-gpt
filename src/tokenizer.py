"""Deterministic character-level tokenizer.

Vocabulary policy
-----------------
* id 0 = <pad>, id 1 = <unk> (special tokens, always present).
* All remaining ids are the sorted unique characters of the training text,
  so building the vocab from the same corpus always gives the same mapping.
* Characters unseen at training time are encoded as <unk>.
"""
from __future__ import annotations

import json

PAD, UNK = "<pad>", "<unk>"


class CharTokenizer:
    def __init__(self, chars: list[str]):
        self.itos: list[str] = [PAD, UNK] + list(chars)
        self.stoi: dict[str, int] = {c: i for i, c in enumerate(self.itos)}
        self.pad_id = self.stoi[PAD]
        self.unk_id = self.stoi[UNK]

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        return cls(sorted(set(text)))

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def encode(self, text: str) -> list[int]:
        return [self.stoi.get(c, self.unk_id) for c in text]

    def decode(self, ids) -> str:
        specials = {self.pad_id, self.unk_id}
        return "".join(self.itos[int(i)] for i in ids if int(i) not in specials)

    def to_dict(self) -> dict:
        return {"chars": self.itos[2:]}

    @classmethod
    def from_dict(cls, d: dict) -> "CharTokenizer":
        return cls(d["chars"])

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f)

    @classmethod
    def load(cls, path: str) -> "CharTokenizer":
        with open(path) as f:
            return cls.from_dict(json.load(f))
