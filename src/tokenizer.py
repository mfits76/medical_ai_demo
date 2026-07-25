"""Minimal character/word tokenizer + vocabulary for the demo."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass


PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"


@dataclass
class Vocab:
    stoi: dict[str, int]
    itos: list[str]

    @property
    def pad_id(self) -> int:
        return self.stoi[PAD_TOKEN]

    @property
    def unk_id(self) -> int:
        return self.stoi[UNK_TOKEN]

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, tokens: list[str], max_len: int) -> list[int]:
        ids = [self.stoi.get(t, self.unk_id) for t in tokens[:max_len]]
        if len(ids) < max_len:
            ids.extend([self.pad_id] * (max_len - len(ids)))
        return ids


def tokenize(text: str) -> list[str]:
    return text.lower().replace(",", " ").replace(".", " ").split()


def build_vocab(texts: list[str], min_freq: int = 2) -> Vocab:
    counts: Counter[str] = Counter()
    for text in texts:
        counts.update(tokenize(text))

    itos = [PAD_TOKEN, UNK_TOKEN]
    for token, freq in counts.most_common():
        if freq >= min_freq:
            itos.append(token)

    stoi = {token: i for i, token in enumerate(itos)}
    return Vocab(stoi=stoi, itos=itos)
