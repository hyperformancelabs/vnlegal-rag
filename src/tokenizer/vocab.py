"""Vocabulary building, serialisation, and loading.

All vocab operations are independent of the embedding backend so they remain
testable without a 3.4 GB word2vec file.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from .constants import PAD_TOKEN, TOKENIZER_BACKEND, UNK_TOKEN
from .tokenizer import simple_tokenize


def build_vocab(
    texts: list[str],
    max_vocab: int = 100_000,
    min_freq: int = 1,
    *,
    pad_token: str = PAD_TOKEN,
    unk_token: str = UNK_TOKEN,
) -> dict[str, int]:
    """Build a ``stoi`` dict ordered ``[PAD, UNK, most_frequent...]``."""
    counter: Counter[str] = Counter()
    for text in texts:
        counter.update(simple_tokenize(text))

    stoi: dict[str, int] = {pad_token: 0, unk_token: 1}
    for token, freq in counter.most_common(max_vocab - 2):
        if freq < min_freq:
            break
        if token not in stoi:
            stoi[token] = len(stoi)
    return stoi


def save_vocab(
    stoi: dict[str, int],
    path: Path,
    *,
    build_metadata: dict | None = None,
) -> None:
    """Persist ``stoi`` and metadata as ``tokenizer_vocab.json``."""
    itos = {str(idx): token for token, idx in stoi.items()}
    payload: dict = {
        "stoi": stoi,
        "itos": itos,
        "tokenizer_backend": TOKENIZER_BACKEND,
    }
    if build_metadata:
        payload["build_metadata"] = build_metadata
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_vocab(path: Path) -> dict[str, int]:
    """Read ``tokenizer_vocab.json`` and return the ``stoi`` dict.

    Emits a :class:`UserWarning` when the on-disk vocab was built with a
    different tokenizer backend.
    """
    with open(path, encoding="utf-8") as f:
        data: dict = json.load(f)

    backend = data.get("tokenizer_backend", "unknown")
    if backend not in ("simple_tokenize", "regex_word", "unknown"):
        import warnings

        warnings.warn(
            f"Vocab at {path} was built with '{backend}' tokenizer, "
            f"but current pipeline uses 'simple_tokenize'. "
            f"Consider rebuilding the vocab for consistent results.",
            UserWarning,
            stacklevel=2,
        )
    return data["stoi"]
