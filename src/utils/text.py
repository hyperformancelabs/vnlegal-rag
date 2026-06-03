"""Canonical tokenizer for the VNLegal RAG pipeline.

Self-contained copy of the shared ``pipeline_v1.3/tokenizer.py`` so the ``src``
package has no sys.path bootstrap dependency. Rules:

* lowercase + strip, then regex ``\\w+`` (Unicode — keeps Vietnamese diacritics)
* no pyvi / underthesea dependency → runs anywhere

Keep in sync with ``model/tokenizer.py`` and ``pipeline_v1.3/tokenizer.py``.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Optional

TOKENIZER_BACKEND: str = "simple_tokenize"
PAD_TOKEN: str = "<PAD>"
UNK_TOKEN: str = "<UNK>"

__all__ = [
    "simple_tokenize",
    "encode_text",
    "encode_with_mask",
    "build_vocab",
    "load_vocab",
    "save_vocab",
    "TOKENIZER_BACKEND",
    "PAD_TOKEN",
    "UNK_TOKEN",
]


def simple_tokenize(text: str) -> list[str]:
    """Lowercase + regex ``\\w+`` tokenizer (Unicode-aware)."""
    return re.findall(r"\w+", str(text).lower().strip(), flags=re.UNICODE)


def encode_text(
    text: str,
    stoi: dict[str, int],
    max_len: int,
    pad_token: str = PAD_TOKEN,
    unk_token: str = UNK_TOKEN,
) -> list[int]:
    """Tokenize → map to ids → pad/truncate to exactly ``max_len``."""
    unk_id = stoi.get(unk_token, stoi.get("UNK", 1))
    pad_id = stoi.get(pad_token, stoi.get("PAD", 0))
    ids = [stoi.get(t, unk_id) for t in simple_tokenize(text)[:max_len]]
    ids += [pad_id] * (max_len - len(ids))
    return ids


def encode_with_mask(
    text: str,
    stoi: dict[str, int],
    max_len: int,
    pad_token: str = PAD_TOKEN,
    unk_token: str = UNK_TOKEN,
) -> tuple[list[int], list[float]]:
    """Like :func:`encode_text` but also returns a float mask (1.0=token, 0.0=pad)."""
    unk_id = stoi.get(unk_token, stoi.get("UNK", 1))
    pad_id = stoi.get(pad_token, stoi.get("PAD", 0))
    toks = simple_tokenize(text)[:max_len]
    ids = [stoi.get(t, unk_id) for t in toks]
    length = len(ids)
    ids += [pad_id] * (max_len - length)
    mask = [1.0] * length + [0.0] * (max_len - length)
    return ids, mask


def build_vocab(
    texts: list[str],
    max_vocab: int = 100_000,
    min_freq: int = 1,
    pad_token: str = PAD_TOKEN,
    unk_token: str = UNK_TOKEN,
) -> dict[str, int]:
    """Build a ``stoi`` mapping ordered ``[PAD, UNK, most_frequent...]``."""
    counter: Counter = Counter()
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
    build_metadata: Optional[dict] = None,
) -> None:
    """Save vocab JSON with ``stoi``/``itos``/backend (and optional metadata)."""
    itos = {str(idx): token for token, idx in stoi.items()}
    payload: dict = {"stoi": stoi, "itos": itos, "tokenizer_backend": TOKENIZER_BACKEND}
    if build_metadata:
        payload["build_metadata"] = build_metadata
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_vocab(path: Path) -> dict[str, int]:
    """Read ``tokenizer_vocab.json`` and return the ``stoi`` dict."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    backend = data.get("tokenizer_backend", "unknown")
    if backend not in ("simple_tokenize", "regex_word", "unknown"):
        import warnings

        warnings.warn(
            f"Vocab at {path} built with '{backend}' tokenizer but pipeline uses "
            "'simple_tokenize'. Consider rebuilding for consistent results.",
            UserWarning,
            stacklevel=2,
        )
    return data["stoi"]
