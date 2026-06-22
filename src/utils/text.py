"""Backward-compatibility shim — re-exports everything from :mod:`src.tokenizer`.

New code should ``from src.tokenizer import ...`` directly.
This module exists only so old import paths don't break.
"""

from __future__ import annotations

from ..tokenizer import (
    PAD_TOKEN,
    TOKENIZER_BACKEND,
    UNK_TOKEN,
    build_vocab,
    encode_text,
    encode_with_mask,
    load_vocab,
    save_vocab,
    simple_tokenize,
)

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
