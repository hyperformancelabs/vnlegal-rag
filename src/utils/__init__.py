"""Shared utilities: text tokenization, device resolution, file IO."""

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
from .device import default_device
from .io import read_json, read_ready_csv

__all__ = [
    "default_device",
    "read_json",
    "read_ready_csv",
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
