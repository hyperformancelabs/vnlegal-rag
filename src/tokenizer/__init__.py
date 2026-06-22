"""Tokenizer subpackage: tokeniser, vocabulary, and embedding-matrix construction.

Canonical home for all text-to-ids logic in the VNLegal RAG pipeline.
Import from here — no :mod:`experiments.tokenizer` copy needed.
"""

from __future__ import annotations

from .constants import PAD_IDX, PAD_TOKEN, TOKENIZER_BACKEND, UNK_IDX, UNK_TOKEN
from .embeddings import build_embedding_matrix, load_word2vec_subset
from .tokenizer import encode_text, encode_with_mask, normalize_vietnamese_syllables, simple_tokenize
from .vocab import build_vocab, load_vocab, save_vocab

__all__ = [
    # tokenizer
    "simple_tokenize",
    "normalize_vietnamese_syllables",
    "encode_text",
    "encode_with_mask",
    # vocab
    "build_vocab",
    "load_vocab",
    "save_vocab",
    # embeddings
    "build_embedding_matrix",
    "load_word2vec_subset",
    # constants
    "PAD_TOKEN",
    "UNK_TOKEN",
    "PAD_IDX",
    "UNK_IDX",
    "TOKENIZER_BACKEND",
]
