"""Embedding-matrix construction from pre-trained word2vec + random fallback.

The word2vec file (standard text format: first line ``vocab_size dim``, then
``word v1 v2 ... v300`` per line) is streamed so only vectors for tokens present
in the project vocabulary are kept in memory.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .constants import PAD_IDX, PAD_TOKEN, UNK_IDX, UNK_TOKEN


def load_word2vec_subset(
    w2v_path: str | Path,
    vocab: set[str],
    *,
    encoding: str = "utf-8",
) -> dict[str, np.ndarray]:
    """Stream a word2vec text file; return vectors only for words in ``vocab``.

    Returns
    -------
    dict[str, np.ndarray]
        ``{word: float32 vector}`` for every vocab word found in the file.
        Words not found in the file are absent from the dict.
    """
    path = Path(w2v_path)
    word_vectors: dict[str, np.ndarray] = {}
    found = 0

    with open(path, encoding=encoding, errors="replace") as f:
        header = f.readline()
        parts = header.split()
        if len(parts) != 2:
            raise ValueError(
                f"Expected word2vec header 'vocab_size dim', got: {header.strip()!r}"
            )
        file_dim = int(parts[1])

        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            # The word may contain spaces; split from the right.
            tokens = line.rsplit(maxsplit=file_dim)
            if len(tokens) != file_dim + 1:
                continue
            word = tokens[0]
            if word in vocab:
                word_vectors[word] = np.array(tokens[1:], dtype=np.float32)
                found += 1
                if found >= len(vocab):
                    break  # all vocab words found — stop early

    return word_vectors


def build_embedding_matrix(
    stoi: dict[str, int],
    embed_dim: int,
    *,
    w2v_path: str | Path | None = None,
    w2v_encoding: str = "utf-8",
    random_seed: int = 42,
) -> tuple[torch.Tensor, int]:
    """Build an embedding matrix for the given vocabulary.

    Strategy (per token):
    * ``<PAD>`` → zero vector
    * ``<UNK>`` → random normal init
    * in word2vec   → pre-trained vector
    * not in word2vec → random normal init

    Parameters
    ----------
    stoi:
        String-to-index vocabulary dict.
    embed_dim:
        Desired embedding dimension.  When a word2vec file is provided this
        *must* match the dimension in the file header.
    w2v_path:
        Optional path to a word2vec text file.  Omit to random-init all vectors.
    w2v_encoding:
        Text encoding of the word2vec file (default ``utf-8``).
    random_seed:
        Seed for the random normal init (reproducible).

    Returns
    -------
    tuple[torch.Tensor, int]
        ``(matrix, hit_count)`` where ``matrix`` has shape ``(vocab_size, embed_dim)``
        and ``hit_count`` is how many pre-trained vectors were used.
    """
    rng = np.random.default_rng(random_seed)
    vocab_size = len(stoi)
    matrix = rng.normal(0.0, 0.02, (vocab_size, embed_dim)).astype(np.float32)

    # PAD is always zero.
    pad_idx = stoi.get(PAD_TOKEN, PAD_IDX)
    matrix[pad_idx] = 0.0

    if w2v_path is not None:
        vocab_set = set(stoi) - {PAD_TOKEN, UNK_TOKEN}
        w2v = load_word2vec_subset(w2v_path, vocab_set, encoding=w2v_encoding)

        file_dim = matrix.shape[1]
        hits = 0
        for word, vec in w2v.items():
            idx = stoi.get(word)
            if idx is not None and len(vec) == file_dim:
                matrix[idx] = vec
                hits += 1
        return torch.from_numpy(matrix), hits

    return torch.from_numpy(matrix), 0
