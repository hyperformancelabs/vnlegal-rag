"""Score-fusion primitives for hybrid retrieval.

Pure NumPy helpers — no model/IO dependency. Used to combine a sparse (TF-IDF)
and a dense (Siamese) score vector that live on different scales.
"""

from __future__ import annotations

import numpy as np


def minmax_norm(scores: np.ndarray) -> np.ndarray:
    """Min-max scale scores to ``[0, 1]``.

    Required before weighted-sum fusion because TF-IDF cosine and Siamese
    dot-product scores occupy different ranges. A degenerate vector (all values
    equal, e.g. a single candidate) maps to zeros — it carries no ranking signal.
    """
    s = np.asarray(scores, dtype=np.float64)
    if s.size == 0:
        return s
    lo = float(s.min())
    hi = float(s.max())
    span = hi - lo
    if span <= 1e-12:
        return np.zeros_like(s)
    return (s - lo) / span


def weighted_sum(
    sparse_scores: np.ndarray,
    dense_scores: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Fuse two **already-normalized** score vectors: ``α·sparse + (1−α)·dense``.

    ``alpha`` weights the sparse (TF-IDF) side; ``alpha=1`` → sparse only,
    ``alpha=0`` → dense only.
    """
    a = float(np.clip(alpha, 0.0, 1.0))
    return a * np.asarray(sparse_scores, dtype=np.float64) + (1.0 - a) * np.asarray(
        dense_scores, dtype=np.float64
    )
