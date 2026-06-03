"""Retrieval subpackage: sparse (TF-IDF + topic filter) and dense (Siamese) paths.

Shared ranking helpers (:func:`format_results`, :func:`top_k_order`) live here so
both paths reuse them. ``fusion`` / ``hybrid`` modules are not yet implemented.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .dense import (
    encode_query_siamese,
    encode_siamese_texts,
    precompute_doc_embeddings,
    siamese_scores,
)
from .fusion import minmax_norm, weighted_sum
from .hybrid import hybrid_scores
from .sparse import (
    batch_encode_textcnn,
    build_label_to_indices,
    build_tfidf_index,
    candidate_indices_for_labels,
    predict_topic_topk,
    tfidf_scores,
)


def format_results(
    corpus_df: pd.DataFrame,
    text_col: str,
    indices: np.ndarray,
    scores: np.ndarray,
    extra: dict | None = None,
) -> pd.DataFrame:
    """Build a ranked result frame from corpus rows, scores, and optional columns."""
    out = corpus_df.iloc[indices][["passage_id", "macro_domain", text_col]].copy()
    out.insert(0, "score", np.asarray(scores, dtype=float))
    if extra:
        for key, values in extra.items():
            out[key] = values
    return out.reset_index(drop=True)


def top_k_order(scores: np.ndarray, k: int) -> np.ndarray:
    """Indices (into ``scores``) of the top-k values, highest first."""
    order = np.argsort(scores)[::-1]
    return order[:k]


__all__ = [
    # sparse
    "build_tfidf_index",
    "build_label_to_indices",
    "tfidf_scores",
    "predict_topic_topk",
    "candidate_indices_for_labels",
    "batch_encode_textcnn",
    # dense
    "encode_siamese_texts",
    "precompute_doc_embeddings",
    "encode_query_siamese",
    "siamese_scores",
    # hybrid (sparse + dense fusion)
    "hybrid_scores",
    "minmax_norm",
    "weighted_sum",
    # shared ranking
    "format_results",
    "top_k_order",
]
