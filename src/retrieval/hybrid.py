"""Hybrid retrieval: weighted-sum fusion of sparse (TF-IDF) and dense (Siamese).

Both score vectors are computed over the **same** candidate pool (the topic
filter's indices), min-max normalized, then fused. Stateless — the pipeline
supplies the index, vectorizer, models, and embeddings.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.sparse import spmatrix
from sklearn.feature_extraction.text import TfidfVectorizer

from .dense import siamese_scores
from .fusion import minmax_norm, weighted_sum
from .sparse import tfidf_scores


def hybrid_scores(
    question: str,
    indices: np.ndarray,
    vectorizer: TfidfVectorizer,
    tfidf_matrix: spmatrix,
    siamese_bundle: dict,
    doc_embeddings: np.ndarray,
    device: torch.device,
    alpha: float = 0.7,
) -> np.ndarray:
    """Fused score per candidate: ``α·norm(tfidf) + (1−α)·norm(siamese)``.

    ``alpha`` weights the (stronger) TF-IDF side; both vectors are min-max
    normalized first so the two scales are comparable.
    """
    s_sparse = tfidf_scores(question, indices, vectorizer, tfidf_matrix)
    s_dense = siamese_scores(question, indices, siamese_bundle, doc_embeddings, device)
    return weighted_sum(minmax_norm(s_sparse), minmax_norm(s_dense), alpha)
