"""Sparse (TF-IDF) retrieval plus the TextCNN topic filter.

The sparse path narrows candidates by predicted macro_domain (TextCNN), then
ranks them by cosine TF-IDF similarity. All functions are stateless: callers
pass the corpus, index, and models explicitly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from scipy.sparse import spmatrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from ..utils.text import encode_text, simple_tokenize


# ── Index construction ───────────────────────────────────────────────────────


def build_tfidf_index(
    corpus_texts: list[str],
    max_features: int,
) -> tuple[TfidfVectorizer, spmatrix]:
    """Fit a TF-IDF vectorizer on tokenized corpus text; return L2-normalized matrix."""
    norm_texts = [" ".join(simple_tokenize(t)) for t in corpus_texts]
    vectorizer = TfidfVectorizer(max_features=max_features, token_pattern=r"\S+")
    matrix = vectorizer.fit_transform(norm_texts)
    matrix = normalize(matrix, norm="l2", axis=1)
    return vectorizer, matrix


def build_label_to_indices(
    corpus_df: pd.DataFrame, label_col: str = "macro_domain"
) -> dict[str, list[int]]:
    """Map each macro_domain label to the corpus row indices it covers."""
    label_to_indices: dict[str, list[int]] = {}
    for i, label in enumerate(corpus_df[label_col].astype(str).tolist()):
        label_to_indices.setdefault(label, []).append(i)
    return label_to_indices


# ── Scoring / topic filter ───────────────────────────────────────────────────


def tfidf_scores(
    question: str,
    indices: np.ndarray,
    vectorizer: TfidfVectorizer,
    tfidf_matrix: spmatrix,
) -> np.ndarray:
    """Cosine TF-IDF scores between the query and the corpus rows at ``indices``."""
    q_norm = " ".join(simple_tokenize(question))
    q_vec = normalize(vectorizer.transform([q_norm]), norm="l2", axis=1)
    sub = tfidf_matrix[indices]
    return np.asarray(sub.dot(q_vec.T).todense()).ravel()


@torch.no_grad()
def predict_topic_topk(
    question: str,
    textcnn_bundle: dict,
    device: torch.device,
    k: int = 3,
) -> tuple[list[int], list[str], np.ndarray]:
    """Return top-k macro_domain (ids, labels, probs) from the TextCNN classifier."""
    ids = encode_text(question, textcnn_bundle["topic_stoi"], textcnn_bundle["cnn_max_len"])
    x = torch.tensor([ids], dtype=torch.long, device=device)
    probs = torch.softmax(textcnn_bundle["model"](x), dim=1).squeeze(0).detach().cpu().numpy()
    top_ids = probs.argsort()[::-1][: min(k, len(probs))].tolist()
    id2label = textcnn_bundle["id2label"]
    labels = [id2label[int(i)] for i in top_ids]
    return top_ids, labels, probs[top_ids]


def candidate_indices_for_labels(
    labels: list[str],
    label_to_indices: dict[str, list[int]],
    corpus_size: int,
) -> np.ndarray:
    """Union of corpus indices for the given labels; full corpus if none match."""
    indices: list[int] = []
    for label in labels:
        indices.extend(label_to_indices.get(str(label), []))
    if not indices:
        return np.arange(corpus_size, dtype=np.int64)
    return np.array(sorted(set(indices)), dtype=np.int64)


@torch.no_grad()
def batch_encode_textcnn(
    texts: list[str],
    stoi: dict[str, int],
    max_len: int,
    device: torch.device,
) -> torch.Tensor:
    """Encode texts to a batched long tensor for TextCNN input."""
    ids = [encode_text(t, stoi, max_len) for t in texts]
    return torch.tensor(ids, dtype=torch.long, device=device)
