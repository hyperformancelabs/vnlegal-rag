"""
Shared retrieval evaluation for pipeline .

Evaluation policy
-----------------
For benchmarks, always use **raw top-k** TextCNN labels (softmax rank).
Do not apply probability thresholds (e.g. demo ``select_topic_labels(min_prob=0.12)``).

Data contract
-------------
- ``passage_matrix[i]`` aligns with ``corpus_df.iloc[i]``
- ``pid_to_idx[passage_id] -> i`` for ground-truth lookup and ``gold_in_candidates``
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch

from tokenizer_bootstrap import encode_text

EncodeQueryFn = Callable[[str], np.ndarray]


@dataclass
class RetrievalEvalConfig:
    topic_topk: int = 3
    retrieve_ks: tuple[int, ...] = (1, 3, 5, 10)
    max_queries: int | None = 1500
    passage_id_col: str = "passage_id"
    question_col: str = "question"


def build_pid_to_idx(corpus_df: pd.DataFrame, id_col: str = "passage_id") -> dict[str, int]:
    return {str(pid): i for i, pid in enumerate(corpus_df[id_col].tolist())}


def build_label_to_indices(
    corpus_df: pd.DataFrame, label_col: str = "macro_domain"
) -> dict[str, list[int]]:
    label_to_indices: dict[str, list[int]] = {}
    for i, label in enumerate(corpus_df[label_col].astype(str).tolist()):
        label_to_indices.setdefault(label, []).append(i)
    return label_to_indices


def assert_passage_alignment(
    corpus_df: pd.DataFrame,
    passage_matrix: np.ndarray,
    pid_to_idx: dict[str, int],
    *,
    id_col: str = "passage_id",
) -> None:
    n = len(corpus_df)
    if passage_matrix.shape[0] != n:
        raise ValueError(
            f"passage_matrix rows ({passage_matrix.shape[0]}) != corpus_df rows ({n})"
        )
    if len(pid_to_idx) != n:
        raise ValueError(f"pid_to_idx size ({len(pid_to_idx)}) != corpus_df rows ({n})")
    for i in range(min(n, 5)):
        pid = str(corpus_df.iloc[i][id_col])
        if pid_to_idx.get(pid) != i:
            raise ValueError(f"pid_to_idx misaligned at row {i} for passage_id={pid!r}")


def candidate_indices_for_labels(
    labels: list[str],
    label_to_indices: dict[str, list[int]],
    corpus_size: int,
) -> np.ndarray:
    indices: list[int] = []
    for label in labels:
        indices.extend(label_to_indices.get(str(label), []))
    if not indices:
        return np.arange(corpus_size, dtype=np.int64)
    return np.array(sorted(set(indices)), dtype=np.int64)


@torch.no_grad()
def predict_topic_topk_labels(
    question: str,
    topic_model: torch.nn.Module,
    topic_stoi: dict[str, int],
    id2label: dict[int, str],
    cnn_max_len: int,
    device: torch.device,
    k: int = 3,
) -> list[str]:
    """Raw top-k macro_domain labels by softmax rank (no probability threshold)."""
    ids = encode_text(question, topic_stoi, cnn_max_len)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    probs = torch.softmax(topic_model(x), dim=1).squeeze(0).detach().cpu().numpy()
    top_ids = probs.argsort()[::-1][: min(k, len(probs))].tolist()
    return [id2label[int(i)] for i in top_ids]


def _sample_qa(qa_df: pd.DataFrame, max_queries: int | None) -> pd.DataFrame:
    if max_queries is None or len(qa_df) <= max_queries:
        return qa_df
    return qa_df.sample(max_queries, random_state=42)


def _as_1d(vec: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(vec, torch.Tensor):
        vec = vec.detach().cpu().numpy()
    arr = np.asarray(vec, dtype=np.float32).reshape(-1)
    return arr


def _scores_for_indices(
    encode_query: EncodeQueryFn,
    passage_matrix: np.ndarray,
    indices: np.ndarray,
    question: str,
) -> np.ndarray:
    q_emb = _as_1d(encode_query(question))
    filtered = passage_matrix[indices]
    return filtered @ q_emb


def _metrics_from_rank(rank: int | None, retrieve_ks: tuple[int, ...]) -> tuple[dict[str, float], float]:
    metrics: dict[str, float] = {}
    for k in retrieve_ks:
        metrics[f"Recall@{k}"] = 1.0 if rank is not None and rank <= k else 0.0
    mrr = 0.0 if rank is None else 1.0 / rank
    return metrics, mrr


def _aggregate_metrics(per_query: list[dict[str, float]], mrrs: list[float]) -> dict[str, Any]:
    n = max(len(per_query), 1)
    out: dict[str, Any] = {k: 0.0 for k in per_query[0]} if per_query else {}
    for row in per_query:
        for k, v in row.items():
            out[k] = out.get(k, 0.0) + v
    for k in list(out):
        if k.startswith("Recall@"):
            out[k] = out[k] / n
    out["MRR"] = float(sum(mrrs) / n) if mrrs else 0.0
    out["n_queries"] = n
    return out


def evaluate_siamese_only(
    encode_query: EncodeQueryFn,
    qa_df: pd.DataFrame,
    passage_matrix: np.ndarray,
    pid_to_idx: dict[str, int],
    cfg: RetrievalEvalConfig,
    *,
    corpus_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    passage_matrix = np.asarray(passage_matrix, dtype=np.float32)
    if corpus_df is not None:
        assert_passage_alignment(corpus_df, passage_matrix, pid_to_idx, id_col=cfg.passage_id_col)

    qa_eval = _sample_qa(qa_df, cfg.max_queries)
    per_query: list[dict[str, float]] = []
    mrrs: list[float] = []

    for _, row in qa_eval.iterrows():
        true_idx = pid_to_idx.get(str(row[cfg.passage_id_col]))
        if true_idx is None:
            continue
        scores = passage_matrix @ _as_1d(encode_query(str(row[cfg.question_col])))
        order = np.argsort(scores)[::-1]
        rank_pos = np.where(order == true_idx)[0]
        rank = int(rank_pos[0]) + 1 if len(rank_pos) else None
        row_metrics, mrr = _metrics_from_rank(rank, cfg.retrieve_ks)
        per_query.append(row_metrics)
        mrrs.append(mrr)

    return _aggregate_metrics(per_query, mrrs)


def evaluate_siamese_textcnn(
    encode_query: EncodeQueryFn,
    qa_df: pd.DataFrame,
    passage_matrix: np.ndarray,
    pid_to_idx: dict[str, int],
    label_to_indices: dict[str, list[int]],
    topic_model: torch.nn.Module,
    topic_stoi: dict[str, int],
    id2label: dict[int, str],
    cnn_max_len: int,
    device: torch.device,
    cfg: RetrievalEvalConfig,
    *,
    corpus_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    passage_matrix = np.asarray(passage_matrix, dtype=np.float32)
    corpus_size = passage_matrix.shape[0]
    if corpus_df is not None:
        assert_passage_alignment(corpus_df, passage_matrix, pid_to_idx, id_col=cfg.passage_id_col)

    qa_eval = _sample_qa(qa_df, cfg.max_queries)
    per_query: list[dict[str, float]] = []
    mrrs: list[float] = []
    gold_in_candidates = 0
    candidate_counts: list[int] = []
    n_scored = 0

    for _, row in qa_eval.iterrows():
        true_idx = pid_to_idx.get(str(row[cfg.passage_id_col]))
        if true_idx is None:
            continue
        question = str(row[cfg.question_col])
        labels = predict_topic_topk_labels(
            question,
            topic_model,
            topic_stoi,
            id2label,
            cnn_max_len,
            device,
            k=cfg.topic_topk,
        )
        indices = candidate_indices_for_labels(labels, label_to_indices, corpus_size)
        candidate_counts.append(len(indices))
        if true_idx in set(indices.tolist()):
            gold_in_candidates += 1

        scores = _scores_for_indices(encode_query, passage_matrix, indices, question)
        order = np.argsort(scores)[::-1]
        local_true = np.where(indices == true_idx)[0]
        rank = int(np.where(order == local_true[0])[0][0]) + 1 if len(local_true) else None

        row_metrics, mrr = _metrics_from_rank(rank, cfg.retrieve_ks)
        per_query.append(row_metrics)
        mrrs.append(mrr)
        n_scored += 1

    out = _aggregate_metrics(per_query, mrrs)
    n = max(n_scored, 1)
    out["filter_stats"] = {
        "gold_in_candidates": gold_in_candidates / n,
        "mean_candidates": float(np.mean(candidate_counts)) if candidate_counts else 0.0,
        "topic_topk": cfg.topic_topk,
    }
    return out


def compare_modes(metrics_only: dict[str, Any], metrics_filtered: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for name, m in [("siamese_only", metrics_only), ("siamese_textcnn", metrics_filtered)]:
        row = {"mode": name, "MRR": m.get("MRR", 0.0), "n_queries": m.get("n_queries", 0)}
        for k, v in m.items():
            if k.startswith("Recall@"):
                row[k] = v
        fs = m.get("filter_stats")
        if fs:
            row["gold_in_candidates"] = fs.get("gold_in_candidates")
            row["mean_candidates"] = fs.get("mean_candidates")
        rows.append(row)
    return pd.DataFrame(rows)
