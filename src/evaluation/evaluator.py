"""Retrieval evaluation harness.

A retriever is any callable ``(question: str, k: int) -> DataFrame`` whose first
column ranks passages and which exposes a ``passage_id`` column. Ground truth is
the ``passage_id`` on each QA row. The harness collects per-query ranks and
delegates aggregation to :mod:`src.evaluation.metrics`.
"""

from __future__ import annotations

from typing import Callable

import pandas as pd

from .metrics import DEFAULT_KS, aggregate_metrics, rank_of

Retriever = Callable[[str, int], pd.DataFrame]


def _sample_qa(qa_df: pd.DataFrame, sample_n: int | None, seed: int) -> pd.DataFrame:
    if sample_n is None or len(qa_df) <= sample_n:
        return qa_df
    return qa_df.sample(sample_n, random_state=seed)


def evaluate_retriever(
    pipeline_name: str,
    retriever: Retriever,
    qa_df: pd.DataFrame,
    *,
    k: int = 10,
    ks: tuple[int, ...] = DEFAULT_KS,
    sample_n: int | None = 200,
    seed: int = 42,
    question_col: str = "question",
    passage_id_col: str = "passage_id",
) -> dict:
    """Run a retriever over sampled QA rows; return Recall@k and MRR metrics."""
    # Recall@kk is only meaningful when the retriever returns >= kk results;
    # drop ks above the retrieval depth so we never report a mislabeled metric.
    ks = tuple(kk for kk in ks if kk <= k)
    qa_eval = _sample_qa(qa_df, sample_n, seed)

    ranks: list[int | None] = []
    for _, row in qa_eval.iterrows():
        gold = str(row[passage_id_col])
        results = retriever(str(row[question_col]), k)
        ranked_ids = [str(pid) for pid in results["passage_id"].tolist()]
        ranks.append(rank_of(gold, ranked_ids))

    return aggregate_metrics(pipeline_name, ranks, ks)
