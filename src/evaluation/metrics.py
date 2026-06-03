"""Retrieval metric primitives: Recall@k, MRR, and markdown rendering.

Pure functions over already-ranked results — no model or IO dependency — so they
are trivially unit-testable and reusable by any evaluator.
"""

from __future__ import annotations

import pandas as pd

DEFAULT_KS: tuple[int, ...] = (1, 3, 5, 10)


def rank_of(gold: str, ranked_ids: list[str]) -> int | None:
    """1-based rank of ``gold`` in ``ranked_ids``, or ``None`` if absent."""
    return ranked_ids.index(gold) + 1 if gold in ranked_ids else None


def reciprocal_rank(rank: int | None) -> float:
    """Reciprocal rank contribution for a single query (0.0 if not found)."""
    return 0.0 if rank is None else 1.0 / rank


def aggregate_metrics(
    pipeline_name: str,
    ranks: list[int | None],
    ks: tuple[int, ...] = DEFAULT_KS,
) -> dict:
    """Aggregate per-query ranks into Recall@k + MRR for one pipeline."""
    n = len(ranks)
    denom = max(n, 1)
    metrics: dict = {"pipeline": pipeline_name, "n_queries": n}
    for kk in ks:
        hits = sum(1 for r in ranks if r is not None and r <= kk)
        metrics[f"Recall@{kk}"] = hits / denom
    metrics["MRR"] = sum(reciprocal_rank(r) for r in ranks) / denom
    return metrics


def build_markdown_table(rows: list[dict]) -> str:
    """Render evaluation metric dicts as a markdown table."""
    if not rows:
        return "_(no results)_"
    df = pd.DataFrame(rows)
    return df.to_markdown(index=False, floatfmt=".4f")
