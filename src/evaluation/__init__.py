"""Evaluation subpackage: metric primitives and the retrieval evaluator."""

from __future__ import annotations

from .evaluator import evaluate_retriever
from .metrics import DEFAULT_KS, aggregate_metrics, build_markdown_table

__all__ = [
    "evaluate_retriever",
    "build_markdown_table",
    "aggregate_metrics",
    "DEFAULT_KS",
]
