"""Corpus post-load processing: text-column detection and deduplication.

Pure DataFrame transforms applied after raw CSVs are read. Kept separate from
:mod:`src.data.loaders` so the "where do rows come from" concern is decoupled
from the "how are rows cleaned" concern.
"""

from __future__ import annotations

import pandas as pd

# Possible text-column names in the corpus (first present wins).
TEXT_COL_CANDIDATES: tuple[str, ...] = ("article_content", "text")


def detect_text_col(df: pd.DataFrame) -> str:
    """Return the first known text column present in the corpus dataframe."""
    for col in TEXT_COL_CANDIDATES:
        if col in df.columns:
            return col
    raise KeyError(
        f"No text column found (tried {TEXT_COL_CANDIDATES}) in {list(df.columns)}"
    )


def dedupe_corpus(corpus: pd.DataFrame, id_col: str = "passage_id") -> pd.DataFrame:
    """Drop duplicate passages and reset the index.

    Resetting is essential: downstream the row *position* must align with the
    TF-IDF / embedding matrices and with ``pid_to_idx`` lookups.
    """
    return corpus.drop_duplicates(subset=id_col).reset_index(drop=True)
