"""Data subpackage: split resolution, corpus/QA loading, and processing."""

from __future__ import annotations

from .loaders import (
    first_dir_with_files,
    load_corpus,
    load_label_list,
    load_qa,
    locate_repo_root,
    resolve_data_dir,
)
from .processors import TEXT_COL_CANDIDATES, dedupe_corpus, detect_text_col

__all__ = [
    "locate_repo_root",
    "first_dir_with_files",
    "resolve_data_dir",
    "load_corpus",
    "load_qa",
    "load_label_list",
    "detect_text_col",
    "dedupe_corpus",
    "TEXT_COL_CANDIDATES",
]
