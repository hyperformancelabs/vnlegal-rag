"""Low-level file IO helpers shared across the pipeline.

Centralizes the on-disk formats the project relies on:

* ready-split CSVs are **tab-separated** (written by ``prepare_data.py``)
* metadata / label maps are JSON

Keeping these in one place avoids re-specifying ``sep="\\t"`` or the UTF-8
encoding at every call site (DRY).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def read_ready_csv(path: Path) -> pd.DataFrame:
    """Read a tab-separated ready-split CSV."""
    return pd.read_csv(path, sep="\t")


def read_json(path: Path) -> dict | list:
    """Read a UTF-8 JSON file."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)
