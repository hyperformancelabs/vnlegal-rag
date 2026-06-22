"""Data loading for the pipeline: repo-root discovery, split resolution, CSV/label IO.

Ready splits are tab-separated CSV files written by ``prepare_data.py``.
Corpus rows are deduplicated by ``passage_id`` (see :mod:`src.data.processors`)
so the matrix aligns with row position for ground-truth lookup during evaluation.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..utils.io import read_json, read_ready_csv
from .processors import dedupe_corpus


def locate_repo_root(start: Path | None = None) -> Path:
    """Walk parents from ``start`` (or this file) until the repo markers appear."""
    here = (start or Path(__file__)).resolve()
    for candidate in [here, *here.parents]:
        if (candidate / "experiments").is_dir() and (candidate / "src").is_dir():
            return candidate
        if (candidate / "model" / "tokenizer.py").is_file():
            return candidate
    # No repo markers found (e.g. installed as a library) — cwd is the best guess.
    import warnings

    warnings.warn(
        "Could not locate repo root from markers; falling back to cwd. "
        "Relative data/artifact paths may not resolve.",
        UserWarning,
        stacklevel=2,
    )
    return Path.cwd()


def first_dir_with_files(candidates: list[Path], required: list[str]) -> Path | None:
    """Return the first candidate directory containing every required filename."""
    for raw in candidates:
        path = Path(raw)
        if path.is_dir() and all((path / name).is_file() for name in required):
            return path
    return None


def resolve_data_dir(candidates: tuple[str, ...], repo_root: Path) -> Path:
    """Resolve the ready-split directory holding qa_train/val/test.csv."""
    abs_candidates = [
        (Path(c) if Path(c).is_absolute() else repo_root / c) for c in candidates
    ]
    found = first_dir_with_files(
        abs_candidates, ["qa_train.csv", "qa_val.csv", "qa_test.csv"]
    )
    if found is None:
        raise FileNotFoundError(
            "No data_ready dir with qa_train/val/test.csv found. Tried: "
            + ", ".join(str(c) for c in abs_candidates)
        )
    return found


def load_corpus(data_dir: Path) -> pd.DataFrame:
    """Load corpus split(s), preferring per-split files then ``corpus_full.csv``.

    Concatenates available train/val/test corpus splits, falling back to
    ``corpus_full.csv``; deduplicates by ``passage_id`` and resets the index so
    row position aligns with the eventual TF-IDF / embedding matrices.
    """
    frames: list[pd.DataFrame] = []
    for name in ("corpus_train.csv", "corpus_val.csv", "corpus_test.csv"):
        path = data_dir / name
        if path.is_file():
            frames.append(read_ready_csv(path))

    if not frames:
        full = data_dir / "corpus_full.csv"
        if not full.is_file():
            raise FileNotFoundError(f"No corpus CSVs in {data_dir}")
        frames.append(read_ready_csv(full))

    corpus = pd.concat(frames, ignore_index=True)
    return dedupe_corpus(corpus)


def load_qa(data_dir: Path) -> dict[str, pd.DataFrame]:
    """Load qa_train/val/test splits into a dict keyed by split name."""
    return {
        split: read_ready_csv(data_dir / f"qa_{split}.csv")
        for split in ("train", "val", "test")
    }


def load_label_list(data_dir: Path) -> list[str]:
    """Load the ordered macro_domain label list from ``label_maps.json``."""
    maps = read_json(data_dir / "label_maps.json")
    # Support a bare list, {"label_list": [...]}, {"id2label": {...}}, or {"label2id": {...}}.
    if isinstance(maps, list):
        return [str(x) for x in maps]
    if "label_list" in maps:
        return [str(x) for x in maps["label_list"]]
    if "id2label" in maps:
        return [maps["id2label"][k] for k in sorted(maps["id2label"], key=int)]
    if "label2id" in maps:
        ordered = sorted(maps["label2id"].items(), key=lambda kv: kv[1])
        return [k for k, _ in ordered]
    raise KeyError(f"Unrecognized label_maps.json structure: {list(maps)}")
