"""
Bootstrap import for the pipeline  tokenizer.

Loads ``tokenizer.py`` from the first available location:

  1. ``experiments/tokenizer.py`` next to this file (Colab / Kaggle notebook cwd)
  2. ``model/tokenizer.py`` under a repo root (local full clone)

Uses ``importlib`` so no ``model.tokenizer`` package import is required.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Optional

_PIPELINE_DIR = Path(__file__).resolve().parent
_TOKENIZER_MODULE: Optional[ModuleType] = None
_TOKENIZER_PATH: Optional[Path] = None


def _search_roots(start: Path | None = None) -> list[Path]:
    roots: list[Path] = []
    if start is not None:
        roots.append(start.resolve())
    roots.extend(
        [
            Path.cwd(),
            Path.cwd().parent,
            _PIPELINE_DIR,
            _PIPELINE_DIR.parent,
            Path("/content"),
            Path("/content/vnlegal-rag"),
            Path("/kaggle/working/vnlegal-rag"),
        ]
    )

    seen: set[Path] = set()
    unique: list[Path] = []
    for root in roots:
        try:
            root = root.resolve()
        except OSError:
            continue
        if root in seen:
            continue
        seen.add(root)
        unique.append(root)
    return unique


def locate_tokenizer_path(start: Path | None = None) -> Path:
    """Return the path to ``tokenizer.py`` used by pipeline ."""
    global _TOKENIZER_PATH
    if _TOKENIZER_PATH is not None:
        return _TOKENIZER_PATH

    candidates: list[Path] = [_PIPELINE_DIR / "tokenizer.py"]
    for root in _search_roots(start):
        candidates.append(root / "experiments" / "tokenizer.py")

    seen: set[Path] = set()
    for path in candidates:
        try:
            path = path.resolve()
        except OSError:
            continue
        if path in seen:
            continue
        seen.add(path)
        if path.is_file():
            _TOKENIZER_PATH = path
            return path

    raise FileNotFoundError(
        "Cannot locate tokenizer.py. Expected experiments/tokenizer.py "
        "(Colab/Kaggle) or model/tokenizer.py (full repo clone)."
    )


def locate_model_dir(start: Path | None = None) -> Path:
    """Back-compat alias: parent directory of the resolved tokenizer file."""
    return locate_tokenizer_path(start).parent


def load_tokenizer_module(start: Path | None = None) -> ModuleType:
    """Load and cache the tokenizer module from disk."""
    global _TOKENIZER_MODULE
    if _TOKENIZER_MODULE is not None:
        return _TOKENIZER_MODULE

    path = locate_tokenizer_path(start)
    spec = importlib.util.spec_from_file_location("_vnlegal_tokenizer", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load tokenizer module from {path}")

    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _TOKENIZER_MODULE = mod
    return mod


_tok = load_tokenizer_module()

simple_tokenize = _tok.simple_tokenize
encode_text = _tok.encode_text
encode_with_mask = _tok.encode_with_mask
build_vocab = _tok.build_vocab
load_vocab = _tok.load_vocab
save_vocab = _tok.save_vocab
TOKENIZER_BACKEND = _tok.TOKENIZER_BACKEND
PAD_TOKEN = _tok.PAD_TOKEN
UNK_TOKEN = _tok.UNK_TOKEN

__all__ = [
    "locate_tokenizer_path",
    "locate_model_dir",
    "load_tokenizer_module",
    "simple_tokenize",
    "encode_text",
    "encode_with_mask",
    "build_vocab",
    "load_vocab",
    "save_vocab",
    "TOKENIZER_BACKEND",
    "PAD_TOKEN",
    "UNK_TOKEN",
]
