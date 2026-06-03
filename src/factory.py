"""Artifact discovery and model assembly (the "factory" for the pipeline).

``load_textcnn`` is required for the topic-filter path. ``load_siamese`` is
optional — it returns ``None`` when no Siamese weights are present, letting the
pipeline degrade gracefully to the TF-IDF + TextCNN retrieval mode.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import torch

from .models.bi_encoder import (
    SiameseLSTM,
    add_encoder_prefix_if_needed,
    infer_siamese_config,
)
from .models.classifier import build_textcnn_from_state_dict
from .utils.io import read_json
from .utils.text import load_vocab

WEIGHT_EXTENSIONS: tuple[str, ...] = (".pt", ".pth")

# Filenames searched inside an artifact directory when loading weights.
TEXTCNN_WEIGHT_NAMES: tuple[str, ...] = ("textcnn_best.pt",)
SIAMESE_WEIGHT_NAMES: tuple[str, ...] = (
    "siamese_bilstm_online_best.pt",
    "siamese_bilstm2L_best.pt",
    "siamese_bilstm1L_best.pt",
    "siamese_lstm_uni_best.pt",
    "siamese_lstm_traditional_cosine_best.pt",
    "siamese_best.pt",
)


def unwrap_state_dict(obj) -> dict:
    """Return the raw tensor dict, unwrapping a ``{'state_dict': ...}`` payload."""
    if isinstance(obj, dict) and "state_dict" in obj:
        return obj["state_dict"]
    return obj


def find_weight_file(artifact_dir: Path, preferred_names: tuple[str, ...]) -> Path | None:
    """Return a preferred weight file, else any ``.pt``/``.pth`` in the directory."""
    for name in preferred_names:
        candidate = artifact_dir / name
        if candidate.is_file():
            return candidate
    for path in sorted(artifact_dir.glob("*")):
        if path.suffix in WEIGHT_EXTENSIONS:
            return path
    return None


def select_artifact_dir(
    candidates: tuple[str, ...],
    repo_root: Path,
    required_files: tuple[str, ...],
    weight_names: tuple[str, ...],
) -> tuple[Path | None, Path | None]:
    """Return (artifact_dir, weight_path) for the first fully valid candidate."""
    for raw in candidates:
        path = Path(raw) if Path(raw).is_absolute() else repo_root / raw
        if not path.is_dir():
            continue
        if not all((path / req).is_file() for req in required_files):
            continue
        weight = find_weight_file(path, weight_names)
        if weight is not None:
            return path, weight
    return None, None


def _load_vocab_stoi(artifact_dir: Path) -> dict[str, int]:
    # Use load_vocab so a tokenizer-backend mismatch emits its UserWarning.
    return load_vocab(artifact_dir / "tokenizer_vocab.json")


def load_textcnn(
    artifact_dir: Path,
    weight_path: Path,
    device: torch.device,
) -> dict[str, Any]:
    """Load the TextCNN topic classifier bundle from a resolved artifact dir."""
    meta = read_json(artifact_dir / "textcnn_meta.json")
    stoi = _load_vocab_stoi(artifact_dir)

    state = unwrap_state_dict(torch.load(weight_path, map_location=device))
    # The final layer's weight is the authoritative class count; fall back to meta.
    if "fc.weight" in state:
        num_classes = int(state["fc.weight"].shape[0])
    else:
        num_classes = int(meta.get("num_classes", len(meta.get("labels", []))))
    if num_classes <= 0:
        raise ValueError(f"Could not determine num_classes for TextCNN in {artifact_dir}")
    model = build_textcnn_from_state_dict(state, num_classes=num_classes).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()

    cnn_max_len = int(meta.get("max_len", 128))
    # Warm up the conv backend before any sklearn/MKL OpenMP runtime initializes.
    # On some Windows hosts, calling a torch conv *after* sklearn's first
    # fit_transform triggers an OpenMP stack-overflow crash; warming up here
    # avoids it and doubles as a load sanity check.
    with torch.no_grad():
        model(torch.zeros(1, cnn_max_len, dtype=torch.long, device=device))

    id2label = {int(k): v for k, v in meta.get("id2label", {}).items()}
    if not id2label and meta.get("labels"):
        id2label = {i: lab for i, lab in enumerate(meta["labels"])}
    if not id2label:
        raise ValueError(
            f"textcnn_meta.json in {artifact_dir} has neither 'id2label' nor 'labels'"
        )

    return {
        "model": model,
        "topic_stoi": stoi,
        "id2label": id2label,
        "cnn_max_len": cnn_max_len,
        "artifact_dir": artifact_dir,
        "weights": weight_path,
    }


def load_siamese(
    artifact_dir: Path | None,
    weight_path: Path | None,
    device: torch.device,
    *,
    default_max_q_len: int = 64,
    default_max_d_len: int = 256,
) -> dict[str, Any] | None:
    """Load the Siamese encoder bundle, or ``None`` if no artifacts were found."""
    if artifact_dir is None or weight_path is None:
        return None

    # Siamese is optional — any artifact problem disables Siamese modes rather
    # than crashing the whole pipeline build (graceful-degrade contract).
    try:
        meta_path = artifact_dir / "siamese_meta.json"
        meta = read_json(meta_path) if meta_path.is_file() else {}
        stoi = _load_vocab_stoi(artifact_dir)

        state = add_encoder_prefix_if_needed(
            unwrap_state_dict(torch.load(weight_path, map_location=device))
        )
        cfg = infer_siamese_config(state, meta)
        model = SiameseLSTM(**cfg).to(device)
        model.load_state_dict(state, strict=True)
        model.eval()
    except Exception as exc:  # noqa: BLE001 - intentional graceful degrade
        warnings.warn(
            f"Siamese load failed ({exc!r}); disabling Siamese retrieval modes.",
            UserWarning,
            stacklevel=2,
        )
        return None

    return {
        "model": model,
        "siamese_stoi": stoi,
        "max_q_len": int(meta.get("max_q_len", default_max_q_len)),
        "max_d_len": int(meta.get("max_d_len", default_max_d_len)),
        "artifact_dir": artifact_dir,
        "weights": weight_path,
    }
