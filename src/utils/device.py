"""Compute-device resolution for the VNLegal RAG pipeline.

Extracted from the original flat ``config.py`` so device logic is reusable
without importing the whole pipeline config.
"""

from __future__ import annotations

import os

import torch


def default_device() -> torch.device:
    """Resolve the compute device.

    Priority: ``VNLEGAL_DEVICE`` env var (``cpu``/``cuda``) → CUDA when available
    → CPU. The env override exists because some hosts report CUDA as available
    while the runtime is actually broken (e.g. cuDNN driver mismatch), which
    hard-crashes uncatchably on the first conv; set ``VNLEGAL_DEVICE=cpu`` there.
    """
    forced = os.environ.get("VNLEGAL_DEVICE", "").strip().lower()
    if forced in ("cpu", "cuda"):
        return torch.device(forced)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
