"""Dense bi-encoder for retrieval — twin LSTM sentence encoder.

* :class:`LSTMEncoder` — masked mean-pooled, L2-normalized sentence encoder.
* :class:`SiameseLSTM` — twin encoder producing query/doc cosine similarity.

Also includes state-dict helpers that infer architecture dimensions directly
from saved weights, so checkpoints load without a separate config file.
"""

from __future__ import annotations

import re

import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMEncoder(nn.Module):
    """Embedding → LSTM → masked mean-pool → L2-normalized vector."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_size: int,
        num_layers: int = 1,
        bidirectional: bool = True,
        dropout: float = 0.0,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        emb = self.embedding(x)
        out, _ = self.lstm(emb)
        if mask is None:
            pooled = out.mean(dim=1)
        else:
            mask = mask.unsqueeze(-1)
            summed = (out * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1e-9)
            pooled = summed / counts
        return F.normalize(pooled, p=2, dim=1)


class SiameseLSTM(nn.Module):
    """Twin :class:`LSTMEncoder`; forward returns query/doc cosine similarity."""

    def __init__(self, **encoder_kwargs):
        super().__init__()
        self.encoder = LSTMEncoder(**encoder_kwargs)

    def encode(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        return self.encoder(x, mask)

    def forward(
        self,
        q_ids: torch.Tensor,
        d_ids: torch.Tensor,
        q_mask: torch.Tensor | None = None,
        d_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q = self.encode(q_ids, q_mask)
        d = self.encode(d_ids, d_mask)
        return (q * d).sum(dim=1)


# ─── State-dict helpers ──────────────────────────────────────────────────────


def add_encoder_prefix_if_needed(state: dict) -> dict:
    """Prefix bare ``embedding.*``/``lstm.*`` keys with ``encoder.`` for SiameseLSTM."""
    if any(k.startswith("encoder.") for k in state):
        return state
    if any(k.startswith(("embedding.", "lstm.")) for k in state):
        return {f"encoder.{k}": v for k, v in state.items()}
    return state


def infer_siamese_config(state: dict, meta: dict | None = None) -> dict:
    """Infer SiameseLSTM encoder kwargs from a state dict (+ optional meta)."""
    meta = meta or {}
    emb_w = state["encoder.embedding.weight"]
    ih_l0 = state["encoder.lstm.weight_ih_l0"]
    hidden_size = int(ih_l0.shape[0] // 4)
    bidirectional = any(k.endswith("_reverse") for k in state)
    layer_idxs = {
        int(m.group(1))
        for m in (re.search(r"weight_ih_l(\d+)", k) for k in state)
        if m is not None
    }
    num_layers = max(layer_idxs) + 1 if layer_idxs else 1
    return {
        "vocab_size": int(emb_w.shape[0]),
        "embed_dim": int(emb_w.shape[1]),
        "hidden_size": hidden_size,
        "num_layers": int(meta.get("num_layers", num_layers)),
        "bidirectional": bidirectional,
        "dropout": float(meta.get("dropout", 0.0)),
        "pad_idx": 0,
    }
