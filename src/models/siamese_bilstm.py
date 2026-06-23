"""Siamese BiLSTM encoder — Neculoiu et al. (2016) contrastive retrieval.

Four bidirectional LSTM layers + dense projection + L2-normalized output.
Trained with contrastive loss: L+ = 0.25*(1-cos)^2, L- = cos^2 (if cos < margin).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class SiameseBiLSTMEncoder(nn.Module):
    """BiLSTM → masked mean-pool → dense projection (no L2-norm on output).

    Paper: hidden_dim=64 / direction → 128-d after concat. The dense layer
    maps to dense_dim (paper: 128). The encoder output is NOT L2-normalized;
    cosine similarity is computed explicitly in :class:`SiameseBiLSTM.forward`.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 300,
        hidden_dim: int = 64,
        num_layers: int = 4,
        dense_dim: int = 128,
        dropout_recurrent: float = 0.2,
        dropout_inter: float = 0.4,
        dropout_out: float = 0.4,
        pad_idx: int = 0,
        embedding_weight: torch.Tensor | None = None,
        freeze_embedding: bool = False,
    ):
        super().__init__()
        if embedding_weight is not None:
            self.embedding = nn.Embedding.from_pretrained(
                embedding_weight, freeze=freeze_embedding, padding_idx=pad_idx,
            )
        else:
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)

        self.lstm = nn.LSTM(
            embed_dim, hidden_dim,
            num_layers=num_layers, batch_first=True, bidirectional=True,
            dropout=dropout_recurrent if num_layers > 1 else 0.0,
        )
        self.inter_dropout = nn.Dropout(dropout_inter)
        self.out_dropout = nn.Dropout(dropout_out)
        self.dense = nn.Linear(hidden_dim * 2, dense_dim)
        self.pad_idx = pad_idx

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mask = (x != self.pad_idx).float()
        return self._pool_and_project(x, mask)

    def encode(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Encode with explicit mask → L2-normalized (for :mod:`src.retrieval.dense`).

        The :mod:`dense` module calls ``model.encode(ids, mask)`` when
        precomputing corpus embeddings and encoding query vectors. This
        method follows that contract and returns unit-norm vectors so dot
        product equals cosine similarity.
        """
        vec = self._pool_and_project(ids, mask.to(dtype=torch.float32))
        return F.normalize(vec, p=2, dim=1)

    def _pool_and_project(
        self, ids: torch.Tensor, mask: torch.Tensor,
    ) -> torch.Tensor:
        emb = self.embedding(ids)
        out, _ = self.lstm(emb)
        out = self.inter_dropout(out)
        mask = mask.unsqueeze(-1)
        summed = (out * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        pooled = summed / counts
        return self.dense(self.out_dropout(pooled))


class SiameseBiLSTM(nn.Module):
    """Twin :class:`SiameseBiLSTMEncoder`; forward returns cosine similarity."""

    def __init__(self, **encoder_kwargs):
        super().__init__()
        self.encoder = SiameseBiLSTMEncoder(**encoder_kwargs)

    def encode(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Proxy to :meth:`SiameseBiLSTMEncoder.encode` for pipeline API."""
        return self.encoder.encode(ids, mask)

    def forward(self, q: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        q_emb = self.encoder(q)
        a_emb = self.encoder(a)
        # Explicit cosine similarity (paper §3.1 eq. for E_W)
        return (q_emb * a_emb).sum(dim=1) / (
            q_emb.norm(p=2, dim=1) * a_emb.norm(p=2, dim=1)
        )


class ContrastiveLoss(nn.Module):
    """Paper §3.1 adapted for word-level: L+ = scale*(1-cos)^2, L- = relu(cos - m)^2.

    The margin can be annealed: start high (m_start) and decay to m_end
    over `m_anneal_epochs` epochs. A higher initial margin pushes the model
    to separate all pairs; decay tightens the requirement over time.
    """

    def __init__(
        self,
        margin: float = 0.5,
        positive_scale: float = 0.25,
        margin_start: float | None = None,
        margin_end: float | None = None,
        margin_anneal_epochs: int = 10,
    ):
        super().__init__()
        self.margin = margin
        self.positive_scale = positive_scale
        self.margin_start = margin_start if margin_start is not None else margin
        self.margin_end = margin_end if margin_end is not None else margin
        self.margin_anneal_epochs = margin_anneal_epochs
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Update the margin based on the current epoch (linear decay)."""
        self._epoch = epoch
        if epoch >= self.margin_anneal_epochs:
            self.margin = self.margin_end
        else:
            t = epoch / max(self.margin_anneal_epochs, 1)
            self.margin = self.margin_start + (self.margin_end - self.margin_start) * t

    def forward(self, cos_sim: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        pos_mask = labels == 1
        neg_mask = labels == 0
        loss_pos = self.positive_scale * (1 - cos_sim[pos_mask]) ** 2
        loss_neg = torch.relu(cos_sim[neg_mask] - self.margin) ** 2
        total = pos_mask.sum().float() + neg_mask.sum().float().clamp(min=1)
        return (loss_pos.sum() + loss_neg.sum()) / total


def load_siamese_from_artifacts(
    artifact_dir: str | Path,
    device: torch.device | str = "cpu",
) -> tuple[SiameseBiLSTM, dict[str, int], dict[str, Any]]:
    """Load a trained SiameseBiLSTM + stoi + metadata from an artifact directory.

    Expected files (as saved by the ``siamese-bilstm*.ipynb`` notebooks):

    * ``stoi.pt`` – ``torch.save(stoi)`` word→index mapping
    * ``siamese_bilstm_best.pt`` – full model state dict
    * ``metadata.json`` – hyperparameters and eval results

    Returns ``(model, stoi, metadata)``. The model is set to ``eval()`` mode
    and moved to ``device``.
    """
    root = Path(artifact_dir)
    stoi: dict[str, int] = torch.load(
        root / "stoi.pt", map_location="cpu", weights_only=False,
    )
    with open(root / "metadata.json", encoding="utf-8") as f:
        meta: dict[str, Any] = json.load(f)

    state = torch.load(
        root / "siamese_bilstm_best.pt", map_location="cpu", weights_only=True,
    )
    model = SiameseBiLSTM(
        vocab_size=meta["vocab_size"],
        embed_dim=meta["embed_dim"],
        hidden_dim=meta["hidden_dim"],
        num_layers=meta["num_layers"],
        dense_dim=meta["dense_dim"],
        dropout_recurrent=meta["dropout_recurrent"],
        dropout_inter=meta["dropout_inter"],
        dropout_out=meta["dropout_out"],
        pad_idx=stoi.get("<PAD>", 0),
    )
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model, stoi, meta
