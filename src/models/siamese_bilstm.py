"""Siamese BiLSTM encoder — Neculoiu et al. (2016) contrastive retrieval.

Four bidirectional LSTM layers + dense projection + L2-normalized output.
Trained with contrastive loss: L+ = 0.25*(1-cos)^2, L- = cos^2 (if cos < margin).
"""

from __future__ import annotations

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
        emb = self.embedding(x)
        out, _ = self.lstm(emb)
        out = self.inter_dropout(out)
        mask = mask.unsqueeze(-1)
        summed = (out * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        pooled = summed / counts
        projected = self.dense(self.out_dropout(pooled))
        return projected


class SiameseBiLSTM(nn.Module):
    """Twin :class:`SiameseBiLSTMEncoder`; forward returns cosine similarity."""

    def __init__(self, **encoder_kwargs):
        super().__init__()
        self.encoder = SiameseBiLSTMEncoder(**encoder_kwargs)

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
