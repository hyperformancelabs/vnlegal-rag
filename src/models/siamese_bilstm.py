"""Siamese BiLSTM encoder — Neculoiu et al. (2016) contrastive retrieval.

Four bidirectional LSTM layers + dense projection + L2-normalized output.
Trained with contrastive loss: L+ = 0.25*(1-cos)^2, L- = cos^2 (if cos < margin).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SiameseBiLSTMEncoder(nn.Module):
    """4 BiLSTM layers → masked mean-pool → dense → L2-normalized embedding.

    Paper: hidden_dim=64 per direction → 128-d after concat, dense_dim=128.
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
        return F.normalize(projected, p=2, dim=1)


class SiameseBiLSTM(nn.Module):
    """Twin :class:`SiameseBiLSTMEncoder`; forward returns cosine similarity."""

    def __init__(self, **encoder_kwargs):
        super().__init__()
        self.encoder = SiameseBiLSTMEncoder(**encoder_kwargs)

    def forward(self, q: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        q_emb = self.encoder(q)
        a_emb = self.encoder(a)
        return (q_emb * a_emb).sum(dim=1)


class ContrastiveLoss(nn.Module):
    """Paper §3.1 contrastive loss + BCE fallback for dead-zone safety.

    The original formulation (L- = cos² when cos < margin, 0 otherwise)
    has a dead zone: once all negatives exceed the margin, gradients stop.
    BCE mode (`use_bce=True`) replaces it with binary cross-entropy on
    cosine similarity, which always provides gradient — recommended for
    random-negative setups where the model can trivially push all cos > margin.
    """

    def __init__(
        self,
        margin: float = 0.5,
        positive_scale: float = 0.25,
        use_bce: bool = True,
        bce_scale: float = 5.0,
    ):
        super().__init__()
        self.margin = margin
        self.positive_scale = positive_scale
        self.use_bce = use_bce
        self.bce_scale = bce_scale

    def forward(self, cos_sim: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if self.use_bce:
            # BCE: cos ∈ [-1,1] → scale into logit space → always has gradient
            logits = cos_sim * self.bce_scale
            return F.binary_cross_entropy_with_logits(logits, labels)

        # Original contrastive (Neculoiu 2016) — dead zone risk
        pos_mask = labels == 1
        neg_mask = labels == 0
        loss_pos = self.positive_scale * (1 - cos_sim[pos_mask]) ** 2
        cos_neg = cos_sim[neg_mask]
        below = torch.where(cos_neg < self.margin)[0]
        loss_neg = cos_neg[below] ** 2 if below.numel() > 0 else torch.zeros(1, device=cos_sim.device).sum()
        total = pos_mask.sum().float() + neg_mask.sum().float().clamp(min=1)
        return (loss_pos.sum() + loss_neg.sum()) / total
