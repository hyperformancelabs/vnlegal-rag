"""Topic (macro_domain) classifier — Kim-style TextCNN.

Used by the retrieval pipeline as a *topic filter*: it predicts the top-k
macro_domain labels for a query so candidate passages can be narrowed before
scoring. Includes a state-dict helper that infers architecture dimensions
directly from saved weights, so checkpoints load without a separate config file.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    """Convolutional topic classifier (filter sizes 3/4/5, max-over-time pool)."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_classes: int,
        filter_sizes: tuple[int, ...] = (3, 4, 5),
        num_filters: int = 100,
        dropout: float = 0.0,
        embed_dropout: float = 0.0,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.embed_dropout = nn.Dropout(embed_dropout)
        self.convs = nn.ModuleList(
            [nn.Conv1d(embed_dim, num_filters, fs) for fs in filter_sizes]
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(filter_sizes), num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embed_dropout(self.embedding(x)).transpose(1, 2)
        pooled = []
        for conv in self.convs:
            feat = F.relu(conv(emb))
            pooled.append(F.max_pool1d(feat, kernel_size=feat.shape[-1]).squeeze(-1))
        return self.fc(self.dropout(torch.cat(pooled, dim=1)))


def build_textcnn_from_state_dict(state: dict, num_classes: int) -> TextCNN:
    """Construct a :class:`TextCNN` whose dims match a saved state dict."""
    emb_w = state["embedding.weight"]
    conv_keys = sorted(
        k for k in state if k.startswith("convs.") and k.endswith(".weight")
    )
    filter_sizes = tuple(int(state[k].shape[-1]) for k in conv_keys)
    num_filters = int(state[conv_keys[0]].shape[0])
    return TextCNN(
        vocab_size=int(emb_w.shape[0]),
        embed_dim=int(emb_w.shape[1]),
        num_classes=num_classes,
        filter_sizes=filter_sizes,
        num_filters=num_filters,
        dropout=0.0,
        embed_dropout=0.0,
        pad_idx=0,
    )
