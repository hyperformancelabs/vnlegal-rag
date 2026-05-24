"""Load TextCNN topic classifier artifacts for retrieval evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_classes: int,
        filter_sizes: tuple[int, ...],
        num_filters: int,
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


def build_textcnn_from_state_dict(state: dict[str, torch.Tensor], num_classes: int) -> TextCNN:
    emb_w = state["embedding.weight"]
    conv_keys = sorted(k for k in state if k.startswith("convs.") and k.endswith(".weight"))
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


def load_textcnn_classifier(
    artifact_dir: Path,
    device: torch.device,
    *,
    weight_names: tuple[str, ...] = ("textcnn_best.pt",),
) -> dict[str, Any]:
    artifact_dir = Path(artifact_dir)
    weights = None
    for name in weight_names:
        candidate = artifact_dir / name
        if candidate.is_file():
            weights = candidate
            break
    if weights is None:
        raise FileNotFoundError(f"No TextCNN weights in {artifact_dir} (tried {weight_names})")

    with open(artifact_dir / "textcnn_meta.json", encoding="utf-8") as f:
        meta = json.load(f)
    with open(artifact_dir / "tokenizer_vocab.json", encoding="utf-8") as f:
        vocab = json.load(f)

    state = torch.load(weights, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    num_classes = int(meta.get("num_classes", len(meta.get("labels", []))))
    model = build_textcnn_from_state_dict(state, num_classes=num_classes).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()

    id2label = {int(k): v for k, v in meta.get("id2label", {}).items()}
    if not id2label and "labels" in meta:
        id2label = {i: lab for i, lab in enumerate(meta["labels"])}

    return {
        "model": model,
        "topic_stoi": vocab["stoi"],
        "id2label": id2label,
        "cnn_max_len": int(meta.get("max_len", 128)),
        "artifact_dir": artifact_dir,
        "weights": weights,
    }
