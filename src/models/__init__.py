"""Model definitions: topic classifier (TextCNN) and dense bi-encoder (Siamese LSTM).

``cross_encoder`` / reranking models are not yet implemented; see the project
roadmap before adding them here.
"""

from __future__ import annotations

from .bi_encoder import (
    LSTMEncoder,
    SiameseLSTM,
    add_encoder_prefix_if_needed,
    infer_siamese_config,
)
from .classifier import TextCNN, build_textcnn_from_state_dict, load_textcnn_from_artifacts
from .siamese_bilstm import SiameseBiLSTM, SiameseBiLSTMEncoder, ContrastiveLoss, load_siamese_from_artifacts

__all__ = [
    "TextCNN",
    "build_textcnn_from_state_dict",
    "LSTMEncoder",
    "SiameseLSTM",
    "add_encoder_prefix_if_needed",
    "infer_siamese_config",
    "SiameseBiLSTM",
    "SiameseBiLSTMEncoder",
    "ContrastiveLoss",
    "load_siamese_from_artifacts",
    "load_textcnn_from_artifacts",
]
