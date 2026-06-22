"""End-to-end retrieval pipeline for VNLegal RAG — config + orchestrator.

``RetrievalPipeline.from_config`` resolves data + artifacts, loads models, builds
the TF-IDF index, and (when Siamese weights exist) precomputes document
embeddings. It then exposes three retrieval modes plus an evaluation helper:

    pipe = RetrievalPipeline.from_config(PipelineConfig())
    pipe.retrieve_tfidf_textcnn("câu hỏi ...")
    pipe.retrieve_siamese_textcnn("câu hỏi ...")   # needs Siamese weights
    pipe.retrieve_siamese_only("câu hỏi ...")      # needs Siamese weights
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import torch

from .data.loaders import (
    load_corpus,
    load_label_list,
    load_qa,
    locate_repo_root,
    resolve_data_dir,
)
from .data.processors import detect_text_col
from .evaluation.evaluator import evaluate_retriever
from .factory import (
    SIAMESE_WEIGHT_NAMES,
    TEXTCNN_WEIGHT_NAMES,
    load_siamese,
    load_textcnn,
    select_artifact_dir,
)
from .retrieval import (
    build_label_to_indices,
    build_tfidf_index,
    candidate_indices_for_labels,
    format_results,
    hybrid_scores,
    precompute_doc_embeddings,
    predict_topic_topk,
    siamese_scores,
    tfidf_scores,
    top_k_order,
)
from .utils.device import default_device


# ── Configuration ─────────────────────────────────────────────────────────────


@dataclass
class PipelineConfig:
    """Tunable settings for building and running the retrieval pipeline.

    Path candidates are tried in order; the first existing directory wins.
    Relative paths are resolved against the located repo root.
    """

    # Reproducibility / inference
    seed: int = 42
    topic_topk: int = 3
    retrieve_k: int = 10
    eval_sample_n: int = 200
    tfidf_max_features: int = 200_000
    precompute_doc_embeddings: bool = True

    # Hybrid fusion: score = α·norm(tfidf) + (1−α)·norm(siamese).
    # Default leans toward TF-IDF (the stronger retriever on current weights).
    hybrid_alpha: float = 0.7

    # Sequence lengths (overridden by artifact meta when available)
    cnn_max_len: int = 128
    max_q_len: int = 64
    max_d_len: int = 256

    # Data split directory candidates (need qa_train/val/test.csv).
    # ``data/data_ready`` (8-label split) is the default — it matches the
    # ``model/`` artifacts below; the k-merged (6-label) splits remain as fallback.
    data_ready_candidates: tuple[str, ...] = (
        "data/data_ready",
        "data/data_ready_k3",
        "data/data_ready_k2",
        "data/data_ready_resplit",
        "/kaggle/input/datasets/hngphtrn/legals-v3",
        "/kaggle/input/legals-v3",
    )

    # TextCNN topic-classifier artifact directory candidates.
    # ``experiments/textcnn_artifacts`` (6-label, max_len=128) is the default.
    textcnn_artifact_candidates: tuple[str, ...] = (
        "experiments/textcnn_artifacts",
        "experiments/textcnn_artifacts_legacy",
        "artifacts/textcnn",
        "/kaggle/working/textcnn_artifacts",
    )

    # Siamese encoder artifact directory candidates (optional).
    # ``experiments/siamese_bilstm1l_artifacts`` is the default.
    siamese_artifact_candidates: tuple[str, ...] = (
        "experiments/siamese_bilstm1l_artifacts",
        "experiments/siamese_bilstm2l_artifacts",
        "experiments/siamese_artifacts",
    )

    device: torch.device = field(default_factory=default_device)

    @property
    def encode_batch_size(self) -> int:
        """Larger batches on GPU, smaller on CPU."""
        return 128 if self.device.type == "cuda" else 64


def seed_everything(seed: int) -> None:
    """Seed Python / NumPy / Torch RNGs for reproducible sampling."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Orchestrator ──────────────────────────────────────────────────────────────


@dataclass
class RetrievalPipeline:
    config: PipelineConfig
    repo_root: object
    corpus_df: pd.DataFrame
    text_col: str
    qa: dict
    label_list: list
    textcnn: dict
    vectorizer: Any
    tfidf_matrix: Any
    label_to_indices: dict
    siamese: dict | None = None
    doc_embeddings: np.ndarray | None = None
    _candidate_attrs: tuple = field(default=(), repr=False)

    # ── Construction ─────────────────────────────────────────────────────────
    @classmethod
    def from_config(cls, config: PipelineConfig, *, verbose: bool = True) -> "RetrievalPipeline":
        seed_everything(config.seed)
        repo_root = locate_repo_root()

        data_dir = resolve_data_dir(config.data_ready_candidates, repo_root)
        corpus_df = load_corpus(data_dir)
        text_col = detect_text_col(corpus_df)
        qa = load_qa(data_dir)
        label_list = load_label_list(data_dir)
        if verbose:
            print(f"[data] dir={data_dir.name}  corpus={len(corpus_df)}  "
                  f"qa_test={len(qa['test'])}  labels={len(label_list)}")

        cnn_dir, cnn_weight = select_artifact_dir(
            config.textcnn_artifact_candidates, repo_root,
            ("textcnn_meta.json", "tokenizer_vocab.json"), TEXTCNN_WEIGHT_NAMES,
        )
        if cnn_dir is None:
            raise FileNotFoundError("No valid TextCNN artifact dir found.")
        textcnn = load_textcnn(cnn_dir, cnn_weight, config.device)
        if verbose:
            print(f"[textcnn] {cnn_dir.name}  max_len={textcnn['cnn_max_len']}")

        corpus_texts = corpus_df[text_col].astype(str).tolist()
        vectorizer, tfidf_matrix = build_tfidf_index(corpus_texts, config.tfidf_max_features)
        label_to_indices = build_label_to_indices(corpus_df)
        if verbose:
            print(f"[tfidf] matrix={tfidf_matrix.shape}")

        sia_dir, sia_weight = select_artifact_dir(
            config.siamese_artifact_candidates, repo_root,
            ("tokenizer_vocab.json",), SIAMESE_WEIGHT_NAMES,
        )
        siamese = load_siamese(
            sia_dir, sia_weight, config.device,
            default_max_q_len=config.max_q_len, default_max_d_len=config.max_d_len,
        )
        doc_embeddings = None
        if siamese is not None and config.precompute_doc_embeddings:
            doc_embeddings = precompute_doc_embeddings(
                corpus_texts, siamese, config.device, batch_size=config.encode_batch_size,
            )
        if verbose:
            print(f"[siamese] {'loaded ' + sia_dir.name if siamese else 'absent — Siamese modes disabled'}")

        return cls(
            config=config, repo_root=repo_root, corpus_df=corpus_df, text_col=text_col,
            qa=qa, label_list=label_list, textcnn=textcnn, vectorizer=vectorizer,
            tfidf_matrix=tfidf_matrix, label_to_indices=label_to_indices,
            siamese=siamese, doc_embeddings=doc_embeddings,
        )

    # ── Helpers ──────────────────────────────────────────────────────────────
    @property
    def has_siamese(self) -> bool:
        return self.siamese is not None and self.doc_embeddings is not None

    def _topic_candidates(self, question: str, topic_topk: int) -> np.ndarray:
        _, labels, _ = predict_topic_topk(question, self.textcnn, self.config.device, k=topic_topk)
        return candidate_indices_for_labels(labels, self.label_to_indices, len(self.corpus_df))

    def _rank(self, indices: np.ndarray, scores: np.ndarray, k: int) -> pd.DataFrame:
        order = top_k_order(scores, k)
        return format_results(self.corpus_df, self.text_col, indices[order], scores[order])

    # ── Public retrieval modes ────────────────────────────────────────────────
    def retrieve_tfidf_textcnn(self, question: str, k: int | None = None, topic_topk: int | None = None) -> pd.DataFrame:
        k = k if k is not None else self.config.retrieve_k
        topic_topk = topic_topk if topic_topk is not None else self.config.topic_topk
        indices = self._topic_candidates(question, topic_topk)
        scores = tfidf_scores(question, indices, self.vectorizer, self.tfidf_matrix)
        return self._rank(indices, scores, k)

    def retrieve_siamese_only(self, question: str, k: int | None = None) -> pd.DataFrame:
        self._require_siamese()
        k = k if k is not None else self.config.retrieve_k
        indices = np.arange(len(self.corpus_df), dtype=np.int64)
        scores = siamese_scores(question, indices, self.siamese, self.doc_embeddings, self.config.device)
        return self._rank(indices, scores, k)

    def retrieve_siamese_textcnn(self, question: str, k: int | None = None, topic_topk: int | None = None) -> pd.DataFrame:
        self._require_siamese()
        k = k if k is not None else self.config.retrieve_k
        topic_topk = topic_topk if topic_topk is not None else self.config.topic_topk
        indices = self._topic_candidates(question, topic_topk)
        scores = siamese_scores(question, indices, self.siamese, self.doc_embeddings, self.config.device)
        return self._rank(indices, scores, k)

    def retrieve_hybrid_textcnn(
        self,
        question: str,
        k: int | None = None,
        topic_topk: int | None = None,
        alpha: float | None = None,
    ) -> pd.DataFrame:
        """Topic-filtered weighted-sum fusion of TF-IDF and Siamese scores."""
        self._require_siamese()
        k = k if k is not None else self.config.retrieve_k
        topic_topk = topic_topk if topic_topk is not None else self.config.topic_topk
        alpha = alpha if alpha is not None else self.config.hybrid_alpha
        indices = self._topic_candidates(question, topic_topk)
        scores = hybrid_scores(
            question, indices, self.vectorizer, self.tfidf_matrix,
            self.siamese, self.doc_embeddings, self.config.device, alpha,
        )
        return self._rank(indices, scores, k)

    def _require_siamese(self) -> None:
        if not self.has_siamese:
            raise RuntimeError(
                "Siamese model/embeddings unavailable — no Siamese weights were found. "
                "Use retrieve_tfidf_textcnn instead."
            )

    # ── Evaluation ─────────────────────────────────────────────────────────────
    def evaluate(self, split: str = "test", *, include_siamese: bool = True) -> list[dict]:
        """Evaluate available retrieval modes on a QA split; return metric rows."""
        qa_df = self.qa[split]
        cfg = self.config
        rows = [evaluate_retriever(
            "tfidf_textcnn", self.retrieve_tfidf_textcnn, qa_df,
            k=cfg.retrieve_k, sample_n=cfg.eval_sample_n, seed=cfg.seed,
        )]
        if include_siamese and self.has_siamese:
            rows.append(evaluate_retriever(
                "siamese_textcnn", self.retrieve_siamese_textcnn, qa_df,
                k=cfg.retrieve_k, sample_n=cfg.eval_sample_n, seed=cfg.seed,
            ))
            rows.append(evaluate_retriever(
                "siamese_only", self.retrieve_siamese_only, qa_df,
                k=cfg.retrieve_k, sample_n=cfg.eval_sample_n, seed=cfg.seed,
            ))
            rows.append(evaluate_retriever(
                "hybrid_textcnn", self.retrieve_hybrid_textcnn, qa_df,
                k=cfg.retrieve_k, sample_n=cfg.eval_sample_n, seed=cfg.seed,
            ))
        return rows
