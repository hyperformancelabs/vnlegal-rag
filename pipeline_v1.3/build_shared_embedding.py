"""
Build shared vocab + random embedding init from data_ready_v1_3 (pipeline v1.3).

Self-contained for Colab/Kaggle (no dependency on model/build_shared_embedding.py).
Default output: pipeline_v1.3/shared_embedding_artifacts/

Run from pipeline_v1.3/:
  python build_shared_embedding.py
  python build_shared_embedding.py --max-vocab 8000 --embed-dim 200

Run from repo root:
  python pipeline_v1.3/build_shared_embedding.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn

PIPELINE_DIR = Path(__file__).resolve().parent
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from tokenizer_bootstrap import PAD_TOKEN, UNK_TOKEN, build_vocab, save_vocab


def detect_data_dir() -> Path:
    """Locate data_ready_v1_3 (same candidates as training notebooks)."""
    candidates = [
        Path("data/data_ready_v1_3"),
        Path("../data/data_ready_v1_3"),
        PIPELINE_DIR.parent / "data" / "data_ready_v1_3",
        Path("/content/data/data_ready_v1_3"),
        Path("/kaggle/input/vnlegal-rag/data/data_ready_v1_3"),
        Path("/kaggle/working/vnlegal-rag/data/data_ready_v1_3"),
        Path("/kaggle/input/datasets/hngphtrn/legals/data_ready_v1_3"),
        Path("/kaggle/input/datasets/hngphtrn/legals-v3"),
        Path("/kaggle/input/datasets/hngphtrn/legals-v1-3"),
        Path("/kaggle/input/datasets/hngphtrn/legals_v1_3"),
    ]
    for path in candidates:
        path = path.resolve()
        if path.is_dir() and (path / "qa_train.csv").is_file() and (path / "corpus_train.csv").is_file():
            return path
    raise FileNotFoundError(
        "Cannot find data_ready_v1_3 with qa_train.csv and corpus_train.csv. "
        "Run: python prepare_data.py --out-dir data/data_ready_v1_3 "
        "(Colab: place data under /content/data/data_ready_v1_3/)"
    )


def load_texts(qa_path: Path, corpus_path: Path) -> list[str]:
    texts: list[str] = []

    if qa_path.is_file():
        df_qa = pd.read_csv(qa_path, sep="\t")
        for col in ["question", "answer"]:
            if col in df_qa.columns:
                texts.extend(df_qa[col].dropna().astype(str).tolist())
        print(f"   Loaded QA data ({len(df_qa)} rows) from {qa_path}")
    else:
        print(f"   [WARN] QA path not found: {qa_path}")

    if corpus_path.is_file():
        df_corpus = pd.read_csv(corpus_path, sep="\t")
        if "article_content" in df_corpus.columns:
            texts.extend(df_corpus["article_content"].dropna().astype(str).tolist())
        print(f"   Loaded corpus data ({len(df_corpus)} rows) from {corpus_path}")
    else:
        print(f"   [WARN] Corpus path not found: {corpus_path}")

    if not texts:
        raise FileNotFoundError(
            f"No text loaded from {qa_path} and {corpus_path}. "
            "Check paths or pass --qa-path / --corpus-path explicitly."
        )
    return texts


def build_shared_embedding(
    qa_path: Path,
    corpus_path: Path,
    out_dir: Path,
    max_vocab: int,
    min_freq: int,
    embed_dim: int,
) -> None:
    print("1. Loading datasets...")
    texts = load_texts(qa_path, corpus_path)

    print(f"\n2. Building vocabulary (max_vocab={max_vocab}, min_freq={min_freq})...")
    stoi = build_vocab(
        texts=texts,
        max_vocab=max_vocab,
        min_freq=min_freq,
        pad_token=PAD_TOKEN,
        unk_token=UNK_TOKEN,
    )
    vocab_size = len(stoi)
    print(f"   Actual vocab size: {vocab_size:,} tokens")

    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n3. Saving vocabulary...")
    vocab_path = out_dir / "tokenizer_vocab.json"
    build_meta = {
        "qa_path": str(qa_path.resolve()),
        "corpus_path": str(corpus_path.resolve()),
        "qa_text_cols": ["question", "answer"],
        "corpus_text_cols": ["article_content"],
        "tokenizer": "simple_tokenize",
        "pipeline": "v1.3",
    }
    save_vocab(stoi, vocab_path, build_metadata=build_meta)
    print(f"   Saved to {vocab_path}")

    print(f"\n4. Generating embedding matrix (vocab_size={vocab_size}, embed_dim={embed_dim})...")
    embedding_weight = torch.empty(vocab_size, embed_dim)
    nn.init.normal_(embedding_weight, mean=0.0, std=0.02)

    pad_idx = stoi.get(PAD_TOKEN, 0)
    with torch.no_grad():
        embedding_weight[pad_idx].fill_(0)

    embed_path = out_dir / "embedding.pt"
    torch.save(
        {
            "embedding_weight": embedding_weight,
            "pad_idx": pad_idx,
            "vocab_size": vocab_size,
            "embed_dim": embed_dim,
        },
        embed_path,
    )
    print(f"   Saved embedding to {embed_path}")
    print("\nShared embedding build completed successfully.")


def main() -> None:
    data_dir = detect_data_dir()
    default_qa = data_dir / "qa_train.csv"
    default_corpus = data_dir / "corpus_train.csv"
    default_out = PIPELINE_DIR / "shared_embedding_artifacts"

    parser = argparse.ArgumentParser(description="Shared embedding for v1.3 QA + corpus train split.")
    parser.add_argument("--qa-path", type=str, default=str(default_qa))
    parser.add_argument("--corpus-path", type=str, default=str(default_corpus))
    parser.add_argument("--out-dir", type=str, default=str(default_out))
    parser.add_argument("--max-vocab", type=int, default=8000)
    parser.add_argument("--min-freq", type=int, default=1)
    parser.add_argument("--embed-dim", type=int, default=200)
    args = parser.parse_args()

    print(f"DATA_DIR = {data_dir}")
    print(f"OUT_DIR  = {args.out_dir}")

    build_shared_embedding(
        qa_path=Path(args.qa_path),
        corpus_path=Path(args.corpus_path),
        out_dir=Path(args.out_dir),
        max_vocab=args.max_vocab,
        min_freq=args.min_freq,
        embed_dim=args.embed_dim,
    )


if __name__ == "__main__":
    main()
