"""
Script để build chung vocabulary và embedding khởi tạo cho toàn bộ pipeline.
Sử dụng canonical tokenizer từ tokenizer.py.
"""
import json
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent))
from tokenizer import simple_tokenize, build_vocab, save_vocab, PAD_TOKEN, UNK_TOKEN

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa_path", type=str, default="data/data_ready/qa_train.csv")
    parser.add_argument("--corpus_path", type=str, default="data/data_ready/corpus_train.csv")
    parser.add_argument("--out_dir", type=str, default="model/shared_embedding_artifacts")
    parser.add_argument("--max_vocab", type=int, default=6227) # Dựa trên vocab size cũ
    parser.add_argument("--min_freq", type=int, default=1)
    parser.add_argument("--embed_dim", type=int, default=200)
    args = parser.parse_args()

    qa_path = Path(args.qa_path)
    corpus_path = Path(args.corpus_path)
    out_dir = Path(args.out_dir)

    print("1. Loading datasets...")
    texts = []
    
    if qa_path.exists():
        df_qa = pd.read_csv(qa_path, sep="\t")
        for col in ["question", "answer"]:
            if col in df_qa.columns:
                texts.extend(df_qa[col].dropna().astype(str).tolist())
        print(f"   Loaded QA data ({len(df_qa)} rows)")
    else:
        print(f"   [WARN] QA path not found: {qa_path}")

    if corpus_path.exists():
        df_corpus = pd.read_csv(corpus_path, sep="\t")
        if "article_content" in df_corpus.columns:
            texts.extend(df_corpus["article_content"].dropna().astype(str).tolist())
        print(f"   Loaded Corpus data ({len(df_corpus)} rows)")
    else:
        print(f"   [WARN] Corpus path not found: {corpus_path}")

    print(f"\n2. Building vocabulary (max_vocab={args.max_vocab}, min_freq={args.min_freq})...")
    stoi = build_vocab(
        texts=texts,
        max_vocab=args.max_vocab,
        min_freq=args.min_freq,
        pad_token=PAD_TOKEN,
        unk_token=UNK_TOKEN
    )
    vocab_size = len(stoi)
    print(f"   Actual vocab size: {vocab_size:,} tokens")

    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n3. Saving vocabulary...")
    vocab_path = out_dir / "tokenizer_vocab.json"
    build_meta = {
        "qa_path": str(qa_path),
        "corpus_path": str(corpus_path),
        "qa_text_cols": ["question", "answer"],
        "corpus_text_cols": ["article_content"],
        "tokenizer": "simple_tokenize",
    }
    save_vocab(stoi, vocab_path, build_metadata=build_meta)
    print(f"   Saved to {vocab_path}")

    print(f"\n4. Generating embedding matrix (vocab_size={vocab_size}, embed_dim={args.embed_dim})...")
    # Khởi tạo embedding với chuẩn Normal phân phối N(0, std=0.02)
    # Trùng khớp với L2 norm ~ 0.28 trong phiên bản cũ
    embedding_weight = torch.empty(vocab_size, args.embed_dim)
    nn.init.normal_(embedding_weight, mean=0.0, std=0.02)
    
    pad_idx = stoi.get(PAD_TOKEN, 0)
    
    # Zero out PAD token
    with torch.no_grad():
        embedding_weight[pad_idx].fill_(0)

    embed_path = out_dir / "embedding.pt"
    torch.save({
        "embedding_weight": embedding_weight,
        "pad_idx": pad_idx,
        "vocab_size": vocab_size,
        "embed_dim": args.embed_dim,
    }, embed_path)
    
    print(f"   Saved embedding to {embed_path}")
    print("\nShared embedding build completed successfully.")

if __name__ == "__main__":
    main()
