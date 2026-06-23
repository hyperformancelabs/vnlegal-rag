"""Integration test: load both best models from artifacts and validate inference.

Run from repo root:
    conda run -n vnlegal python experiments/test_model_artifacts.py

Tests:
  1. Load SiameseBiLSTM via load_siamese_from_artifacts()
  2. Load TextCNN via load_textcnn_from_artifacts()
  3. Use src.tokenizer.simple_tokenize to encode real Vietnamese legal text
  4. Run Siamese retrieval on a small subset
  5. Run TextCNN classification on a small subset
  6. Verify model outputs are reasonable (no NaN, expected shapes)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from src import (
    load_siamese_from_artifacts,
    load_textcnn_from_artifacts,
    simple_tokenize,
    default_device,
)
from src.data.loaders import load_qa, load_corpus

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_DIR = Path("data/data_ready_k4")
SIAMESE_DIR = Path("experiments/siamese_256_artifacts")
TEXTCNN_DIR = Path("experiments/textcnn_k4_artifacts")
MAX_LEN_SIAMESE = 256
MAX_LEN_TEXTCNN = 256
DEVICE = default_device()

MSGS: list[str] = []


def check(desc: str, ok: bool) -> None:
    tag = "✅" if ok else "❌"
    MSGS.append(f"  {tag} {desc}")
    if not ok:
        MSGS.append(f"     FAILED: {desc}")


# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
print("1. Loading data...")
qa = load_qa(DATA_DIR)
corpus = load_corpus(DATA_DIR)

check("qa splits loaded", len(qa) == 3 and "train" in qa)
check("corpus loaded", len(corpus) > 0)
check("question column present", "question" in qa["test"].columns)
check("article_content column present", "article_content" in corpus.columns)
check("macro_domain column present", "macro_domain" in qa["test"].columns)
print(f"   qa_test: {len(qa['test'])} rows, corpus: {len(corpus)} rows")

# ---------------------------------------------------------------------------
# 2. Load Siamese model
# ---------------------------------------------------------------------------
print("\n2. Loading SiameseBiLSTM...")
siamese, siamese_stoi, siamese_meta = load_siamese_from_artifacts(SIAMESE_DIR, device=DEVICE)

check("siamese model loaded", siamese is not None)
check("siamese stoi loaded", len(siamese_stoi) > 0)
check(f"siamese vocab matches meta", len(siamese_stoi) == siamese_meta["vocab_size"])
check("siamese meta has retrieval results", "test_retrieval" in siamese_meta)
check(
    "siamese MRR matches expected range",
    0.2 <= siamese_meta["test_retrieval"]["mrr"] <= 0.5,
)
n_params = sum(p.numel() for p in siamese.parameters())
check(f"siamese params ({n_params:,}) > 1M", n_params > 1_000_000)
print(f"   vocab={len(siamese_stoi)}, MRR={siamese_meta['test_retrieval']['mrr']:.4f}")

# ---------------------------------------------------------------------------
# 3. Load TextCNN model
# ---------------------------------------------------------------------------
print("\n3. Loading TextCNN k4...")
textcnn, textcnn_stoi, textcnn_labels, textcnn_meta = load_textcnn_from_artifacts(TEXTCNN_DIR, device=DEVICE)

check("textcnn model loaded", textcnn is not None)
check("textcnn stoi loaded", len(textcnn_stoi) > 0)
check(f"textcnn vocab matches meta", len(textcnn_stoi) == textcnn_meta["vocab_size"])
check("textcnn labels present", len(textcnn_labels) == 5)
check("textcnn has num_classes=5", textcnn_meta["num_classes"] == 5)
check("textcnn has filter_sizes", "filter_sizes" in textcnn_meta)
check(
    "textcnn F1 in expected range",
    0.6 <= textcnn_meta["test_macro_f1"] <= 0.9,
)
n_tc = sum(p.numel() for p in textcnn.parameters())
check(f"textcnn params ({n_tc:,}) ≈ 1M", abs(n_tc - 1_061_000) < 10_000)
print(f"   vocab={len(textcnn_stoi)}, F1={textcnn_meta['test_macro_f1']:.4f}, labels={textcnn_labels}")


# ---------------------------------------------------------------------------
# 4. Test Siamese inference
# ---------------------------------------------------------------------------
print("\n4. Testing Siamese inference...")

def encode_siamese(texts: list[str]) -> torch.Tensor:
    pad = siamese_stoi.get("<PAD>", 0)
    unk = siamese_stoi.get("<UNK>", 1)
    batch = []
    for t in texts:
        tokens = simple_tokenize(str(t))
        ids = [siamese_stoi.get(tk, unk) for tk in tokens[:MAX_LEN_SIAMESE]]
        ids += [pad] * (MAX_LEN_SIAMESE - len(ids))
        batch.append(ids)
    return torch.tensor(batch, device=DEVICE)


qs = qa["test"]["question"].iloc[:10].tolist()
cs = corpus["article_content"].iloc[:200].tolist()

q_ids = encode_siamese(qs)
c_ids = encode_siamese(cs)

with torch.no_grad():
    q_emb = siamese.encoder(q_ids)
    c_emb = siamese.encoder(c_ids)

check("q_emb shape (10, 128)", q_emb.shape == (10, 128))
check("c_emb shape (200, 128)", c_emb.shape == (200, 128))
check("q_emb not NaN", not torch.isnan(q_emb).any().item())
check("c_emb not NaN", not torch.isnan(c_emb).any().item())

# Cosine similarity
q_n = q_emb / q_emb.norm(dim=1, keepdim=True).clamp(min=1e-9)
c_n = c_emb / c_emb.norm(dim=1, keepdim=True).clamp(min=1e-9)
sim = q_n @ c_n.T

check("similarity matrix shape (10, 200)", sim.shape == (10, 200))
check("similarities in [-1, 1]", (sim >= -1).all().item() and (sim <= 1).all().item())
max_sim = sim.max(dim=1).values
check(f"max similarity per Q > 0.5 (max={max_sim.mean():.3f})", (max_sim > 0.5).all().item())

# Test explicit-cosine forward
cosines = siamese(q_ids[:4], q_ids[:4])  # same text → should be ~1
check("cosine(q,q) ≈ 1", all(abs(c - 1.0) < 0.01 for c in cosines.tolist()))

# Test encode() method (for dense.py pipeline)
with torch.no_grad():
    masks = (c_ids != siamese_stoi["<PAD>"]).float()
    enc_emb = siamese.encode(c_ids, masks)
check("encode() returns L2-normalized", all(abs(e.norm().item() - 1.0) < 0.01 for e in enc_emb))


# ---------------------------------------------------------------------------
# 5. Test TextCNN inference
# ---------------------------------------------------------------------------
print("\n5. Testing TextCNN inference...")

def encode_textcnn(texts: list[str]) -> torch.Tensor:
    pad = textcnn_stoi.get("<PAD>", 0)
    unk = textcnn_stoi.get("<UNK>", 1)
    batch = []
    for t in texts:
        tokens = simple_tokenize(str(t))
        ids = [textcnn_stoi.get(tk, unk) for tk in tokens[:MAX_LEN_TEXTCNN]]
        ids += [pad] * (MAX_LEN_TEXTCNN - len(ids))
        batch.append(ids)
    return torch.tensor(batch, device=DEVICE)


t_ids = encode_textcnn(qs)
with torch.no_grad():
    logits = textcnn(t_ids)
    probs = torch.softmax(logits, dim=1)
    preds = logits.argmax(dim=1)

check("logits shape (10, 5)", logits.shape == (10, 5))
check("logits not NaN", not torch.isnan(logits).any().item())
check("probabilities sum to 1", all(abs(p.sum() - 1.0) < 0.01 for p in probs))

for i in range(min(5, len(qs))):
    actual = qa["test"]["macro_domain"].iloc[i]
    pred = textcnn_labels[preds[i].item()]
    ok = pred == actual
    tag = "✅" if ok else "❌"
    print(f"  {tag} Q{i}: pred={pred:30s} actual={actual:30s} conf={probs[i].max().item():.3f}")


# ---------------------------------------------------------------------------
# 6. Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
failed = sum(1 for m in MSGS if m.startswith("  ❌"))
passed = sum(1 for m in MSGS if m.startswith("  ✅"))
print(f"Results: {passed} passed, {failed} failed, {passed + failed} total")
for m in MSGS:
    if m.startswith("  ❌") or "FAILED:" in m:
        print(m)
if failed == 0:
    print("\n✅ All tests PASSED!")
else:
    print(f"\n❌ {failed} test(s) FAILED")
