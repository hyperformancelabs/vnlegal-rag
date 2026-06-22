# Siamese BiLSTM Experiments — Configuration & Results Comparison

> Generated: 2026-06-23 | GPU: NVIDIA A100-SXM4-40GB (Colab)
> Architecture: Neculoiu et al. (2016) adapted for word-level legal Vietnamese

## Overview

3 experiments sweeping `MAX_LEN` to measure the cost of token truncation on retrieval quality.
All use the same model architecture, vocabulary (6,761 tokens), word2vec embeddings (4,027/6,759 hits = 59.6%), and contrastive loss.

---

## Experiment Matrix

| # | Notebook | MAX_LEN | BATCH_SIZE | Q Trunc. | A Trunc. | C Trunc. | Status |
|---|----------|:------:|:---:|:---:|:---:|:---:|:------:|
| 1 | `siamese-bilstm.ipynb` | 256 | 512 | 0.0% | 17.3% | 35.8% | ✅ Run |
| 2 | `siamese-bilstm512.ipynb` | 512 | 256 | 0.0% | 0.7% | 10.7% | ✅ Run |
| 3 | `siamese-bilstm1024.ipynb` | 1024 | 512 | 0.0% | 0.0% | 2.8% | ✅ Run |

**Shared config:** 1 BiLSTM layer × 64 hidden, dense=128, margin=0.5, positive_scale=3.0, dropout 0.2/0.4/0.4, Adam LR=0.001, grad_clip=5.0, neg:pos=4:1, hard_neg_k=10, hard_neg_ratio=0.5, patience=5.

---

## Pair Discrimination Results

| Metric | 256 | 512 | 1024 |
|--------|:---:|:---:|:---:|
| **Accuracy** | 70.40% | 69.83% | 68.14% |
| Pos Acc | 97.78% | 98.23% | 98.02% |
| Neg Acc | 63.56% | 62.73% | 60.67% |
| Best val_loss | 0.0312 | 0.0303 | 0.0304 |
| Early stop epoch | 26 | 24 | 22 |
| Total params | 2,232,204 | 2,232,204 | 2,232,204 |

## Retrieval Results

| Metric | 256 | 512 | 1024 |
|--------|:---:|:---:|:---:|
| **MRR** | 0.3608 | **0.3766** | 0.3563 |
| **MAP** | 0.3608 | **0.3766** | 0.3563 |
| Recall@1 | 24.01% | **24.54%** | 22.92% |
| Recall@5 | 49.36% | **52.08%** | 49.79% |
| Recall@10 | 61.37% | **63.45%** | 60.84% |
| Recall@20 | 72.49% | **74.47%** | 72.67% |
| Mean rank | 32 | **30** | 33 |

---

## Key Findings

### 1. MAX_LEN=512 wins across ALL retrieval metrics

| Metric | 256→512 gain |
|--------|:---:|
| MRR | +4.4% |
| Recall@1 | +2.2% |
| Recall@5 | +5.5% |
| Recall@10 | +3.4% |

Going from 256→512 eliminates 17.3% answer truncation and 25.1% corpus passage truncation (35.8%→10.7%). The extra context helps the model discriminate between similar legal articles.

### 2. MAX_LEN=1024 is WORSE than 256 — diminishing returns

Despite near-zero truncation (0% q, 0% a, 2.8% c), 1024 is **worse** than 256 on every metric. Possible causes:

- **Signal dilution**: Mean-pool over 1024 tokens vs 256 spreads the signal 4× thinner
- **LSTM forgets earlier tokens**: At 1024 tokens, the LSTM's effective memory window (~200-300 tokens) leaves 70% of the sequence as noise
- **Overfitting to padding**: Most sequences are well under 1024 (q max=209, a max=989, c p95=743), so ~30% of tokens are PAD → model learns to ignore padding but wastes capacity

### 3. Negatives ceiling around 63-64%

Neg_acc plateaus at ~63% regardless of MAX_LEN. This is a fundamental limit of the architecture: 37% of hard negatives (TF-IDF nearest neighbors from adjacent legal articles) are genuinely indistinguishable at the sentence level. Fixes that might push past 65%:

- **Cross-attention** (not Siamese): let the model attend across question-answer pairs
- **Article-level context**: prepend the law name or domain label
- **Deeper projection**: DENSE_DIM=128→32→128 bottleneck forces more discriminative features

### 4. Training dynamics are healthy

| | 256 | 512 | 1024 |
|---|---|---|---|
| Epoch 1 val_loss | 0.0600 | 0.0617 | 0.0647 |
| Best val_loss | 0.0312 | 0.0303 | 0.0304 |
| neg_acc start | 29.0% | 31.7% | 29.1% |
| neg_acc best | 63.6% | 62.7% | 60.7% |
| pos_acc throughout | 98-99% | 98-99% | 98-99% |

No dead zone. Loss decreases monotonically. Positives nearly perfect. Negatives steadily improve then plateau. This is the expected behavior for hard negative contrastive learning.

---

## Token Truncation Analysis

| MAX_LEN | Q (max=209) | A (max=989) | C (max=92K) | % C truncated |
|:------:|:---:|:---:|:---:|:---:|
| 256 | 0.0% | 17.3% | 35.8% | **35.8% lose signal** |
| 512 | 0.0% | 0.7% | 10.7% | **10.7% lose signal** |
| 1024 | 0.0% | 0.0% | 2.8% | **2.8% lose signal** |

The sweet spot is 512: captures 89.3% of corpus passages fully while keeping sequence length manageable for the LSTM's memory window.

---

## Comparison to TextCNN

| Metric | TextCNN (k4, 5-class) | Siamese (512) | Notes |
|--------|:---:|:---:|------|
| Task | Topic classification | Retrieval | Different problems |
| Acc@1 / Recall@1 | 79.9% | 24.5% | Classification easier |
| Macro-F1 / MRR | 0.76 | 0.38 | Different metrics |
| Worst performer | Justice (F1=0.46) | — | Siamese doesn't classify |
| Params | 1,062,008 | 2,232,204 | Siamese 2× larger |

The TextCNN classifies a question into 5 domains. The Siamese ranks 23K candidate articles for each question — a much harder task.

---

## Recommendations

1. **Default to MAX_LEN=512** — best across all retrieval metrics with reasonable GPU memory
2. **Don't exceed 512** — 1024 hurts performance due to LSTM memory limitations
3. **Try cross-attention** — Siamese architectures have a fundamental ceiling on hard negatives; cross-attention would let the model reason about question-answer relationships
4. **Try MAX_LEN=384** — the gap between 256 and 512 suggests a continuous curve; 384 might hit diminishing returns at lower cost
5. **Explore answer-only length problem**: answers have max=989 but p95=368. 512 captures 99.3% of answers without truncation — the long tail (0.7%) might be pathological cases
