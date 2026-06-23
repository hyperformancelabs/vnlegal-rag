# Master Comparison — All VnLegal Experiments

> Generated: 2026-06-23 | GPU: NVIDIA A100-SXM4-40GB (Colab)

## Summary Table — All 11 Runs

| # | Notebook | Task | Classes | MAX_LEN | DROPOUT | EMB_DROP | Metric | Score | Best Ep | Epochs |
|---|----------|------|:------:|:---:|:---:|:---:|--------|:---:|:---:|:---:|
| 1 | `textcnn` | Classify | 9 | 256 | 0.5 | 0.0 | F1 | 0.6082 | 5 | 9 |
| 2 | `textcnn_k2` | Classify | 7 | 256 | 0.5 | 0.0 | F1 | 0.6250 | 9 | 13 |
| 3 | `textcnn_k3` | Classify | 6 | 256 | 0.5 | 0.0 | F1 | 0.6342 | 10 | 14 |
| 4 | `textcnn_k4` | Classify | 5 | 256 | 0.5 | 0.0 | F1 | **0.7602** | 5 | 9 |
| 5 | `textcnn_k4_d02` | Classify | 5 | 256 | 0.2 | 0.0 | F1 | 0.7438 | 5 | 9 |
| 6 | `textcnn_k4_d03` | Classify | 5 | 256 | 0.3 | 0.0 | F1 | 0.7386 | 5 | 9 |
| 7 | `textcnn_k4_d07` | Classify | 5 | 256 | 0.7 | 0.0 | F1 | 0.7408 | 5 | 9 |
| 8 | `textcnn_k4_ed03` | Classify | 5 | 256 | 0.5 | 0.3 | F1 | 0.7160 | 5 | 9 |
| — | — | — | — | — | — | — | — | — | — | — |
| 9 | `siamese-bilstm` | Retrieve | — | 256 | 0.2† | — | MRR | 0.3608 | 21 | 26 |
| 10 | `siamese-bilstm512` | Retrieve | — | 512 | 0.2† | — | MRR | **0.3766** | 19 | 24 |
| 11 | `siamese-bilstm1024` | Retrieve | — | 1024 | 0.2† | — | MRR | 0.3563 | 17 | 22 |

> † Siamese dropout: embed=0.2, recurrent=0.2, output=0.4 (3 separate dropout rates, not a single DROPOUT param)

---

## TextCNN — Classification Leaderboard

```
🥇 textcnn_k4          F1=0.7602   5 classes, DROPOUT=0.5
🥈 textcnn_k4_d02      F1=0.7438   5 classes, DROPOUT=0.2  (overfits)
🥉 textcnn_k4_d07      F1=0.7408   5 classes, DROPOUT=0.7  (underfits)
 4  textcnn_k4_d03     F1=0.7386   5 classes, DROPOUT=0.3
 5  textcnn_k4_ed03    F1=0.7160   5 classes, EMBED_DROP=0.3  (hurts)
 6  textcnn_k3         F1=0.6342   6 classes
 7  textcnn_k2         F1=0.6250   7 classes
 8  textcnn            F1=0.6082   9 classes
```

## Siamese — Retrieval Leaderboard

```
🥇 siamese-bilstm512    MRR=0.3766   R@5=52.1%   R@10=63.5%
🥈 siamese-bilstm       MRR=0.3608   R@5=49.4%   R@10=61.4%
🥉 siamese-bilstm1024   MRR=0.3563   R@5=49.8%   R@10=60.8%
```

---

## Key Insights

| Dimension | TextCNN | Siamese BiLSTM |
|-----------|---------|----------------|
| **Task** | 5-way topic classification | 23K-article ranking |
| **Best config** | k4 (5 classes), D=0.5 | MAX_LEN=512 |
| **Difficulty ceiling** | Justice class (recall=31%) | Neg accuracy (63%) |
| **Diminishing returns** | Merging > k4 hurts F1 | MAX_LEN > 512 hurts MRR |
| **Dropout sensitivity** | 0.5 optimal (U-curve) | 3-way dropout, not swept |
| **Embedding** | word2vec 300d, non-static | word2vec 300d, non-static |
| **Params** | 1,062,008 | 2,232,204 |
| **Overfitting gap** | F1 drops 0.08 val→test | neg_acc plateaus 63% |
| **Ready for production?** | Yes — 76% F1 on 5 classes | No — MRR 0.38 < BM25 ~0.45 |

---

## Per-Class Difficulty (TextCNN k4)

| Class | Test samples | F1 | Model behavior |
|-------|:---:|:---:|------|
| Transportation | 339 | 0.96 | Near-perfect |
| Finance & Banking | 918 | 0.85 | Strong, majority class |
| other | 603 | 0.81 | Good, merged 4 rare classes |
| State Organization | 567 | 0.75 | Decent |
| **Justice** | 405 | **0.43** | Recall=28%, confused with State Org |

---

## Per-MAX_LEN Retrieval (Siamese)

| MAX_LEN | MRR | R@1 | R@5 | R@10 | R@20 | Mean Rank | Median Rank |
|:------:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 256 | 0.361 | 24.0% | 49.4% | 61.4% | 72.5% | 32 | 6 |
| **512** | **0.377** | **24.5%** | **52.1%** | **63.5%** | **74.5%** | **30** | **5** |
| 1024 | 0.356 | 22.9% | 49.8% | 60.8% | 72.7% | 33 | 6 |

---

## Training Efficiency

| Notebook | Epochs to best | Wall time (est.) | Checkpoint size |
|----------|:---:|:---:|:---:|
| textcnn (8 runs) | 5-10 | ~2 min each | 12.7 MB |
| siamese 256 | 21 | ~45 min | 26.8 MB |
| siamese 512 | 19 | ~60 min | 26.8 MB |
| siamese 1024 | 17 | ~90 min | 26.8 MB |

---

## What Worked

| Technique | Impact | Evidence |
|-----------|:---:|------|
| **Class merging (9→5)** | +15.2 F1 | TextCNN: 0.608 → 0.760 |
| **DROPOUT=0.5** | Optimal | U-curve: 0.2=0.744, 0.5=0.760, 0.7=0.741 |
| **1-layer BiLSTM (not 4)** | Avoids signal collapse | 4-layer → var/dim = 3e-6 (dead) |
| **relu(cos-m)² (not cos² if cos<m)** | Fixes dead zone | Training converges from epoch 1 |
| **TF-IDF hard negatives** | +8-13% neg_acc | Neg=50% → 63% |
| **Question-only for TextCNN** | Cleaner signal | Q max 209 tokens vs Q+A max 1071 |

## What Didn't Work

| Technique | Impact | Evidence |
|-----------|:---:|------|
| **EMBED_DROPOUT** | −4.4 F1 | 0.760 → 0.716 with static w2v |
| **MAX_LEN=1024** | −0.021 MRR | Worse than 256 despite zero truncation |
| **Margin annealing (0.85→0.5)** | Dead zone | cos_init=0.84 < 0.85 → L-=0 |
| **4-layer BiLSTM** | Signal collapse | Word-level input needs 1 layer |
| **Raw edit .ipynb** | JSON corruption | Must use nbformat |

## What To Try Next

| Idea | Expected gain | Risk |
|------|:---:|------|
| **BM25 + Siamese hybrid** | MRR 0.42-0.48 | Medium — integration complexity |
| **Focal loss for Justice class** | Justice F1 0.43→0.55 | Low — standard technique |
| **Cross-attention (cross-encoder)** | Neg ceiling 63%→75% | High — 10× slower inference |
| **MAX_LEN=384 for Siamese** | MRR 0.37-0.38 | Low — minor sweep |
| **Mixup for TextCNN** | F1 +0.02-0.05 | Low — well-studied |
| **DENSE_DIM bottleneck (128→32→128)** | Discriminative features | Low — simple architecture change |
| **Article metadata prepend** | Neg ceiling +2-5% | Low — just text prepend |

---

> **Detailed reports:** [TEXTCNN_COMPARISON.md](TEXTCNN_COMPARISON.md) | [SIAMESE_COMPARISON.md](SIAMESE_COMPARISON.md)
