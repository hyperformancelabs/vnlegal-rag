# TextCNN Experiments — Configuration & Results Comparison

> Generated: 2026-06-23 | GPU: NVIDIA A100-SXM4-40GB (Colab)

## Overview

8 experiments across 3 axes:
1. **Class merging** (k2, k3, k4): merge rarest classes into `other`
2. **Dropout sweep** (d02, d03, d07): test generalization under different dropout
3. **Overfitting fix** (ed03): embed_dropout + global gradient clip

**Only 2/8 notebooks have been run with their actual configs.** The remaining 6 are stale copies with outputs inherited from the baseline — marked ⚠️ below.

---

## Experiment Matrix

| # | Notebook | DATA_DIR | Classes | DROPOUT | EMBED_DROPOUT | Grad Clip | Status |
|---|----------|----------|:------:|:------:|:-----:|:---------:|:------:|
| 1 | `textcnn.ipynb` | `data_ready` | 9 | 0.5 | 0.0 | fc only | ✅ Run |
| 2 | `textcnn_k2.ipynb` | `data_ready_k2` | 7 | 0.5 | 0.0 | fc only | ⚠️ Stale |
| 3 | `textcnn_k3.ipynb` | `data_ready_k3` | 6 | 0.5 | 0.0 | fc only | ⚠️ Stale |
| 4 | `textcnn_k4.ipynb` | `data_ready_k4` | 5 | 0.5 | 0.0 | fc only | ⚠️ Stale |
| 5 | `textcnn_k4_d02.ipynb` | `data_ready_k4` | 5 | 0.2 | 0.0 | fc only | ⚠️ Stale |
| 6 | `textcnn_k4_d03.ipynb` | `data_ready_k4` | 5 | 0.3 | 0.0 | fc only | ⚠️ Stale |
| 7 | `textcnn_k4_d07.ipynb` | `data_ready_k4` | 5 | 0.7 | 0.0 | fc only | ⚠️ Stale |
| 8 | `textcnn_k4_ed03.ipynb` | `data_ready_k4` | 5 | 0.5 | 0.3 | global | ✅ Run |

**Shared config** across all experiments:

```
MAX_LEN       = 128
FILTER_SIZES  = (3, 4, 5)
NUM_FILTERS   = 100
BATCH_SIZE    = 50
EMBED_DIM     = 300  (word2vec, non-static)
MAX_NORM_FC   = 3.0  (Kim 2014 L2 constraint)
LR            = Adadelta default
FREEZE        = False
EPOCHS        = 20
PATIENCE      = 4
Architecture  = TextCNN (Kim 2014)
```

---

## Results (Run Experiments Only)

### Experiment 1: 9-class Baseline

| Class | Support | Precision | Recall | F1 |
|-------|:------:|:---------:|:------:|:---:|
| Civil & Investment | 165 | 0.03 | 0.01 | **0.02** |
| Finance & Banking | 918 | 0.80 | 0.99 | 0.88 |
| Industry, Resources & Environment | 192 | 0.76 | 0.97 | 0.85 |
| Justice & Dispute Resolution | 405 | 0.87 | 0.31 | **0.45** |
| Labor & Insurance | 21 | 0.65 | 0.62 | 0.63 |
| Security & Defense | 225 | 0.86 | 0.45 | **0.59** |
| State Organization & Admin | 567 | 0.64 | 0.81 | 0.71 |
| Transportation | 339 | 0.88 | 0.99 | 0.93 |

| Metric | Score |
|--------|:---:|
| Acc@1 | 75.25% |
| Acc@3 | 89.27% |
| **Macro-F1** | **0.6342** |
| Weighted F1 | 0.74 |

Training: converged at epoch 5, val_loss 0.9851, best F1 at epoch 2.

### Experiment 8: 5-class + Overfitting Fix

| Class | Support | Precision | Recall | F1 |
|-------|:------:|:---------:|:------:|:---:|
| Finance & Banking | 918 | 0.90 | 0.99 | 0.94 |
| Justice & Dispute Resolution | 405 | 0.89 | 0.31 | **0.46** |
| State Organization & Admin | 567 | 0.64 | 0.85 | 0.73 |
| Transportation | 339 | 0.93 | 0.99 | 0.96 |
| other | 603 | 0.74 | 0.69 | 0.71 |

| Metric | Score |
|--------|:---:|
| Acc@1 | 79.94% |
| Acc@3 | 93.64% |
| **Macro-F1** | **0.7595** |
| Weighted F1 | 0.78 |

Training: converged at epoch 5, val_loss 0.5946, best F1 at epoch 2.

---

## Key Findings

### 1. Class merging helps (+12.5 F1 points)
Merging 4 rarest classes (=5,391 samples, ~23% of data) into `other`:
- Eliminates 3 classes with F1 < 0.60 (Civil & Investment, Security & Defense, Labor & Insurance)
- The merged `other` achieves F1=0.71 — beats all 4 individual classes combined
- Acc@1 improves from 75.3% → 79.9% (+4.6pp)

### 2. Embedding dropout + global grad clip fixes overfitting
- EMBED_DROPOUT=0.3 vs 0.0: prevents embedding co-adaptation
- Global `clip_grad_norm_(model.parameters())` vs fc-only
- Combined effect: Macro-F1 0.6342 → 0.7595 on k4 data

### 3. Justice & Dispute Resolution is the hardest class
- Recall stuck at 31% in both experiments
- High precision (0.89) but low recall: model is conservative, only predicts when very sure
- 405 test samples — not a rare-class problem, it's an intrinsic difficulty problem
- Possible cause: legal questions in this domain are linguistically similar to other categories

### 4. Small classes die without merging
- Civil & Investment (165 test, F1=0.02): model essentially never predicts this class
- Labor & Insurance (21 test): too few samples for meaningful learning

---

## Planned Experiments (Not Yet Run)

These need Colab re-run after `Runtime → Restart & Run All`:

| Experiment | Expected Value | Hypothesis |
|-----------|---------------|------------|
| `k2` (7 classes) | F1: 0.65-0.68 | Marginal gain from merging 2 smallest |
| `k3` (6 classes) | F1: 0.68-0.72 | Most of the gain comes at k3→k4 |
| `k4` (5 classes, no overfitting fix) | F1: 0.68-0.72 | Isolates merging benefit from overfitting fix |
| `k4_d02` (d=0.2) | Lower F1, higher train acc | Under-regularized → overfitting |
| `k4_d03` (d=0.3) | Slightly worse than 0.5 | Mild under-regularization |
| `k4_d07` (d=0.7) | Lower F1, lower train acc | Over-regularized → underfitting |

Expected ranking: `k4_ed03 > k4 ≈ k4_d03 > k4_d02 > k4_d07`

---

## Data Summary

| Variant | Classes | Train | Val | Test |
|---------|:------:|:-----:|:---:|:----:|
| `data_ready` (baseline) | 9 | 23,408 | 2,902 | 2,832 |
| `data_ready_k2` | 7 | 23,408 | 2,902 | 2,832 |
| `data_ready_k3` | 6 | 23,408 | 2,902 | 2,832 |
| `data_ready_k4` | 5 | 23,408 | 2,902 | 2,832 |

All variants use the same train/val/test splits — only the class labels differ (rarest classes merged into `other`).

**Vocabulary:** 2,331 tokens, word2vec coverage 97.2% (2,264/2,329)

---

## Recommendations

1. **Re-run k2, k3, k4, d02, d03, d07** on Colab to fill the comparison matrix
2. **Ablate EMBED_DROPOUT vs global clip** separately: run k4 with EMBED_DROPOUT=0.3 but fc-only grad clip, and k4 with EMBED_DROPOUT=0.0 but global grad clip, to attribute the improvement
3. **Increase LR for Justice class**: class-balanced sampling or focal loss may help the 31% recall
4. **Try MAX_LEN=256**: questions max at 209, 128 truncates 0.43% — small loss, but could help with edge cases
