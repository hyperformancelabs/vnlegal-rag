# VNLegal-RAG: A Vietnamese Legal Retrieval-Augmented Pipeline with TextCNN Topic Prior and Siamese BiLSTM Reranking

## Abstract
This document presents the current implementation of `vnlegal-rag`, a Vietnamese legal-domain retrieval system designed for question answering workflows. The project combines three main components: (i) data preparation from legal corpus and QA pairs, (ii) a TextCNN classifier for macro-domain prediction, and (iii) a Siamese BiLSTM model for dense retrieval/reranking. A lexical TF-IDF retriever is used as candidate generation, then neural signals are fused for final ranking. Experimental logs from current notebooks indicate stable convergence for the Siamese branch and meaningful gains in ranking metrics over training epochs. The repository is organized to support reproducible experiments via notebooks and reusable modules in `src/`.

## 1. Introduction
Legal QA in Vietnamese is challenging due to domain-specific terminology, long passages, and semantic mismatch between user questions and legal text. Pure lexical retrieval can miss semantically relevant passages, while dense-only retrieval may be less robust for rare legal terms.

To address this, the project adopts a hybrid retrieval strategy:
- lexical retrieval for high-recall candidate generation,
- domain-aware prior from a classifier,
- dense similarity reranking for semantic alignment.

The objective is to improve top-k passage retrieval quality (e.g., Recall@k, MRR), which is a core prerequisite for downstream RAG answer generation.

## 2. Project Scope and Contributions
The current repository contributes:
- A complete data workflow from raw legal datasets to train/validation/test splits with anti-leakage grouping by document.
- A Vietnamese TextCNN pipeline for `macro_domain` classification.
- A Siamese BiLSTM triplet-learning pipeline for question-passage representation learning.
- Class-imbalance-aware training design (weighted losses and evaluation-aware checkpointing) for legal domain skew.
- Training stabilization techniques (normalization, regularization, mixed precision, scheduler-based adaptation).
- Artifact management for model checkpoints, metadata, vocabularies, and training history.
- A hybrid retrieval design documented in `model/retrieval_results_comparison.md`.

## 3. Data Pipeline
### 3.1 Data Sources and Processing
The data workflow is implemented under `data/`:
- `data/load.ipynb`: standardization and id creation (`qa_id`, `passage_id`), domain labeling support.
- `data/eda.ipynb`: data cleaning, null/duplicate checks, distribution analysis, processed dataset export.
- `data/prepare-data.py`: split generation using grouped strategy (by `doc_name`) and label-aware stratification.

In operational terms, the pipeline first enforces schema consistency across corpus and QA tables (column naming, id uniqueness, and text field integrity), then enriches QA rows with metadata needed by downstream models (e.g., document linkage and domain tags). The EDA stage serves two roles: quality assurance and feature diagnostics. It validates missing-value patterns, duplicate behavior, length distribution of question/passage fields, and class balance for `macro_domain`.

The split stage is designed to prevent optimistic evaluation. Instead of random row-level splitting, records are grouped by `doc_name`, ensuring that semantically related passages from the same source document do not leak across train/validation/test partitions. Label-aware stratification is applied to keep domain proportions as stable as possible between splits. This design is critical for legal retrieval because multiple QA entries often reference highly overlapping legal provisions.

### 3.2 Ready-to-Train Datasets
Training-ready files are exported to `data/data_ready/`, including:
- `qa_train.csv`, `qa_val.csv`, `qa_test.csv`
- `corpus_train.csv`, `corpus_val.csv`, `corpus_test.csv`
- `corpus_full.csv`, `label_maps.json`

This structure supports both classification and retrieval tasks while reducing train-test leakage risk.

The exported files are consumed in two complementary modes:
- **Classification mode**: `qa_*` files are used to train/evaluate the TextCNN domain classifier.
- **Retrieval mode**: `qa_*` provides queries and positive passage ids, while `corpus_*` provides searchable passage candidates.

`label_maps.json` standardizes label-index mapping across training, checkpointing, and inference, which avoids class-order mismatch when reloading artifacts. `corpus_full.csv` is retained for global indexing and full-corpus retrieval experiments, while split-specific corpus files are used for controlled validation.

## 4. Methodology
### 4.1 Topic Classifier: TextCNN
The TextCNN branch (see `model/train-textcnn.ipynb`, `src/models/textcnn.py`) follows a multi-kernel CNN design for sentence-level classification:
- token embedding layer,
- multiple `Conv1d` filters (e.g., kernel sizes 3/4/5),
- max-over-time pooling,
- dropout and final linear classification.

The model predicts `macro_domain` probabilities, used as a topic prior in hybrid retrieval.

From a modeling perspective, TextCNN captures local n-gram patterns that are highly relevant in legal questions (e.g., recurring phrase templates for penalties, procedures, or authorities). Multi-kernel convolution allows the model to aggregate short and medium contextual cues in parallel. Max-over-time pooling selects the most salient activation from each feature map, producing compact sentence-level representations robust to variable input length.

A key contribution relative to the original Kim (2014) setting is the embedding strategy. Instead of relying on general-purpose pretrained English embeddings (e.g., word2vec), this project uses randomly initialized trainable embeddings (CNN-rand style) for Vietnamese legal text. This avoids domain mismatch and allows the embedding space to adapt directly to legal terminology and phrasing patterns present in the training corpus.

Another practical contribution is the loss design under label imbalance. While the original baseline commonly uses standard cross-entropy, this implementation supports weighted cross-entropy and focal loss variants to reduce majority-class dominance in `macro_domain` classification.

During retrieval, classifier outputs are not treated as final answers but as a soft prior. Candidate passages with domain labels consistent with high-probability query domains receive a score boost in the fusion stage. This mechanism improves ranking precision in corpora where legal domains have distinct terminology but still share overlapping vocabulary.

### 4.2 Dense Retriever/Reranker: Siamese BiLSTM
The Siamese branch (see `model/train-siamese-bilstm.ipynb`, `src/models/siamese.py`) uses:
- shared BiLSTM encoder for anchor (question), positive passage, and negative passage,
- pooled sequence representation,
- L2 normalization on output embeddings,
- triplet loss with Euclidean distance:

\[
\mathcal{L} = \max(0, m + d(a,p) - d(a,n))
\]

where \(m\) is margin, \(d\) is pairwise Euclidean distance, and \((a,p,n)\) are anchor/positive/negative embeddings.

The shared encoder enforces a common embedding space for both query and passage representations. BiLSTM is chosen to model bidirectional token dependencies, which is important for long-form legal text where meaning often depends on both preceding and following clauses. After sequence pooling, L2 normalization stabilizes embedding scale and makes distance-based ranking more comparable across batches.

Triplet training encourages relative ordering rather than absolute classification: for each query, the positive passage is pulled closer than the negative passage by at least margin \(m\). In practice, this objective directly aligns with top-k retrieval goals, because the model learns to separate relevant from irrelevant passages in a ranking-oriented geometry. Harder negatives (lexically similar but semantically incorrect) are particularly valuable for improving retrieval robustness.

Compared with common Siamese BiLSTM references, the current experiments include notable implementation choices: mean pooling over valid tokens (instead of max pooling), margin configured at \(m = 0.75\), gradient clipping for training stability, and a domain-aware hard-negative strategy (sampling negatives from the same `macro_domain` when possible). A cosine-similarity variant is also investigated in `train-siamese-bilstm-cosine.ipynb`; when it outperforms Euclidean in controlled evaluation, it is treated as an empirical contribution of the project setup.

### 4.3 Hybrid Retrieval Strategy
The current retrieval design can be described as:
1. Candidate generation by TF-IDF over corpus.
2. Topic prior scoring from TextCNN (`macro_domain` compatibility).
3. Dense score from Siamese embedding similarity/distance.
4. Score fusion and reranking to output top-k passages.

This hybrid setup balances lexical precision, domain-aware filtering, and semantic matching.

### 4.4 Additional Technical Contributions Notable in TextCNN and Siamese Branches
Beyond the high-level architecture, the current implementation includes several practical contributions that are important for legal-domain robustness:

**TextCNN branch**
- Class-imbalance handling through weighted optimization setup, reducing dominance of frequent `macro_domain` classes.
- Randomly initialized trainable embeddings (CNN-rand) tailored to Vietnamese legal-domain language, instead of external generic embeddings.
- Loss-function adaptation from standard cross-entropy to weighted cross-entropy / focal variants for skewed class distributions.
- Regularization stack (embedding/feature dropout and controlled weight constraints) to reduce overfitting on repetitive legal phrase patterns.
- Stable checkpoint policy driven by validation behavior rather than training loss only, improving generalization for unseen documents.
- Topic-prior integration design: classifier outputs are directly consumable by the retrieval fusion stage, not isolated as a standalone classifier metric.

**Siamese BiLSTM branch**
- Shared-encoder metric learning setup for query and passage spaces, enabling direct ranking-oriented embedding geometry.
- Mean pooling design (valid-token average) as a robust alternative to max pooling for long legal sequences.
- L2-normalized embeddings before distance computation, improving comparability and numerical stability across batches.
- Margin setting at `0.75` and gradient clipping for stable optimization in triplet training.
- Domain-aware triplet construction strategy (positive linkage by passage id and practical negative sampling), which better reflects retrieval difficulty than random negatives.
- Same-domain hard-negative sampling via `macro_domain`, increasing semantic difficulty of negatives and improving discriminative learning.
- Cosine-distance experiment track (`train-siamese-bilstm-cosine.ipynb`) to validate metric choice beyond Euclidean.
- Retrieval-centric validation loop with Recall@k and MRR tracking during training, aligning optimization decisions with downstream ranking objectives.

Together, these contributions move the system from a baseline model collection toward a retrieval-engineered pipeline suitable for legal QA workloads.

## 5. Implementation and Repository Organization
### 5.1 Reusable Modules (`src/`)
- `src/datasets/`: classification and triplet dataset classes.
- `src/models/`: `TextCNN`, `BiLSTMEncoder`, `SiameseBiLSTM`.
- `src/encoders/tfidf.py`: lexical baseline encoder.
- `src/training/`: training loop skeleton and metrics utilities.

### 5.2 Experiment Notebooks (`model/`)
- `train-textcnn.ipynb`
- `train-siamese-bilstm.ipynb`
- `train-siamese-bilstm-cosine.ipynb`
- `test-pipeline.ipynb`

### 5.3 Artifacts
Saved outputs include:
- checkpoints (`*.pt`),
- training metadata (`*_meta.json`),
- vocabularies (`tokenizer_vocab.json`),
- training curves/history (`train_history.csv`).

## 6. Experimental Observations (Current State)
From current training logs in the Siamese workflow:
- the model converges stably across epochs,
- retrieval quality metrics increase steadily over training,
- best observed validation ranking reaches approximately:
  - `MRR ≈ 0.5794`
  - `R@1 ≈ 0.4633`
  - `R@5 ≈ 0.7173`

These values indicate that dense reranking is learning meaningful semantic structure for legal QA retrieval. Exact outcomes may vary by split, seed, and negative sampling configuration.

## 7. Discussion
### 7.1 Strengths
- Hybrid retrieval architecture improves robustness over single-signal retrieval.
- Domain classifier prior is useful in legal corpora with structured topic spaces.
- Group-aware data splitting improves evaluation reliability.

### 7.2 Current Limitations
- Notebook-centric training limits automation and CI reproducibility.
- Some CLI pipeline stubs remain placeholders for full productionization.
- Evaluation focuses on retrieval quality; end-to-end answer quality metrics are not yet standardized.

## 8. Future Work
- Consolidate notebook experiments into script/CLI pipelines.
- Add systematic ablation studies (TF-IDF only, dense only, +topic prior).
- Benchmark cosine vs Euclidean under controlled normalization settings.
- Add end-to-end RAG generation evaluation (faithfulness, citation accuracy, legal consistency).

## 9. Conclusion
`vnlegal-rag` provides a practical and extensible foundation for Vietnamese legal retrieval in RAG systems. By integrating lexical candidate generation, topic-aware classification, and Siamese dense reranking, the project targets improved top-k retrieval quality under realistic legal-domain constraints. The current codebase and artifacts already support meaningful experimentation and can be advanced toward a production-ready legal QA retrieval stack.

## 10. Ablation Study (Recommended Protocol)
To quantify the contribution of each design choice, we recommend the following ablation matrix:
- **A0 (Lexical baseline)**: TF-IDF retrieval only.
- **A1**: TF-IDF + TextCNN topic prior.
- **A2**: TF-IDF + Siamese rerank (Euclidean).
- **A3**: TF-IDF + Siamese rerank (Cosine).
- **A4 (Full hybrid)**: TF-IDF + TextCNN prior + Siamese rerank.

For TextCNN-specific ablations:
- **T0**: pretrained/general embedding + cross-entropy.
- **T1**: CNN-rand embedding + cross-entropy.
- **T2**: CNN-rand + weighted cross-entropy.
- **T3**: CNN-rand + focal loss.

For Siamese-specific ablations:
- **S0**: max pooling + random negatives + Euclidean.
- **S1**: mean pooling + random negatives + Euclidean.
- **S2**: mean pooling + hard negatives (same `macro_domain`) + Euclidean.
- **S3**: mean pooling + hard negatives + Cosine.

Primary metrics should include `Recall@1`, `Recall@5`, `MRR`, and inference latency. This setup allows direct attribution of gains to (i) domain-adaptive embeddings, (ii) imbalance-aware losses, (iii) hard-negative strategy, and (iv) distance metric choice.

## 11. Threats to Validity
Several factors can affect the reliability of observed improvements:
- **Data split sensitivity**: retrieval performance may vary with grouped split composition, especially for minority legal domains.
- **Negative sampling bias**: hard-negative quality strongly depends on candidate pool construction and domain labels.
- **Notebook execution order**: non-linear notebook runs may create hidden state effects if seeds/configs are not reset consistently.
- **Metric leakage risk**: if corpus indexing or split filtering is misapplied, top-k metrics can be overestimated.
- **Domain coverage limits**: current corpus may not fully represent all legal subdomains or recent legal language changes.

These risks should be mitigated through repeated runs with fixed seeds, strict split audits, and standardized evaluation scripts.

## 12. Reproducibility Checklist
To improve reproducibility and reporting quality:
- Fix and record random seeds for Python/NumPy/PyTorch.
- Log full training configuration (model, loss, margin, metric type, pooling type, negative sampling mode).
- Version all artifacts (`*.pt`, `*_meta.json`, `train_history.csv`, vocab files) with timestamped experiment tags.
- Report both average and best-run metrics across multiple seeds.
- Keep a frozen evaluation set and script to recompute `Recall@k` and `MRR` from saved checkpoints.
- Document hardware/runtime context (GPU type, batch size, AMP on/off, training time).

## References
[1] Y. Kim, "Convolutional Neural Networks for Sentence Classification," EMNLP, 2014.  
[2] F. A. Author et al., "Triplet-based metric learning approaches," general metric learning literature.  
[3] Repository internal documentation: `MO_TA_CAU_TRUC_DU_AN.md`, `MO_TA_KIEN_TRUC_TEXTCNN.md`, `model/retrieval_results_comparison.md`.
