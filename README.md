# VNLegal RAG

This repo now includes a PyTorch macro-domain classifier for Vietnamese legal queries. The classifier uses a Vietnamese legal encoder to turn a query into contextual token vectors, then applies the original Kim-style TextCNN head for classification.

## Data Pipeline Recap

The existing data workflow is:

1. `data/load.ipynb`
Creates `passage_id`/`qa_id`, expands each article into QA rows, and assigns `macro_domain` heuristically from `doc_name`.

2. `data/eda.ipynb`
Checks nulls and duplicates, normalizes `question_type` and `difficulty`, inspects label/length distributions, and saves cleaned Hugging Face datasets under `data/processed/`.

3. `data/prepare-data.py`
Deduplicates corpus and QA rows, joins `doc_name`, performs grouped stratified splitting by `doc_name`, writes TSV files to `data/data_ready/`, and builds `label_maps.json`.

## Classifier Design

- Input: question text only
- Target: `macro_domain`
- Encoder: `ntphuc149/ViLegalBERT` by default, with fallback to `vinai/phobert-base-v2`
- CNN head: original TextCNN pattern from Kim (2014)
  - filter sizes `3,4,5`
  - `100` feature maps per filter size
  - ReLU
  - max-over-time pooling
  - dropout `0.5`
  - output-layer max-norm constraint `3.0`
- Validation: 5-fold `StratifiedGroupKFold` on `qa_train + qa_val`
- Final evaluation: retrain on full `train+val`, then evaluate once on untouched `qa_test`
- Imbalance handling: `CrossEntropyLoss` with per-fold balanced class weights

## Why K-Fold Here

The current fixed validation split is group-safe, but it is small and does not contain every domain. Grouped k-fold gives a more stable validation estimate while still keeping all questions from the same `doc_name` together.

## Install

```bash
python -m pip install -r requirements.txt
```

## Train

```bash
python -m src.trainers.domain_classifier
```

Useful overrides:

```bash
python -m src.trainers.domain_classifier \
  --model-name vinai/phobert-base-v2 \
  --num-folds 5 \
  --max-epochs 8 \
  --train-batch-size 16 \
  --eval-batch-size 32
```

Artifacts are written to `artifacts/domain_classifier/`.

## Predict

```bash
python -m src.inference.predict_domain \
  --checkpoint artifacts/domain_classifier/final/model.pt \
  --text "Hợp đồng lao động thời vụ có bắt buộc đóng bảo hiểm xã hội không?"
```

## Output Files

- `artifacts/domain_classifier/config.json`
- `artifacts/domain_classifier/fold_assignments.csv`
- `artifacts/domain_classifier/cross_validation_results.csv`
- `artifacts/domain_classifier/cross_validation_summary.json`
- `artifacts/domain_classifier/folds/fold_*/best_model.pt`
- `artifacts/domain_classifier/final/model.pt`
- `artifacts/domain_classifier/final/test_metrics.json`
