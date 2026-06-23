# VNLegal RAG

Legal retrieval-augmented generation for Vietnamese statutory law. Cascaded hybrid retrieval (TF-IDF + Siamese BiLSTM) with TextCNN topic classification and Groq LLM answer synthesis.

## Quick start

```bash
# Web demo
python experiments/demo_siamese_bilstm1l_ui.py
# → http://127.0.0.1:8000
```

## Structure

```
src/              # Package: tokenizer, models, retrieval, data, evaluation
experiments/      # Colab notebooks, web demo, model artifacts
data/
  data_ready_k4/  # Final 5-class dataset variant
  word2vec/       # Pretrained Vietnamese syllable embeddings (300d)
docs/
  final_reports.pdf  # Full academic report
```

## Models

| Model | Location | Key result |
|-------|----------|-----------|
| TextCNN (k4, 5-class) | `experiments/textcnn_k4_artifacts/` | Macro F1 = 0.7602 |
| Siamese BiLSTM (256) | `experiments/siamese_256_artifacts/` | MRR = 0.361 |

## Requirements

- Python 3.11+
- PyTorch, pandas, numpy, scikit-learn
- Groq API key (`.env`: `GROQ_API_KEY=...`)

## Report

See [`docs/final_reports.pdf`](docs/final_reports.pdf) for the full academic report.
