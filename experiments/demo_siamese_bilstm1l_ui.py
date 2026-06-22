"""
Browser demo for pipeline  Siamese BiLSTM (1 layer).

Run from repo root:
  python experiments/demo_siamese_bilstm1l_ui.py

Then open:
  http://127.0.0.1:8000

Configuration:
  Sensitive values (e.g. GROQ_API_KEY) and tunables (e.g. GROQ_MODEL) are read
  from environment variables. A ``.env`` file placed in the repo root or in the
  ``experiments/`` directory will be loaded automatically at startup. See
  ``experiments/.env.example`` for the available keys.
"""

from __future__ import annotations

import argparse
import json
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import os
from pathlib import Path
import sys
from typing import Any
import urllib.error
import urllib.request

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
DEFAULT_ARTIFACT_DIR = REPO_ROOT / "experiments" / "siamese_bilstm1L_artifacts"
DEFAULT_TEXTCNN_ARTIFACT_DIR = REPO_ROOT / "experiments" / "textcnn_artifacts"
_corpus_full = REPO_ROOT / "data" / "data_ready" / "corpus_ready_full.csv"
_corpus_train = REPO_ROOT / "data" / "data_ready" / "corpus_train.csv"
DEFAULT_DATA_PATH = _corpus_full if _corpus_full.is_file() else _corpus_train

from src.tokenizer import simple_tokenize


def select_topic_labels(path: Path, *, override: bool = False) -> None:
    """Minimal .env loader (KEY=VALUE per line). Does not require python-dotenv.

    Supports:
      - Comments starting with '#'
      - Optional 'export KEY=VALUE'
      - Surrounding single/double quotes around the value
    Existing environment variables are preserved unless ``override`` is True.
    """
    if not path.is_file():
        return
    try:
        with open(path, encoding="utf-8") as fh:
            for raw_line in fh:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("export "):
                    line = line[len("export ") :].lstrip()
                if "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip()
                if (len(value) >= 2) and (
                    (value[0] == value[-1] == '"') or (value[0] == value[-1] == "'")
                ):
                    value = value[1:-1]
                if not key:
                    continue
                if override or key not in os.environ:
                    os.environ[key] = value
    except OSError:
        pass


def _bootstrap_env() -> None:
    """Load .env from common locations before any env-var lookups.

    Preference order (later files override earlier ones only if values are missing):
      1. ``<repo_root>/.env``
      2. ``<script_dir>/.env`` (i.e. ``experiments/.env``)
    If ``python-dotenv`` is installed it is used; otherwise the minimal loader
    above is used.
    """
    candidates = [REPO_ROOT / ".env", Path(__file__).resolve().parent / ".env"]
    try:
        from dotenv import load_dotenv

        for env_path in candidates:
            if env_path.is_file():
                load_dotenv(env_path, override=False)
    except ImportError:
        for env_path in candidates:
            _load_env_file(env_path, override=False)


_bootstrap_env()


def select_topic_labels(topic_prediction: dict[str, Any], min_prob: float = 0.12, max_topics: int = 3) -> list[str]:
    """Deployment heuristic only. Benchmarks use raw top-3 via retrieval_eval.predict_topic_topk_labels."""
    top_probs = topic_prediction.get("top_probs", [])
    selected = [
        p["label"]
        for p in top_probs[:max_topics]
        if float(p.get("probability", 0.0)) >= min_prob
    ]
    if not selected and top_probs:
        selected = [top_probs[0]["label"]]
    return selected


HTML_PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>VNLegal  - Siamese BiLSTM</title>
  <style>
    :root {
      --primary: #2563eb;
      --primary-hover: #1d4ed8;
      --bg: #f8fafc;
      --surface: #ffffff;
      --text-main: #0f172a;
      --text-muted: #64748b;
      --border: #e2e8f0;
      --radius: 12px;
    }

    * {
      box-sizing: border-box;
    }

    body {
      font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
      background-color: var(--bg);
      color: var(--text-main);
      margin: 0;
      padding: 40px 20px;
      line-height: 1.5;
    }

    .container {
      max-width: 900px;
      margin: 0 auto;
      background: var(--surface);
      padding: 30px;
      border-radius: var(--radius);
      box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05);
    }

    .header {
      margin-bottom: 24px;
      border-bottom: 2px solid var(--bg);
      padding-bottom: 16px;
    }

    .header h2 {
      margin: 0 0 8px 0;
      color: #1e293b;
      font-weight: 600;
    }

    .header p {
      margin: 0;
      color: var(--text-muted);
      font-size: 14px;
    }

    .input-group {
      margin-bottom: 20px;
    }

    textarea {
      width: 100%;
      min-height: 120px;
      padding: 16px;
      font-size: 15px;
      border: 1px solid var(--border);
      border-radius: 8px;
      resize: vertical;
      font-family: inherit;
      transition: all 0.2s ease;
      background-color: #fcfcfc;
    }

    textarea:focus {
      outline: none;
      border-color: var(--primary);
      box-shadow: 0 0 0 4px rgba(37, 99, 235, 0.1);
      background-color: var(--surface);
    }

    .actions {
      display: flex;
      align-items: center;
      gap: 16px;
    }

    button {
      background-color: var(--primary);
      color: white;
      border: none;
      padding: 10px 24px;
      font-size: 15px;
      font-weight: 500;
      border-radius: 8px;
      cursor: pointer;
      transition: background-color 0.2s ease;
      display: flex;
      align-items: center;
      gap: 8px;
    }

    button:hover {
      background-color: var(--primary-hover);
    }

    button:active {
      transform: translateY(1px);
    }

    .shortcut-hint {
      font-size: 13px;
      color: var(--text-muted);
    }

    #status {
      margin: 20px 0;
      font-size: 14px;
      font-weight: 500;
      color: var(--primary);
    }

    #topic {
      margin-bottom: 16px;
    }

    /* Result Cards */
    .card {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 20px;
      margin-bottom: 16px;
      transition: transform 0.2s, box-shadow 0.2s;
    }

    .card:hover {
      box-shadow: 0 8px 16px rgba(0, 0, 0, 0.04);
      border-color: #cbd5e1;
    }

    .meta {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-bottom: 16px;
      padding-bottom: 12px;
      border-bottom: 1px solid var(--border);
    }

    .badge {
      display: inline-flex;
      align-items: center;
      padding: 4px 10px;
      border-radius: 6px;
      font-size: 12px;
      font-weight: 600;
    }

    .badge.score { background: #dcfce7; color: #166534; }
    .badge.doc { background: #e0f2fe; color: #0369a1; }
    .badge.id { background: #f1f5f9; color: #475569; }
    .badge.rating { background: #fef3c7; color: #92400e; }

    pre {
      white-space: pre-wrap;
      word-wrap: break-word;
      margin: 0;
      font-family: Consolas, Monaco, monospace;
      font-size: 14px;
      line-height: 1.6;
      color: #334155;
      background-color: #f8fafc;
      padding: 16px;
      border-radius: 8px;
      border: 1px solid var(--border);
    }
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <h2>⚖️ VNLegal </h2>
      <p>Hệ thống tra cứu pháp luật sử dụng mô hình Siamese BiLSTM (1 Layer)</p>
    </div>

    <div class="input-group">
      <textarea id="question" placeholder="Nhập câu hỏi pháp lý của bạn vào đây (ví dụ: Tội trộm cắp tài sản bị phạt như thế nào?)..."></textarea>
    </div>
    
    <div class="actions">
      <button id="askBtn">
        <svg width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24" stroke-linecap="round" stroke-linejoin="round">
          <circle cx="11" cy="11" r="8"></circle>
          <line x1="21" y1="21" x2="16.65" y2="16.65"></line>
        </svg>
        Tìm kiếm
      </button>
      <span class="shortcut-hint">Mẹo: Nhấn <strong>Ctrl + Enter</strong> để gửi</span>
    </div>

    <div id="status">Sẵn sàng.</div>
    <div id="topic"></div>
    <div id="answer"></div>
    <div id="results"></div>
  </div>

  <script>
    const askBtn = document.getElementById("askBtn");
    const statusDiv = document.getElementById("status");
    const topicDiv = document.getElementById("topic");
    const answerDiv = document.getElementById("answer");
    const resultsDiv = document.getElementById("results");
    const questionBox = document.getElementById("question");

    function esc(s) {
      return String(s)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;");
    }

    async function ask() {
      const question = questionBox.value.trim();
      if (!question) {
        statusDiv.textContent = "⚠️ Vui lòng nhập câu hỏi trước khi tìm kiếm.";
        statusDiv.style.color = "#dc2626"; // red
        return;
      }

      statusDiv.textContent = "⏳ Đang tìm kiếm tài liệu phù hợp...";
      statusDiv.style.color = "#2563eb"; // blue
      topicDiv.innerHTML = "";
      answerDiv.innerHTML = "";
      resultsDiv.innerHTML = "";
      askBtn.disabled = true;

      try {
        const res = await fetch("/ask", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ question }),
        });
        const data = await res.json();
        if (!res.ok) {
          throw new Error(data.error || "Request failed");
        }

        statusDiv.textContent = `✅ Kết quả.`;
        statusDiv.style.color = "#166534"; // green
        const topic = data.topic_prediction;
        const probs = (topic.top_probs || [])
          .map((p) => `${esc(p.label)}: ${(Number(p.probability) * 100).toFixed(2)}%`)
          .join(" | ");
        topicDiv.innerHTML = `
          <div class="card">
            <div class="meta">
              <span class="badge score">🏷️ Chủ đề: ${esc(topic.label)}</span>
              <span class="badge doc">Độ tin cậy: ${(Number(topic.confidence) * 100).toFixed(2)}%</span>
            </div>
            <div style="font-size:13px;color:#475569;">${probs}</div>
          </div>
        `;
        
        if (!data.results.length) {
          resultsDiv.innerHTML = "<p style='color: #64748b;'>Không tìm thấy kết quả nào phù hợp.</p>";
          return;
        }
        if (data.answer) {
          const confidencePct = Number(data.answer.confidence || 0) * 100;
          const verified = Boolean(data.answer.verified);
          const reason = data.answer.fallback_reason || "";
          const titleBadge = verified
            ? '<span class="badge score">🧠 Trả lời có kiểm chứng (LLM)</span>'
            : '<span class="badge rating">⚠️ Chưa kiểm chứng / từ chối trả lời</span>';
          const reasonLabel = verified
            ? ""
            : `<span class="badge id">Lý do: ${esc(reason || "fallback")}</span>`;
          answerDiv.innerHTML = `
            <div class="card">
              <div class="meta">
                ${titleBadge}
                <span class="badge doc">Độ tin cậy: ${confidencePct.toFixed(2)}%</span>
                <span class="badge id">Nguồn: ${Number(data.answer.used_sources || 0)} đoạn</span>
                ${reasonLabel}
              </div>
              <pre>${esc(data.answer.text || "")}</pre>
            </div>
          `;
        }

        let html = "";
        data.results.forEach((item, idx) => {
          const retrievalRating = item.retrieval_rating ? Number(item.retrieval_rating) : null;
          html += `
            <div class="card">
              <div class="meta">
                <span class="badge score">#${idx + 1} | Độ tương đồng: ${Number(item.siamese_score).toFixed(4)}</span>
                <span class="badge doc">📄 ${esc(item.doc_name || "unknown_doc")}</span>
                <span class="badge id">🔖 ID: ${esc(item.passage_id || "N/A")}</span>
                <span class="badge id">🏷️ Lớp: ${esc(item.macro_domain || "")}</span>
                ${retrievalRating === null ? "" : `<span class="badge rating">⭐ Relevance: ${retrievalRating.toFixed(1)}/10</span>`}
              </div>
              <pre>${esc(item.article_content || "")}</pre>
            </div>
          `;
        });
        resultsDiv.innerHTML = html;
      } catch (err) {
        statusDiv.textContent = "❌ Lỗi: " + err.message;
        statusDiv.style.color = "#dc2626"; // red
      } finally {
        askBtn.disabled = false;
      }
    }

    askBtn.addEventListener("click", ask);
    questionBox.addEventListener("keydown", (e) => {
      if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
        ask();
      }
    });
  </script>
</body>
</html>
"""

class SiameseBiLSTMEncoder(nn.Module):
    def __init__(
        self,
        embedding_weight: torch.Tensor,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
        pad_idx: int,
    ) -> None:
        super().__init__()
        self.pad_idx = int(pad_idx)
        self.embedding = nn.Embedding.from_pretrained(
            embedding_weight, freeze=False, padding_idx=self.pad_idx
        )
        self.lstm = nn.LSTM(
            input_size=embedding_weight.shape[1],
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
            batch_first=True,
        )

    def forward(self, token_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.embedding(token_ids)
        out, _ = self.lstm(x)
        mask = mask.unsqueeze(-1)
        pooled = (out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
        return F.normalize(pooled, p=2, dim=-1)


class TextCNNClassifier(nn.Module):
    def __init__(
        self,
        embedding_weight: torch.Tensor,
        num_classes: int,
        kernel_sizes: list[int],
        num_filters: int,
        dropout: float,
        pad_idx: int,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(
            embedding_weight, freeze=False, padding_idx=pad_idx
        )
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=embedding_weight.shape[1],
                    out_channels=num_filters,
                    kernel_size=k,
                )
                for k in kernel_sizes
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_classes)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(token_ids)
        x = x.transpose(1, 2)
        features = []
        for conv in self.convs:
            c = F.relu(conv(x))
            p = F.max_pool1d(c, kernel_size=c.size(2)).squeeze(2)
            features.append(p)
        cat = torch.cat(features, dim=1)
        return self.fc(self.dropout(cat))


class SiameseBiLSTMDemo:
    def __init__(
        self,
        artifact_dir: Path,
        textcnn_artifact_dir: Path,
        corpus_path: Path,
        max_docs: int,
        max_d_len: int,
        max_q_len: int,
        device: str,
    ) -> None:
        self.artifact_dir = artifact_dir
        self.textcnn_artifact_dir = textcnn_artifact_dir
        self.corpus_path = corpus_path
        self.max_docs = int(max_docs)
        self.max_d_len = int(max_d_len)
        self.max_q_len = int(max_q_len)
        self.device = torch.device(device)

        self.stoi: dict[str, int] = {}
        self.pad_idx = 0
        self.unk_idx = 1
        self.model: SiameseBiLSTMEncoder | None = None
        self.textcnn_model: TextCNNClassifier | None = None
        self.textcnn_stoi: dict[str, int] = {}
        self.textcnn_pad_idx = 0
        self.textcnn_unk_idx = 1
        self.textcnn_max_len = 128
        self.textcnn_labels: list[str] = []
        self.corpus_df = pd.DataFrame()
        self.doc_embeddings: torch.Tensor | None = None
        self.groq_api_key = os.environ.get("GROQ_API_KEY", "").strip()
        self.groq_model = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile").strip()
        self.answer_top_k = int(os.environ.get("ANSWER_TOP_K", "3"))
        self.answer_min_rating = float(os.environ.get("ANSWER_MIN_RATING", "6"))
        self.answer_ctx_char_limit = int(os.environ.get("ANSWER_CTX_CHAR_LIMIT", "1800"))
        self.min_query_words = int(os.environ.get("MIN_QUERY_WORDS", "4"))
        self.min_topic_confidence = float(os.environ.get("MIN_TOPIC_CONFIDENCE", "0.35"))
        self.min_retrieval_score = float(os.environ.get("MIN_RETRIEVAL_SCORE", "0.45"))

    def _encode_text(self, text: str, max_len: int) -> tuple[list[int], list[float]]:
        tokens = simple_tokenize(text)
        ids = [self.stoi.get(tok, self.unk_idx) for tok in tokens[:max_len]]
        length = len(ids)
        if length < max_len:
            ids.extend([self.pad_idx] * (max_len - length))
        mask = [1.0] * length + [0.0] * (max_len - length)
        return ids, mask

    def _batch_encode_docs(self, texts: list[str], batch_size: int = 256) -> torch.Tensor:
        assert self.model is not None
        encoded_batches: list[torch.Tensor] = []
        self.model.eval()

        with torch.no_grad():
            for start in range(0, len(texts), batch_size):
                chunk = texts[start : start + batch_size]
                pair = [self._encode_text(t, self.max_d_len) for t in chunk]
                ids = torch.tensor([p[0] for p in pair], dtype=torch.long, device=self.device)
                mask = torch.tensor([p[1] for p in pair], dtype=torch.float32, device=self.device)
                embs = self.model(ids, mask).cpu()
                encoded_batches.append(embs)

        return torch.cat(encoded_batches, dim=0) if encoded_batches else torch.empty(0, 1)

    def load(self) -> None:
        meta_path = self.artifact_dir / "siamese_meta.json"
        vocab_path = self.artifact_dir / "tokenizer_vocab.json"
        ckpt_path = self.artifact_dir / "siamese_bilstm_best.pt"

        if not meta_path.is_file():
            raise FileNotFoundError(f"Missing metadata file: {meta_path}")
        if not vocab_path.is_file():
            raise FileNotFoundError(f"Missing vocab file: {vocab_path}")
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Missing checkpoint file: {ckpt_path}")
        if not self.corpus_path.is_file():
            raise FileNotFoundError(f"Missing corpus file: {self.corpus_path}")

        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        with open(vocab_path, encoding="utf-8") as f:
            vocab_data = json.load(f)

        self.stoi = vocab_data["stoi"]
        self.pad_idx = self.stoi.get("<PAD>", self.stoi.get("PAD", 0))
        self.unk_idx = self.stoi.get("<UNK>", self.stoi.get("UNK", 1))

        raw_state_dict = torch.load(ckpt_path, map_location="cpu")
        if not isinstance(raw_state_dict, dict):
            raise ValueError(f"Unsupported checkpoint format at {ckpt_path}")

        if "encoder.embedding.weight" in raw_state_dict:
            state_dict = {
                key.removeprefix("encoder."): value for key, value in raw_state_dict.items()
            }
        else:
            state_dict = raw_state_dict

        embedding_weight = state_dict["embedding.weight"]
        ckpt_vocab_size, ckpt_embed_dim = embedding_weight.shape
        loaded_vocab_size = len(self.stoi)
        if ckpt_vocab_size != loaded_vocab_size:
            raise ValueError(
                f"Siamese checkpoint vocab_size={ckpt_vocab_size} != loaded vocab size={loaded_vocab_size}"
            )
        expected_embed_dim = int(meta.get("embedding_dim", meta.get("embed_dim", ckpt_embed_dim)))
        if ckpt_embed_dim != expected_embed_dim:
            raise ValueError(
                f"Siamese checkpoint embed_dim={ckpt_embed_dim} != meta embed_dim={expected_embed_dim}"
            )
        model = SiameseBiLSTMEncoder(
            embedding_weight=embedding_weight,
            hidden_size=int(meta["hidden_size_per_direction"]),
            num_layers=int(meta.get("num_layers", 1)),
            dropout=float(meta.get("dropout", 0.0)),
            bidirectional=bool(meta.get("bidirectional", True)),
            pad_idx=self.pad_idx,
        )
        model.load_state_dict(state_dict, strict=True)
        model.to(self.device)
        model.eval()
        self.model = model

        df = pd.read_csv(self.corpus_path, sep="\t")
        if "article_content" not in df.columns:
            raise ValueError("Corpus file must have an 'article_content' column.")
        if self.max_docs > 0:
            df = df.head(self.max_docs).copy()
        self.corpus_df = df.reset_index(drop=True)
        texts = self.corpus_df["article_content"].fillna("").astype(str).tolist()
        self.doc_embeddings = self._batch_encode_docs(texts)

        self.label_to_indices: dict[str, list[int]] = {}
        if "macro_domain" in self.corpus_df.columns:
            for i, label in enumerate(self.corpus_df["macro_domain"].astype(str)):
                self.label_to_indices.setdefault(label, []).append(i)

        textcnn_meta_path = self.textcnn_artifact_dir / "textcnn_meta.json"
        textcnn_vocab_path = self.textcnn_artifact_dir / "tokenizer_vocab.json"
        textcnn_ckpt_path = self.textcnn_artifact_dir / "textcnn_best.pt"
        if not textcnn_meta_path.is_file():
            raise FileNotFoundError(f"Missing TextCNN metadata file: {textcnn_meta_path}")
        if not textcnn_vocab_path.is_file():
            raise FileNotFoundError(f"Missing TextCNN vocab file: {textcnn_vocab_path}")
        if not textcnn_ckpt_path.is_file():
            raise FileNotFoundError(f"Missing TextCNN checkpoint file: {textcnn_ckpt_path}")

        with open(textcnn_meta_path, encoding="utf-8") as f:
            textcnn_meta = json.load(f)
        with open(textcnn_vocab_path, encoding="utf-8") as f:
            textcnn_vocab_data = json.load(f)

        self.textcnn_stoi = textcnn_vocab_data["stoi"]
        self.textcnn_pad_idx = self.textcnn_stoi.get("<PAD>", self.textcnn_stoi.get("PAD", 0))
        self.textcnn_unk_idx = self.textcnn_stoi.get("<UNK>", self.textcnn_stoi.get("UNK", 1))
        self.textcnn_max_len = int(textcnn_meta.get("max_len", 128))
        self.textcnn_labels = list(textcnn_meta["labels"])

        textcnn_state = torch.load(textcnn_ckpt_path, map_location="cpu")
        if not isinstance(textcnn_state, dict):
            raise ValueError(f"Unsupported TextCNN checkpoint format at {textcnn_ckpt_path}")
        textcnn_embedding = textcnn_state["embedding.weight"]
        tcnn_vocab_size, tcnn_embed_dim = textcnn_embedding.shape
        loaded_tcnn_vocab_size = len(self.textcnn_stoi)
        if tcnn_vocab_size != loaded_tcnn_vocab_size:
            raise ValueError(
                f"TextCNN checkpoint vocab_size={tcnn_vocab_size} != loaded vocab size={loaded_tcnn_vocab_size}"
            )
        if "filter_sizes" in textcnn_meta:
            kernel_sizes = list(textcnn_meta["filter_sizes"])
        else:
            conv_keys = sorted(
                k for k in textcnn_state if k.startswith("convs.") and k.endswith(".weight")
            )
            kernel_sizes = [int(textcnn_state[k].shape[-1]) for k in conv_keys]
        num_filters = int(
            textcnn_meta.get("num_filters", textcnn_state["convs.0.weight"].shape[0])
        )
        textcnn_dropout = float(
            textcnn_meta.get(
                "dropout",
                textcnn_meta.get("train_strategy", {}).get("dropout", 0.5),
            )
        )

        textcnn_model = TextCNNClassifier(
            embedding_weight=textcnn_embedding,
            num_classes=len(self.textcnn_labels),
            kernel_sizes=kernel_sizes,
            num_filters=num_filters,
            dropout=textcnn_dropout,
            pad_idx=self.textcnn_pad_idx,
        )
        textcnn_model.load_state_dict(textcnn_state, strict=True)
        textcnn_model.to(self.device)
        textcnn_model.eval()
        self.textcnn_model = textcnn_model

    def classify_topic(self, query: str, top_n: int = 3) -> dict[str, Any]:
        if self.textcnn_model is None:
            raise RuntimeError("TextCNN model is not loaded.")
        q = query.strip()
        if not q:
            return {"label": "", "confidence": 0.0, "top_probs": []}

        tokens = simple_tokenize(q)
        ids = [self.textcnn_stoi.get(tok, self.textcnn_unk_idx) for tok in tokens[: self.textcnn_max_len]]
        if len(ids) < self.textcnn_max_len:
            ids.extend([self.textcnn_pad_idx] * (self.textcnn_max_len - len(ids)))
        x = torch.tensor([ids], dtype=torch.long, device=self.device)

        with torch.no_grad():
            logits = self.textcnn_model(x)
            probs = F.softmax(logits, dim=1).squeeze(0).cpu()

        best_idx = int(torch.argmax(probs).item())
        top_vals, top_idx = torch.topk(probs, k=min(top_n, len(self.textcnn_labels)))
        top_probs = [
            {"label": self.textcnn_labels[int(i)], "probability": float(v)}
            for v, i in zip(top_vals.tolist(), top_idx.tolist())
        ]
        return {
            "label": self.textcnn_labels[best_idx],
            "confidence": float(probs[best_idx].item()),
            "top_probs": top_probs,
        }

    def search(self, query: str, top_k: int, topic_labels: list[str] | None = None) -> list[dict[str, Any]]:
        if self.model is None or self.doc_embeddings is None:
            raise RuntimeError("Model is not loaded.")
        q = query.strip()
        if not q:
            return []

        if topic_labels and hasattr(self, "label_to_indices") and self.label_to_indices:
            candidate_idx = []
            for label in topic_labels:
                candidate_idx.extend(self.label_to_indices.get(label, []))
            candidate_idx = sorted(set(candidate_idx))
            if not candidate_idx:
                candidate_idx = list(range(len(self.corpus_df)))
        else:
            candidate_idx = list(range(len(self.corpus_df)))

        ids, mask = self._encode_text(q, self.max_q_len)
        q_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
        q_mask = torch.tensor([mask], dtype=torch.float32, device=self.device)

        with torch.no_grad():
            q_emb = self.model(q_ids, q_mask).cpu()
            filtered_embeddings = self.doc_embeddings[candidate_idx]
            scores = torch.matmul(filtered_embeddings, q_emb.t()).squeeze(1)

        k = min(int(top_k), len(candidate_idx))
        top_scores, top_local_idx = torch.topk(scores, k=k)
        original_idx = [candidate_idx[i] for i in top_local_idx.tolist()]
        out = self.corpus_df.iloc[original_idx].copy()
        out["siamese_score"] = top_scores.numpy()

        results: list[dict[str, Any]] = []
        for row in out.itertuples(index=False):
            article_content = str(getattr(row, "article_content", ""))
            results.append(
                {
                    "doc_name": str(getattr(row, "doc_name", "unknown_doc")),
                    "passage_id": str(getattr(row, "passage_id", "N/A")),
                    "macro_domain": str(getattr(row, "macro_domain", "")),
                    "siamese_score": float(getattr(row, "siamese_score")),
                    "article_content": article_content,
                }
            )
        return results

    def _call_groq_chat(self, system_prompt: str, user_prompt: str, temperature: float = 0.0) -> str:
        if not self.groq_api_key:
            raise RuntimeError("Missing GROQ_API_KEY environment variable.")

        payload = {
            "model": self.groq_model,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "response_format": {"type": "json_object"},
        }
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url="https://api.groq.com/openai/v1/chat/completions",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.groq_api_key}",
                "Accept": "application/json",
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0.0.0 Safari/537.36"
                ),
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                raw = resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"Groq API HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Cannot reach Groq API: {exc}") from exc

        data = json.loads(raw)
        return str(data["choices"][0]["message"]["content"]).strip()

    def _fallback_extractive_answer(
        self,
        query: str,
        results: list[dict[str, Any]],
        *,
        error_detail: str | None = None,
    ) -> dict[str, Any]:
        if not results:
            return {
                "text": "Không tìm thấy ngữ cảnh phù hợp để trả lời chắc chắn từ dữ liệu đã truy xuất.",
                "confidence": 0.0,
                "used_sources": 0,
                "verified": False,
                "fallback_reason": "no_results",
            }

        best = results[0]
        text = best.get("article_content", "").strip()
        if not text:
            return {
                "text": "Không có đủ nội dung nguồn để tạo câu trả lời không suy diễn.",
                "confidence": 0.0,
                "used_sources": 0,
                "verified": False,
                "fallback_reason": "empty_content",
            }
        excerpt = text[:1400]
        if not self.groq_api_key:
            banner = (
                "⚠️ LLM xác minh chưa được bật (chưa cấu hình GROQ_API_KEY) — "
                "chỉ hiển thị đoạn được retriever cho điểm cao nhất; có thể không "
                "trả lời đúng câu hỏi."
            )
            reason = "no_groq_api_key"
        else:
            detail_line = f" Chi tiết: {error_detail}" if error_detail else ""
            banner = (
                "⚠️ Đã cấu hình GROQ_API_KEY nhưng gọi Groq API thất bại — "
                f"chỉ hiển thị đoạn retriever cho điểm cao nhất.{detail_line}\n"
                "Hãy kiểm tra: (1) API key còn hiệu lực, (2) tên model GROQ_MODEL "
                "còn được hỗ trợ, (3) kết nối mạng / firewall."
            )
            reason = "llm_call_failed"
        return {
            "text": f"{banner}\n\n{excerpt}",
            "confidence": 0.2,
            "used_sources": 1,
            "verified": False,
            "fallback_reason": reason,
        }

    def rate_retrieval_with_groq(self, query: str, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not results:
            return results
        if not self.groq_api_key:
            for item in results:
                item["retrieval_rating"] = None
            return results

        passages = []
        for idx, item in enumerate(results, start=1):
            passages.append(
                {
                    "rank": idx,
                    "doc_name": item.get("doc_name"),
                    "passage_id": item.get("passage_id"),
                    "article_content": item.get("article_content", ""),
                }
            )

        system_prompt = (
            "Bạn là chuyên gia đánh giá chất lượng retrieval cho hệ thống hỏi đáp pháp luật. "
            "Chấm điểm mức liên quan của từng đoạn với câu hỏi theo thang 0-10. "
            "Chỉ trả về JSON hợp lệ."
        )
        user_prompt = json.dumps(
            {
                "question": query,
                "passages": passages,
                "output_schema": {
                    "ratings": [
                        {
                            "rank": 1,
                            "retrieval_rating": "number 0-10",
                            "reason": "short reason in Vietnamese",
                        }
                    ]
                },
            },
            ensure_ascii=False,
        )

        try:
            content = self._call_groq_chat(system_prompt=system_prompt, user_prompt=user_prompt, temperature=0.0)
            rating_data = json.loads(content)
            ratings_by_rank: dict[int, float] = {}
            for row in rating_data.get("ratings", []):
                rank = int(row.get("rank", 0))
                raw_score = float(row.get("retrieval_rating", 0.0))
                ratings_by_rank[rank] = max(0.0, min(10.0, raw_score))
            for i, item in enumerate(results, start=1):
                item["retrieval_rating"] = ratings_by_rank.get(i)
        except Exception as exc:  # noqa: BLE001
            print(
                f"[groq] retrieval rating call failed: {exc}",
                file=sys.stderr,
            )
            traceback.print_exc()
            for item in results:
                item["retrieval_rating"] = None

        return results

    def _select_answer_contexts(self, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        rated = [
            r
            for r in results
            if isinstance(r.get("retrieval_rating"), (int, float))
            and float(r["retrieval_rating"]) >= self.answer_min_rating
        ]
        pool = rated if rated else results[: self.answer_top_k]
        pool = pool[: self.answer_top_k]
        return pool

    def answer_grounded_with_groq(
        self,
        query: str,
        results: list[dict[str, Any]],
        topic_prediction: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        query_words = [w for w in query.split() if len(w) > 1]
        if len(query_words) < self.min_query_words:
            return {
                "text": (
                    f"Câu hỏi quá ngắn hoặc mơ hồ (chỉ {len(query_words)} từ). "
                    "Vui lòng đặt câu hỏi cụ thể hơn, ví dụ: "
                    "'Tội trộm cắp tài sản bị phạt bao nhiêu năm tù theo Bộ luật Hình sự?'"
                ),
                "confidence": 0.0,
                "used_sources": 0,
                "verified": False,
                "fallback_reason": "query_too_short",
            }

        topic_conf = float((topic_prediction or {}).get("confidence", 1.0))
        if topic_conf < self.min_topic_confidence:
            return {
                "text": (
                    f"Bộ phân loại chủ đề không đủ tự tin (độ tin cậy {topic_conf:.0%} < "
                    f"{self.min_topic_confidence:.0%}). Câu hỏi có thể nằm ngoài phạm vi dữ liệu, "
                    "hoặc cần thêm ngữ cảnh để phân loại đúng lĩnh vực."
                ),
                "confidence": 0.0,
                "used_sources": 0,
                "verified": False,
                "fallback_reason": "low_topic_confidence",
            }

        top_score = float(results[0].get("siamese_score", 0.0)) if results else 0.0
        if top_score < self.min_retrieval_score:
            return {
                "text": (
                    f"Không có đoạn nào đủ liên quan trong corpus (điểm cao nhất {top_score:.3f} < "
                    f"{self.min_retrieval_score:.2f}). Hãy thử đặt câu hỏi cụ thể, kèm tên luật hoặc điều khoản."
                ),
                "confidence": 0.0,
                "used_sources": 0,
                "verified": False,
                "fallback_reason": "low_retrieval_score",
            }

        if not self.groq_api_key:
            return self._fallback_extractive_answer(query, results)

        pool = self._select_answer_contexts(results)
        if not pool:
            return {
                "text": "Không tìm thấy ngữ cảnh đủ liên quan để trả lời. Vui lòng đặt lại câu hỏi cụ thể hơn.",
                "confidence": 0.0,
                "used_sources": 0,
                "verified": False,
                "fallback_reason": "no_qualified_context",
            }

        contexts = []
        for idx, item in enumerate(pool, start=1):
            content = str(item.get("article_content", ""))[: self.answer_ctx_char_limit]
            contexts.append(
                {
                    "source_id": idx,
                    "doc_name": item.get("doc_name"),
                    "passage_id": item.get("passage_id"),
                    "content": content,
                }
            )

        system_prompt = (
            "Bạn là trợ lý pháp lý nghiêm ngặt về tính trung thực. Quy tắc bắt buộc:\n"
            "1) Chỉ được dùng thông tin XUẤT HIỆN NGUYÊN VĂN trong các đoạn `contexts` được cung cấp. "
            "Cấm suy luận ngoài văn bản, cấm dùng kiến thức nền hay 'thường lệ pháp luật'.\n"
            "2) Mọi khẳng định trong câu trả lời PHẢI có thể truy được về một `source_id` cụ thể. "
            "Nếu không, không được nói.\n"
            "3) Nếu các `contexts` KHÔNG chứa đủ thông tin để trả lời câu hỏi, "
            "BẮT BUỘC trả về `answer = \"Không đủ thông tin trong nguồn để trả lời câu hỏi này.\"`, "
            "`used_source_ids = []`, `confidence <= 0.2`. Không được đoán.\n"
            "4) Khi trả lời, ưu tiên TRÍCH DẪN nguyên văn ngắn từ `contexts` và ghi rõ điều/khoản nếu có. "
            "Sau đó tóm gọn lại bằng tiếng Việt.\n"
            "5) `used_source_ids` chỉ chứa các `source_id` thực sự được dùng trong câu trả lời. Không bịa.\n"
            "6) Chỉ trả về JSON đúng schema, không kèm văn bản thừa."
        )
        user_prompt = json.dumps(
            {
                "question": query,
                "contexts": contexts,
                "output_schema": {
                    "answer": "string (tiếng Việt, có trích dẫn từ contexts)",
                    "confidence": "number 0..1",
                    "used_source_ids": [1, 2],
                },
                "abstain_rule": (
                    "Nếu contexts không đủ thông tin trực tiếp cho câu hỏi, "
                    "trả về câu từ chối ở mục 3) thay vì đoán."
                ),
            },
            ensure_ascii=False,
        )

        def _is_grounded(answer_text: str, used_ids: list[int]) -> bool:
            if not used_ids:
                return False
            for sid in used_ids:
                if not isinstance(sid, int) or not (1 <= sid <= len(contexts)):
                    continue
                ctx_content = contexts[sid - 1]["content"]
                snippet = " ".join(ctx_content.split())
                ans_norm = " ".join(answer_text.split())
                for w in range(8, 4, -1):
                    tokens = snippet.split()
                    for i in range(0, max(0, len(tokens) - w + 1), 1):
                        ngram = " ".join(tokens[i : i + w])
                        if len(ngram) >= 20 and ngram in ans_norm:
                            return True
            return False

        try:
            content = self._call_groq_chat(system_prompt=system_prompt, user_prompt=user_prompt, temperature=0.0)
            answer_data = json.loads(content)
            answer_text = str(answer_data.get("answer", "")).strip()
            confidence = float(answer_data.get("confidence", 0.0))
            raw_ids = answer_data.get("used_source_ids", []) or []
            used_source_ids = [int(x) for x in raw_ids if isinstance(x, (int, float))]
            if not answer_text:
                raise ValueError("Empty answer from Groq")

            abstain_marker = "không đủ thông tin"
            looks_like_abstain = abstain_marker in answer_text.lower()

            if not looks_like_abstain and not _is_grounded(answer_text, used_source_ids):
                return {
                    "text": (
                        "Không đủ thông tin trong nguồn để trả lời câu hỏi này. "
                        "Mô hình đã từ chối do câu trả lời không bám sát ngữ cảnh được truy xuất."
                    ),
                    "confidence": 0.1,
                    "used_sources": 0,
                    "verified": False,
                    "fallback_reason": "llm_not_grounded",
                }

            if looks_like_abstain:
                return {
                    "text": answer_text,
                    "confidence": max(0.0, min(1.0, confidence)),
                    "used_sources": 0,
                    "verified": False,
                    "fallback_reason": "llm_abstained",
                }

            return {
                "text": answer_text,
                "confidence": max(0.0, min(1.0, confidence)),
                "used_sources": len(used_source_ids),
                "verified": True,
            }
        except Exception as exc:  # noqa: BLE001
            print(
                f"[groq] grounded answer call failed: {exc}",
                file=sys.stderr,
            )
            traceback.print_exc()
            return self._fallback_extractive_answer(
                query, pool, error_detail=str(exc)
            )


class DemoHandler(BaseHTTPRequestHandler):
    engine: SiameseBiLSTMDemo | None = None
    top_k: int = 5

    def _json_response(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path != "/":
            self.send_error(404, "Not Found")
            return
        body = HTML_PAGE.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/ask":
            self.send_error(404, "Not Found")
            return
        if self.engine is None:
            self._json_response(500, {"error": "Engine is not initialized"})
            return

        try:
            content_len = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(content_len)
            payload = json.loads(raw.decode("utf-8")) if raw else {}
            question = str(payload.get("question", "")).strip()
            if not question:
                self._json_response(400, {"error": "question is required"})
                return
            topic_prediction = self.engine.classify_topic(question, top_n=3)
            if topic_prediction.get("confidence", 0.0) < 0.25:
                topic_labels = None
            else:
                topic_labels = select_topic_labels(
                    topic_prediction,
                    min_prob=0.12,
                    max_topics=3,
                )
            results = self.engine.search(question, top_k=self.top_k, topic_labels=topic_labels)
            results = self.engine.rate_retrieval_with_groq(question, results)
            answer = self.engine.answer_grounded_with_groq(
                question, results, topic_prediction=topic_prediction
            )
            self._json_response(200, {"topic_prediction": topic_prediction, "answer": answer, "results": results})
        except Exception as exc:  # noqa: BLE001
            self._json_response(500, {"error": str(exc)})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HTML demo for Siamese BiLSTM 1-layer (pipeline ).")
    parser.add_argument("--artifact-dir", type=str, default=str(DEFAULT_ARTIFACT_DIR))
    parser.add_argument("--textcnn-artifact-dir", type=str, default=str(DEFAULT_TEXTCNN_ARTIFACT_DIR))
    parser.add_argument("--corpus-path", type=str, default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--max-docs", type=int, default=3000, help="How many corpus rows to index for demo speed.")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-q-len", type=int, default=64)
    parser.add_argument("--max-d-len", type=int, default=256)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    engine = SiameseBiLSTMDemo(
        artifact_dir=Path(args.artifact_dir),
        textcnn_artifact_dir=Path(args.textcnn_artifact_dir),
        corpus_path=Path(args.corpus_path),
        max_docs=args.max_docs,
        max_d_len=args.max_d_len,
        max_q_len=args.max_q_len,
        device=args.device,
    )
    print("Loading model + corpus index...")
    engine.load()
    print(f"Ready. Indexed {len(engine.corpus_df)} documents.")

    DemoHandler.engine = engine
    DemoHandler.top_k = int(args.top_k) if args.top_k is not None else 3
    server = ThreadingHTTPServer((args.host, args.port), DemoHandler)
    print(f"Serving HTML demo at http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
