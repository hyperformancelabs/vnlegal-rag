"""
tokenizer.py — Canonical tokenizer cho toàn bộ vnlegal-rag pipeline.

Dùng chung cho:
  - shared_embedding (build vocab)
  - train-textcnn        (encode_text, build vocab)
  - train-siamese-lstm-* (encode_text, hard negative mining)
  - test-pipeline        (TextCNN encode + Siamese encode)

Quy tắc:
  • lowercase + strip trước
  • regex \\w+ (bắt ký tự Unicode bao gồm tiếng Việt có dấu)
  • KHÔNG phụ thuộc pyvi / underthesea → chạy ổn trên mọi môi trường

Cách import trong pipeline  (notebook / Colab / Kaggle):
    # Thêm experiments vào sys.path, rồi:
    from tokenizer_bootstrap import simple_tokenize, encode_text, build_vocab

Bản sao Colab: experiments/tokenizer.py (giữ đồng bộ với file này).

Cách import trực tiếp (script trong thư mục model/):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from tokenizer import simple_tokenize, encode_text, build_vocab
"""

from __future__ import annotations

import re
import json
from pathlib import Path
from collections import Counter
from typing import Dict, List, Optional, Tuple

__all__ = [
    "simple_tokenize",
    "encode_text",
    "build_vocab",
    "load_vocab",
    "save_vocab",
    "TOKENIZER_BACKEND",
]

# ─── Hằng số ────────────────────────────────────────────────────────────────
TOKENIZER_BACKEND: str = "simple_tokenize"   # canonical, không phụ thuộc gói ngoài
PAD_TOKEN: str = "<PAD>"
UNK_TOKEN: str = "<UNK>"

# ─── Core tokenizer ─────────────────────────────────────────────────────────

def simple_tokenize(text: str) -> List[str]:
    """
    Tokenizer chuẩn cho toàn bộ pipeline.

    Quy trình:
      1. Ép kiểu str, lowercase, strip whitespace thừa
      2. Regex \\w+ trích xuất từ (bao gồm ký tự Unicode / tiếng Việt có dấu)

    Ví dụ:
      >>> simple_tokenize("Luật Dân sự 2015, điều 12!")
      ['luật', 'dân', 'sự', '2015', 'điều', '12']
    """
    return re.findall(r"\w+", str(text).lower().strip(), flags=re.UNICODE)


# ─── Encoding ────────────────────────────────────────────────────────────────

def encode_text(
    text: str,
    stoi: Dict[str, int],
    max_len: int,
    pad_token: str = PAD_TOKEN,
    unk_token: str = UNK_TOKEN,
) -> List[int]:
    """
    Tokenize → map stoi → pad / truncate đến max_len.
    Trả về list[int] độ dài đúng bằng max_len.
    """
    unk_id = stoi.get(unk_token, stoi.get("UNK", 1))
    pad_id = stoi.get(pad_token, stoi.get("PAD", 0))
    ids = [stoi.get(t, unk_id) for t in simple_tokenize(text)[:max_len]]
    ids += [pad_id] * (max_len - len(ids))
    return ids


def encode_with_mask(
    text: str,
    stoi: Dict[str, int],
    max_len: int,
    pad_token: str = PAD_TOKEN,
    unk_token: str = UNK_TOKEN,
) -> Tuple[List[int], List[float]]:
    """
    Giống encode_text nhưng trả thêm float mask (1.0 = real token, 0.0 = pad).
    Dùng cho Siamese LSTM (masked mean-pooling).
    """
    unk_id = stoi.get(unk_token, stoi.get("UNK", 1))
    pad_id = stoi.get(pad_token, stoi.get("PAD", 0))
    toks = simple_tokenize(text)[:max_len]
    ids = [stoi.get(t, unk_id) for t in toks]
    length = len(ids)
    ids += [pad_id] * (max_len - length)
    mask = [1.0] * length + [0.0] * (max_len - length)
    return ids, mask


# ─── Vocab utilities ─────────────────────────────────────────────────────────

def build_vocab(
    texts: List[str],
    max_vocab: int = 100_000,
    min_freq: int = 1,
    pad_token: str = PAD_TOKEN,
    unk_token: str = UNK_TOKEN,
) -> Dict[str, int]:
    """
    Build stoi từ danh sách văn bản.

    Thứ tự: [PAD, UNK, token_thường_xuyên_nhất, ...]
    Trả về dict stoi (token → index).
    """
    counter: Counter = Counter()
    for text in texts:
        counter.update(simple_tokenize(text))

    stoi: Dict[str, int] = {pad_token: 0, unk_token: 1}
    for token, freq in counter.most_common(max_vocab - 2):
        if freq < min_freq:
            break
        if token not in stoi:
            stoi[token] = len(stoi)
    return stoi


def save_vocab(
    stoi: Dict[str, int],
    path: Path,
    build_metadata: Optional[dict] = None,
) -> None:
    """
    Lưu vocab ra JSON với format chuẩn:
      {
        "stoi": {token: idx, ...},
        "itos": {"0": token, "1": token, ...},
        "tokenizer_backend": "simple_tokenize",
        "build_metadata": {...}   # tuỳ chọn
      }
    """
    itos = {str(idx): token for token, idx in stoi.items()}
    payload = {
        "stoi": stoi,
        "itos": itos,
        "tokenizer_backend": TOKENIZER_BACKEND,
    }
    if build_metadata:
        payload["build_metadata"] = build_metadata
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_vocab(path: Path) -> Dict[str, int]:
    """
    Đọc tokenizer_vocab.json và trả về stoi dict.
    Cảnh báo nếu file được build bằng tokenizer khác.
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    backend = data.get("tokenizer_backend", "unknown")
    if backend not in ("simple_tokenize", "regex_word", "unknown"):
        import warnings
        warnings.warn(
            f"Vocab at {path} was built with '{backend}' tokenizer, "
            f"but current pipeline uses 'simple_tokenize'. "
            f"Consider rebuilding the vocab for consistent results.",
            UserWarning,
            stacklevel=2,
        )
    return data["stoi"]


# ─── Notebook snippet (dán vào đầu mỗi notebook) ────────────────────────────
_NOTEBOOK_SNIPPET = '''
# ── Canonical tokenizer — dùng chung cho tất cả models ─────────────────────
import re

def simple_tokenize(text: str) -> list:
    """Tokenizer chuẩn: regex \\w+, lowercase. Không cần pyvi/underthesea."""
    return re.findall(r"\\w+", str(text).lower().strip(), flags=re.UNICODE)
# ────────────────────────────────────────────────────────────────────────────
'''


if __name__ == "__main__":
    # Smoke test
    samples = [
        "Luật Dân sự 2015, điều 12 khoản 3!",
        "Quy định về bảo hiểm xã hội...",
        "   THÔNG TƯ số 01/2024/TT-BCA   ",
        "",
        123,
    ]
    print("=== simple_tokenize smoke test ===")
    for s in samples:
        print(f"  {repr(s)!s:45s} -> {simple_tokenize(s)}")

    # encode_text test
    stoi_test = {PAD_TOKEN: 0, UNK_TOKEN: 1, "luật": 2, "dân": 3, "sự": 4}
    enc = encode_text("Luật Dân sự 2015", stoi_test, max_len=6)
    print(f"\nencode_text test: {enc}  (expected [2,3,4,1,0,0])")
    assert enc == [2, 3, 4, 1, 0, 0], f"Got {enc}"

    print("\n✓ All checks passed.")
