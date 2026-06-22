"""Canonical regex-based tokenizer for the VNLegal RAG pipeline.

Steps applied in order:
1. Vietnamese syllable diacritic normalisation (VinAI 2021 dict)
2. Lowercase + strip
3. Unicode ``\\w+`` regex split

No pyvi / underthesea dependency.
"""

from __future__ import annotations

import re

from .constants import PAD_TOKEN, UNK_TOKEN

# ── Vietnamese syllable normalisation (VinAI Research, 2021) ────────────────
#
# In legacy encodings and older documents the tone mark is sometimes placed on
# the *second* vowel of a diphthong (e.g. "hoà" instead of "hoà").  Modern
# Vietnamese orthography places the tone on the first vowel ("hoà").  The
# word2vec vocabulary uses the modern form; this dict normalises both cases so
# tokens consistently match pre-trained vectors.
#
# Derived from VinAI Research's tokenizer (Apache 2.0):
# https://github.com/VinAIResearch/PhoBERT

_VI_SYLLABLE_NORMALIZE: dict[str, str] = {
    # oa
    "òa": "oà", "Òa": "Oà", "ÒA": "OÀ",
    "óa": "oá", "Óa": "Oá", "ÓA": "OÁ",
    "ỏa": "oả", "Ỏa": "Oả", "ỎA": "OẢ",
    "õa": "oã", "Õa": "Oã", "ÕA": "OÃ",
    "ọa": "oạ", "Ọa": "Oạ", "ỌA": "OẠ",
    # oe
    "òe": "oè", "Òe": "Oè", "ÒE": "OÈ",
    "óe": "oé", "Óe": "Oé", "ÓE": "OÉ",
    "ỏe": "oẻ", "Ỏe": "Oẻ", "ỎE": "OẺ",
    "õe": "oẽ", "Õe": "Oẽ", "ÕE": "OẼ",
    "ọe": "oẹ", "Ọe": "Oẹ", "ỌE": "OẸ",
    # uy
    "ùy": "uỳ", "Ùy": "Uỳ", "ÙY": "UỲ",
    "úy": "uý", "Úy": "Uý", "ÚY": "UÝ",
    "ủy": "uỷ", "Ủy": "Uỷ", "ỦY": "UỶ",
    "ũy": "uỹ", "Ũy": "Uỹ", "ŨY": "UỸ",
    "ụy": "uỵ", "Ụy": "Uỵ", "ỤY": "UỴ",
}


def normalize_vietnamese_syllables(text: str) -> str:
    """Replace legacy diacritic placement with modern standard form.

    >>> normalize_vietnamese_syllables("hòa bình")
    'hoà bình'
    >>> normalize_vietnamese_syllables("thùy")
    'thuỳ'
    """
    for old, new in _VI_SYLLABLE_NORMALIZE.items():
        text = text.replace(old, new)
    return text


def simple_tokenize(text: str) -> list[str]:
    """Normalise Vietnamese syllables → lowercase → Unicode ``\\w+`` regex split.

    >>> simple_tokenize("Luật Dân sự 2015, điều 12!")
    ['luật', 'dân', 'sự', '2015', 'điều', '12']
    >>> simple_tokenize("hòa bình")
    ['hoà', 'bình']
    """
    text = normalize_vietnamese_syllables(str(text).lower().strip())
    return re.findall(r"\w+", text, flags=re.UNICODE)


def encode_text(
    text: str,
    stoi: dict[str, int],
    max_len: int,
    *,
    pad_token: str = PAD_TOKEN,
    unk_token: str = UNK_TOKEN,
) -> list[int]:
    """Tokenize → map ``stoi`` → pad/truncate to exactly ``max_len``."""
    unk_id = stoi.get(unk_token, stoi.get("UNK", 1))
    pad_id = stoi.get(pad_token, stoi.get("PAD", 0))
    ids = [stoi.get(t, unk_id) for t in simple_tokenize(text)[:max_len]]
    ids += [pad_id] * (max_len - len(ids))
    return ids


def encode_with_mask(
    text: str,
    stoi: dict[str, int],
    max_len: int,
    *,
    pad_token: str = PAD_TOKEN,
    unk_token: str = UNK_TOKEN,
) -> tuple[list[int], list[float]]:
    """Like :func:`encode_text` but also returns a float mask (1.0=token, 0.0=pad).

    Used for Siamese LSTM masked mean-pooling.
    """
    unk_id = stoi.get(unk_token, stoi.get("UNK", 1))
    pad_id = stoi.get(pad_token, stoi.get("PAD", 0))
    toks = simple_tokenize(text)[:max_len]
    ids = [stoi.get(t, unk_id) for t in toks]
    length = len(ids)
    ids += [pad_id] * (max_len - length)
    mask = [1.0] * length + [0.0] * (max_len - length)
    return ids, mask
