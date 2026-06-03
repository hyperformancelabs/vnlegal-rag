"""Dense retrieval via the Siamese LSTM bi-encoder.

Documents are encoded once (precomputed embeddings); a query is encoded on the
fly and scored by dot product against the L2-normalized document embeddings.
"""

from __future__ import annotations

import numpy as np
import torch

from ..utils.text import encode_with_mask


@torch.no_grad()
def encode_siamese_texts(
    texts: list[str],
    model: torch.nn.Module,
    stoi: dict[str, int],
    max_len: int,
    device: torch.device,
    batch_size: int = 64,
) -> np.ndarray:
    """Batched Siamese encoding → stacked float32 L2-normalized embeddings."""
    out: list[np.ndarray] = []
    for start in range(0, len(texts), batch_size):
        chunk = texts[start : start + batch_size]
        encoded = [encode_with_mask(t, stoi, max_len) for t in chunk]
        ids = torch.tensor([e[0] for e in encoded], dtype=torch.long, device=device)
        mask = torch.tensor([e[1] for e in encoded], dtype=torch.float, device=device)
        emb = model.encode(ids, mask).detach().cpu().numpy().astype(np.float32)
        out.append(emb)
    if not out:
        return np.zeros((0, 0), dtype=np.float32)
    return np.vstack(out)


def precompute_doc_embeddings(
    corpus_texts: list[str],
    siamese_bundle: dict,
    device: torch.device,
    batch_size: int = 64,
) -> np.ndarray:
    """Precompute Siamese embeddings for the whole corpus (document side)."""
    return encode_siamese_texts(
        corpus_texts,
        siamese_bundle["model"],
        siamese_bundle["siamese_stoi"],
        siamese_bundle["max_d_len"],
        device,
        batch_size=batch_size,
    )


@torch.no_grad()
def encode_query_siamese(
    question: str,
    siamese_bundle: dict,
    device: torch.device,
) -> np.ndarray:
    """Encode a single query into an L2-normalized Siamese embedding (1-D)."""
    ids, mask = encode_with_mask(
        question, siamese_bundle["siamese_stoi"], siamese_bundle["max_q_len"]
    )
    ids_t = torch.tensor([ids], dtype=torch.long, device=device)
    mask_t = torch.tensor([mask], dtype=torch.float, device=device)
    vec = siamese_bundle["model"].encode(ids_t, mask_t).detach().cpu().numpy().astype(np.float32)
    return vec.reshape(-1)


def siamese_scores(
    question: str,
    indices: np.ndarray,
    siamese_bundle: dict,
    doc_embeddings: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """Dot-product scores between the query embedding and ``doc_embeddings[indices]``."""
    q_emb = encode_query_siamese(question, siamese_bundle, device)
    return doc_embeddings[indices] @ q_emb
