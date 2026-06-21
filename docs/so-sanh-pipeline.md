# So sanh pipeline v1 vs v2

## Tong quan khac biet

- `v1`: pipeline co dinh `TextCNN topic filter -> Siamese rerank`, dung `topic_topk=2`.
- `v2`: giu nhanh Siamese va bo sung nhanh `Hybrid (TF-IDF + Siamese)`, co sweep `topic_topk` va `alpha`.
- Diem chung: cung artifact model, tokenizer/encode va bo metric retrieval.

## Cau hinh danh gia

### Cau hinh chung

- `RETRIEVE_K = 10` (danh gia top-10 passage tra ve).
- Topic router: `TextCNN` du doan `macro_domain`; candidate pool loc theo `topic_topk`.
- Neural scorer: cosine giua embedding query-doc tu `SiameseLSTM`.
- Tokenizer/encode giu nguyen giua v1 va v2.

### Khong gian cau hinh theo nhom pipeline

| Nhom pipeline | Candidate pool | Ham tinh diem | Khong gian cau hinh |
|---|---|---|---|
| `TF-IDF full corpus` | Toan bo corpus | TF-IDF cosine | Co dinh (giong nhau o v1/v2) |
| `TF-IDF + topic filter` | Top-`topic_topk` topic tu TextCNN | TF-IDF cosine | v1: `topic_topk=2`; v2: `topic_topk in {3,4}` |
| `Siamese retrieval` | Top-`topic_topk` topic tu TextCNN | Siamese cosine | v1: `topic_topk=2`; v2: `topic_topk in {3,4}` |
| `Hybrid (TF-IDF + Siamese)` | Top-`topic_topk` topic tu TextCNN | `alpha * tfidf_norm + (1-alpha) * siamese_norm` | v2: `topic_topk in {3,4}`, `alpha in {0,0.25,0.5,0.75,1}`; auto-search `alpha=0.00..1.00` step `0.05` |

## Ket qua so sanh chinh

| Pipeline/Baseline | Cau hinh | MRR | Recall@1 | Recall@5 | Recall@10 | nDCG@10 |
|---|---|---:|---:|---:|---:|---:|
| TF-IDF full corpus | v1/v2 (chung) | 0.6768 | 0.5670 | 0.8251 | 0.8887 | 0.7282 |
| TF-IDF + topic filter | v1, top-2 | 0.5827 | 0.4865 | 0.7121 | 0.7670 | 0.6274 |
| TF-IDF + topic filter | v2, top-3 | 0.6127 | 0.5115 | 0.7499 | 0.8058 | 0.6597 |
| TF-IDF + topic filter | v2, top-4 | 0.6307 | 0.5272 | 0.7706 | 0.8288 | 0.6788 |
| Siamese retrieval | v1, top-2 | 0.4165 | 0.3213 | 0.5453 | 0.6142 | 0.4641 |
| Siamese retrieval | v2, top-3 | 0.4289 | 0.3320 | 0.5603 | 0.6326 | 0.4779 |
| Siamese retrieval | v2, top-4 | 0.4303 | 0.3333 | 0.5607 | 0.6352 | 0.4796 |
| Hybrid (TF-IDF + Siamese) | v2, top-4, alpha=0.5 | 0.6766 | 0.5777 | 0.8081 | 0.8623 | 0.7218 |

## Nhan xet nhanh

- `TF-IDF full corpus` van la moc lexical manh nhat theo MRR (`0.6768`), khong doi giua v1/v2.
- `TF-IDF + topic filter` tang deu khi tang `topic_topk` o v2 (`MRR: 0.5827 -> 0.6307` so voi v1).
- Nhanh `Siamese retrieval` trong v2 tot hon v1 nhung khoang cach khong lon (`MRR +0.0138`, `Recall@10 +0.0210`).
- `Hybrid v2` cai thien ro so voi Siamese thuần, va gan bang `TF-IDF full` theo MRR (`0.6766` vs `0.6768`).

## Khuyen nghi su dung

- Chon `v1` neu can baseline gon nhe, setup nhanh, de debug.
- Chon `v2` neu muc tieu la tuning co he thong va toi uu metric retrieval cuoi.

## Luu y tinh cong bang khi so sanh

- v1 va v2 dung default `topic_topk` khac nhau (`2` vs `3/4`), nen chua phai so sanh apples-to-apples.
- Neu can so sanh cong bang tuyet doi: chay lai cung test set, cung seed va cung `topic_topk`.

