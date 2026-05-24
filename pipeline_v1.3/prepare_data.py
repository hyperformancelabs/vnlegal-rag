"""
Pipeline v1.3: cùng split StratifiedGroupKFold như pipeline_v1.1,
nhưng gộp **k lớp macro_domain có ít mẫu trên train nhất** (mặc định k=3)
vào một nhãn "other" — không dùng ngưỡng min_train_samples.

Chạy từ thư mục gốc repo:
  python pipeline_v1.3/prepare_data.py
  python pipeline_v1.3/prepare_data.py --merge-bottom-k 3 --out-dir data/data_ready_v1_3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from datasets import load_from_disk
from sklearn.model_selection import StratifiedGroupKFold

OTHER_LABEL = "other"
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "processed"
CORPUS_PATH = DATA_DIR / "vnlegal_corpus"
QA_PATH = DATA_DIR / "vnlegal_qa"
SEED = 42


def save_split(out_root: Path, name: str, df: pd.DataFrame) -> None:
    out_path = out_root / f"{name}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, sep="\t", index=False)
    print(f"Saved: {out_path} | shape={df.shape}")


def build_bottom_k_merge_map(train_domains: pd.Series, k: int) -> tuple[dict[str, str], list[str]]:
    """Map the k rarest macro_domain values on train to OTHER_LABEL; others map to themselves."""
    k = int(k)
    if k < 1:
        raise ValueError("merge_bottom_k must be >= 1")

    counts = train_domains.value_counts()
    order = counts.reset_index()
    order.columns = ["domain", "cnt"]
    order = order.sort_values(["cnt", "domain"], ascending=[True, True])
    n_unique = len(order)
    if k >= n_unique:
        raise ValueError(
            f"merge_bottom_k ({k}) must be < number of unique macro_domain on train ({n_unique})"
        )

    to_merge = order.head(k)["domain"].astype(str).tolist()
    mapping: dict[str, str] = {str(dom): str(dom) for dom in counts.index}
    for d in to_merge:
        mapping[d] = OTHER_LABEL
    return mapping, to_merge


def apply_domain_merge(df: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    out = df.copy()
    out["macro_domain"] = out["macro_domain"].astype(str).map(lambda d: mapping.get(d, d))
    out["strat_label"] = out["macro_domain"] + "||" + out["question_type"].astype(str)
    return out


def ordered_label_list(labels: set[str]) -> list[str]:
    rest = sorted(l for l in labels if l != OTHER_LABEL)
    if OTHER_LABEL in labels:
        return rest + [OTHER_LABEL]
    return rest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare data_ready v1.3: merge k rarest train macro_domain into other."
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="data/data_ready_v1_3",
        help="Output dir for CSV + label_maps (relative to repo root unless absolute).",
    )
    parser.add_argument(
        "--merge-bottom-k",
        type=int,
        default=3,
        help="Number of least frequent macro_domain classes on train to merge into 'other'.",
    )
    args = parser.parse_args()
    data_ready = Path(args.out_dir)
    if not data_ready.is_absolute():
        data_ready = REPO_ROOT / data_ready
    data_ready.mkdir(parents=True, exist_ok=True)
    merge_k = int(args.merge_bottom_k)

    corpus_ds = load_from_disk(str(CORPUS_PATH))
    qa_ds = load_from_disk(str(QA_PATH))

    corpus_df = corpus_ds.to_pandas().copy()
    qa_df = qa_ds.to_pandas().copy()

    corpus_df = corpus_df.drop_duplicates(subset=["doc_name", "article_content"]).reset_index(drop=True)
    qa_df = qa_df.drop_duplicates(subset=["passage_id", "question"]).reset_index(drop=True)
    qa_df = qa_df.drop_duplicates(subset=["question", "answer"]).reset_index(drop=True)

    qa_df = qa_df.merge(
        corpus_df[["passage_id", "doc_name"]],
        on="passage_id",
        how="inner",
        validate="many_to_one",
    )

    qa_df["strat_label"] = qa_df["macro_domain"].astype(str) + "||" + qa_df["question_type"].astype(str)

    sgkf_outer = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=SEED)
    train_val_idx, test_idx = next(
        sgkf_outer.split(qa_df, y=qa_df["strat_label"], groups=qa_df["doc_name"])
    )
    train_val_df = qa_df.iloc[train_val_idx].reset_index(drop=True)
    test_df = qa_df.iloc[test_idx].reset_index(drop=True)

    sgkf_inner = StratifiedGroupKFold(n_splits=9, shuffle=True, random_state=SEED)
    train_idx, val_idx = next(
        sgkf_inner.split(train_val_df, y=train_val_df["strat_label"], groups=train_val_df["doc_name"])
    )
    train_df = train_val_df.iloc[train_idx].reset_index(drop=True)
    val_df = train_val_df.iloc[val_idx].reset_index(drop=True)

    raw_counts = train_df["macro_domain"].value_counts()
    print("\nTrain macro_domain counts (before merge):")
    for dom, cnt in raw_counts.sort_values(ascending=False).items():
        print(f"  {int(cnt):6d}  {dom}")

    merge_map, merged_from = build_bottom_k_merge_map(train_df["macro_domain"], merge_k)
    print(f"\nMerging {merge_k} rarest classes into '{OTHER_LABEL}':")
    for d in merged_from:
        print(f"  {raw_counts[d]:6d}  {d} -> {OTHER_LABEL}")

    train_df = apply_domain_merge(train_df, merge_map)
    val_df = apply_domain_merge(val_df, merge_map)
    test_df = apply_domain_merge(test_df, merge_map)

    label_list = ordered_label_list(set(train_df["macro_domain"].unique().tolist()))
    label2id = {label: i for i, label in enumerate(label_list)}
    id2label = {str(i): label for label, i in label2id.items()}

    with open(data_ready / "label_maps.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "label_list": label_list,
                "label2id": label2id,
                "id2label": id2label,
                "other_merge": {
                    "strategy": "bottom_k_train_counts",
                    "merge_bottom_k": merge_k,
                    "other_label": OTHER_LABEL,
                    "domains_merged_to_other": merged_from,
                    "train_counts_before_merge": {str(k): int(v) for k, v in raw_counts.items()},
                },
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    with open(data_ready / "label_merge_map.json", "w", encoding="utf-8") as f:
        json.dump(merge_map, f, ensure_ascii=False, indent=2)

    save_split(data_ready, "qa_train", train_df)
    save_split(data_ready, "qa_val", val_df)
    save_split(data_ready, "qa_test", test_df)

    save_split(data_ready, "corpus_full", corpus_df)

    def remap_corpus(df: pd.DataFrame) -> pd.DataFrame:
        c = df.copy()
        c["macro_domain"] = c["macro_domain"].astype(str).map(lambda d: merge_map.get(d, d))
        return c

    train_docs = set(train_df["doc_name"])
    val_docs = set(val_df["doc_name"])
    test_docs = set(test_df["doc_name"])

    corpus_train = remap_corpus(corpus_df[corpus_df["doc_name"].isin(train_docs)].reset_index(drop=True))
    corpus_val = remap_corpus(corpus_df[corpus_df["doc_name"].isin(val_docs)].reset_index(drop=True))
    corpus_test = remap_corpus(corpus_df[corpus_df["doc_name"].isin(test_docs)].reset_index(drop=True))

    save_split(data_ready, "corpus_train", corpus_train)
    save_split(data_ready, "corpus_val", corpus_val)
    save_split(data_ready, "corpus_test", corpus_test)

    corpus_ready_full = pd.concat(
        [corpus_train, corpus_val, corpus_test], ignore_index=True
    ).drop_duplicates(subset=["passage_id"]).reset_index(drop=True)
    assert corpus_ready_full["passage_id"].is_unique, "Duplicate passage_id in corpus_ready_full"
    save_split(data_ready, "corpus_ready_full", corpus_ready_full)

    print("\nQA split sizes:")
    print("train:", len(train_df))
    print("val  :", len(val_df))
    print("test :", len(test_df))

    print("\nmacro_domain distribution (%) after merge:")
    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        print(f"\n{name}")
        print((df["macro_domain"].value_counts(normalize=True) * 100).round(2))

    print(f"\nLabel list ({len(label_list)}): {label_list}")


if __name__ == "__main__":
    main()
