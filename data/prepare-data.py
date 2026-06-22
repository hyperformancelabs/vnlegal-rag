"""Prepare corpus and QA data for vnlegal RAG pipeline.

Loads clean (deduped) data from processed/,
merges doc_name into QA, then performs stratified group split by doc_name.
Saves train/val/test splits as TSV files and label maps as JSON.

Usage:
  conda run -n vnlegal python data/prepare-data.py
  conda run -n vnlegal python data/prepare-data.py --out-dir data/data_ready_k2 --merge-bottom-k 2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast

import pandas as pd
from datasets import Dataset, load_from_disk
from sklearn.model_selection import StratifiedGroupKFold

OTHER_LABEL = "other"

DATA_DIR = Path("data/processed")
CORPUS_PATH = DATA_DIR / "vnlegal_corpus"
QA_PATH = DATA_DIR / "vnlegal_qa"
SEED = 42


def save_split(out_dir: Path, name: str, df: pd.DataFrame) -> None:
    """Save a split dataframe as TSV to {out_dir}/{name}.csv."""
    out_path = out_dir / f"{name}.csv"
    df.to_csv(out_path, sep="\t", index=False)
    print(f"Saved: {out_path} | shape={df.shape}")


def _build_label_map(out_dir: Path, train_df: pd.DataFrame) -> dict[str, list[str]]:
    """Build and save label maps (label_list, label2id, id2label) as JSON."""
    label_list = sorted(train_df["macro_domain"].unique())
    label2id = {label: i for i, label in enumerate(label_list)}
    id2label = {i: label for label, i in label2id.items()}

    maps = {"label_list": label_list, "label2id": label2id, "id2label": id2label}
    with open(out_dir / "label_maps.json", "w", encoding="utf-8") as f:
        json.dump(maps, f, ensure_ascii=False, indent=2)
    return maps


def _print_split_stats(
    train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame
) -> None:
    """Print QA split sizes and distribution statistics."""
    print("\nQA split sizes:")
    print("train:", len(train_df))
    print("val  :", len(val_df))
    print("test :", len(test_df))

    print("\nmacro_domain distribution (%):")
    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        print(f"\n{name}")
        print((df["macro_domain"].value_counts(normalize=True) * 100).round(2))

    print("\nquestion_type distribution (%):")
    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        print(f"\n{name}")
        print((df["question_type"].value_counts(normalize=True) * 100).round(2))


def _save_corpus_splits(
    out_dir: Path,
    corpus_df: pd.DataFrame,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    """Save corpus subsets filtered by doc_name of each QA split."""
    train_docs = list(set(train_df["doc_name"]))
    val_docs = list(set(val_df["doc_name"]))
    test_docs = list(set(test_df["doc_name"]))

    corpus_train = corpus_df.loc[
        corpus_df["doc_name"].isin(train_docs)
    ].reset_index(drop=True)
    corpus_val = corpus_df.loc[
        corpus_df["doc_name"].isin(val_docs)
    ].reset_index(drop=True)
    corpus_test = corpus_df.loc[
        corpus_df["doc_name"].isin(test_docs)
    ].reset_index(drop=True)

    save_split(out_dir, "corpus_train", corpus_train)
    save_split(out_dir, "corpus_val", corpus_val)
    save_split(out_dir, "corpus_test", corpus_test)


def _build_merge_map(
    train_domains: pd.Series, merge_k: int
) -> tuple[dict[str, str], list[str]]:
    """Build mapping that merges the k rarest macro_domain values into OTHER_LABEL."""
    counts = train_domains.value_counts()
    order = counts.reset_index()
    order.columns = ["domain", "cnt"]
    order = order.sort_values(["cnt", "domain"], ascending=[True, True])
    n_unique = len(order)
    if merge_k >= n_unique:
        raise ValueError(
            f"merge_k ({merge_k}) must be < number of unique macro_domain on train ({n_unique})"
        )

    to_merge = order.head(merge_k)["domain"].astype(str).tolist()
    mapping: dict[str, str] = {str(dom): str(dom) for dom in counts.index}
    for d in to_merge:
        mapping[d] = OTHER_LABEL
    return mapping, to_merge


def _apply_merge(df: pd.DataFrame, merge_map: dict[str, str]) -> pd.DataFrame:
    """Apply domain merge mapping to a dataframe."""
    out = df.copy()
    out["macro_domain"] = out["macro_domain"].astype(str).map(
        lambda d: merge_map.get(str(d), str(d))
    )
    return out


def main() -> None:
    """Load clean data and perform stratified group split."""
    parser = argparse.ArgumentParser(
        description="Prepare QA + corpus splits for vnlegal RAG."
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="data/data_ready",
        help="Output directory (default: data/data_ready).",
    )
    parser.add_argument(
        "--merge-bottom-k",
        type=int,
        default=0,
        help="Merge k rarest macro_domain classes into 'other' (0 = no merge).",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    corpus_ds = load_from_disk(str(CORPUS_PATH))
    qa_ds = load_from_disk(str(QA_PATH))
    assert isinstance(corpus_ds, Dataset)
    assert isinstance(qa_ds, Dataset)

    corpus_df = cast(pd.DataFrame, corpus_ds.to_pandas()).rename(
        columns={"article_len_word": "article_token"}
    )
    qa_df = cast(pd.DataFrame, qa_ds.to_pandas()).rename(
        columns={"question_len_word": "question_token", "answer_len_word": "answer_token"}
    )

    # Merge doc_name from corpus into QA
    qa_df = qa_df.merge(
        corpus_df[["passage_id", "doc_name"]],
        on="passage_id",
        how="inner",
        validate="many_to_one",
    )

    # Outer split: train+val vs test (~10%)
    sgkf_outer = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=SEED)
    train_val_idx, test_idx = next(
        sgkf_outer.split(
            qa_df, y=qa_df["macro_domain"], groups=qa_df["doc_name"]
        )
    )
    train_val_df = qa_df.iloc[train_val_idx].reset_index(drop=True)
    test_df = qa_df.iloc[test_idx].reset_index(drop=True)

    # Inner split: train vs val (~10%)
    sgkf_inner = StratifiedGroupKFold(n_splits=9, shuffle=True, random_state=SEED)
    train_idx, val_idx = next(
        sgkf_inner.split(
            train_val_df,
            y=train_val_df["macro_domain"],
            groups=train_val_df["doc_name"],
        )
    )
    train_df = train_val_df.iloc[train_idx].reset_index(drop=True)
    val_df = train_val_df.iloc[val_idx].reset_index(drop=True)

    # Optional merge: combine k rarest macro_domain into "other"
    if args.merge_bottom_k > 0:
        merge_map, merged_from = _build_merge_map(
            train_df["macro_domain"], args.merge_bottom_k
        )
        print(f"\nMerging {args.merge_bottom_k} rarest classes into '{OTHER_LABEL}':")
        raw_counts = train_df["macro_domain"].value_counts()
        for d in merged_from:
            print(f"  {raw_counts[d]:6d}  {d} -> {OTHER_LABEL}")

        train_df = _apply_merge(train_df, merge_map)
        val_df = _apply_merge(val_df, merge_map)
        test_df = _apply_merge(test_df, merge_map)

        with open(out_dir / "label_merge_map.json", "w", encoding="utf-8") as f:
            json.dump(merge_map, f, ensure_ascii=False, indent=2)

    # Pre-build combined text field for retrieval tasks.
    for df in (train_df, val_df, test_df):
        df["text"] = df["question"].fillna("") + " " + df["answer"].fillna("")

    # Save everything
    _build_label_map(out_dir, train_df)
    for split_name, df in [
        ("qa_train", train_df),
        ("qa_val", val_df),
        ("qa_test", test_df),
    ]:
        save_split(out_dir, split_name, df)
    save_split(out_dir, "corpus_full", corpus_df)
    _save_corpus_splits(out_dir, corpus_df, train_df, val_df, test_df)

    # Sanity check
    _print_split_stats(train_df, val_df, test_df)


if __name__ == "__main__":
    main()
