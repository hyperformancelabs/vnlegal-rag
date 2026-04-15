# prepare_data.py
import json
from pathlib import Path

import pandas as pd
from datasets import load_from_disk
from sklearn.model_selection import StratifiedGroupKFold


DATA_DIR = Path("data/processed")
DATA_READY = Path("data/data_ready")
DATA_READY.mkdir(parents=True, exist_ok=True)

CORPUS_PATH = DATA_DIR / "vnlegal_corpus"
QA_PATH = DATA_DIR / "vnlegal_qa"
SEED = 42


def save_split(name: str, df: pd.DataFrame):
    out_path = DATA_READY / f"{name}.csv"
    df.to_csv(out_path, sep='\t', index=False)
    print(f"Saved: {out_path} | shape={df.shape}")


def main():
    corpus_ds = load_from_disk(str(CORPUS_PATH))
    qa_ds = load_from_disk(str(QA_PATH))

    corpus_df = corpus_ds.to_pandas().copy()
    qa_df = qa_ds.to_pandas().copy()

    # clean
    corpus_df = corpus_df.drop_duplicates(subset=["doc_name", "article_content"]).reset_index(drop=True)
    qa_df = qa_df.drop_duplicates(subset=["passage_id", "question"]).reset_index(drop=True)
    qa_df = qa_df.drop_duplicates(subset=["question", "answer"]).reset_index(drop=True)

    # join doc_name from corpus
    qa_df = qa_df.merge(
        corpus_df[["passage_id", "doc_name"]],
        on="passage_id",
        how="inner",
        validate="many_to_one",
    )

    # stratify: domain + qtype
    qa_df["strat_label"] = qa_df["macro_domain"] + "||" + qa_df["question_type"]

    # outer split: test ~10%
    sgkf_outer = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=SEED)
    train_val_idx, test_idx = next(
        sgkf_outer.split(
            qa_df,
            y=qa_df["strat_label"],
            groups=qa_df["doc_name"],
        )
    )

    train_val_df = qa_df.iloc[train_val_idx].reset_index(drop=True)
    test_df = qa_df.iloc[test_idx].reset_index(drop=True)

    # inner split: val ~10%
    sgkf_inner = StratifiedGroupKFold(n_splits=9, shuffle=True, random_state=SEED)
    train_idx, val_idx = next(
        sgkf_inner.split(
            train_val_df,
            y=train_val_df["strat_label"],
            groups=train_val_df["doc_name"],
        )
    )

    train_df = train_val_df.iloc[train_idx].reset_index(drop=True)
    val_df = train_val_df.iloc[val_idx].reset_index(drop=True)

    # build label map
    label_list = sorted(train_df["macro_domain"].unique().tolist())
    label2id = {label: i for i, label in enumerate(label_list)}
    id2label = {i: label for label, i in label2id.items()}

    with open(DATA_READY / "label_maps.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "label_list": label_list,
                "label2id": label2id,
                "id2label": id2label,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    # save QA splits
    save_split("qa_train", train_df)
    save_split("qa_val", val_df)
    save_split("qa_test", test_df)

    # save full corpus
    save_split("corpus_full", corpus_df)

    # save corpus theo doc_name của từng split
    train_docs = set(train_df["doc_name"])
    val_docs = set(val_df["doc_name"])
    test_docs = set(test_df["doc_name"])

    corpus_train = corpus_df[corpus_df["doc_name"].isin(train_docs)].reset_index(drop=True)
    corpus_val = corpus_df[corpus_df["doc_name"].isin(val_docs)].reset_index(drop=True)
    corpus_test = corpus_df[corpus_df["doc_name"].isin(test_docs)].reset_index(drop=True)

    save_split("corpus_train", corpus_train)
    save_split("corpus_val", corpus_val)
    save_split("corpus_test", corpus_test)

    # sanity check
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


if __name__ == "__main__":
    main()