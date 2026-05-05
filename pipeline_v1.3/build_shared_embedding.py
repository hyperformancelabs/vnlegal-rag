"""
Build shared vocab + random embedding init from data_ready_v1_3 (pipeline v1.3).

Delegates to model/build_shared_embedding.py; default output:
  pipeline_v1.3/shared_embedding_artifacts/

Run from repo root:
  python pipeline_v1.3/build_shared_embedding.py
  python pipeline_v1.3/build_shared_embedding.py --max-vocab 8000 --embed-dim 200

Then train in `pipeline_v1.3/train-textcnn_v3.ipynb`, `train-siamese-lstm-v3.ipynb`, or `train-siamese-bilstm-online-v3.ipynb` (Kaggle: e.g. `legal-embedding-v1-3`).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_SCRIPT = REPO_ROOT / "model" / "build_shared_embedding.py"


def main() -> None:
    default_qa = REPO_ROOT / "data" / "data_ready_v1_3" / "qa_train.csv"
    default_corpus = REPO_ROOT / "data" / "data_ready_v1_3" / "corpus_train.csv"
    default_out = REPO_ROOT / "pipeline_v1.3" / "shared_embedding_artifacts"

    parser = argparse.ArgumentParser(description="Shared embedding for v1.3 QA + corpus train split.")
    parser.add_argument("--qa-path", type=str, default=str(default_qa))
    parser.add_argument("--corpus-path", type=str, default=str(default_corpus))
    parser.add_argument("--out-dir", type=str, default=str(default_out))
    parser.add_argument("--max-vocab", type=int, default=8000)
    parser.add_argument("--min-freq", type=int, default=1)
    parser.add_argument("--embed-dim", type=int, default=200)
    args = parser.parse_args()

    if not MODEL_SCRIPT.is_file():
        raise FileNotFoundError(f"Missing {MODEL_SCRIPT}")

    cmd = [
        sys.executable,
        str(MODEL_SCRIPT),
        "--qa_path",
        args.qa_path,
        "--corpus_path",
        args.corpus_path,
        "--out_dir",
        args.out_dir,
        "--max_vocab",
        str(args.max_vocab),
        "--min_freq",
        str(args.min_freq),
        "--embed_dim",
        str(args.embed_dim),
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))


if __name__ == "__main__":
    main()
