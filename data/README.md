# Extract dataset

## Requirement: 7-Zip

- **macOS:** `brew install p7zip`
- **Linux:** `sudo apt install p7zip-full`
- **Windows:** https://www.7-zip.org/

## macOS / Linux

```bash
cd data/word2vec
cat word2vec_part_* > word2vec_vi_syllables_300dims.zip
7z x word2vec_vi_syllables_300dims.zip
```

## Windows (cmd)

```cmd
cd data\word2vec
copy /b word2vec_part_aa + word2vec_part_ab + word2vec_part_ac + word2vec_part_ad + word2vec_part_ae + word2vec_part_af + word2vec_part_ag + word2vec_part_ah + word2vec_part_ai + word2vec_part_aj + word2vec_part_ak + word2vec_part_al + word2vec_part_am + word2vec_part_an + word2vec_part_ao + word2vec_part_ap + word2vec_part_aq + word2vec_part_ar + word2vec_part_as + word2vec_part_at + word2vec_part_au + word2vec_part_av + word2vec_part_aw + word2vec_part_ax + word2vec_part_ay + word2vec_part_az + word2vec_part_ba + word2vec_part_bb word2vec_vi_syllables_300dims.zip
7z x word2vec_vi_syllables_300dims.zip
```

## After extraction

The file `word2vec_vi_syllables_300dims.txt` (3.4 GB) is used by `src.tokenizer.build_embedding_matrix`:

```python
from src.tokenizer import build_vocab, build_embedding_matrix

stoi = build_vocab(texts)
matrix, hits = build_embedding_matrix(
    stoi, embed_dim=300,
    w2v_path="data/word2vec/word2vec_vi_syllables_300dims.txt",
)
```
