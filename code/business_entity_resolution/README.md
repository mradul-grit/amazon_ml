# Business Entity Resolution: end-to-end pipeline (ML Challenge 2026), configuration exp_023

exp_023 = exp_022_wide (wide bidirectional blocking + LightGBM) plus a one-to-one **exclusivity** step at decision time. This regenerates `matching_results.tsv` and `candidate_pairs.tsv` from the official training and test files. It uses only the provided challenge data. There are no external data, APIs or pretrained models.

## Environment
- Python 3.14.3 (tested on Windows 11). The code uses `pathlib` only.
- `pip install -r requirements.txt` installs numpy, pandas, scipy, scikit-learn, lightgbm and rapidfuzz. All are BSD-3 or MIT licensed; there are no model weights.
- Hardware used: 12 logical cores and 15.7 GB RAM. Peak RSS is about 8 GB, with about 80 GB of scratch disk for caches.

## Setup
1. Copy `paths.example.json` to `paths.json` and set `dataset_dir` (the official `student_resource/dataset` folder, containing `train/` and `test/`) and `work_dir` (a scratch folder for caches). Without `paths.json`, the code looks for the dataset in `data/original/student_resource/dataset/` and writes caches to `data/working/`, both inside this folder (`src/io_utils.py`).
2. `final_train_ids.tsv` (420,000 labelled train S1 ids; see below) is included in this folder.

## Run (one command)
```
python src/final_v2.py --config final_config.json --train-ids final_train_ids.tsv --out output
```
`final_config.json` sets the shipped configuration: forward top-10 (`Kf`) ∪ reverse top-3 (`rev_r`), `cap` 25, `neg_rate` 0.4 and `"exclusivity": true` (the one-to-one decision step below).
This writes `output/matching_results.tsv`, `output/candidate_pairs.tsv`, `output/final_report.json`, `output/final_model.pkl` and `output/final_log.txt`. A verification run of this exact package from an empty cache took 7,952 s (about 2 h 13 min) and reproduced both submitted files byte for byte.

## Pipeline (data → blocking → matching → output)
1. **Load** every TSV with `sep="\t"`, all columns as strings, no quoting and no NA parsing (`src/io_utils.py`).
2. **Normalise** (`src/normalize.py`, `src/translit.py`):
   - NFKD accent folding, lower-casing, `&` → `and`, punctuation removed
   - rule-based transliteration of every Indic script to Latin
   - no country-specific rules and no hand word lists
3. **Blocking** within each country partition (`src/blocking.py`, `src/prep.py`, `src/pipeline.py`):
   - Hashed binary features: name tokens, address tokens, name and address word bigrams, and 4-grams of the space-free name. IDF weights come from the unlabelled records of the partition; features with pool document frequency above 3000 are dropped. Scored by sparse cosine.
   - Forward: each S1's top-10 S2 and top-10 S3 records.
   - Reverse: each S2/S3 record's top-3 S1 entities across the partition, inverted into per-S1 lists.
   - Union, capped at 25 candidates per source per S1 by blocking score. This exact set is scored by the model and written to `candidate_pairs.tsv`.
4. **Pair features** (59, country-agnostic): set similarities of names, addresses and digits; RapidFuzz ratios; length and script ratios; blocking score, rank and gaps; one-to-one features (the S1's rank in the record's own best-S1 list, gap to the record's best S1); entity context.
5. **Model** (`src/model.py`): LightGBM (800 trees, 127 leaves, min_child_samples 50, seed 2026, deterministic). It is trained on all positive pairs and a deterministic 40% sample of negative pairs, with kept negatives weighted 1/0.4.
6. **Decision** (`model.decide_exclusive`):
   - One global probability threshold from 5-fold grouped out-of-fold predictions.
   - Then **exclusivity**: an S2/S3 record accepted by several S1 entities is kept only for the highest-probability S1 (ties go to the smallest S1 id). In the training data every S2/S3 record belongs to at most one S1.

## Training entities
`final_train_ids.tsv` holds 420,000 labelled train S1 ids: the development samples (150k fit + 60k dev + 60k lockbox, seed 2026) plus 150k more from the train split (seed 2026).
