# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** VibeCoders  
**Team Members:** Aditya Pratap Singh Bhadauria (Team Leader), Anshika Dixit, Mradul Tripathi, Preeti Kumari  
**Submission Date:** 27 September 2026

---

## 1. Executive Summary

The shipped pipeline is `src/final_v2.py`, run with `final_config.json`. It builds candidates separately within each country partition using sparse, hashed TF-IDF-style vectors and bidirectional top-k retrieval. A LightGBM binary classifier scores candidate pairs from name, address, blocking, reverse-retrieval, and entity-context features. A global probability threshold is selected from grouped out-of-fold predictions by maximizing macro F0.5. At test decision time, the configured exclusivity step assigns each candidate S2/S3 record to at most one S1 entity.

This submission package does not contain the referenced `output/` directory, so output row counts, contents, and reported run metrics cannot be independently verified from this checkout. Performance figures and experimental comparisons are therefore omitted here.

## 2. Methodology

### 2.1 Problem and data representation

The inputs are tab-separated tables for three record sources (S1, S2, S3), plus train ground truth. The loader reads fields as strings and disables automatic NA parsing so literal strings are retained. The pipeline compares business names and business addresses. Candidate generation and unlabelled statistics operate within each country value, treating the values as partitions; the country label is not included as a model feature.

`final_train_ids.tsv` supplies the S1 entities used to build the final training pairs. The README describes this list as 420,000 labelled training S1 IDs. Ground-truth matched IDs are parsed as comma-separated lists.

### 2.2 End-to-end stages

1. **Load and prepare:** `io_utils.py` reads the source TSVs. `prep.py` selects the S1/S2/S3 records for each country partition and constructs sparse hashed feature matrices.
2. **Normalize and transliterate:** `normalize.py` lowercases, applies NFKD accent/mark removal, changes `&` to `and`, maps punctuation and other non-word runs to spaces, and collapses whitespace. The shipped feature setting selects transliteration v2 for Indic script characters before normalisation.
3. **Generate candidates:** `blocking.py` and `pipeline.py` perform forward and reverse cosine retrieval within the same partition. Candidate pairs from both directions are unioned and deduplicated, then capped per S1 and source.
4. **Engineer pair features:** `features.py` computes similarities for name and address tokens, character n-grams, and address digits, alongside string similarity, blocking, reverse-retrieval, source, and entity-context features.
5. **Train and calibrate a decision threshold:** `final_v2.py` builds labels from ground truth, trains five grouped folds, collects predictions for held-out S1 groups, and searches a global threshold grid for macro F0.5. It then fits the final classifier on the configured sampled training rows.
6. **Infer and decide:** the final classifier scores the test candidate set in chunks. Candidates at or above the OOF-selected threshold are retained. With `exclusivity: true`, collisions are resolved so each S2/S3 candidate is assigned to the S1 with the greatest predicted probability.
7. **Write results:** `final.py`'s `write_lists` function writes one row per S1 ID with comma-separated candidate or matched IDs.

### 2.3 Which implementation is final

The package README gives the reproduction command `python src/final_v2.py --config final_config.json --train-ids final_train_ids.tsv --out output`, and the shipped `final_config.json` names `exp_023_consistency (exp_022_wide + exclusivity)` as its source experiment. `final_v2.py` imports `write_lists` from `final.py` but owns the configured negative sampling, OOF threshold selection, final fit, test scoring, and configured exclusivity decision. `final.py` is an earlier entry point: it fits without the `final_v2.py` negative-subsampling path and calls the non-exclusive decision function. `translit_v1.py` is retained for older feature settings; configuration value 4 selects v2. Other branches in `pipeline.py` and `model.py` are optional experiment hooks and are not enabled by the shipped config.

## 3. Candidate Generation / Blocking

### 3.1 Partition and representation

Blocking is performed independently for each country partition. For each record, `blocking.make_analyzer` emits binary hashed features:

- `n:` prefixed normalized name tokens;
- `a:` prefixed normalized address tokens;
- name word bigrams and address word bigrams when `char_ngrams >= 2`;
- 4-character n-grams of the normalized, space-free name when `char_ngrams` is enabled (or the entire shorter nonempty name if shorter than four characters).

The shipped `char_ngrams: 4` activates both word-bigram and name-character-gram features. `HashingVectorizer` uses 2^22 dimensions, binary values, and unsigned feature hashing (`alternate_sign=False`). `prep.idf_and_keep` calculates IDF from all three sources in the partition, then excludes features whose combined S2/S3 pool document frequency exceeds `max_df` (3,000). The remaining vectors are IDF-weighted and L2-normalized; pairwise sparse dot products are cosine scores. The vectorizer processes records in chunks of 300,000. No phonetic key, email, phone, website, or separate fuzzy-string blocking pass is implemented.

### 3.2 Forward retrieval

**What:** For each S1 query and separately for S2 and S3, retrieve the top 10 records by cosine score.  
**How:** `pipeline.candidates` weights the S1 query vectors and each source pool with the same partition statistics; `blocking.topk` calculates sparse query–pool products in chunks and retains the highest nonzero scores.  
**Why:** This efficiently narrows a large cross-source comparison space to records sharing informative name/address features.

### 3.3 Reverse retrieval

**What:** For every S2 and S3 record, retrieve its top 3 S1 records in the same country partition.  
**How:** `pipeline.reverse_all` uses the same binary features, IDF weights, and cosine calculation in the reverse query direction. The reverse results are inverted into S1 candidate lists and combined with the forward lists.  
**Why:** A record can be a strong choice for an S1 even when it falls outside that S1's forward top 10; reverse retrieval allows that pair into the candidate set.

### 3.4 Union, cap, and deduplication

For each source separately, the pipeline takes the union of forward top-10 and reverse top-3 pairs. `numpy.unique` removes duplicate query/pool index pairs produced by both passes. It then retains at most 25 candidates per S1 per source, ordered by the pair's cosine blocking score. When a pair is in both directions, the forward score is used; cosine is symmetric. Thus the maximum configured candidate count is 50 per S1 across S2 and S3 combined. Zero-score forward results are excluded by top-k retrieval. There is no fallback pass when a query has few or no candidates.

The repository does not include the generated candidate output in this checkout, so realized candidate counts and candidate recall are not verifiable here.

## 4. Model Architecture and Feature Engineering

### 4.1 Model and scoring

The configured model is LightGBM's `LGBMClassifier`, a gradient-boosted decision-tree binary classifier, implemented in `model.fit_arrays`. The default parameters in code are overridden by `final_config.json` where specified. Effective configured parameters are:

- `n_estimators=800`
- `learning_rate=0.05`
- `num_leaves=127`
- `min_child_samples=50`
- `subsample=0.8`, `subsample_freq=1`
- `colsample_bytree=0.8`
- `reg_lambda=1.0`
- `n_jobs=10`, deterministic mode enabled, row-wise training forced
- random seed `2026`

The classifier uses its standard binary objective (no alternate objective is configured). `predict_proba` supplies the positive-class probability for each candidate pair. This is the model score used by the decision threshold; the repository does not document it as a calibrated probability.

All positive training candidate pairs are retained. `final_v2.subsample` deterministically retains 40% of negative pairs using a seeded random generator and assigns each retained negative weight 2.5; positive weights are 1.0. The training process therefore addresses the computational effect of numerous candidate negatives by sampling and weighting. No ensemble is used by the shipped entry point.

### 4.2 Pair features

The base pair features from `features.pair_features` comprise 46 columns; enabling `ctx` adds 7, and the bidirectional candidate union adds 6, for 59 configured features in total.

**Name and address token/character/digit overlap.** For each of five binary hashed sets—name word tokens (`NT`), name character 3-grams of the space-free normalized name (`NC`), address word tokens (`AT`), address character 3-grams (`AC`), and address digit strings (`AD`)—the pipeline calculates:

- Jaccard similarity (intersection divided by union);
- containment (intersection divided by the smaller set size);
- binary cosine similarity;
- intersection count;
- an empty-set indicator.

These quantify exact token, character-fragment, and address-number overlap at different levels of strictness. Hashing uses 2^20 dimensions.

**String and length features.** The model receives normalized name and address length ratios; an address-empty flag for the pool record; name ASCII-character shares for each side and their absolute difference; exact equality of normalized space-free names; RapidFuzz `ratio`, `token_sort_ratio`, `token_set_ratio`, and `partial_ratio` for names; RapidFuzz `ratio` for space-free names; and `token_sort_ratio`, `token_set_ratio`, and `partial_ratio` for addresses. These capture edit similarity, token reordering, token containment, partial strings, and script composition.

**Blocking and source features.** These include the blocking cosine score, forward rank, blocking score relative to the query's forward top score, gap from that top score, gap between the forward top and second scores, and a flag indicating whether the candidate is from S3 (versus S2).

**Reverse-retrieval features.** Because the configured candidate set uses reverse retrieval, the model also receives reverse rank, the candidate record's best reverse score, the gap between that best score and this pair's score, the gap between the best and second-best reverse scores, and flags for inclusion in the configured forward and reverse lists. These encode whether each side ranks the pair highly.

**Entity-context features (`ctx: true`).** The code calculates log frequency of each normalized space-free name across all records in the partition, for both query and candidate; per-S1 counts of candidates with name token-set similarity at least 0.95 and address token-set similarity at least 0.90; within-S1 ranks by name and address token-set similarity; and the difference between a candidate's name token-set similarity and the maximum for that S1. These capture name ambiguity and competition among candidates for a query.

No country, city, postal code, email, phone, or website feature is used. Optional core-name, S2–S3 consistency, and token-IDF feature branches exist in `pipeline.py`, but the shipped config does not enable them.

## 5. Normalization and Transliteration

`normalize.norm` performs the following operations in order: optional configured transliteration; lowercase; Unicode NFKD decomposition and removal of combining marks; `&` to `and`; replace runs of non-word characters with spaces; replace underscore runs with spaces; and collapse whitespace and trim. `squash` removes spaces from a normalized string. `digits` extracts digit runs from normalized text for address digit features.

With `char_ngrams: 4`, `pipeline.configure` selects `translit.py` (v2). It maps characters from Devanagari, Bengali, Gurmukhi, Gujarati, Oriya, Tamil, Telugu, Kannada, and Malayalam Unicode blocks to Latin using a hand-coded, shared offset-based mapping. It handles consonants, vowel signs, virama, selected marks, script digits, joiner removal, and word-final inherent-vowel deletion. Non-Indic characters pass through. This is rule-based transliteration rather than language identification or dictionary transliteration.

The legacy `translit_v1.py` remains in the repository for feature version 3 and earlier reproducibility; it is not selected by the shipped configuration. There are no dedicated phone, email, domain, or address parsers, no number canonicalization beyond punctuation cleaning and digit extraction, and no country-specific address rules. A limitation of the configured normalizer is that removing combining marks also affects scripts where marks encode meaningful distinctions; transliteration is approximate and can merge or separate variant spellings.

## 6. Training, Validation, and Threshold Selection

The training IDs are read from `final_train_ids.tsv`; pair labels are created by membership in parsed training ground truth. Candidate generation is performed before pair classification, so pairs outside the blocking set are unavailable to the model. `final_v2.py` assigns each selected S1 entity to one of five folds using a seeded random assignment. All candidate pairs for an S1 stay in the same fold, preventing the same query entity from appearing in both fold training and held-out predictions. Each fold model predicts every candidate pair of its held-out S1 entities.

Threshold selection evaluates the OOF predictions at thresholds from 0.400 through 0.900 in increments of 0.025 and chooses the value with the highest macro F0.5 as computed by `evaluate.py`. The metric is calculated per S1, including an empty-prediction correctness rule for ground-truth singletons, and then averaged across S1 entities. The selected threshold is global, not country-specific. The final model is fit after threshold selection on all configured training pairs after deterministic negative sampling and weighting. At inference, `final_v2.py` applies the same threshold and then, because `exclusivity` is true, resolves each candidate ID claimed by multiple S1 entities in favor of the largest pair probability (ties go to lexically smallest S1 ID under the implemented sort).

The repository code therefore explicitly optimizes/evaluates macro F0.5 for threshold selection. It does not implement a separate validation split or cross-validation model-selection comparison in the shipped entry point; its five-fold OOF procedure is used to choose the decision threshold. Reported threshold value and OOF score are not present as generated artifacts in this checkout and are not verifiable from the current repository files.

## 7. End-to-End Pipeline and Code Roles

```text
Train/test TSV records
        ↓
io_utils: read string fields and cache tables
        ↓
prep: country partitions and sparse record matrices
        ↓
normalize + translit v2
        ↓
blocking: forward top-10 ∪ reverse top-3, cap 25 per source
        ↓
pipeline: deduplicated candidate pairs and pair features
        ↓
final_v2: labels, negative sampling/weights, five-fold OOF and LightGBM fit
        ↓
LightGBM candidate scores → global OOF-selected threshold
        ↓
configured test exclusivity across S1 claims
        ↓
final.write_lists: candidate_pairs.tsv and matching_results.tsv
```

- `blocking.py`: hashed sparse vectors, IDF weighting, cosine top-k retrieval.
- `prep.py`: country partitions, record selections, matrix persistence, IDF and frequent-feature mask.
- `normalize.py`, `translit.py`: text normalization and configured transliteration.
- `features.py`: base pair-level overlap and string features.
- `pipeline.py`: candidate union/capping, optional feature branches, context and pair-table construction.
- `model.py`: LightGBM fit/prediction helpers, threshold policy, and exclusive decision policy; it also contains unused optional model classes/branches.
- `final_v2.py`: shipped orchestration for training, threshold selection, inference, and configured exclusivity.
- `final.py`: earlier orchestration and the shared output writer imported by `final_v2.py`.
- `evaluate.py`: per-entity and macro F0.5 metric implementation and related reports.
- `io_utils.py`: input loading, caching, and ground-truth parsing.
- `translit_v1.py`: earlier transliteration implementation, not selected by shipped config.

## 8. Outputs

The requested `output/candidate_pairs.tsv` and `output/matching_results.tsv` files are not present in the inspected repository checkout. Their actual rows and counts therefore cannot be audited here. The writer in `final.py` defines their expected format:

- **`candidate_pairs.tsv`:** columns `source1_entity_id` and `candidate_entity_ids`; one row per test S1, containing the comma-separated deduplicated IDs in that S1's generated candidate set. These are the records scored by the model, whether or not they pass the decision threshold.
- **`matching_results.tsv`:** columns `source1_entity_id` and `matched_entity_ids`; one row per test S1, containing the comma-separated IDs selected after thresholding and configured exclusivity. An empty list is written as an empty second field.

The matching output is derived from candidate scores, so it is a subset of candidate pairs, subject to the exclusivity step removing candidate claims that lose a cross-S1 conflict.

## 9. Computational Efficiency

The implementation reduces all-pairs comparisons using country partitioning and top-k blocking. It uses `HashingVectorizer` and SciPy CSR sparse matrices for feature representation and cosine products, prunes pool-common features above document frequency 3,000, and processes vectorization, sparse retrieval, pair-feature generation, and inference in chunks. Candidate lists are capped at 25 per source per S1. `prep.py` and `pipeline.py` persist matrices, top-k results, reverse results, and pair tables to disk for reuse. `final_v2.py` samples negatives with importance-style weights, retains one float32 feature matrix during its OOF loop, and deletes large intermediate objects with garbage collection. RapidFuzz computes aligned string similarity with multiple workers.

These mechanisms bound expensive string comparison and classifier scoring to the candidate set and manage intermediate memory. Multiprocessing beyond RapidFuzz worker use is not implemented in the inspected pipeline.

## 10. Design Decisions

| Decision | Implementation | Purpose |
|---|---|---|
| Partition by country | Build vectors and candidate pools independently for each country value | Restrict comparisons to the same partition; country itself is not a model feature. |
| Combine forward and reverse retrieval | Forward top-10 per S1/source union reverse top-3 S1 per pool record | Include pairs ranked highly from either side. |
| Drop very common blocking features | Remove features with S2+S3 pool frequency above 3,000 | Reduce sparse-product work and limit the value of broadly shared tokens. |
| Use a capped candidate set | Keep up to 25 candidates per S1 per source by cosine score | Bound per-query pair count passed to feature engineering and classification. |
| Use a country-agnostic feature set | Compare names, addresses, and digits; omit country-coded model features | Apply the same configured feature logic to all partitions. |
| Sample and weight negative training pairs | Keep all positives and a seeded 40% of negatives; weight sampled negatives 2.5 | Reduce training rows while compensating for sampled negative frequency. |
| Choose one global threshold on OOF predictions | Search the configured threshold grid using macro F0.5 | Select the operating point against the challenge metric using held-out S1 predictions. |
| Enforce one-to-one ownership at inference | Keep the highest-probability S1 claim for each S2/S3 candidate ID | Apply the one-to-one constraint encoded by the configured decision policy. |

## 11. Limitations and Edge Cases

- Blocking can omit true matches that share no retained hashed features or fall outside both top-k directions and the per-source cap; the classifier cannot recover a pair it never receives.
- Name and address fields can be empty, noisy, abbreviated, or reordered. The code uses similarity measures but does not expand abbreviations or parse structured address components.
- Transliteration is a fixed rule-based mapping. It does not use language identification, dictionaries, or contextual disambiguation; other scripts are not transliterated by these functions.
- Hashing maps tokens/grams into a finite feature space, so collisions are possible. Frequent pool features are removed, which can also remove a shared feature useful to an unusual true match.
- Country partitioning assumes relevant matches share the same country label. A missing or inconsistent country value can place a true pair outside the comparison partition.
- The global threshold and one-to-one exclusivity are fixed decision rules. A wrong high-scoring claim can win a conflict, while an S1 is allowed to have no selected candidates if all scores are below threshold.
- Pair similarity is based on name/address text. No dedicated phone, email, website, geocoding, or external knowledge features are implemented.
- The package's referenced generated output files are absent in this checkout, so test behavior and realized performance cannot be confirmed from these files.

## 12. Conclusion

The shipped approach combines within-country sparse cosine blocking in both retrieval directions with a LightGBM pair classifier and a macro-F0.5-selected global threshold. The configured test policy then enforces exclusive ownership of each S2/S3 record. The implementation and configuration establish these mechanics; quantitative submission results should be treated as unverified unless the corresponding run report or output artifacts are supplied.

## Appendix

### A. Code Artefacts

The implementation is in `code/business_entity_resolution/`. The documented reproduction command from its README is:

```text
python src/final_v2.py --config final_config.json --train-ids final_train_ids.tsv --out output
```

A local `paths.json` can point the loader to the challenge dataset and a scratch work directory; `paths.example.json` provides the expected keys. No API or external lookup is used by the code paths documented above.

### B. Dependencies

`requirements.txt` pins NumPy, pandas, SciPy, scikit-learn, LightGBM, and RapidFuzz. The matching classifier is LightGBM; scikit-learn supplies hashing/vectorization and other utilities, and RapidFuzz supplies string-similarity functions.

### C. Verification Gaps

The current checkout does not contain the `output/` directory or generated `final_report.json`. Consequently, realized candidate counts, selected threshold value, OOF macro F0.5 value, test match counts, runtime, and peak memory from the claimed run are **Not verifiable from the current repository.** Experimental results and comparisons not encoded in the supplied implementation/configuration are not reported.

---

## FINAL AUDIT SUMMARY

- **Final pipeline identified:** `src/final_v2.py` with `final_config.json`; `final.py` supplies the shared output writer and is also an earlier pipeline.
- **Blocking strategy:** Country partitioning; hashed binary name/address tokens, optional word bigrams and name 4-grams; IDF-weighted sparse cosine; forward top-10 union reverse top-3; deduplication and cap 25 per source per S1.
- **Model:** LightGBM `LGBMClassifier`, 800 estimators, 127 leaves, 50 minimum child samples, learning rate 0.05, seed 2026; all positives plus deterministic 40% negative sample weighted 2.5.
- **Main feature groups:** Name/address token, character-gram and digit overlaps; RapidFuzz string similarities; length/script indicators; blocking and reverse-rank features; S2/S3 source flag; partition name frequency and within-S1 candidate context.
- **Normalization/transliteration:** Lowercase, NFKD mark removal, ampersand expansion, punctuation-to-space, whitespace collapse; shipped config uses rule-based Indic-to-Latin transliteration v2.
- **Validation/evaluation:** Five seeded S1-grouped OOF folds; global threshold grid selected by the implemented per-entity macro F0.5 metric. Numeric OOF result/threshold artifact absent, so values are not verifiable.
- **Output generation:** `final_v2.py` writes candidate IDs and thresholded/exclusive matched IDs through `final.write_lists`; requested output files are absent from checkout.
- **Any uncertainties:** Realized outputs, run metrics, exact selected threshold, performance claims, and experimental outcomes cannot be verified from current files. The selected entry point/config are identified by README and configuration evidence.
- **Documentation completed:** YES
