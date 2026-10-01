"""End-to-end final pipeline: data -> blocking -> features -> model -> output files.

Usage (from the project / package root):
    python src/final.py [--config final_config.json] [--train-ids final_train_ids.tsv] [--out output]

Steps
1. Training entities: the S1 ids listed in --train-ids (labelled train S1; see README).
2. Candidate generation per country partition (forward top-Kf per source UNION reverse top-r, capped),
   pair features, labels from train_ground_truth.tsv.
3. 5-fold out-of-fold (grouped by S1) predictions -> one global threshold maximising macro F0.5.
4. Final LightGBM fitted on all training pairs with the identical configuration.
5. Test: the same candidate generation over ALL test S1; candidate_pairs.tsv is exactly the scored set;
   matching_results.tsv = candidates with probability >= threshold (a subset).
Writes <out>/matching_results.tsv, <out>/candidate_pairs.tsv and <out>/final_report.json.
"""
import argparse
import json
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import evaluate as ev  # noqa: E402
import io_utils as io  # noqa: E402
import model as M  # noqa: E402
import pipeline  # noqa: E402


def write_lists(path, header, queries, lists):
    """Write one TSV row per S1 id: '<s1>\t<comma-separated ids>' (empty list allowed)."""
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("source1_entity_id\t" + header + "\n")
        for q in queries:
            f.write(q + "\t" + ",".join(dict.fromkeys(lists.get(q, ()))) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "final_config.json"))
    ap.add_argument("--train-ids", default=str(ROOT / "final_train_ids.tsv"))
    ap.add_argument("--out", default=str(ROOT / "output"))
    ap.add_argument("--folds", type=int, default=5)
    a = ap.parse_args()
    cfg = json.loads(pathlib.Path(a.config).read_text())
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    logf = out / "final_log.txt"
    L = lambda m: pipeline.log(m, logf)  # noqa: E731
    t0 = time.time()
    seed = int(cfg.get("seed", 2026))
    rep = {"config": cfg}

    # 1-2. training pairs
    ids = pd.read_csv(a.train_ids, sep="\t", dtype=str).entity_id.tolist()
    L(f"training S1 entities: {len(ids)}")
    tag = pipeline.pair_tag("final_train", cfg)
    paths = pipeline.build_pairs("train", ids, cfg["K"], cfg["max_df"], tag, cfg["char_ngrams"], logf,
                                 **pipeline.pair_opts(cfg))
    pairs, queries = pipeline.load_pairs(paths)
    truth = io.parse_gt(io.load("train_gt"))
    pairs = M.add_labels(pairs, truth)
    cols = M.feature_cols(pairs)
    L(f"training pairs {len(pairs)}, positives {int(pairs.label.sum())}, features {len(cols)}")
    tr = {q: truth[q] for q in queries}
    cand = M.candidates(pairs, queries)
    rep["train_candidate_recall"] = ev.report(tr, {q: [] for q in queries}, cand, queries)["candidate_recall"]

    # 3. out-of-fold threshold
    rng = np.random.default_rng(seed)
    fold_of = dict(zip(queries, rng.integers(0, a.folds, len(queries))))
    fold = pairs.s1.map(fold_of).values
    oof = np.zeros(len(pairs), dtype=np.float32)
    for k in range(a.folds):
        m = M.fit(pairs[fold != k], cols, cfg.get("model_params"), seed)
        oof[fold == k] = M.predict(m, pairs[fold == k], cols)
        L(f"OOF fold {k + 1}/{a.folds} done")
    pairs["prob"] = oof
    grid = np.round(np.arange(0.40, 0.901, 0.025), 3)
    curve = [(float(t), ev.macro_f05(tr, M.decide(pairs, queries, t), queries)) for t in grid]
    thr, f_best = max(curve, key=lambda x: x[1])
    rep["oof_threshold"] = thr
    rep["oof_macro_f05"] = f_best
    rep["oof_threshold_curve"] = curve
    L(f"OOF threshold {thr} (OOF macro F0.5 {f_best:.5f}); curve: " +
      ", ".join(f"{t:.3f}:{f:.5f}" for t, f in curve))
    s1c = io.load("train_s1").set_index("entity_id").country
    rep["oof_prob_quantiles_by_country"] = {
        c: np.round(np.quantile(g.prob.values, [0.5, 0.75, 0.9, 0.95, 0.99]), 4).tolist()
        for c, g in pairs.assign(c=pairs.s1.map(s1c)).groupby("c")}

    # 4. final model on all training pairs
    final = M.fit(pairs, cols, cfg.get("model_params"), seed)
    pd.to_pickle(final, out / "final_model.pkl")
    L("final model fitted")
    del pairs, oof
    import gc
    gc.collect()

    # 5. test inference
    scorer = lambda fdf: M.predict(final, fdf, cols)  # noqa: E731
    ttag = pipeline.pair_tag("final_test", cfg)
    tpaths = pipeline.build_pairs("test", None, cfg["K"], cfg["max_df"], ttag, cfg["char_ngrams"], logf,
                                  scorer=scorer, **pipeline.pair_opts(cfg))
    tp, tq = pipeline.load_pairs(tpaths)
    all_s1 = io.load("test_s1").entity_id.tolist()
    assert len(set(tq)) == len(tq) and set(tq) == set(all_s1), "query set != test S1"
    pred = M.decide(tp, all_s1, thr)
    tcand = M.candidates(tp, all_s1)
    write_lists(out / "matching_results.tsv", "matched_entity_ids", all_s1, pred)
    write_lists(out / "candidate_pairs.tsv", "candidate_entity_ids", all_s1, tcand)
    sizes = np.array([len(tcand[q]) for q in all_s1])
    npred = np.array([len(pred[q]) for q in all_s1])
    tc = io.load("test_s1").set_index("entity_id").country
    rep["test"] = {
        "n_s1": len(all_s1), "avg_candidates": float(sizes.mean()), "max_candidates": int(sizes.max()),
        "avg_matches": float(npred.mean()), "empty_match_rate": float((npred == 0).mean()),
        "prob_quantiles_by_country": {
            c: np.round(np.quantile(g.prob.values, [0.5, 0.75, 0.9, 0.95, 0.99]), 4).tolist()
            for c, g in tp.assign(c=tp.s1.map(tc)).groupby("c")},
        "avg_candidates_by_country": pd.Series(sizes, index=all_s1).groupby(tc.reindex(all_s1).values).mean().round(3).to_dict(),
        "empty_match_rate_by_country": pd.Series(npred == 0, index=all_s1).groupby(tc.reindex(all_s1).values).mean().round(4).to_dict(),
    }
    rep["runtime_seconds"] = round(time.time() - t0, 1)
    rep["peak_rss_gb"] = io.peak_rss_gb()
    (out / "final_report.json").write_text(json.dumps(rep, indent=2))
    L(f"done in {rep['runtime_seconds']}s: {json.dumps(rep['test'])}")


if __name__ == "__main__":
    main()
