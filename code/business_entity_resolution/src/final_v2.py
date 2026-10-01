"""Final pipeline v2 (exp_022_wide): as final.py, plus a deterministic negative subsample of
TRAINING rows (rate cfg["neg_rate"], kept negatives weighted 1/rate) and a memory-lean OOF loop
(one float32 feature matrix; folds sliced by index). Held-out folds are always scored on ALL their
pairs, so the OOF threshold is chosen on the true candidate sets.

Usage: python src/final_v2.py --config <cfg.json> --train-ids final_train_ids.tsv --out <dir>
Writes <out>/matching_results.tsv, <out>/candidate_pairs.tsv, <out>/final_report.json.
"""
import argparse
import gc
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
from final import write_lists  # noqa: E402


def subsample(idx, y, rate, seed):
    """Deterministic: keep all positives, a `rate` share of negatives; weights 1/rate for kept negatives."""
    if not rate:
        return idx, np.ones(len(idx), np.float32)
    u = np.random.default_rng(seed + 7).random(len(idx))
    keep = (y[idx] == 1) | (u < rate)
    k = idx[keep]
    w = np.where(y[k] == 1, 1.0, 1.0 / rate).astype(np.float32)
    return k, w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--train-ids", default=str(ROOT / "final_train_ids.tsv"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--folds", type=int, default=5)
    a = ap.parse_args()
    cfg = json.loads(pathlib.Path(a.config).read_text())
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    logf = out / "final_log.txt"
    L = lambda m: pipeline.log(m, logf)  # noqa: E731
    t0 = time.time()
    seed = int(cfg.get("seed", 2026))
    rate = cfg.get("neg_rate")
    rep = {"config": cfg}

    ids = pd.read_csv(a.train_ids, sep="\t", dtype=str).entity_id.tolist()
    L(f"training S1 entities: {len(ids)}")
    tag = pipeline.pair_tag("final_train", cfg)
    paths = pipeline.build_pairs("train", ids, cfg["K"], cfg["max_df"], tag, cfg["char_ngrams"], logf,
                                 **pipeline.pair_opts(cfg))
    pairs, queries = pipeline.load_pairs(paths)
    truth = io.parse_gt(io.load("train_gt"))
    pairs = M.add_labels(pairs, truth)
    cols = M.feature_cols(pairs)
    X = pairs[cols].to_numpy(np.float32)
    y = pairs.label.values.astype(np.int8)
    meta = pairs[["s1", "cand"]].copy()
    del pairs
    gc.collect()
    L(f"training pairs {len(y)}, positives {int(y.sum())}, features {len(cols)}, free={io.free_ram_gb():.1f}GB")
    tr = {q: truth[q] for q in queries}
    rep["train_candidate_recall"] = ev.report(tr, {q: [] for q in queries}, M.candidates(meta, queries),
                                              queries)["candidate_recall"]

    rng = np.random.default_rng(seed)
    fold_of = dict(zip(queries, rng.integers(0, a.folds, len(queries))))
    fold = meta.s1.map(fold_of).values
    oof = np.zeros(len(y), dtype=np.float32)
    for k in range(a.folds):
        tr_idx, w = subsample(np.flatnonzero(fold != k), y, rate, seed + k)
        m = M.fit_arrays(X[tr_idx], y[tr_idx], w, cfg.get("model_params"), seed)
        te = np.flatnonzero(fold == k)
        oof[te] = m.predict_proba(X[te])[:, 1]
        del m, tr_idx, w
        gc.collect()
        L(f"OOF fold {k + 1}/{a.folds} done, free={io.free_ram_gb():.1f}GB")
    meta["prob"] = oof
    grid = np.round(np.arange(0.40, 0.901, 0.025), 3)
    curve = [(float(t), ev.macro_f05(tr, M.decide(meta, queries, t), queries)) for t in grid]
    thr, f_best = max(curve, key=lambda x: x[1])
    rep.update({"oof_threshold": thr, "oof_macro_f05": f_best, "oof_threshold_curve": curve})
    L(f"OOF threshold {thr} (OOF macro F0.5 {f_best:.5f}); curve: " + ", ".join(f"{t:.3f}:{f:.5f}" for t, f in curve))
    s1c = io.load("train_s1").set_index("entity_id").country
    rep["oof_prob_quantiles_by_country"] = {
        c: np.round(np.quantile(g.prob.values, [0.5, 0.75, 0.9, 0.95, 0.99]), 4).tolist()
        for c, g in meta.assign(c=meta.s1.map(s1c)).groupby("c")}

    all_idx, w = subsample(np.arange(len(y)), y, rate, seed + 100)
    final = M.fit_arrays(X[all_idx], y[all_idx], w, cfg.get("model_params"), seed)
    rep["final_fit_rows"] = int(len(all_idx))
    pd.to_pickle(final, out / "final_model.pkl")
    L(f"final model fitted on {len(all_idx)} rows (neg_rate {rate})")
    del X, y, meta, oof, all_idx, w
    gc.collect()

    scorer = lambda fdf: final.predict_proba(fdf[cols].to_numpy(np.float32))[:, 1]  # noqa: E731
    ttag = pipeline.pair_tag("final_test", cfg)
    tpaths = pipeline.build_pairs("test", None, cfg["K"], cfg["max_df"], ttag, cfg["char_ngrams"], logf,
                                  scorer=scorer, **pipeline.pair_opts(cfg))
    tp, tq = pipeline.load_pairs(tpaths)
    all_s1 = io.load("test_s1").entity_id.tolist()
    assert len(set(tq)) == len(tq) and set(tq) == set(all_s1), "query set != test S1"
    if cfg.get("exclusivity"):
        pred = M.decide_exclusive(tp, all_s1, thr)
        L("exclusivity post-processing applied (one S1 per S2/S3 record)")
    else:
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
