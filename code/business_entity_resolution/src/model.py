"""Matching model, decision policy and validation run helpers."""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

import evaluate as ev

META = ("s1", "cand")


def feature_cols(pairs):
    return [c for c in pairs.columns if c not in META and c not in ("label", "prob", "w")]


def add_labels(pairs, truth):
    tset = {(s, m) for s, ms in truth.items() for m in ms}
    pairs["label"] = np.fromiter(((s, c) in tset for s, c in zip(pairs.s1.values, pairs.cand.values)),
                                 dtype=np.int8, count=len(pairs))
    return pairs


class RankModel:
    """LightGBM lambdarank grouped by S1 entity, mapped to probabilities with Platt scaling
    (fitted on train_fit calib pairs) so the global-threshold decision still applies."""

    def __init__(self, ranker):
        self.ranker = ranker
        self.a, self.b = 1.0, 0.0

    def raw(self, X):
        return self.ranker.predict(X)

    def calibrate(self, X, y):
        from sklearn.linear_model import LogisticRegression
        lr = LogisticRegression(C=1e6, max_iter=1000)
        lr.fit(self.raw(X).reshape(-1, 1), y)
        self.a, self.b = float(lr.coef_[0, 0]), float(lr.intercept_[0])

    def predict_proba(self, X):
        p = 1.0 / (1.0 + np.exp(-(self.a * self.raw(X) + self.b)))
        return np.column_stack([1 - p, p])


class AvgModel:
    """Average of two models' probabilities; each model sees its own feature columns."""

    def __init__(self, ma, cols_a, mb, cols_b, all_cols):
        self.ma, self.mb = ma, mb
        self.ia = [all_cols.index(c) for c in cols_a]
        self.ib = [all_cols.index(c) for c in cols_b]

    def predict_proba(self, X):
        p = 0.5 * (self.ma.predict_proba(X[:, self.ia])[:, 1] + self.mb.predict_proba(X[:, self.ib])[:, 1])
        return np.column_stack([1 - p, p])


def fit_arrays(X, y, w, params, seed):
    """LightGBM on numpy arrays (memory-lean path used by final_v2)."""
    import lightgbm as lgb
    p = dict(n_estimators=400, learning_rate=0.05, num_leaves=63, min_child_samples=100,
             subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
             n_jobs=10, deterministic=True, force_row_wise=True, verbose=-1)
    p.update(dict(params or {}))
    p.pop("algo", None)
    clf = lgb.LGBMClassifier(random_state=seed, **p)
    clf.fit(X, y, sample_weight=w)
    return clf


def fit(pairs, cols, params, seed):
    """LightGBM (MIT) by default; params {"algo": "hgb", ...} selects sklearn HistGradientBoosting;
    {"algo": "rank"} a LightGBM lambdarank model grouped by S1 (calibrate() must be called after)."""
    params = dict(params or {})
    algo = params.pop("algo", "lgbm")
    if algo == "rank":
        import lightgbm as lgb
        ps = pairs.sort_values("s1", kind="stable")
        groups = ps.groupby("s1", sort=False).size().values
        p = dict(n_estimators=400, learning_rate=0.05, num_leaves=63, min_child_samples=100,
                 subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                 n_jobs=10, deterministic=True, force_row_wise=True, verbose=-1)
        p.update(params)
        rk = lgb.LGBMRanker(objective="lambdarank", random_state=seed, **p)
        rk.fit(ps[cols].values.astype(np.float32), ps.label.values, group=groups)
        return RankModel(rk)
    X = pairs[cols].values.astype(np.float32)
    y = pairs.label.values
    w = pairs["w"].values if "w" in pairs.columns else None
    if algo == "hgb":
        p = dict(max_iter=300, learning_rate=0.1, max_leaf_nodes=63, min_samples_leaf=100,
                 l2_regularization=1.0, early_stopping=False)
        p.update(params)
        clf = HistGradientBoostingClassifier(random_state=seed, **p)
        clf.fit(X, y)
        return clf
    import lightgbm as lgb
    p = dict(n_estimators=400, learning_rate=0.05, num_leaves=63, min_child_samples=100,
             subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
             n_jobs=10, deterministic=True, force_row_wise=True, verbose=-1)
    p.update(params)
    clf = lgb.LGBMClassifier(random_state=seed, **p)
    clf.fit(X, y, sample_weight=w)
    return clf


def predict(clf, pairs, cols, chunk=2_000_000):
    out = np.empty(len(pairs), dtype=np.float32)
    for s in range(0, len(pairs), chunk):
        out[s:s + chunk] = clf.predict_proba(pairs[cols].values[s:s + chunk].astype(np.float32))[:, 1]
    return out


def decide(pairs, queries, threshold):
    """Global threshold policy: each S1 gets every candidate with prob >= threshold."""
    sel = pairs[pairs.prob >= threshold]
    pred = {q: [] for q in queries}
    for s, c in zip(sel.s1.values, sel.cand.values):
        pred[s].append(c)
    return pred


def decide_exclusive(pairs, queries, threshold):
    """Global threshold, then one-to-one exclusivity: an S2/S3 record accepted by several S1 is kept
    only for the highest-probability S1 (ties: smallest S1 id). Every S2/S3 record belongs to at most
    one S1 in the training data."""
    acc = pairs[pairs.prob >= threshold][["s1", "cand", "prob"]]
    acc = acc.sort_values(["cand", "prob", "s1"], ascending=[True, False, True], kind="stable")
    win = acc.drop_duplicates("cand", keep="first")
    pred = {q: [] for q in queries}
    for s, c in zip(win.s1.values, win.cand.values):
        pred[s].append(c)
    return pred


def candidates(pairs, queries):
    cand = {q: [] for q in queries}
    for s, c in zip(pairs.s1.values, pairs.cand.values):
        cand[s].append(c)
    return cand


def best_threshold(pairs, queries, truth, grid=None):
    grid = grid if grid is not None else np.round(np.arange(0.05, 0.96, 0.025), 3)
    res = [(t, ev.macro_f05(truth, decide(pairs, queries, t), queries)) for t in grid]
    t_best, f_best = max(res, key=lambda x: x[1])
    return float(t_best), float(f_best), res
