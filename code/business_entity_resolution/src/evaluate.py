"""Official metric: per-S1-entity F0.5, macro-averaged over all S1 entities.

Per entity (truth T, prediction P, both de-duplicated sets):
  - T empty: 1.0 if P empty else 0.0                       (Problem_Statement: singletons)
  - T non-empty, P empty: 0.0                               (recall 0 -> F = 0)
  - else: prec = |P&T|/|P|, rec = |P&T|/|T|,
          F0.5 = 1.25*prec*rec / (0.25*prec + rec), 0 if prec+rec == 0.
Entities missing from the predictions are scored as empty predictions (the real
scorer would reject such a file; our harness scores them so bugs show up as losses).
Duplicate predicted IDs are de-duplicated (the real scorer rejects them).
"""
import numpy as np

BETA2 = 0.25


def entity_f05(truth, pred):
    t = set(truth)
    p = set(pred)
    if not t:
        return 1.0 if not p else 0.0
    if not p:
        return 0.0
    tp = len(t & p)
    if tp == 0:
        return 0.0
    prec = tp / len(p)
    rec = tp / len(t)
    return (1 + BETA2) * prec * rec / (BETA2 * prec + rec)


def per_entity(truth, pred, entities=None):
    """truth, pred: dict s1_id -> iterable of ids. Returns np.array of per-entity F0.5
    over `entities` (default: all keys of truth), in that order."""
    ents = list(truth.keys()) if entities is None else list(entities)
    return np.array([entity_f05(truth.get(e, ()), pred.get(e, ())) for e in ents], dtype=np.float64)


def macro_f05(truth, pred, entities=None):
    s = per_entity(truth, pred, entities)
    return float(s.mean()) if len(s) else float("nan")


def report(truth, pred, candidates=None, entities=None, pool_size=None):
    """Full report: macro F0.5, micro precision/recall, singleton accuracy,
    candidate recall, average/max candidates per S1, reduction ratio."""
    ents = list(truth.keys()) if entities is None else list(entities)
    scores = per_entity(truth, pred, ents)
    tp = fp = fn = 0
    sing_n = sing_ok = 0
    for e in ents:
        t, p = set(truth.get(e, ())), set(pred.get(e, ()))
        tp += len(t & p)
        fp += len(p - t)
        fn += len(t - p)
        if not t:
            sing_n += 1
            sing_ok += int(not p)
    out = {
        "n_entities": len(ents),
        "macro_f05": float(scores.mean()) if len(ents) else float("nan"),
        "precision_micro": tp / (tp + fp) if tp + fp else 1.0,
        "recall_micro": tp / (tp + fn) if tp + fn else 1.0,
        "singleton_n": sing_n,
        "singleton_accuracy": sing_ok / sing_n if sing_n else float("nan"),
        "pred_nonempty_rate": float(np.mean([bool(pred.get(e)) for e in ents])) if ents else float("nan"),
    }
    if candidates is not None:
        ctp = ctot = 0
        sizes = []
        for e in ents:
            c = set(candidates.get(e, ()))
            t = set(truth.get(e, ()))
            ctp += len(t & c)
            ctot += len(t)
            sizes.append(len(c))
        sizes = np.array(sizes)
        out.update({
            "candidate_recall": ctp / ctot if ctot else 1.0,
            "avg_candidates": float(sizes.mean()) if len(sizes) else 0.0,
            "max_candidates": int(sizes.max()) if len(sizes) else 0,
        })
        if pool_size:
            out["reduction_ratio"] = 1.0 - float(sizes.sum()) / (len(ents) * pool_size)
    return out


def bootstrap_se(scores, n_boot=1000, seed=0):
    """Bootstrap standard error of the mean over entities."""
    rng = np.random.default_rng(seed)
    n = len(scores)
    means = np.array([scores[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    return float(means.std(ddof=1))


def paired_diff_se(scores_a, scores_b, n_boot=1000, seed=0):
    """Mean and bootstrap SE of the paired per-entity difference a - b."""
    d = np.asarray(scores_a) - np.asarray(scores_b)
    return float(d.mean()), bootstrap_se(d, n_boot, seed)
