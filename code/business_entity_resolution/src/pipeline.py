"""Candidate generation + pair features for a set of S1 queries, per country partition.

build_pairs(part, query_ids or None, K, max_df, tag, scorer=None) writes
  data/working/pairs/<tag>_<country>.pkl = {"pairs": DataFrame, "queries": [s1 ids]}
If scorer is None the DataFrame holds s1, cand and every feature (for training and
validation). If scorer(features_df) -> prob is given, only s1, cand, score, prob are
kept (for test inference; bounded memory). query_ids=None means every S1 of the
partition. The candidate pairs stored here are EXACTLY the pairs the model scores.
"""
import gc
import time

import numpy as np
import pandas as pd

import blocking
import features
import io_utils as io
import prep

PAIRS = io.WORK / "pairs"


def log(msg, logfile=None):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if logfile:
        with open(logfile, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def _pair_table(ci, cs, src):
    n, k = ci.shape
    qrow = np.repeat(np.arange(n), k)
    rank = np.tile(np.arange(k), n)
    pj = ci.ravel()
    top = np.repeat(cs[:, 0], k)
    second = np.repeat(cs[:, 1] if k > 1 else np.zeros(n, np.float32), k)
    m = pj >= 0
    return pd.DataFrame({"q": qrow[m].astype(np.int64), "pj": pj[m].astype(np.int64), "src": src,
                         "score": cs.ravel()[m], "rank": rank[m].astype(np.int16),
                         "top": top[m], "second": second[m]})


def configure(char_ngrams):
    """Feature version: 1 tokens+name 4-grams, 2 = +word bigrams, 3 = 2 + Indic transliteration v1,
    4 = 2 + transliteration v2 (script-agnostic fixes)."""
    import normalize
    cg = int(char_ngrams)
    normalize.TRANSLIT = 2 if cg >= 4 else (1 if cg == 3 else 0)


def candidates(part, country, q_idx, K, max_df, char_ngrams, recs, logfile=None, full=False):
    """Top-K per source for the given S1 row indices, cached on disk by blocking config and
    query set (full=True: all S1 of the partition, e.g. test inference)."""
    configure(char_ngrams)
    if ("cands", part, country) in OVERRIDE:
        return OVERRIDE[("cands", part, country)]
    if full:
        cpath = PAIRS / f"topk_{part}_{country}_K{K}_df{max_df}_c{int(char_ngrams)}.npz"
    else:
        import hashlib
        qh = hashlib.sha256(np.ascontiguousarray(q_idx, dtype=np.int64).tobytes()).hexdigest()[:12]
        cpath = PAIRS / f"topk_{part}_{country}_q{qh}_K{K}_df{max_df}_c{int(char_ngrams)}.npz"
    if cpath.exists():
        z = np.load(cpath)
        log(f"topk cache hit {cpath.name}", logfile)
        return {src: (z[f"{src}_i"], z[f"{src}_s"]) for src in ("s2", "s3")}
    Xs = prep.matrices(part, country, char_ngrams, recs=recs, log=lambda m: log(m, logfile))
    idf, keep = prep.idf_and_keep(Xs, max_df)
    Q = blocking.weight(Xs["s1"][q_idx], idf, keep)
    cands = {}
    for src in ("s2", "s3"):
        P = blocking.weight(Xs[src], idf, keep)
        PT = P.T.tocsr()
        del P
        PT._is_transposed_pool = True
        t = time.time()
        cands[src] = blocking.topk(Q, PT, K)
        log(f"{part} {country} {src}: top{K} for {len(q_idx)} queries in {time.time()-t:.0f}s", logfile)
        del PT
        gc.collect()
    del Xs, Q
    gc.collect()
    tmp = cpath.with_name(cpath.stem + ".tmp.npz")
    np.savez(tmp, **{f"{s}_{k}": v for s, (i, sc) in cands.items() for k, v in (("i", i), ("s", sc))})
    tmp.replace(cpath)
    return cands


def name_freq(recs):
    """Frequency of each normalised space-free name over all records of the partition (unlabelled)."""
    from collections import Counter

    from normalize import norm, squash
    cnt = Counter()
    for src in ("s1", "s2", "s3"):
        cnt.update(squash(norm(n)) for n in recs[src].business_name.values)
    return cnt


def common_name_tokens(recs, top=200):
    """The partition's most frequent normalised name tokens (legal forms, generic words).
    Unlabelled and computed per partition, so it adapts to an unseen country (e.g. SARL, SAS)."""
    from collections import Counter

    from normalize import norm
    cnt = Counter()
    for src in ("s1", "s2", "s3"):
        for n in recs[src].business_name.values:
            cnt.update(set(norm(n).split()))
    return {t for t, _ in cnt.most_common(top)}


def _core(s, stop):
    toks = [t for t in s.split() if t not in stop and not t.isdigit()]
    return " ".join(toks)


def _first_num(s):
    for t in s.split():
        if t[:1].isdigit():
            return t
    return ""


def add_core(fdf, qn, pn, qa, pa, stop):
    """Core-name similarity (without the partition's common tokens) and house-number agreement."""
    from rapidfuzz import fuzz, process
    qc = [_core(x, stop) for x in qn]
    pc = [_core(x, stop) for x in pn]
    fdf["core_token_set"] = (process.cpdist(qc, pc, scorer=fuzz.token_set_ratio, workers=8, dtype=np.float32) / 100).astype(np.float32)
    fdf["core_ratio"] = (process.cpdist(qc, pc, scorer=fuzz.ratio, workers=8, dtype=np.float32) / 100).astype(np.float32)
    fdf["core_empty"] = np.fromiter(((not a) or (not b) for a, b in zip(qc, pc)), np.float32, len(qc))
    fdf["core_first_eq"] = np.fromiter(((a.split()[:1] == b.split()[:1]) and bool(a) for a, b in zip(qc, pc)), np.float32, len(qc))
    qf = [_first_num(x) for x in qa]
    pf = [_first_num(x) for x in pa]
    fdf["addr_num1_eq"] = np.fromiter((a == b and a != "" for a, b in zip(qf, pf)), np.float32, len(qf))
    fdf["addr_num1_missing"] = np.fromiter((a == "" or b == "" for a, b in zip(qf, pf)), np.float32, len(qf))
    g = fdf.groupby("s1", sort=False)
    fdf["ent_core_rank"] = g.core_token_set.rank(ascending=False, method="min").astype(np.float32)
    return fdf


def add_consistency(fdf, pn, pa):
    """Collective S2-S3 consistency: compare each candidate with the S1's top-scoring candidate
    (by blocking score) from the OTHER source, and record whether that record's own best S1 is this S1."""
    from rapidfuzz import fuzz, process
    n = len(fdf)
    key = fdf.s1.values
    other = np.full(n, -1, dtype=np.int64)
    top_idx = fdf.groupby(["s1", "src_is_s3"], sort=False).block_score.idxmax()
    top_map = {(s, int(t)): i for (s, t), i in top_idx.items()}
    src = fdf.src_is_s3.values.astype(int)
    for i in range(n):
        j = top_map.get((key[i], 1 - src[i]))
        if j is not None:
            other[i] = j
    has = other >= 0
    oi = np.where(has, other, 0)
    fdf["cons_has_other"] = has.astype(np.float32)
    fdf["cons_name"] = np.where(has, process.cpdist(list(pn), list(pn[oi]), scorer=fuzz.token_set_ratio,
                                                    workers=8, dtype=np.float32) / 100, -1).astype(np.float32)
    fdf["cons_addr"] = np.where(has, process.cpdist(list(pa), list(pa[oi]), scorer=fuzz.token_set_ratio,
                                                    workers=8, dtype=np.float32) / 100, -1).astype(np.float32)
    if "rev_rank" in fdf.columns:
        fdf["cons_other_rev_rank"] = np.where(has, fdf.rev_rank.values[oi], -1).astype(np.float32)
    return fdf


def name_token_idf(recs):
    """IDF of normalised name tokens over all records of the partition (unlabelled, per partition)."""
    from collections import Counter

    from normalize import norm
    cnt = Counter()
    n = 0
    for src in ("s1", "s2", "s3"):
        for x in recs[src].business_name.values:
            cnt.update(set(norm(x).split()))
            n += 1
    if IDF_MODE == "pct":
        # rarity percentile within the partition, occurrence-weighted: share of token occurrences
        # that are MORE frequent than this token (0 = most common token, ~1 = rarest). Bounded
        # and comparable across partitions/countries.
        items = sorted(cnt.items(), key=lambda kv: -kv[1])
        total = float(sum(cnt.values()))
        w, cum = {}, 0.0
        for t, c in items:
            w[t] = cum / total
            cum += c
        return w, 1.0
    return {t: float(np.log((n + 1) / (c + 1)) + 1) for t, c in cnt.items()}, float(np.log(n + 1) + 1)


def _idf_side(a_toks, b_toks, idf, idf_max, fuzz_ratio):
    """IDF share of a's tokens matched (exactly, or fuzzily >= 85) in b, and the max IDF unmatched."""
    tot = got = 0.0
    worst = 0.0
    bset = set(b_toks)
    for t in a_toks:
        w = idf.get(t, idf_max)
        tot += w
        if t in bset or any(fuzz_ratio(t, u) >= FUZZ_T for u in b_toks if abs(len(u) - len(t)) <= 2):
            got += w
        elif w > worst:
            worst = w
    return (got / tot if tot else 0.0), worst / idf_max


def add_idf_name(fdf, qn, pn, idf_pack):
    """Soft TF-IDF style name agreement features (country-agnostic, transfer-safe weights)."""
    from rapidfuzz.fuzz import ratio
    idf, idf_max = idf_pack
    n = len(qn)
    qs = np.empty(n, np.float32)
    ps = np.empty(n, np.float32)
    qw = np.empty(n, np.float32)
    pw = np.empty(n, np.float32)
    for i in range(n):
        a, b = qn[i].split(), pn[i].split()
        qs[i], qw[i] = _idf_side(a, b, idf, idf_max, ratio)
        ps[i], pw[i] = _idf_side(b, a, idf, idf_max, ratio)
    fdf["idf_match_q"] = qs
    fdf["idf_match_p"] = ps
    fdf["idf_unmatched_max_q"] = qw
    fdf["idf_unmatched_max_p"] = pw
    fdf["idf_match_min"] = np.minimum(qs, ps)
    return fdf


def add_context(fdf, qn_sq, pn_sq, freq):
    """Entity-context features (country-agnostic): name ambiguity and within-entity competition."""
    fdf["q_name_freq"] = np.log1p(np.fromiter((freq.get(x, 0) for x in qn_sq), np.float32, len(qn_sq)))
    fdf["p_name_freq"] = np.log1p(np.fromiter((freq.get(x, 0) for x in pn_sq), np.float32, len(pn_sq)))
    fdf["_nc"] = (fdf.name_token_set >= 0.95).astype(np.float32)
    fdf["_ac"] = (fdf.addr_token_set >= 0.9).astype(np.float32)
    g = fdf.groupby("s1", sort=False)
    fdf["ent_n_name_close"] = g._nc.transform("sum").astype(np.float32)
    fdf["ent_n_addr_close"] = g._ac.transform("sum").astype(np.float32)
    fdf.drop(columns=["_nc", "_ac"], inplace=True)
    fdf["ent_name_rank"] = g.name_token_set.rank(ascending=False, method="min").astype(np.float32)
    fdf["ent_addr_rank"] = g.addr_token_set.rank(ascending=False, method="min").astype(np.float32)
    fdf["ent_name_max_other"] = (g.name_token_set.transform("max") - fdf.name_token_set).astype(np.float32)
    return fdf


REV_K = 3
FUZZ_T = 85  # fuzzy token-match threshold for the IDF name features; cfg "idfn_fuzz"
IDF_MODE = "idf"  # "idf" (exp_019) or "pct" (partition rarity percentile, exp_020); set via cfg "idfn_mode"
OVERRIDE = {}  # stress evaluations only: {("cands"|"rev", part, country): precomputed arrays}
REV_COLS = ["rev_rank", "rev_top", "rev_gap", "rev_top_second_gap", "in_forward_k", "in_reverse_r"]


def reverse_all(part, country, K_rev, max_df, char_ngrams, recs, logfile=None):
    """Reverse retrieval: for EVERY S2/S3 record of the partition, its top-K_rev S1 records
    (same country, same features/IDF; cosine is symmetric so scores are comparable).
    Cached on disk. Returns {src: (ri [n_pool, K_rev] S1 row idx or -1, rs scores)}."""
    configure(char_ngrams)
    if ("rev", part, country) in OVERRIDE:
        return OVERRIDE[("rev", part, country)]
    cpath = PAIRS / f"rev_{part}_{country}_R{K_rev}_df{max_df}_c{int(char_ngrams)}.npz"
    if cpath.exists():
        z = np.load(cpath)
        log(f"reverse cache hit {cpath.name}", logfile)
        return {src: (z[f"{src}_i"], z[f"{src}_s"]) for src in ("s2", "s3")}
    Xs = prep.matrices(part, country, char_ngrams, recs=recs, log=lambda m: log(m, logfile))
    idf, keep = prep.idf_and_keep(Xs, max_df)
    S1T = blocking.weight(Xs["s1"], idf, keep).T.tocsr()
    S1T._is_transposed_pool = True
    out = {}
    for src in ("s2", "s3"):
        t = time.time()
        Q = blocking.weight(Xs[src], idf, keep)
        out[src] = blocking.topk(Q, S1T, K_rev)
        del Q
        gc.collect()
        log(f"{part} {country} {src}: reverse top{K_rev} for {Xs[src].shape[0]} records in {time.time()-t:.0f}s", logfile)
    del Xs, S1T
    gc.collect()
    tmp = cpath.with_name(cpath.stem + ".tmp.npz")
    np.savez(tmp, **{f"{s}_{k}": v for s, (i, sc) in out.items() for k, v in (("i", i), ("s", sc))})
    tmp.replace(cpath)
    return out


def _union_table(ci, cs, rev, q_rows_chunk, pos_lo, rev_sorted, Kf, r, src, cap=None):
    """Candidate pairs for one chunk of queries and one source: forward top-Kf UNION reverse top-r.
    ci/cs: forward top-K arrays for the chunk (K >= Kf); q_rows_chunk: S1 row index of each query;
    rev: (ri, rs) reverse arrays for all pool records; rev_sorted: (qpos, pj, rrank) of all reverse
    pairs pointing to a query, sorted by qpos; pos_lo: first query position of the chunk."""
    n, K = ci.shape
    ri, rs = rev
    parts = []
    # forward part
    if Kf > 0:
        qf = np.repeat(np.arange(n), Kf)
        pf = ci[:, :Kf].ravel()
        m = pf >= 0
        parts.append(np.stack([qf[m], pf[m]], 1))
    # reverse part
    if r > 0:
        qpos, pj, _ = rev_sorted
        a, b = np.searchsorted(qpos, [pos_lo, pos_lo + n])
        parts.append(np.stack([qpos[a:b] - pos_lo, pj[a:b]], 1))
    qp = np.concatenate(parts) if parts else np.zeros((0, 2), np.int64)
    qp = np.unique(qp.astype(np.int64), axis=0)
    q, pj = qp[:, 0], qp[:, 1]
    # forward rank / score (from the full forward top-K list), K if absent
    eq = ci[q] == pj[:, None]
    has_f = eq.any(1)
    frank = np.where(has_f, eq.argmax(1), K).astype(np.int16)
    fscore = np.where(has_f, cs[q, np.minimum(eq.argmax(1), K - 1)], 0).astype(np.float32)
    # reverse rank of this S1 in the record's list, R if absent
    s1row = q_rows_chunk[q]
    req = ri[pj] == s1row[:, None]
    has_r = req.any(1)
    rrank = np.where(has_r, req.argmax(1), ri.shape[1]).astype(np.int16)
    rscore = np.where(has_r, rs[pj, np.minimum(req.argmax(1), ri.shape[1] - 1)], 0).astype(np.float32)
    score = np.where(has_f, fscore, rscore).astype(np.float32)  # cosine is symmetric
    rev_top = rs[pj, 0].astype(np.float32)
    rev_second = (rs[pj, 1] if rs.shape[1] > 1 else np.zeros(len(pj))).astype(np.float32)
    if cap:
        # keep at most `cap` candidates per query and source, by blocking score (deterministic)
        order = np.lexsort((-score, q))
        q, pj, score, frank, rrank, rev_top, rev_second = (x[order] for x in (q, pj, score, frank, rrank, rev_top, rev_second))
        first = np.r_[0, np.flatnonzero(q[1:] != q[:-1]) + 1]
        within = np.arange(len(q)) - np.repeat(first, np.diff(np.r_[first, len(q)]))
        k = within < cap
        q, pj, score, frank, rrank, rev_top, rev_second = (x[k] for x in (q, pj, score, frank, rrank, rev_top, rev_second))
    return pd.DataFrame({
        "q": q, "pj": pj, "src": src, "score": score,
        "rank": frank, "top": cs[q, 0], "second": cs[q, 1] if K > 1 else 0,
        "rev_rank": rrank.astype(np.float32), "rev_top": rev_top,
        "rev_gap": (rev_top - score).astype(np.float32),
        "rev_top_second_gap": (rev_top - rev_second).astype(np.float32),
        "in_forward_k": (frank < Kf).astype(np.float32), "in_reverse_r": (rrank < r).astype(np.float32),
    })


def build_pairs(part, query_ids, K, max_df, tag, char_ngrams=True, logfile=None,
                scorer=None, qchunk=60_000, ctx=False, core=False, Kf=None, rev_r=0, cap=None, cons=False,
                idfn=False):
    """Kf: forward candidates kept per source (default K, all); rev_r: reverse top-r union (0 = off).
    With Kf < K or rev_r > 0 the candidate set is forward top-Kf UNION reverse top-r, and the
    one-to-one features REV_COLS are added."""
    PAIRS.mkdir(parents=True, exist_ok=True)
    configure(char_ngrams)
    written = []
    qset = set(query_ids) if query_ids is not None else None
    for country in prep.countries(part):
        out_path = PAIRS / f"{tag}_{country}.pkl"
        if out_path.exists():
            written.append(out_path)
            log(f"{tag} {country}: cached", logfile)
            continue
        t0 = time.time()
        recs = prep.records(part, country)
        s1 = recs["s1"]
        q_idx = np.arange(len(s1)) if qset is None else np.flatnonzero(s1.entity_id.isin(qset).values)
        if len(q_idx) == 0:
            continue
        cands = candidates(part, country, q_idx, K, max_df, char_ngrams, recs, logfile,
                           full=qset is None)
        q_ids = s1.entity_id.values[q_idx]
        union = rev_r > 0 or (Kf is not None and Kf < K) or bool(cap)
        rev, rev_sorted = None, {}
        if union:
            Kf_ = K if Kf is None else Kf
            Kf = Kf_
            rev = reverse_all(part, country, REV_K, max_df, char_ngrams, recs, logfile)
            pos = np.full(len(s1), -1, dtype=np.int64)
            pos[q_idx] = np.arange(len(q_idx))
            for src in ("s2", "s3"):
                ri = rev[src][0][:, :rev_r] if rev_r > 0 else np.zeros((0, 0), np.int32)
                pj = np.repeat(np.arange(ri.shape[0]), ri.shape[1])
                rr = ri.ravel()
                ok = rr >= 0
                qp = np.where(ok, pos[np.where(ok, rr, 0)], -1)
                m = qp >= 0
                order = np.argsort(qp[m], kind="stable")
                rev_sorted[src] = (qp[m][order], pj[m][order], None)
        freq = name_freq(recs) if ctx else None
        idf_pack = name_token_idf(recs) if idfn else None
        stop = common_name_tokens(recs) if core else None
        outs = []
        for cs_ in range(0, len(q_idx), qchunk):
            ce = min(cs_ + qchunk, len(q_idx))
            qrec = s1.iloc[q_idx[cs_:ce]]
            QM = features.record_matrices(qrec.business_name.tolist(), qrec.business_address.tolist())
            chunk_parts, qsq, psq, strs = [], [], [], {"qn": [], "pn": [], "qa": [], "pa": []}
            for src in ("s2", "s3"):
                ci, cs = cands[src]
                if union:
                    sub = _union_table(ci[cs_:ce], cs[cs_:ce], rev[src], q_idx[cs_:ce], cs_, rev_sorted[src],
                                       Kf, rev_r, src, cap)
                else:
                    sub = _pair_table(ci[cs_:ce], cs[cs_:ce], src)
                if len(sub) == 0:
                    continue
                uniq, inv = np.unique(sub.pj.values, return_inverse=True)
                prec = recs[src].iloc[uniq]
                PM = features.record_matrices(prec.business_name.tolist(), prec.business_address.tolist())
                F, names = features.pair_features(
                    QM, PM, sub.q.values, inv.astype(np.int64), sub.score.values, sub["rank"].values,
                    np.full(len(sub), src == "s3"), sub.top.values, sub.second.values)
                fdf = pd.DataFrame(F, columns=names)
                fdf.insert(0, "s1", q_ids[cs_:ce][sub.q.values])
                fdf.insert(1, "cand", prec.entity_id.values[inv])
                if union:
                    for col in REV_COLS:
                        fdf[col] = sub[col].values
                chunk_parts.append(fdf)
                qsq.append(QM["_nsq"][sub.q.values])
                psq.append(PM["_nsq"][inv])
                if core or cons or idfn:
                    strs["qn"].append(QM["_nn"][sub.q.values])
                    strs["pn"].append(PM["_nn"][inv])
                    strs["qa"].append(QM["_na"][sub.q.values])
                    strs["pa"].append(PM["_na"][inv])
                del PM, F
            if chunk_parts:
                fdf = pd.concat(chunk_parts, ignore_index=True)
                if ctx:
                    fdf = add_context(fdf, np.concatenate(qsq), np.concatenate(psq), freq)
                if core:
                    fdf = add_core(fdf, *(np.concatenate(strs[k]) for k in ("qn", "pn", "qa", "pa")), stop)
                if cons:
                    fdf = add_consistency(fdf, np.concatenate(strs["pn"]), np.concatenate(strs["pa"]))
                if idfn:
                    fdf = add_idf_name(fdf, np.concatenate(strs["qn"]), np.concatenate(strs["pn"]), idf_pack)
                if scorer is not None:
                    fdf = pd.DataFrame({"s1": fdf.s1.values, "cand": fdf.cand.values,
                                        "score": fdf.block_score.values, "prob": scorer(fdf)})
                outs.append(fdf)
            del QM, chunk_parts
            gc.collect()
            log(f"{tag} {country}: features {ce}/{len(q_idx)} queries, free={io.free_ram_gb():.1f}GB", logfile)
        out = pd.concat(outs, ignore_index=True)
        pd.to_pickle({"pairs": out, "queries": q_ids.tolist()}, out_path)
        written.append(out_path)
        log(f"{tag} {country}: {len(q_idx)} queries, {len(out)} pairs, {time.time()-t0:.0f}s, free={io.free_ram_gb():.1f}GB", logfile)
        del recs, out, outs, cands
        gc.collect()
    return written


def pair_opts(cfg):
    """Pair-building options from an experiment config."""
    return dict(ctx=bool(cfg.get("ctx", False)), core=bool(cfg.get("core", False)), Kf=cfg.get("Kf"),
                rev_r=int(cfg.get("rev_r", 0)), cap=cfg.get("cap"), cons=bool(cfg.get("cons", False)),
                idfn=bool(cfg.get("idfn", False)))


def pair_tag(prefix, cfg):
    """Cache tag for the pairs built with this config (must match across validation and test).
    Also applies module-level modes from the config (IDF_MODE)."""
    global IDF_MODE, FUZZ_T
    IDF_MODE = cfg.get("idfn_mode", "idf")
    FUZZ_T = int(cfg.get("idfn_fuzz", 85))
    o = pair_opts(cfg)
    return (f"{prefix}_K{cfg['K']}_df{cfg['max_df']}_c{int(cfg.get('char_ngrams', True))}"
            + ("_x1" if o["ctx"] else "") + ("_k1" if o["core"] else "")
            + (f"_f{o['Kf']}r{o['rev_r']}" if (o["rev_r"] or o["Kf"] is not None) else "")
            + (f"_cap{o['cap']}" if o["cap"] else "") + ("_s23" if o["cons"] else "")
            + (("_idf" if IDF_MODE == "idf" else "_idfp") + ("" if FUZZ_T == 85 else f"f{FUZZ_T}")
               if o["idfn"] else ""))


def load_pairs(paths):
    ps, qs = [], []
    for p in paths:
        d = pd.read_pickle(p)
        ps.append(d["pairs"])
        qs += d["queries"]
    return pd.concat(ps, ignore_index=True), qs
