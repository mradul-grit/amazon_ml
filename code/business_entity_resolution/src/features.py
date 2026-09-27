"""Country-agnostic pair features, computed with sparse row-wise products (scales to
tens of millions of pairs). No feature uses the country name.

Per record we build binary hashed matrices:
  NT name word tokens, NC char 3-grams of the space-free name,
  AT address word tokens, AC char 3-grams of the space-free address, AD address digit tokens.
For a pair (i, j): inter = <X_i, X_j>, sizes |X_i|, |X_j| -> jaccard, containment
(inter / min size), dice-style cosine for binary vectors.
"""
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import HashingVectorizer

from normalize import norm, squash, digits

NF = 2 ** 20


def _grams(s, n=3):
    if len(s) < n:
        return [s] if s else []
    return list({s[i:i + n] for i in range(len(s) - n + 1)})


ANALYZERS = {
    "NT": lambda r: list(set(r[0].split())),
    "NC": lambda r: _grams(r[2]),
    "AT": lambda r: list(set(r[1].split())),
    "AC": lambda r: _grams(r[3]),
    "AD": lambda r: list(set(digits(r[1]))),
}


def record_matrices(names, addrs):
    nn = [norm(n) for n in names]
    na = [norm(a) for a in addrs]
    nsq = [squash(x) for x in nn]
    recs = list(zip(nn, na, nsq, [squash(x) for x in na]))
    out = {}
    for k, an in ANALYZERS.items():
        hv = HashingVectorizer(n_features=NF, analyzer=an, alternate_sign=False, norm=None,
                               binary=True, dtype=np.float32)
        X = hv.transform(recs).tocsr()
        X.indices = X.indices.astype(np.int32)
        out[k] = X
    out["_nlen"] = np.array([len(x) for x in nn], dtype=np.float32)
    out["_alen"] = np.array([len(x) for x in na], dtype=np.float32)
    out["_nascii"] = np.array([sum(c.isascii() for c in n) / max(len(n), 1) for n in names], dtype=np.float32)
    out["_nsq"] = np.array(nsq, dtype=object)
    out["_nn"] = np.array(nn, dtype=object)
    out["_na"] = np.array(na, dtype=object)
    return out


def _fuzz(a, b, scorer, sort_tokens=False):
    """Element-wise RapidFuzz similarity (0..1) for aligned string arrays."""
    from rapidfuzz import process
    return (process.cpdist(list(a), list(b), scorer=scorer, workers=8, dtype=np.float32) / 100.0).astype(np.float32)


def _rowdot(A, B):
    return np.asarray(A.multiply(B).sum(axis=1)).ravel().astype(np.float32)


def _rowsum(A):
    return np.diff(A.indptr).astype(np.float32)


FEATURE_NAMES = None


def pair_features(QM, PM, qi, pj, block_score, rank, src_is_s3, top_score, second_score, chunk=500_000):
    """QM, PM: record_matrices of queries and pool; qi, pj: int arrays of pair indices."""
    global FEATURE_NAMES
    cols = []
    names = []
    n = len(qi)
    feats = []
    for s in range(0, n, chunk):
        a, b = qi[s:s + chunk], pj[s:s + chunk]
        f = {}
        for k in ("NT", "NC", "AT", "AC", "AD"):
            A = QM[k][a]
            B = PM[k][b]
            inter = _rowdot(A, B)
            sa, sb = _rowsum(A), _rowsum(B)
            union = sa + sb - inter
            f[k + "_jac"] = np.where(union > 0, inter / np.maximum(union, 1), 0).astype(np.float32)
            f[k + "_cont"] = np.where(np.minimum(sa, sb) > 0, inter / np.maximum(np.minimum(sa, sb), 1), 0).astype(np.float32)
            f[k + "_cos"] = np.where(sa * sb > 0, inter / np.sqrt(np.maximum(sa * sb, 1)), 0).astype(np.float32)
            f[k + "_inter"] = inter
            f[k + "_empty"] = ((sa == 0) | (sb == 0)).astype(np.float32)
        nl_a, nl_b = QM["_nlen"][a], PM["_nlen"][b]
        f["name_len_ratio"] = (np.minimum(nl_a, nl_b) / np.maximum(np.maximum(nl_a, nl_b), 1)).astype(np.float32)
        al_a, al_b = QM["_alen"][a], PM["_alen"][b]
        f["addr_len_ratio"] = (np.minimum(al_a, al_b) / np.maximum(np.maximum(al_a, al_b), 1)).astype(np.float32)
        f["addr_empty_pool"] = (al_b == 0).astype(np.float32)
        f["name_ascii_q"] = QM["_nascii"][a]
        f["name_ascii_p"] = PM["_nascii"][b]
        f["name_ascii_diff"] = np.abs(f["name_ascii_q"] - f["name_ascii_p"])
        f["name_sq_equal"] = (QM["_nsq"][a] == PM["_nsq"][b]).astype(np.float32)
        from rapidfuzz import fuzz
        qn, pn = QM["_nn"][a], PM["_nn"][b]
        qa, pa = QM["_na"][a], PM["_na"][b]
        f["name_ratio"] = _fuzz(qn, pn, fuzz.ratio)
        f["name_token_sort"] = _fuzz(qn, pn, fuzz.token_sort_ratio)
        f["name_token_set"] = _fuzz(qn, pn, fuzz.token_set_ratio)
        f["name_partial"] = _fuzz(qn, pn, fuzz.partial_ratio)
        f["name_sq_ratio"] = _fuzz(QM["_nsq"][a], PM["_nsq"][b], fuzz.ratio)
        f["addr_token_sort"] = _fuzz(qa, pa, fuzz.token_sort_ratio)
        f["addr_token_set"] = _fuzz(qa, pa, fuzz.token_set_ratio)
        f["addr_partial"] = _fuzz(qa, pa, fuzz.partial_ratio)
        f["block_score"] = block_score[s:s + chunk].astype(np.float32)
        f["block_rank"] = rank[s:s + chunk].astype(np.float32)
        f["block_rel_top"] = (block_score[s:s + chunk] / np.maximum(top_score[s:s + chunk], 1e-6)).astype(np.float32)
        f["block_gap_top"] = (top_score[s:s + chunk] - block_score[s:s + chunk]).astype(np.float32)
        f["block_top_second_gap"] = (top_score[s:s + chunk] - second_score[s:s + chunk]).astype(np.float32)
        f["src_is_s3"] = src_is_s3[s:s + chunk].astype(np.float32)
        if names == []:
            names = list(f.keys())
        feats.append(np.column_stack([f[k] for k in names]))
    FEATURE_NAMES = names
    return (np.vstack(feats) if feats else np.zeros((0, len(names)), np.float32)), names
