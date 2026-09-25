"""Candidate generation (blocking).

Within each country partition (matched pairs always share the country label in
training), every record becomes a sparse vector of hashed features:
  n:<name token>, a:<address token>, and optionally c:<4-gram of the space-free name>.
Weights are IDF computed on the unlabelled records of the same partition (queries +
pool). Features whose pool document frequency exceeds max_df are dropped (they make
the sparse product expensive and carry little identity information). Rows are
L2-normalised; the score is the cosine. For each S1 entity we keep the top-K S2 and
the top-K S3 records separately (K per source), dropping zero scores.

Nothing here uses labels; no country name is a feature.
"""
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import HashingVectorizer

from normalize import norm, squash

N_FEATURES = 2 ** 22


def make_analyzer(char_ngrams=True, ng=4):
    """char_ngrams: 0 = tokens only, 1 = + name char 4-grams, 2 = + word bigrams of name and address."""
    def analyze(rec):
        name, addr = rec
        nn = norm(name)
        na = norm(addr)
        nt, at = nn.split(), na.split()
        feats = ["n:" + t for t in set(nt)]
        feats += ["a:" + t for t in set(at)]
        if char_ngrams >= 2:
            feats += list({"nb:" + a + "_" + b for a, b in zip(nt, nt[1:])})
            feats += list({"ab:" + a + "_" + b for a, b in zip(at, at[1:])})
        if char_ngrams:
            sq = squash(nn)
            if len(sq) >= ng:
                feats += ["c:" + g for g in {sq[i:i + ng] for i in range(len(sq) - ng + 1)}]
            elif sq:
                feats.append("c:" + sq)
        return feats
    return analyze


def vectorize(names, addrs, char_ngrams=True, chunk=300_000):
    hv = HashingVectorizer(n_features=N_FEATURES, analyzer=make_analyzer(char_ngrams),
                           alternate_sign=False, norm=None, binary=True, dtype=np.float32)
    names, addrs = list(names), list(addrs)
    parts = []
    for s in range(0, len(names), chunk):
        X = hv.transform(list(zip(names[s:s + chunk], addrs[s:s + chunk]))).tocsr()
        X.indices = X.indices.astype(np.int32)
        X.indptr = X.indptr.astype(np.int64)
        parts.append(X)
    return sp.vstack(parts, format="csr") if parts else sp.csr_matrix((0, N_FEATURES), dtype=np.float32)


def doc_freq(X):
    return np.bincount(X.indices, minlength=X.shape[1]).astype(np.int64)


def weight(X, idf, keep):
    """Apply idf * keep-mask to binary X and L2-normalise rows."""
    w = (idf * keep).astype(np.float32)
    X = X.tocsr(copy=True)
    X.data = X.data * w[X.indices]
    X.eliminate_zeros()
    nrm = np.sqrt(np.asarray(X.multiply(X).sum(axis=1)).ravel())
    nrm[nrm == 0] = 1.0
    return sp.diags(1.0 / nrm).dot(X).tocsr().astype(np.float32)


def topk(Q, P, k, chunk=4000):
    """Top-k columns of Q @ P.T per row. Returns (idx int32 [n,k], score float32 [n,k]),
    idx=-1 where fewer than k non-zero scores exist."""
    n = Q.shape[0]
    out_i = np.full((n, k), -1, dtype=np.int32)
    out_s = np.zeros((n, k), dtype=np.float32)
    PT = P if getattr(P, "_is_transposed_pool", False) else P.T.tocsr()
    for start in range(0, n, chunk):
        R = (Q[start:start + chunk] @ PT).tocsr()
        indptr, ind, dat = R.indptr, R.indices, R.data
        for r in range(R.shape[0]):
            a, b = indptr[r], indptr[r + 1]
            if a == b:
                continue
            d = dat[a:b]
            if b - a > k:
                sel = np.argpartition(-d, k - 1)[:k]
            else:
                sel = np.arange(b - a)
            sel = sel[np.argsort(-d[sel], kind="stable")]
            out_i[start + r, :len(sel)] = ind[a:b][sel]
            out_s[start + r, :len(sel)] = d[sel]
    return out_i, out_s
