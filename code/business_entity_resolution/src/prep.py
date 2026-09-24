"""Per (partition, country) record tables and binary blocking matrices, cached on disk.

partition is "train" or "test". Countries are read from the data (open set).
"""
import gc
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp

import blocking
import io_utils as io

VEC = io.WORK / "vec"
SOURCES = ("s1", "s2", "s3")


def countries(part):
    return sorted(io.load(f"{part}_s1").country.unique().tolist())


def records(part, country):
    """Return {src: DataFrame(entity_id, business_name, business_address)} for one country."""
    out = {}
    for src in SOURCES:
        d = io.load(f"{part}_{src}")
        out[src] = d.loc[d.country == country, ["entity_id", "business_name", "business_address"]].reset_index(drop=True)
        del d
        gc.collect()
    return out


def matrices(part, country, char_ngrams=True, recs=None, log=print):
    """Binary hashed feature matrices per source (cached)."""
    VEC.mkdir(parents=True, exist_ok=True)
    tag = f"{part}_{country}_c{int(char_ngrams)}"
    paths = {src: VEC / f"{tag}_{src}.npz" for src in SOURCES}
    if all(p.exists() for p in paths.values()):
        return {src: sp.load_npz(p).tocsr() for src, p in paths.items()}
    recs = recs or records(part, country)
    import pipeline
    pipeline.configure(char_ngrams)
    out = {}
    for src in SOURCES:
        if paths[src].exists():
            out[src] = sp.load_npz(paths[src]).tocsr()
            continue
        t = time.time()
        X = blocking.vectorize(recs[src].business_name, recs[src].business_address, char_ngrams)
        tmp = paths[src].with_name(paths[src].stem + ".tmp.npz")
        sp.save_npz(tmp, X, compressed=False)
        tmp.replace(paths[src])
        out[src] = X
        log(f"vectorized {tag} {src}: {X.shape[0]} rows nnz={X.nnz} {time.time()-t:.0f}s free={io.free_ram_gb():.1f}GB")
    return out


def idf_and_keep(Xs, max_df):
    """IDF over all records of the partition (unlabelled); keep features with pool df <= max_df."""
    df_all = sum(blocking.doc_freq(X) for X in Xs.values())
    df_pool = blocking.doc_freq(Xs["s2"]) + blocking.doc_freq(Xs["s3"])
    n = sum(X.shape[0] for X in Xs.values())
    idf = (np.log((n + 1) / (df_all + 1)) + 1).astype(np.float32)
    keep = ((df_pool <= max_df) & (df_all > 0)).astype(np.float32)
    return idf, keep
