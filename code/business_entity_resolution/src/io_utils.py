"""Data loading and caching.

All TSVs are read with sep="\t", every column as str, no quoting and no NA parsing
(so names such as "NA" or "NULL" stay literal strings). Each file is converted once
to a pickle cache in data/working/cache/, keyed by the file's SHA-256 prefix.
"""
import csv
import hashlib
import pathlib

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "original" / "student_resource" / "dataset"
WORK = ROOT / "data" / "working"
# Optional override (used by the packaged copy): <root>/paths.json with
# {"dataset_dir": "<folder containing train/ and test/>", "work_dir": "<cache folder>"}
_paths = ROOT / "paths.json"
if _paths.exists():
    import json as _json
    _p = _json.loads(_paths.read_text())
    DATA = pathlib.Path(_p.get("dataset_dir", DATA))
    WORK = pathlib.Path(_p.get("work_dir", WORK))
CACHE = WORK / "cache"

FILES = {
    "train_s1": DATA / "train" / "train_source1.tsv",
    "train_s2": DATA / "train" / "train_source2.tsv",
    "train_s3": DATA / "train" / "train_source3.tsv",
    "train_gt": DATA / "train" / "train_ground_truth.tsv",
    "test_s1": DATA / "test" / "test_source1.tsv",
    "test_s2": DATA / "test" / "test_source2.tsv",
    "test_s3": DATA / "test" / "test_source3.tsv",
}


def free_ram_gb():
    """Free physical memory in GB (Windows via ctypes; None elsewhere if unavailable)."""
    try:
        import ctypes

        class MS(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        m = MS()
        m.dwLength = ctypes.sizeof(MS)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        return m.ullAvailPhys / 1e9
    except Exception:
        return None


def peak_rss_gb():
    """Peak working set of this process in GB (Windows), None if unavailable."""
    try:
        import ctypes
        from ctypes import wintypes

        class PMC(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
        c = PMC()
        c.cb = ctypes.sizeof(PMC)
        k32 = ctypes.windll.kernel32
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        k32.K32GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
        if not k32.K32GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(c), c.cb):
            return None
        return c.PeakWorkingSetSize / 1e9
    except Exception:
        return None


def file_hash(path, n=16):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()[:n]


def read_tsv(path):
    return pd.read_csv(path, sep="\t", dtype=str, quoting=csv.QUOTE_NONE,
                       keep_default_na=False, na_filter=False, encoding="utf-8")


def load(key):
    """Load one file as a DataFrame of str columns, using the pickle cache."""
    path = FILES[key]
    CACHE.mkdir(parents=True, exist_ok=True)
    hkey = CACHE / f"{key}.hash"
    pkl = CACHE / f"{key}.pkl"
    if pkl.exists() and hkey.exists():
        return pd.read_pickle(pkl)
    df = read_tsv(path)
    df.to_pickle(pkl)
    hkey.write_text(file_hash(path))
    return df


def parse_gt(gt):
    """Ground truth -> dict s1_id -> list of matched ids (possibly empty)."""
    out = {}
    for s1, m in zip(gt["source1_entity_id"], gt["matched_entity_ids"]):
        out[s1] = [x for x in m.split(",") if x] if m else []
    return out
