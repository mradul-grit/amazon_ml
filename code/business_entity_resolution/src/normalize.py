"""Country-agnostic string normalisation (no country-specific rules)."""
import re
import unicodedata

_NON_WORD = re.compile(r"[^\w]+", re.UNICODE)
_US = re.compile(r"_+")
_WS = re.compile(r"\s+")
_DIGITS = re.compile(r"\d+")


def strip_accents(s):
    """NFKD + drop combining marks (é -> e). Marks in Indic scripts are dropped too."""
    s = unicodedata.normalize("NFKD", s)
    return "".join(ch for ch in s if not unicodedata.combining(ch))


TRANSLIT = False  # set by pipeline.configure(): 0/False none, 1 = translit_v1, 2 = translit (v2)


def norm(s):
    """Lowercase, strip accents, '&' -> 'and', punctuation -> space, collapse spaces."""
    if not s:
        return ""
    if TRANSLIT == 2:
        from translit import translit
        s = translit(s)
    elif TRANSLIT:
        from translit_v1 import translit as translit1
        s = translit1(s)
    s = strip_accents(s.lower()).replace("&", " and ")
    s = _NON_WORD.sub(" ", s)
    s = _US.sub(" ", s)
    return _WS.sub(" ", s).strip()


def tokens(s):
    return s.split() if s else []


def digits(s):
    return _DIGITS.findall(s) if s else []


def squash(s):
    """Normalised string with spaces removed (handles 'colonialfellowship.com')."""
    return s.replace(" ", "")
