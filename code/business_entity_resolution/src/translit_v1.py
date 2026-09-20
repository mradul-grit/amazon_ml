"""Transliteration v1 (frozen copy used by feature version char_ngrams=3, i.e. exp_001/exp_004).
Kept only for reproducibility; v2 (translit.py) fixes ZWNJ splitting, explicit long-a deletion,
Gurmukhi tippi and Malayalam chillu letters."""
import re

_BLOCK_STARTS = [0x0900, 0x0980, 0x0A00, 0x0A80, 0x0B00, 0x0B80, 0x0C00, 0x0C80, 0x0D00]

_CONS = {
    0x15: "k", 0x16: "kh", 0x17: "g", 0x18: "gh", 0x19: "n", 0x1A: "ch", 0x1B: "chh", 0x1C: "j",
    0x1D: "jh", 0x1E: "n", 0x1F: "t", 0x20: "th", 0x21: "d", 0x22: "dh", 0x23: "n", 0x24: "t",
    0x25: "th", 0x26: "d", 0x27: "dh", 0x28: "n", 0x29: "n", 0x2A: "p", 0x2B: "ph", 0x2C: "b",
    0x2D: "bh", 0x2E: "m", 0x2F: "y", 0x30: "r", 0x31: "r", 0x32: "l", 0x33: "l", 0x34: "l",
    0x35: "v", 0x36: "sh", 0x37: "sh", 0x38: "s", 0x39: "h",
    0x58: "q", 0x59: "kh", 0x5A: "g", 0x5B: "z", 0x5C: "d", 0x5D: "rh", 0x5E: "f", 0x5F: "y",
}
_VOWELS = {0x05: "a", 0x06: "a", 0x07: "i", 0x08: "i", 0x09: "u", 0x0A: "u", 0x0B: "ri",
           0x0E: "e", 0x0F: "e", 0x10: "ai", 0x12: "o", 0x13: "o", 0x14: "au", 0x0D: "e", 0x11: "o"}
_SIGNS = {0x3E: "a", 0x3F: "i", 0x40: "i", 0x41: "u", 0x42: "u", 0x43: "ri", 0x46: "e", 0x47: "e",
          0x48: "ai", 0x49: "o", 0x4A: "o", 0x4B: "o", 0x4C: "au", 0x45: "e"}
_VIRAMA = 0x4D
_ANUSVARA = {0x01: "n", 0x02: "n", 0x03: "h"}
_NUKTA = 0x3C
_DIGIT0 = 0x66
_INDIC = re.compile(r"[ऀ-ൿ]")


def _off(ch):
    cp = ord(ch)
    for b in _BLOCK_STARTS:
        if b <= cp < b + 0x80:
            return cp - b
    return None


def translit(s):
    if not s or not _INDIC.search(s):
        return s
    out = []
    pending = False
    for ch in s:
        o = _off(ch)
        if o is None:
            if pending:
                out.append("a")
                pending = False
            out.append(ch)
            continue
        if o in _CONS:
            if pending:
                out.append("a")
            out.append(_CONS[o])
            pending = True
        elif o in _SIGNS:
            out.append(_SIGNS[o])
            pending = False
        elif o == _VIRAMA:
            pending = False
        elif o == _NUKTA:
            continue
        elif o in _VOWELS:
            if pending:
                out.append("a")
                pending = False
            out.append(_VOWELS[o])
        elif o in _ANUSVARA:
            if pending:
                out.append("a")
                pending = False
            out.append(_ANUSVARA[o])
        elif _DIGIT0 <= o <= _DIGIT0 + 9:
            if pending:
                out.append("a")
                pending = False
            out.append(str(o - _DIGIT0))
        else:
            if pending:
                out.append("a")
                pending = False
    if pending:
        out.append("a")
    text = "".join(out)
    return re.sub(r"(?<=[bcdfghjklmnpqrstvwxyz])a\b", "", text)
