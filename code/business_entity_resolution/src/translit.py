"""Rule-based transliteration of Indic scripts to Latin (written from scratch; no external data).

All the major Indic Unicode blocks (Devanagari, Bengali, Gurmukhi, Gujarati, Oriya, Tamil, Telugu,
Kannada, Malayalam) share the ISCII-derived layout, so a character is mapped to its Devanagari
equivalent by block offset and then transliterated with one table. Consonants carry an
inherent 'a' unless followed by a vowel sign or virama; the word-final inherent 'a' is dropped
(schwa deletion), so "लिमिटेड" -> "limited". The rule is script-generic, not country-specific.
"""
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
_ANUSVARA = {0x01: "n", 0x02: "n", 0x03: "h", 0x70: "n"}  # 0x70: Gurmukhi tippi
_CHILLU = {0x7A: "n", 0x7B: "n", 0x7C: "r", 0x7D: "l", 0x7E: "l", 0x7F: "k"}  # Malayalam chillu letters
_INHERENT = "\x01"  # placeholder for the inherent vowel, resolved after the pass
_ZW = re.compile("[‌‍]")
_NUKTA = 0x3C
_DIGIT0 = 0x66

_INDIC = re.compile(r"[ऀ-ൿ]")


def _off(ch):
    cp = ord(ch)
    for b in _BLOCK_STARTS:
        if b <= cp < b + 0x80:
            return cp - b
    return None


def has_indic(s):
    return bool(_INDIC.search(s)) if s else False


def translit(s):
    """Transliterate Indic characters in s to Latin; other characters pass through."""
    if not s or not has_indic(s):
        return s
    s = _ZW.sub("", s)
    out = []
    pending = False  # a consonant waiting for its vowel

    def flush():
        nonlocal pending
        if pending:
            out.append(_INHERENT)  # inherent vowel; resolved below (schwa deletion)
            pending = False

    for ch in s:
        o = _off(ch)
        if o is None:
            flush()
            out.append(ch)
            continue
        base = ord(ch) - o
        if o in _CONS:
            flush()
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
            flush()
            out.append(_VOWELS[o])
        elif o in _ANUSVARA and (o != 0x70 or base == 0x0A00):
            flush()
            out.append(_ANUSVARA[o])
        elif o in _CHILLU and base == 0x0D00:
            flush()
            out.append(_CHILLU[o])
        elif _DIGIT0 <= o <= _DIGIT0 + 9:
            flush()
            out.append(str(o - _DIGIT0))
        else:
            flush()
    flush()
    text = "".join(out)
    # schwa deletion: an inherent vowel at the end of a word is dropped; elsewhere it is 'a'.
    # Explicit vowel signs (e.g. the long 'aa' sign) are never dropped.
    text = re.sub(_INHERENT + r"(?=\W|$)", "", text)
    return text.replace(_INHERENT, "a")
