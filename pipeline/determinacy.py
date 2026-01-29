
from typing import Literal, Tuple, Iterable

DefTag = Literal["Def", "Indef", "None"]

# English sets
EN_DEF_STRICT = {"the"}
EN_INDEF_STRICT = {"a", "an"}

EN_DEF_GENERAL = {"the", "this", "that", "these", "those"}
EN_INDEF_GENERAL = {"a", "an", "some"}

# Polish proxies (no articles): demonstratives vs. indefinites
PL_DEF = {"ten", "ta", "to", "ci", "te", "tamten", "tamta", "tamto"}
PL_INDEF = {"jakiś", "jakieś", "jakaś"}

def _has_any(tokens: Iterable[str], vocab: set[str]) -> bool:
    for w in tokens:
        if w.lower() in vocab:
            return True
    return False

def subject_np_determinacy(doc, subj_span: Tuple[int, int], lang: str, mode: str = "general") -> DefTag:
    """
    Determine Def/Indef/None for Subject NP.
    - lang 'en': mode='strict' → only 'the' vs {'a','an'}; mode='general' → broader sets.
    - lang 'pl': use PL_DEF/PL_INDEF lexica; else 'None'.
    """
    if subj_span == (-1, -1):
        return "None"
    start, end = subj_span
    toks = [t.text.lower() for t in doc[start:end]]
    if lang.startswith("en"):
        if mode == "strict":
            if _has_any(toks, EN_DEF_STRICT): return "Def"
            if _has_any(toks, EN_INDEF_STRICT): return "Indef"
            return "None"
        else:
            if _has_any(toks, EN_DEF_GENERAL): return "Def"
            if _has_any(toks, EN_INDEF_GENERAL): return "Indef"
            return "None"
    if lang.startswith("pl"):
        if _has_any(toks, PL_DEF): return "Def"
        if _has_any(toks, PL_INDEF): return "Indef"
        return "None"
    return "None"

def expected_from_pl_order(pl_order_binary: int) -> DefTag:
    """
    Polish order rule (strict):
      SUBJ_before_ROOT → Def
      ROOT_before_SUBJ → Indef
    """
    if pl_order_binary == 1:
        return "Def"
    if pl_order_binary == 0:
        return "Indef"
    return "None"


# PREREG NOTE: For confirmatory analysis, use src_order-based proxy if lexical cues are absent.
