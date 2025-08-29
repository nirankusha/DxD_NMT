
import re
from typing import Tuple, List

def extract_tuples(text: str) -> Tuple[tuple, tuple]:
    words_pattern = r'(\w+)(?=\[)'
    bracketed_pattern = r'(\[[^\]]*\])'
    words = re.findall(words_pattern, text or "")
    pos = re.findall(bracketed_pattern, text or "")
    return tuple(words), tuple(pos)

def map_targets_to_positions(target_words: Tuple[str, ...], tgt_tokens: List[str]) -> List[int]:
    """Return first‑match indices of each target word in the tokenized target (‑1 if missing)."""
    idxs = []
    used = set()
    for w in target_words or []:
        found = -1
        wl = w.lower()
        for i, tok in enumerate(tgt_tokens):
            if i in used: 
                continue
            if tok.lower() == wl:
                found = i
                used.add(i)
                break
        idxs.append(found)
    return idxs
