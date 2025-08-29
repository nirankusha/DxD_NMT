
from typing import List, Tuple

def whitespace_token_spans(text: str) -> List[Tuple[int, int]]:
    """
    Compute (start, end) char spans for whitespace-tokenized words in text.
    Robust to repeated tokens by scanning incrementally.
    """
    spans = []
    i = 0
    for w in text.strip().split():
        # find w starting from i
        j = text.find(w, i)
        if j == -1:
            # fallback: try anywhere after i
            j = text.find(w, i)
        k = j + len(w) if j != -1 else i
        spans.append((j, k))
        i = k
    return spans

def map_ws_index_to_spacy(doc, ws_index: int) -> int:
    """
    Map whitespace token index to closest spaCy token index by max span overlap.
    Returns -1 if mapping fails.
    """
    ws_spans = whitespace_token_spans(doc.text)
    if ws_index < 0 or ws_index >= len(ws_spans):
        return -1
    a, b = ws_spans[ws_index]
    best_i, best_overlap = -1, 0
    for t in doc:
        x, y = t.idx, t.idx + len(t.text)
        overlap = max(0, min(b, y) - max(a, x))
        if overlap > best_overlap:
            best_overlap = overlap
            best_i = t.i
    return best_i
