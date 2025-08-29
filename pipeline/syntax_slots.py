
from typing import Dict, Any, Tuple, List
import spacy

spacy.require_gpu()
nlp = spacy.load("en_core_web_sm")
nlp.require_gpu()

try:
    import benepar
    HAS_BENEPAR = True
except Exception:
    HAS_BENEPAR = False

# --- add near the top of syntax_slots.py ---
def _num_children(tok):
    # Fast path if n_lefts/n_rights are available (spaCy)
    n_lefts = getattr(tok, "n_lefts", None)
    n_rights = getattr(tok, "n_rights", None)
    if n_lefts is not None and n_rights is not None:
        return int(n_lefts) + int(n_rights)
    # Fallback: children is a generator — count it
    return sum(1 for _ in tok.children)


class SyntaxSlots:
    """
    Extract ROOT index and Subject NP span (token indices) with spaCy.
    If benepar is available for English, uses it to refine NP span; otherwise uses dependency subtree.
    """
    def __init__(self, lang: str = "en"):
        self.lang = lang
        if lang.startswith("en"):
            self.nlp = spacy.load("en_core_web_sm")
            # GPU acceleration
            try:
                spacy.require_gpu()
                self.nlp.require_gpu()
            except Exception:
                pass  # Fall back to CPU if GPU unavailable
        
            if HAS_BENEPAR and not self.nlp.has_pipe("benepar"):
                self.nlp.add_pipe("benepar", config={"model": "benepar_en3"})
        elif lang.startswith("pl"):  # This was incorrectly nested
            self.nlp = spacy.load("pl_core_news_sm")
            try:
                spacy.require_gpu()
                self.nlp.require_gpu()
            except Exception:
                pass
        else:
            self.nlp = spacy.blank(lang)
            
    def _subject_np_span_dep(self, doc) -> Tuple[int, int]:
        subj = next((t for t in doc if t.dep_.startswith("nsubj")), None)
        if subj is None:
            return (-1, -1)
        toks = list(subj.subtree)
        return (toks[0].i, toks[-1].i + 1)

    def analyze(self, text: str) -> Dict[str, Any]:
        doc = self.nlp(text)
        root = getattr(doc, "root", None)
        if root is None:
            root = next((t for t in doc if t.head == t), None)
        root_idx = root.i if root is not None else -1
        subj_start, subj_end = self._subject_np_span_dep(doc)

        order = None
        if root_idx >= 0 and subj_start >= 0:
            order = "SUBJ_before_ROOT" if subj_start < root_idx else "ROOT_before_SUBJ"

        subj_head = next((t for t in doc if t.dep_.startswith("nsubj")), None)

        return {
            "doc": doc,
            "root_idx": root_idx,
            "subj_span": (subj_start, subj_end),
            "order": order,
            "root_branching": _num_children(root) if root is not None else 0,
            "subj_branching": len(list(subj_head.children)) if subj_head is not None else 0,
        }
    
    @lru_cache(maxsize=5000)
    def _analyze_cached(self, text_hash: str, text: str) -> Dict[str, Any]:
        """Cached version of analyze - uses hash to enable caching of mutable objects."""
        return self._analyze_uncached(text)

    def _analyze_uncached(self, text: str) -> Dict[str, Any]:
        """Original analyze logic without caching."""
        doc = self.nlp(text)
        root = getattr(doc, "root", None)
        if root is None:
            root = next((t for t in doc if t.head == t), None)
        root_idx = root.i if root is not None else -1
        subj_start, subj_end = self._subject_np_span_dep(doc)
        
        order = None
        if root_idx >= 0 and subj_start >= 0:
            order = "SUBJ_before_ROOT" if subj_start < root_idx else "ROOT_before_SUBJ"

        subj_head = next((t for t in doc if t.dep_.startswith("nsubj")), None)

        return {
            "doc": doc,
            "root_idx": root_idx,
            "subj_span": (subj_start, subj_end),
            "order": order,
            "root_branching": _num_children(root) if root is not None else 0,
            "subj_branching": len(list(subj_head.children)) if subj_head is not None else 0,
            }

    def analyze(self, text: str) -> Dict[str, Any]:
        """Main analyze method with caching."""
        if not text or not text.strip():
            return {"doc": None, "root_idx": -1, "subj_span": (-1, -1), 
                "order": None, "root_branching": 0, "subj_branching": 0}
    
        text_hash = hash(text)
        return self._analyze_cached(text_hash, text)
    
    def analyze_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Batch process multiple texts for better performance."""
        if not texts:
            return []
    
        # Process all texts in batch
        docs = list(self.nlp.pipe(texts, batch_size=32, n_process=1))
        results = []
        
        for doc in docs:
            root = getattr(doc, "root", None)
            if root is None:
                root = next((t for t in doc if t.head == t), None)
            root_idx = root.i if root is not None else -1
            subj_start, subj_end = self._subject_np_span_dep(doc)

            order = None
            if root_idx >= 0 and subj_start >= 0:
                order = "SUBJ_before_ROOT" if subj_start < root_idx else "ROOT_before_SUBJ"

            subj_head = next((t for t in doc if t.dep_.startswith("nsubj")), None)

            results.append({
                "doc": doc,
                "root_idx": root_idx,
                "subj_span": (subj_start, subj_end),
                "order": order,
                "root_branching": _num_children(root) if root is not None else 0,
                "subj_branching": len(list(subj_head.children)) if subj_head is not None else 0,
                })
    
        return results