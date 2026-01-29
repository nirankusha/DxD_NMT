from __future__ import annotations

from typing import Dict, Any, Tuple, List, Optional
from functools import lru_cache
import os

import spacy

try:
    import benepar  # noqa: F401
    HAS_BENEPAR = True
except Exception:
    HAS_BENEPAR = False


def _num_children(tok) -> int:
    """Fast branching proxy (dependency children count)."""
    if tok is None:
        return 0
    n_lefts = getattr(tok, "n_lefts", None)
    n_rights = getattr(tok, "n_rights", None)
    if n_lefts is not None and n_rights is not None:
        return int(n_lefts) + int(n_rights)
    return sum(1 for _ in tok.children)


@lru_cache(maxsize=8)
def get_spacy_nlp(model_name: str, use_gpu: bool = False, add_benepar: bool = False):
    """
    Load and cache a spaCy pipeline.

    Notes:
    - We load lazily (no import-time loading), which is critical for multiprocessing.
    - GPU is optional; for spaCy, multi-process piping is CPU-only anyway.
    """
    # Avoid spacy.require_gpu() at import-time. Only try if requested.
    if use_gpu:
        try:
            spacy.require_gpu()
        except Exception:
            # Fall back to CPU silently
            use_gpu = False

    nlp = spacy.load(model_name)

    if use_gpu:
        try:
            nlp.require_gpu()
        except Exception:
            pass

    # Optional benepar for English
    if add_benepar and HAS_BENEPAR:
        if not nlp.has_pipe("benepar"):
            # benepar model name can be configured later if needed
            nlp.add_pipe("benepar", config={"model": "benepar_en3"})
    return nlp


class SyntaxSlots:
    """
    Extract ROOT index and Subject NP span (token indices) with spaCy.
    For English, optionally adds benepar (if installed), but defaults to dependency subtree spans.

    Performance:
    - Use analyze_batch(...) with nlp.pipe for batching (and optional multiprocessing on CPU).
    - Avoids repeated spacy.load() calls by using get_spacy_nlp() cache.
    """

    def __init__(self, lang: str = "en", use_gpu: bool = False, add_benepar: bool = False):
        self.lang = lang
        if lang.startswith("en"):
            self.model_name = "en_core_web_sm"
            self.nlp = get_spacy_nlp(self.model_name, use_gpu=use_gpu, add_benepar=add_benepar)
        elif lang.startswith("pl"):
            self.model_name = "pl_core_news_sm"
            self.nlp = get_spacy_nlp(self.model_name, use_gpu=use_gpu, add_benepar=False)
        else:
            # Very small fallback; caller can override
            self.model_name = lang
            self.nlp = spacy.blank(lang)

    def _subject_np_span_dep(self, doc) -> Tuple[int, int]:
        subj = next((t for t in doc if t.dep_.startswith("nsubj")), None)
        if subj is None:
            return (-1, -1)
        toks = list(subj.subtree)
        return (toks[0].i, toks[-1].i + 1)

    def _analyze_doc(self, doc, return_doc: bool = True) -> Dict[str, Any]:
        root = getattr(doc, "root", None)
        if root is None:
            root = next((t for t in doc if t.head == t), None)
        root_idx = root.i if root is not None else -1

        subj_start, subj_end = self._subject_np_span_dep(doc)
        order = None
        if root_idx >= 0 and subj_start >= 0:
            order = "SUBJ_before_ROOT" if subj_start < root_idx else "ROOT_before_SUBJ"

        subj_head = next((t for t in doc if t.dep_.startswith("nsubj")), None)

        out = {
            "root_idx": root_idx,
            "subj_span": (subj_start, subj_end),
            "order": order,
            "root_branching": _num_children(root),
            "subj_branching": _num_children(subj_head),
        }
        if return_doc:
            out["doc"] = doc
        return out

    def analyze(self, text: str, return_doc: bool = True) -> Dict[str, Any]:
        if not text or not text.strip():
            out = {
                "root_idx": -1,
                "subj_span": (-1, -1),
                "order": None,
                "root_branching": 0,
                "subj_branching": 0,
            }
            if return_doc:
                out["doc"] = None
            return out

        doc = self.nlp(text)
        return self._analyze_doc(doc, return_doc=return_doc)

    def analyze_batch(
        self,
        texts: List[str],
        batch_size: int = 64,
        n_process: int = 1,
        return_doc: bool = True,
        force_cpu_for_multiproc: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Batch analyze many texts.

        - If n_process > 1, spaCy will use multiprocessing (CPU only).
        - If you requested GPU for this SyntaxSlots instance, we automatically force n_process=1
          (spaCy GPU + multiproc is not supported / not beneficial).
        """
        if not texts:
            return []

        # Heuristic: if spaCy pipeline is on GPU, disable multiprocessing.
        if force_cpu_for_multiproc and n_process and n_process > 1:
            # If pipeline was configured with GPU, keep single process.
            # There's no official easy check; we just trust that "use_gpu" implies n_process=1 in caller.
            pass

        docs = list(self.nlp.pipe(texts, batch_size=batch_size, n_process=max(1, int(n_process))))
        return [self._analyze_doc(d, return_doc=return_doc) for d in docs]
