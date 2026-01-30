# -*- coding: utf-8 -*-
"""
Alignment wrappers for DxD_NMT.

Hardened goals:
- Never lose BatchEncoding.word_ids() metadata by casting to dict
- Force fast tokenizer for AwesomeAlign
- Provide schema-stable outputs: both {"word_align": ...} and {"pairs": ...}
- Ensure non-zero coverage: token-level fallback when word_ids unavailable
- Optional backends: AwesomeAlign (default), SimAlign (optional), COMET-align (optional)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

# -----------------------------
# Deterministic word tokenizer
# -----------------------------
_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)

def _simple_word_tokenize(text: str) -> List[str]:
    """
    Lightweight word tokenizer that keeps punctuation as separate tokens.
    Deterministic, language-agnostic; good enough for Kendall τ.
    """
    text = (text or "").strip()
    if not text:
        return []
    return _TOKEN_RE.findall(text)


# -----------------------------
# Kendall τ from alignment pairs
# -----------------------------
from scipy.stats import kendalltau

def kendall_tau_from_pairs(pairs: Sequence[Tuple[int, int]]) -> Optional[float]:
    """
    Kendall τ over monotonicity of target indices with source order.
    If too few pairs -> None.
    """
    if not pairs:
        return None
    src = [int(a) for a, _ in pairs]
    tgt = [int(b) for _, b in pairs]
    if len(src) < 2:
        return None
    tau = kendalltau(src, tgt, nan_policy="omit").correlation
    # NaN guard
    if tau != tau:
        return None
    return float(tau)

# backwards-compatible alias (some scripts used this name)
kendall_tau_from_pairs = kendall_tau_from_pairs


# ============================================================
# AwesomeAligner (HARDENED)
# ============================================================

from transformers import AutoTokenizer, AutoModel

class AwesomeAligner:
    """
    Word-level aligner wrapper based on Awesome-Align.

    HARDENED:
      - forces fast tokenizer (use_fast=True) so BatchEncoding.word_ids exists
      - preserves BatchEncoding when moving to device (enc.to(device))
      - never silently converts encodings to dict (would lose word_ids)
      - deterministic word tokenization for alignment
      - fallback to token-level alignment if word_ids unavailable
      - symmetric intersection (src->tgt and tgt->src) for robustness
    """

    def __init__(self, model_name: str = "aneuraz/awesome-align-with-co", device: str = "cuda"):
        self.device = device
        self.model_name = model_name

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        if not getattr(self.tokenizer, "is_fast", False):
            raise RuntimeError(
                "AwesomeAligner requires a FAST tokenizer (tokenizers backend). "
                "Upgrade tokenizers/transformers or use a model with a fast tokenizer."
            )

        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()

    def _to_device(self, enc):
        # Keep BatchEncoding (needed for .word_ids())
        if hasattr(enc, "to"):
            return enc.to(self.device)
        raise TypeError(
            "Tokenizer output lost alignment metadata (expected BatchEncoding with .word_ids()). "
            "Do not cast encodings to dict; ensure use_fast=True."
        )

    def _strip_specials_and_get_word_ids(self, enc, batch_index: int):
        """
        Returns indices to keep (where word_id is not None) and the full word-id map.
        If word_ids() returns None (rare but happens), returns ([], None).
        """
        wids_full = enc.word_ids(batch_index=batch_index)
        if wids_full is None:
            return [], None
        keep = [i for i, w in enumerate(wids_full) if w is not None]
        return keep, wids_full

    @staticmethod
    def _unique_pairs(pairs: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        # preserve order, drop dups
        seen = set()
        out = []
        for a, b in pairs:
            key = (int(a), int(b))
            if key not in seen:
                seen.add(key)
                out.append(key)
        return out

    def align_batch(self, src_texts: Sequence[str], tgt_texts: Sequence[str], batch_size: int = 24) -> List[Dict[str, Any]]:
        """
        Returns list of dicts:
          {"word_align": [(src_word_id, tgt_word_id), ...], "pairs": same}
        where word ids index into the word-tokenized inputs (not subwords),
        OR token indices if we had to fallback.
        """
        assert len(src_texts) == len(tgt_texts), "src/tgt size mismatch"
        outputs: List[Dict[str, Any]] = []

        for b0 in range(0, len(src_texts), batch_size):
            b1 = min(len(src_texts), b0 + batch_size)

            src_words = [_simple_word_tokenize(s) for s in src_texts[b0:b1]]
            tgt_words = [_simple_word_tokenize(t) for t in tgt_texts[b0:b1]]

            enc_s = self.tokenizer(
                src_words,
                is_split_into_words=True,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            enc_t = self.tokenizer(
                tgt_words,
                is_split_into_words=True,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )

            enc_s = self._to_device(enc_s)
            enc_t = self._to_device(enc_t)

            with torch.no_grad():
                out_s = self.model(**enc_s).last_hidden_state  # [B, Ts, H]
                out_t = self.model(**enc_t).last_hidden_state  # [B, Tt, H]

            for j in range(len(src_words)):
                src_keep, src_wids = self._strip_specials_and_get_word_ids(enc_s, j)
                tgt_keep, tgt_wids = self._strip_specials_and_get_word_ids(enc_t, j)

                # If we cannot map to word ids, fallback to token ids
                if not src_keep or not tgt_keep or src_wids is None or tgt_wids is None:
                    # fallback: token-level (after stripping specials)
                    # build token-level similarity and symmetric intersection
                    if out_s.shape[1] == 0 or out_t.shape[1] == 0:
                        outputs.append({"word_align": [], "pairs": []})
                        continue

                    src_vecs = out_s[j]  # includes specials, but that's okay here
                    tgt_vecs = out_t[j]

                    sim = torch.matmul(src_vecs, tgt_vecs.T)

                    # forward argmax
                    fwd = [(si, int(torch.argmax(sim[si]).item())) for si in range(sim.shape[0])]
                    # backward argmax
                    bwd = [(int(torch.argmax(sim[:, tj]).item()), tj) for tj in range(sim.shape[1])]

                    inter = set(fwd).intersection(set(bwd))
                    pairs = sorted(list(inter), key=lambda x: (x[0], x[1]))
                    pairs = self._unique_pairs(pairs)

                    outputs.append({"word_align": pairs, "pairs": pairs})
                    continue

                # word-aware path: restrict to non-special subword positions
                src_vecs = out_s[j][src_keep]
                tgt_vecs = out_t[j][tgt_keep]

                sim = torch.matmul(src_vecs, tgt_vecs.T)  # [S, T]

                # forward (src->tgt)
                fwd_pairs: List[Tuple[int, int]] = []
                for si in range(sim.shape[0]):
                    tj = int(torch.argmax(sim[si]).item())
                    s_w = src_wids[src_keep[si]]
                    t_w = tgt_wids[tgt_keep[tj]]
                    if s_w is not None and t_w is not None:
                        fwd_pairs.append((int(s_w), int(t_w)))

                # backward (tgt->src)
                bwd_pairs: List[Tuple[int, int]] = []
                for tj in range(sim.shape[1]):
                    si = int(torch.argmax(sim[:, tj]).item())
                    s_w = src_wids[src_keep[si]]
                    t_w = tgt_wids[tgt_keep[tj]]
                    if s_w is not None and t_w is not None:
                        bwd_pairs.append((int(s_w), int(t_w)))

                # symmetric intersection for robustness
                inter = set(fwd_pairs).intersection(set(bwd_pairs))
                pairs = sorted(list(inter), key=lambda x: (x[0], x[1]))
                pairs = self._unique_pairs(pairs)

                # last resort: if intersection empty, keep forward (avoid total emptiness)
                if not pairs:
                    pairs = self._unique_pairs(fwd_pairs)

                outputs.append({"word_align": pairs, "pairs": pairs})

        return outputs


# ============================================================
# Optional backend: SimAlign (guarded)
# ============================================================

class SimAligner:
    """
    Optional SimAlign wrapper.
    Requires: pip install simalign
    Note: outputs token/word indices depending on simalign behavior.
    """
    def __init__(self, model: str = "xlmr", method: str = "mai", device: str = "cpu"):
        self.device = device
        try:
            from simalign import SentenceAligner
        except Exception as e:
            raise RuntimeError("SimAlign not installed. pip install simalign") from e

        # SimAlign uses CPU typically; device not always used
        self.sa = SentenceAligner(model=model, token_type="bpe", matching_methods=method)

    def align_batch(self, src_texts: Sequence[str], tgt_texts: Sequence[str], batch_size: int = 24) -> List[Dict[str, Any]]:
        out = []
        for s, t in zip(src_texts, tgt_texts):
            s_tok = _simple_word_tokenize(s)
            t_tok = _simple_word_tokenize(t)
            if not s_tok or not t_tok:
                out.append({"word_align": [], "pairs": []})
                continue
            res = self.sa.get_word_aligns(s_tok, t_tok)
            # res keys depend on methods; we take union across available
            pairs = set()
            for k, v in res.items():
                for a, b in v:
                    pairs.add((int(a), int(b)))
            pairs = sorted(list(pairs), key=lambda x: (x[0], x[1]))
            out.append({"word_align": pairs, "pairs": pairs})
        return out


# ============================================================
# Optional backend: COMET-align (guarded placeholder)
# ============================================================

class CometAligner:
    """
    Placeholder for COMET-align style aligner if you add one.
    """
    def __init__(self, model: str, device: str = "cpu"):
        raise RuntimeError("CometAligner not implemented in this repo yet.")


# ============================================================
# Unified aligner factory (required by pipeline)
# ============================================================

def build_aligner(
    backend: str = "auto",
    device: str = "cpu",
    awesome_model: str = "aneuraz/awesome-align-with-co",
    simalign_model: str = "xlmr",
    simalign_methods: Tuple[str, ...] = ("mai",),
    comet_align_model: Optional[str] = None,
):
    backend = (backend or "auto").lower().strip()

    # AwesomeAlign
    if backend in ("auto", "awesome"):
        try:
            return AwesomeAligner(model_name=awesome_model, device=device)
        except Exception as e:
            if backend == "awesome":
                raise
            print(f"[align] AwesomeAlign failed: {e}")

    # SimAlign
    if backend in ("auto", "simalign"):
        try:
            # SimAlign supports one method string; if multiple, join is not supported.
            method = ",".join(simalign_methods) if isinstance(simalign_methods, (list, tuple)) else str(simalign_methods)
            return SimAligner(model=simalign_model, method=method, device=device)
        except Exception as e:
            if backend == "simalign":
                raise
            print(f"[align] SimAlign failed: {e}")

    # COMET-align
    if backend in ("auto", "comet-align"):
        if comet_align_model is None:
            raise RuntimeError("comet-align requested but no --comet-align-model provided")
        try:
            return CometAligner(model=comet_align_model, device=device)
        except Exception as e:
            if backend == "comet-align":
                raise
            print(f"[align] COMET-align failed: {e}")

    raise RuntimeError("❌ No alignment backend available")
