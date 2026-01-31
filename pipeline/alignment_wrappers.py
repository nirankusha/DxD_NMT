# -*- coding: utf-8 -*-
"""
Alignment wrappers for DxD_NMT.

Hardened goals:
- Never lose BatchEncoding.word_ids() metadata by casting to dict
- Force fast tokenizer for AwesomeAlign
- Provide schema-stable outputs: both {"word_align": ...} and {"pairs": ...}
- Ensure non-zero coverage: token-level fallback when word_ids unavailable
- Optional backends: AwesomeAlign (default), SimAlign (optional), COMET-align (optional)

Notes:
- AwesomeAligner here is a lightweight, deterministic wrapper that uses encoder
  representations + argmax matching, then enforces symmetric intersection.
- Kendall τ is computed over (src_index, tgt_index) pairs (word indices preferred).
"""

from __future__ import annotations

from transformers import logging as hf_logging
hf_logging.set_verbosity_error()

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from scipy.stats import kendalltau
from transformers import AutoModel, AutoTokenizer


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
def kendall_tau_from_pairs(pairs: Sequence[Tuple[int, int]]) -> Optional[float]:
    """
    Kendall τ over monotonicity of target indices with source order.
    If too few pairs -> None.
    """
    if pairs is None or len(pairs) < 2:
        return None

    src = []
    tgt = []
    for a, b in pairs:
        if a is None or b is None:
            continue
        src.append(int(a))
        tgt.append(int(b))

    if len(src) < 2:
        return None

    tau = kendalltau(src, tgt, nan_policy="omit").correlation
    if tau is None or tau != tau:  # NaN guard
        return None
    return float(tau)


# backwards-compatible alias (some scripts used this name)
kendall_tau_from_pairs = kendall_tau_from_pairs


# ============================================================
# AwesomeAligner (HARDENED)
# ============================================================
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
      - schema-stable output: {"word_align": pairs, "pairs": pairs}
    """

    def __init__(
        self,
        model_name: str = "aneuraz/awesome-align-with-co",
        device: str = "cuda",
    ):
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
        # Keep BatchEncoding (needed for .word_ids()).
        # BatchEncoding has .to(device). A plain dict does not.
        if hasattr(enc, "to"):
            return enc.to(self.device)
        raise TypeError(
            "Tokenizer output lost alignment metadata (expected BatchEncoding with .word_ids()). "
            "Do not cast encodings to dict; ensure use_fast=True."
        )

    def _strip_specials_and_get_word_ids(self, enc, batch_index: int):
        """
        Returns indices to keep (where word_id is not None) and the full word-id map.
        If word_ids() returns None, returns ([], None).
        """
        wids_full = enc.word_ids(batch_index=batch_index)
        if wids_full is None:
            return [], None
        keep = [i for i, w in enumerate(wids_full) if w is not None]
        return keep, wids_full

    @staticmethod
    def _unique_pairs(pairs: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Preserve order, drop duplicates."""
        seen = set()
        out: List[Tuple[int, int]] = []
        for a, b in pairs:
            key = (int(a), int(b))
            if key not in seen:
                seen.add(key)
                out.append(key)
        return out

    @staticmethod
    def _sym_intersection(
        fwd_pairs: List[Tuple[int, int]],
        bwd_pairs: List[Tuple[int, int]],
        fallback_to_fwd: bool = True,
    ) -> List[Tuple[int, int]]:
        inter = set(fwd_pairs).intersection(set(bwd_pairs))
        if inter:
            pairs = sorted(list(inter), key=lambda x: (x[0], x[1]))
            return AwesomeAligner._unique_pairs(pairs)
        if fallback_to_fwd:
            return AwesomeAligner._unique_pairs(fwd_pairs)
        return []

    def align_batch(
        self,
        src_texts: Sequence[str],
        tgt_texts: Sequence[str],
        batch_size: int = 24,
    ) -> List[Dict[str, Any]]:
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

                # ------------------------------------------------------------
                # Fallback path: if we cannot map to word ids, do token-level.
                # ------------------------------------------------------------
                if (
                    src_wids is None
                    or tgt_wids is None
                    or not src_keep
                    or not tgt_keep
                ):
                    # Token-level: include all tokens (including specials),
                    # but enforce symmetric intersection to reduce garbage.
                    src_vecs = out_s[j]  # [Ts, H]
                    tgt_vecs = out_t[j]  # [Tt, H]
                    if src_vecs.numel() == 0 or tgt_vecs.numel() == 0:
                        outputs.append({"word_align": [], "pairs": []})
                        continue

                    sim = torch.matmul(src_vecs, tgt_vecs.T)  # [Ts, Tt]

                    fwd = [(si, int(torch.argmax(sim[si]).item())) for si in range(sim.shape[0])]
                    bwd = [(int(torch.argmax(sim[:, tj]).item()), tj) for tj in range(sim.shape[1])]

                    pairs = self._sym_intersection(fwd, bwd, fallback_to_fwd=True)
                    outputs.append({"word_align": pairs, "pairs": pairs})
                    continue

                # ------------------------------------------------------------
                # Word-aware path: restrict to non-special subword positions
                # ------------------------------------------------------------
                src_vecs = out_s[j][src_keep]  # [S, H]
                tgt_vecs = out_t[j][tgt_keep]  # [T, H]
                if src_vecs.numel() == 0 or tgt_vecs.numel() == 0:
                    outputs.append({"word_align": [], "pairs": []})
                    continue

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

                pairs = self._sym_intersection(fwd_pairs, bwd_pairs, fallback_to_fwd=True)
                outputs.append({"word_align": pairs, "pairs": pairs})

        return outputs


# ============================================================
# Optional backend: SimAlign (guarded)
# ============================================================
class SimAligner:
    """
    Optional SimAlign wrapper.
    Requires: pip install simalign

    Returns {"word_align": pairs, "pairs": pairs}, where indices are word indices
    over the provided token lists (we use _simple_word_tokenize()).
    """

    def __init__(
        self,
        model: str = "xlmr",
        method: str = "mai",
        device: str = "cpu",
    ):
        self.device = device
        try:
            from simalign import SentenceAligner
        except Exception as e:
            raise RuntimeError("SimAlign not installed. pip install simalign") from e

        # SimAlign typically runs on CPU; device is not always used.
        self.sa = SentenceAligner(model=model, token_type="bpe", matching_methods=method)

    def align_batch(
        self,
        src_texts: Sequence[str],
        tgt_texts: Sequence[str],
        batch_size: int = 24,
    ) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for s, t in zip(src_texts, tgt_texts):
            s_tok = _simple_word_tokenize(s)
            t_tok = _simple_word_tokenize(t)
            if not s_tok or not t_tok:
                out.append({"word_align": [], "pairs": []})
                continue

            res = self.sa.get_word_aligns(s_tok, t_tok)
            pairs_set = set()
            # res keys depend on method; take union across present values
            for _, v in res.items():
                for a, b in v:
                    pairs_set.add((int(a), int(b)))

            pairs = sorted(list(pairs_set), key=lambda x: (x[0], x[1]))
            out.append({"word_align": pairs, "pairs": pairs})
        return out


# ============================================================
# Optional backend: COMET-align (guarded placeholder)
# ============================================================
class CometAligner:
    """
    Placeholder for COMET-align style aligner if you add one.
    Keeping a consistent interface so the pipeline CLI flags are stable.
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
    """
    backend:
      - "auto": try AwesomeAlign -> SimAlign -> COMET-align
      - "awesome": force AwesomeAlign
      - "simalign": force SimAlign
      - "comet-align": force COMET-align (requires comet_align_model)
    """
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
            # SimAlign expects a method string; if multiple, join with comma.
            if isinstance(simalign_methods, (list, tuple)):
                method = ",".join([str(x) for x in simalign_methods])
            else:
                method = str(simalign_methods)
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
