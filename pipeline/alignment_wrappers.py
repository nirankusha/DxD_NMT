from __future__ import annotations

from typing import List, Tuple, Dict, Any, Optional
import torch
from transformers import AutoModel, AutoTokenizer


class AwesomeAligner:
    """
    Word-level alignment using aneuraz/awesome-align-with-co (or compatible).
    Key optimizations vs. the naive version:
    - Single model/tokenizer instance reused across the whole run
    - Fast-tokenizer word mapping via `is_split_into_words=True` + `word_ids()`
    - Optional batch alignment over multiple (src,tgt) pairs
    """

    def __init__(
        self,
        model_name: str = "aneuraz/awesome-align-with-co",
        device: Optional[str] = None,
        align_layer: int = 8,
        threshold: float = 1e-3,
        max_length: Optional[int] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

        self.align_layer = int(align_layer)
        self.threshold = float(threshold)
        self.max_length = max_length or getattr(self.tokenizer, "model_max_length", 512)

    def _encode_wordlist_batch(self, word_lists: List[List[str]]):
        """
        Encode a batch of tokenized sentences (list of word lists).
        Returns:
          enc: BatchEncoding
          hs: hidden states tensor at align_layer, shape (B, L, D)
        """
        enc = self.tokenizer(
            word_lists,
            is_split_into_words=True,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.inference_mode():
            out = self.model(**enc, output_hidden_states=True)
            hs = out.hidden_states[self.align_layer]  # (B, L, D)
        return enc, hs

    @staticmethod
    def _strip_specials_and_get_word_ids(enc, batch_index: int):
        """
        Returns:
          keep_idx: list[int] token positions to keep (exclude specials where word_id is None)
          word_ids: list[Optional[int]] aligned with kept token positions
        """
        wids_full = enc.word_ids(batch_index=batch_index)
        keep = [i for i, wid in enumerate(wids_full) if wid is not None]
        wids = [wids_full[i] for i in keep]
        return keep, wids

    def _align_one_from_embeddings(
        self,
        src_words: List[str],
        tgt_words: List[str],
        src_h: torch.Tensor,  # (Ls, D) incl specials possibly
        tgt_h: torch.Tensor,  # (Lt, D)
        src_keep: List[int],
        tgt_keep: List[int],
        src_wids: List[int],
        tgt_wids: List[int],
    ) -> Dict[str, Any]:
        # Select non-special token embeddings
        src_e = src_h[src_keep, :]
        tgt_e = tgt_h[tgt_keep, :]

        # Similarity matrix
        dot = torch.matmul(src_e, tgt_e.transpose(-1, -2))  # (ls, lt)

        # Symmetric thresholding as in awesome-align
        soft_srctgt = torch.softmax(dot, dim=-1)
        soft_tgtsrc = torch.softmax(dot, dim=-2)
        inter = (soft_srctgt > self.threshold) & (soft_tgtsrc > self.threshold)
        sub_pairs = torch.nonzero(inter, as_tuple=False).tolist()  # pairs in "kept-subtoken space"

        # Map kept-subtoken indices back to word indices
        word_pairs = set()
        for si, ti in sub_pairs:
            wi = src_wids[si]
            wj = tgt_wids[ti]
            if wi is not None and wj is not None:
                word_pairs.add((int(wi), int(wj)))

        return {
            "src_words": src_words,
            "tgt_words": tgt_words,
            "subword_align": sub_pairs,
            "word_align": sorted(word_pairs),
        }

    def align(self, src: str, tgt: str) -> Dict[str, Any]:
        src_words = (src or "").strip().split()
        tgt_words = (tgt or "").strip().split()
        if not src_words or not tgt_words:
            return {"src_words": src_words, "tgt_words": tgt_words, "subword_align": [], "word_align": []}

        enc_s, hs_s = self._encode_wordlist_batch([src_words])
        enc_t, hs_t = self._encode_wordlist_batch([tgt_words])

        src_keep, src_wids = self._strip_specials_and_get_word_ids(enc_s, 0)
        tgt_keep, tgt_wids = self._strip_specials_and_get_word_ids(enc_t, 0)

        return self._align_one_from_embeddings(
            src_words, tgt_words,
            hs_s[0], hs_t[0],
            src_keep, tgt_keep, src_wids, tgt_wids
        )

    def align_batch(self, src_texts: List[str], tgt_texts: List[str], batch_size: int = 16) -> List[Dict[str, Any]]:
        """
        Batch alignment over multiple pairs. Uses separate encoding of src and tgt batches
        (keeps the implementation simple and stable).
        """
        assert len(src_texts) == len(tgt_texts), "src_texts and tgt_texts must be same length"
        out: List[Dict[str, Any]] = []

        n = len(src_texts)
        for b0 in range(0, n, batch_size):
            b1 = min(n, b0 + batch_size)
            src_words_batch = [(src_texts[i] or "").strip().split() for i in range(b0, b1)]
            tgt_words_batch = [(tgt_texts[i] or "").strip().split() for i in range(b0, b1)]

            # Handle empties without encoding
            nonempty_idx = [i for i, (sw, tw) in enumerate(zip(src_words_batch, tgt_words_batch)) if sw and tw]
            if not nonempty_idx:
                for i in range(b0, b1):
                    out.append({"src_words": src_words_batch[i-b0], "tgt_words": tgt_words_batch[i-b0], "subword_align": [], "word_align": []})
                continue

            src_ne = [src_words_batch[i] for i in nonempty_idx]
            tgt_ne = [tgt_words_batch[i] for i in nonempty_idx]

            enc_s, hs_s = self._encode_wordlist_batch(src_ne)
            enc_t, hs_t = self._encode_wordlist_batch(tgt_ne)

            # Reconstruct outputs in original order
            res_block = [None] * (b1 - b0)
            for j, local_i in enumerate(nonempty_idx):
                src_keep, src_wids = self._strip_specials_and_get_word_ids(enc_s, j)
                tgt_keep, tgt_wids = self._strip_specials_and_get_word_ids(enc_t, j)
                res_block[local_i] = self._align_one_from_embeddings(
                    src_words_batch[local_i],
                    tgt_words_batch[local_i],
                    hs_s[j], hs_t[j],
                    src_keep, tgt_keep, src_wids, tgt_wids
                )

            for k in range(b1 - b0):
                if res_block[k] is None:
                    res_block[k] = {"src_words": src_words_batch[k], "tgt_words": tgt_words_batch[k], "subword_align": [], "word_align": []}
                out.append(res_block[k])

        return out
