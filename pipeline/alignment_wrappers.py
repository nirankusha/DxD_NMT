import torch
import re
from transformers import AutoTokenizer, AutoModel

_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)

def _simple_word_tokenize(text: str):
    """
    Lightweight word tokenizer that keeps punctuation as separate tokens.
    This is intentionally simple and deterministic for alignment/τ.
    """
    text = (text or "").strip()
    if not text:
        return []
    return _TOKEN_RE.findall(text)

class AwesomeAligner:
    """
    Word-level aligner wrapper based on Awesome-Align.
    HARDENED:
      - forces fast tokenizer (use_fast=True) so BatchEncoding.word_ids exists
      - preserves BatchEncoding when moving to device (enc.to(device))
      - never silently converts encodings to dict (would lose word_ids)
      - robust tokenization for word-level alignment
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
        wids_full = enc.word_ids(batch_index=batch_index)
        keep = [i for i, w in enumerate(wids_full) if w is not None]
        return keep, wids_full

    def align_batch(self, src_texts, tgt_texts, batch_size: int = 24):
        """
        Returns list of dicts: {"word_align": [(src_word_id, tgt_word_id), ...]}
        where word ids index into the word-tokenized inputs (not subwords).
        """
        assert len(src_texts) == len(tgt_texts), "src/tgt size mismatch"
        outputs = []

        for b0 in range(0, len(src_texts), batch_size):
            b1 = min(len(src_texts), b0 + batch_size)

            src_batch = [_simple_word_tokenize(s) for s in src_texts[b0:b1]]
            tgt_batch = [_simple_word_tokenize(t) for t in tgt_texts[b0:b1]]

            enc_s = self.tokenizer(
                src_batch,
                is_split_into_words=True,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            enc_t = self.tokenizer(
                tgt_batch,
                is_split_into_words=True,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )

            enc_s = self._to_device(enc_s)
            enc_t = self._to_device(enc_t)

            with torch.no_grad():
                out_s = self.model(**enc_s).last_hidden_state
                out_t = self.model(**enc_t).last_hidden_state

            for j in range(len(src_batch)):
                src_keep, src_wids = self._strip_specials_and_get_word_ids(enc_s, j)
                tgt_keep, tgt_wids = self._strip_specials_and_get_word_ids(enc_t, j)

                if not src_keep or not tgt_keep:
                    outputs.append({"word_align": []})
                    continue

                src_vecs = out_s[j][src_keep]
                tgt_vecs = out_t[j][tgt_keep]

                # cosine-ish similarity via dot product (representations are comparable)
                sim = torch.matmul(src_vecs, tgt_vecs.T)

                pairs = []
                for si in range(sim.shape[0]):
                    tj = int(torch.argmax(sim[si]).item())
                    s_w = src_wids[src_keep[si]]
                    t_w = tgt_wids[tgt_keep[tj]]
                    if s_w is not None and t_w is not None:
                        pairs.append((int(s_w), int(t_w)))

                outputs.append({"word_align": pairs})

        return outputs
