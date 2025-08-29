# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
translate_concordances_mbart_only.py — mBART50 only (plain HF + ZurichNLP MBR)

- Loads CSV; input = 'Edited Sentence' or 'Merged Context'
- Forward PL→EN:
    * mBART50 (seq2seq via AutoModelForSeq2SeqLM)
    * mBART50 (MBartForConditionalGeneration)
    * (optional) ZurichNLP MBR decoding wrappers for both

- Back-translation EN→PL (for round-trip metrics):
    * mBART50 (seq2seq), src="en_XX", tgt="pl_PL"

- External MBR selectors: chrF, BLEU, TER (consensus over sampled candidates)
- Adequacy: LaBSE cosine (and LASER if available)
- Round-trip: chrF/BLEU/TER

CLI example:
  python translate_concordances_mbart_only.py \
    --in /content/concordances_selected.csv --out /content/translations_all.csv \
    --mbart-seq --mbart-cg \
    --mbart-seq-mbr --mbart-cg-mbr \
    --mbr-samples 6 --mbr-dec-samples 6 --mbr-dec-metric chrf \
    --beam 5 --batch-size 8 --max-new-tokens 128
"""

import os
import argparse
from typing import List, Optional, Tuple

import pandas as pd
import sacrebleu
import torch
from tqdm import tqdm

from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM,
    MBartForConditionalGeneration,
)

try:
    from transformers import MBart50TokenizerFast as MBartTok
except Exception:
    from transformers import MBartTokenizer as MBartTok

# --- quiet TF + tokenizers (optional) ---
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# ---- optional ZurichNLP MBR ----
HAS_MBR_LIB = True
try:
    from mbr import MBR, MBRConfig
except Exception:
    HAS_MBR_LIB = False

# ---- HF Hub helpers: login+fallback loaders ----
from huggingface_hub import snapshot_download
import transformers

def hf_auth_kwargs(hf_token: str):
    if not hf_token:
        return {}
    return {"use_auth_token": hf_token, "token": hf_token}

def load_tokenizer(model_id: str, hf_token: str):
    kw = hf_auth_kwargs(hf_token)
    try:
        return transformers.AutoTokenizer.from_pretrained(model_id, **kw)
    except Exception:
        local = snapshot_download(repo_id=model_id, token=hf_token, local_files_only=False)
        return transformers.AutoTokenizer.from_pretrained(local)

def load_seq2seq(model_id: str, hf_token: str):
    kw = hf_auth_kwargs(hf_token)
    try:
        return transformers.AutoModelForSeq2SeqLM.from_pretrained(model_id, **kw)
    except Exception:
        local = snapshot_download(repo_id=model_id, token=hf_token, local_files_only=False)
        return transformers.AutoModelForSeq2SeqLM.from_pretrained(local)

def load_mbart_tokenizer(model_id: str, hf_token: str):
    kw = hf_auth_kwargs(hf_token)
    try:
        return MBartTok.from_pretrained(model_id, **kw)
    except Exception:
        try:
            from transformers import MBartTokenizer as _MBTok
            return _MBTok.from_pretrained(model_id, **kw)
        except Exception:
            local = snapshot_download(repo_id=model_id, token=hf_token, local_files_only=False)
            try:
                return MBartTok.from_pretrained(local)
            except Exception:
                from transformers import MBartTokenizer as _MBTok
                return _MBTok.from_pretrained(local)

def load_mbart_cg(model_id: str, hf_token: str):
    kw = hf_auth_kwargs(hf_token)
    try:
        return MBartForConditionalGeneration.from_pretrained(model_id, **kw)
    except Exception:
        local = snapshot_download(repo_id=model_id, token=hf_token, local_files_only=False)
        return MBartForConditionalGeneration.from_pretrained(local)

# ---- optional adequacy models ----
_LABSE = None
_LASER = None
def maybe_load_labse():
    global _LABSE
    if _LABSE is None:
        try:
            from sentence_transformers import SentenceTransformer
            _LABSE = SentenceTransformer("sentence-transformers/LaBSE")
        except Exception:
            _LABSE = False
    return _LABSE

def maybe_load_laser():
    global _LASER
    if _LASER is None:
        try:
            from laserembeddings import Laser
            _LASER = Laser()
        except Exception:
            _LASER = False
    return _LASER

def cosine_rows(a, b):
    return (a * b).sum(axis=1)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------- generation helpers --------------------

def batched(lst, bs: int):
    for i in range(0, len(lst), bs):
        yield lst[i:i+bs], i, min(i+bs, len(lst))

def decode(outputs, tok) -> List[str]:
    return tok.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True)

def generate_beam(model, tok, inputs: List[str], batch_size: int, max_new_tokens=128, num_beams=5, **extra) -> List[str]:
    outs = []
    for batch, _, _ in batched(inputs, batch_size):
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
        with torch.inference_mode():
            gen = model.generate(**enc, max_new_tokens=max_new_tokens, num_beams=num_beams, early_stopping=True, **extra)
        outs.extend(decode(gen, tok))
    return outs

def generate_samples(model, tok, inputs: List[str], *,
                     num_samples=6, batch_size: int = 1,
                     max_new_tokens=128, top_p=0.95, temperature=0.7, **extra) -> List[List[str]]:
    all_cands: List[List[str]] = []
    for text in tqdm(inputs, desc="Sampling candidates"):
        enc = tok([text], return_tensors="pt", padding=True, truncation=True).to(DEVICE)
        cands = []
        with torch.inference_mode():
            for _ in range(num_samples):
                gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=True,
                                     top_p=top_p, temperature=temperature, num_beams=1, **extra)
                cands.append(decode(gen, tok)[0])
        all_cands.append(cands)
    return all_cands

# -------------------- external MBR selectors --------------------

def mbr_select_chrF(cands_per_sent: List[List[str]]) -> List[str]:
    sel = []
    for cands in cands_per_sent:
        if not cands: sel.append(""); continue
        if len(cands) == 1: sel.append(cands[0]); continue
        scores = []
        for i, ci in enumerate(cands):
            sims = [sacrebleu.sentence_chrf(ci, [cj]).score for j, cj in enumerate(cands) if j != i]
            scores.append(sum(sims) / max(1, len(sims)))
        sel.append(cands[max(range(len(cands)), key=lambda k: scores[k])])
    return sel

def mbr_select_bleu(cands_per_sent: List[List[str]]) -> List[str]:
    sel = []
    for cands in cands_per_sent:
        if not cands: sel.append(""); continue
        if len(cands) == 1: sel.append(cands[0]); continue
        scores = []
        for i, ci in enumerate(cands):
            sims = [sacrebleu.sentence_bleu(ci, [cj]).score for j, cj in enumerate(cands) if j != i]
            scores.append(sum(sims) / max(1, len(sims)))
        sel.append(cands[max(range(len(cands)), key=lambda k: scores[k])])
    return sel

def mbr_select_ter(cands_per_sent: List[List[str]]) -> List[str]:
    sel = []
    for cands in cands_per_sent:
        if not cands: sel.append(""); continue
        if len(cands) == 1: sel.append(cands[0]); continue
        avg_ter = []
        for i, ci in enumerate(cands):
            vals = [sacrebleu.sentence_ter(ci, [cj]).score for j, cj in enumerate(cands) if j != i]
            avg_ter.append(sum(vals) / max(1, len(vals)))
        sel.append(cands[min(range(len(cands)), key=lambda k: avg_ter[k])])
    return sel

# -------------------- translators (plain) --------------------

class BaseTranslator:
    def __init__(self, model, tokenizer, forced_bos_token_id: Optional[int] = None):
        self.model = model.to(DEVICE)
        self.tok = tokenizer
        self.extra = {}
        if forced_bos_token_id is not None:
            self.extra["forced_bos_token_id"] = forced_bos_token_id

    def beam(self, inputs: List[str], batch_size: int, max_new_tokens: int, num_beams: int) -> List[str]:
        return generate_beam(self.model, self.tok, inputs, batch_size, max_new_tokens, num_beams, **self.extra)

    def sample(self, inputs: List[str], num_samples: int, batch_size: int, max_new_tokens: int,
               top_p: float, temperature: float) -> List[List[str]]:
        return generate_samples(self.model, self.tok, inputs,
                                num_samples=num_samples, batch_size=batch_size,
                                max_new_tokens=max_new_tokens, top_p=top_p,
                                temperature=temperature, **self.extra)

MBART_ID = "facebook/mbart-large-50-many-to-many-mmt"

def make_mbart_seq_plain(src="pl_PL", tgt="en_XX", hf_token="") -> BaseTranslator:
    tok = load_tokenizer(MBART_ID, hf_token)
    forced_bos = tok.lang_code_to_id[tgt] if hasattr(tok, "lang_code_to_id") else None
    if hasattr(tok, "src_lang"): tok.src_lang = src
    mdl = load_seq2seq(MBART_ID, hf_token)
    return BaseTranslator(mdl, tok, forced_bos)

def make_mbart_cg_plain(src="pl_PL", tgt="en_XX", hf_token="") -> BaseTranslator:
    tok = load_mbart_tokenizer(MBART_ID, hf_token)
    forced_bos = tok.lang_code_to_id[tgt] if hasattr(tok, "lang_code_to_id") else None
    if hasattr(tok, "src_lang"): tok.src_lang = src
    mdl = load_mbart_cg(MBART_ID, hf_token)
    return BaseTranslator(mdl, tok, forced_bos)

def make_en_pl_back_plain(hf_token="") -> BaseTranslator:
    # EN -> PL using mBART
    return make_mbart_seq_plain(src="en_XX", tgt="pl_PL", hf_token=hf_token)

# -------------------- translators (ZurichNLP MBR-wrapped) --------------------

class MBRTranslator(BaseTranslator):
    def __init__(self, model, tokenizer, mbr_config: "MBRConfig", forced_bos_token_id: Optional[int] = None):
        super().__init__(model, tokenizer, forced_bos_token_id)
        self.mbr_config = mbr_config

    def mbr_dec(self, inputs: List[str], batch_size: int, max_new_tokens: int,
                top_p: float, temperature: float) -> List[str]:
        outs = []
        for batch, _, _ in batched(inputs, batch_size):
            enc = self.tok(batch, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
            with torch.inference_mode():
                gen = self.model.generate(**enc,
                                          do_sample=True, num_beams=1,
                                          top_p=top_p, temperature=temperature,
                                          max_new_tokens=max_new_tokens,
                                          mbr_config=self.mbr_config,
                                          tokenizer=self.tok,
                                          **self.extra)
            outs.extend(decode(gen, self.tok))
        return outs

def build_mbr_config(metric="chrf", num_samples=8):
    if not HAS_MBR_LIB:
        raise RuntimeError("ZurichNLP mbr is not installed. pip install 'transformers<4.39' mbr")
    return MBRConfig(metric=metric, num_samples=num_samples)

def make_mbart_seq_mbr(src="pl_PL", tgt="en_XX", metric="chrf", num_samples=8, hf_token=""):
    tok = load_tokenizer(MBART_ID, hf_token)
    forced_bos = tok.lang_code_to_id[tgt] if hasattr(tok, "lang_code_to_id") else None
    if hasattr(tok, "src_lang"): tok.src_lang = src
    MSeq = MBR(AutoModelForSeq2SeqLM)
    kw = hf_auth_kwargs(hf_token)
    try:
        mdl = MSeq.from_pretrained(MBART_ID, **kw)
    except Exception:
        local = snapshot_download(repo_id=MBART_ID, token=hf_token, local_files_only=False)
        mdl = MSeq.from_pretrained(local)
    return MBRTranslator(mdl, tok, build_mbr_config(metric, num_samples), forced_bos)

def make_mbart_cg_mbr(src="pl_PL", tgt="en_XX", metric="chrf", num_samples=8, hf_token=""):
    tok = load_mbart_tokenizer(MBART_ID, hf_token)
    forced_bos = tok.lang_code_to_id[tgt] if hasattr(tok, "lang_code_to_id") else None
    if hasattr(tok, "src_lang"): tok.src_lang = src
    MCG = MBR(MBartForConditionalGeneration)
    kw = hf_auth_kwargs(hf_token)
    try:
        mdl = MCG.from_pretrained(MBART_ID, **kw)
    except Exception:
        local = snapshot_download(repo_id=MBART_ID, token=hf_token, local_files_only=False)
        mdl = MCG.from_pretrained(local)
    return MBRTranslator(mdl, tok, build_mbr_config(metric, num_samples), forced_bos)

# -------------------- adequacy & round-trip --------------------

def labse_scores(pl_texts: List[str], en_texts: List[str]) -> List[float]:
    st = maybe_load_labse()
    if not st:
        return [float("nan")] * len(pl_texts)
    import numpy as np
    pl_emb = st.encode(pl_texts, normalize_embeddings=True, convert_to_numpy=True)
    en_emb = st.encode(en_texts, normalize_embeddings=True, convert_to_numpy=True)
    return cosine_rows(pl_emb, en_emb).tolist()

def laser_scores(pl_texts: List[str], en_texts: List[str]) -> List[float]:
    laser = maybe_load_laser()
    if not laser:
        return [float("nan")] * len(pl_texts)
    import numpy as np
    pl_emb = laser.embed_sentences(pl_texts, lang='pol')
    en_emb = laser.embed_sentences(en_texts, lang='eng')
    pl_norm = pl_emb / (np.linalg.norm(pl_emb, axis=1, keepdims=True) + 1e-12)
    en_norm = en_emb / (np.linalg.norm(en_emb, axis=1, keepdims=True) + 1e-12)
    return cosine_rows(pl_norm, en_norm).tolist()

def roundtrip_scores(src_pl: List[str], hyp_en: List[str], backtrans: BaseTranslator,
                     batch_size: int, max_new_tokens: int, num_beams: int = 5
                     ) -> Tuple[List[float], List[float], List[float]]:
    bt_pl = backtrans.beam(hyp_en, batch_size=batch_size, max_new_tokens=max_new_tokens, num_beams=num_beams)
    chrf = []; bleu = []; ter = []
    for bt, src in zip(bt_pl, src_pl):
        chrf.append(sacrebleu.sentence_chrf(bt, [src]).score)
        bleu.append(sacrebleu.sentence_bleu(bt, [src]).score)
        ter.append(sacrebleu.sentence_ter(bt, [src]).score)
    return chrf, bleu, ter

# -------------------- pipeline --------------------

def main():
    ap = argparse.ArgumentParser(description="PL→EN with mBART50 only (plain + ZurichNLP MBR)")
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--out", dest="out_path", required=True)
    ap.add_argument("--col", default="")  # Edited Sentence or Merged Context

    # auth
    ap.add_argument("--hf-token", default=os.getenv("HF_TOKEN", ""), help="HF Hub token (or set HF_TOKEN env)")

    # which plain models
    ap.add_argument("--mbart-seq", action="store_true")
    ap.add_argument("--mbart-cg", action="store_true")

    # which Zurich MBR models (internal decoding)
    ap.add_argument("--mbart-seq-mbr", action="store_true")
    ap.add_argument("--mbart-cg-mbr", action="store_true")

    # decoding options
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--beam", type=int, default=5)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--temperature", type=float, default=0.7)

    # sampling counts
    ap.add_argument("--mbr-samples", type=int, default=6, help="Samples for external MBR (plain sampling)")
    ap.add_argument("--mbr-dec-samples", type=int, default=6, help="Samples for Zurich-style MBR decoding")
    ap.add_argument("--mbr-dec-metric", default="chrf", help="Metric for Zurich-style MBR decoding (e.g., chrf)")

    args = ap.parse_args()

    df = pd.read_csv(args.in_path)
    col = args.col or ("Edited Sentence" if "Edited Sentence" in df.columns else "Merged Context")
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found. Available: {list(df.columns)}")
    inputs = df[col].fillna("").astype(str).tolist()

    out_df = df.copy()
    out_df["original"] = inputs

    # back-translation model for roundtrip metrics (EN->PL)
    back = make_en_pl_back_plain(hf_token=args.hf_token)

    def compute_metrics(colname: str):
        # adequacy
        out_df[f"{colname}_labse"] = labse_scores(inputs, out_df[colname].tolist())
        out_df[f"{colname}_laser"] = laser_scores(inputs, out_df[colname].tolist())
        # roundtrip
        chrf, bleu, ter = roundtrip_scores(inputs, out_df[colname].tolist(),
                                           back, batch_size=args.batch_size,
                                           max_new_tokens=args.max_new_tokens,
                                           num_beams=args.beam)
        out_df[f"{colname}_rt_chrf"] = chrf
        out_df[f"{colname}_rt_bleu"] = bleu
        out_df[f"{colname}_rt_ter"]  = ter

    def run_plain(tag: str, tr: BaseTranslator):
        print(f"\n[{tag}] beam…")
        out_df[f"{tag}_beam_en"] = tr.beam(inputs, args.batch_size, args.max_new_tokens, args.beam)
        compute_metrics(f"{tag}_beam_en")

        print(f"[{tag}] sampling {args.mbr_samples} for external MBR…")
        cands = tr.sample(inputs, args.mbr_samples, batch_size=1,
                          max_new_tokens=args.max_new_tokens,
                          top_p=args.top_p, temperature=args.temperature)

        out_df[f"{tag}_mbr_chrf_en"] = mbr_select_chrF(cands)
        compute_metrics(f"{tag}_mbr_chrf_en")

        out_df[f"{tag}_mbr_bleu_en"] = mbr_select_bleu(cands)
        out_df[f"{tag}_mbr_ter_en"]  = mbr_select_ter(cands)
        # Uncomment if you want metrics for BLEU/TER selections too:
        # compute_metrics(f"{tag}_mbr_bleu_en")
        # compute_metrics(f"{tag}_mbr_ter_en")

    def run_mbr(tag: str, mtr: MBRTranslator):
        if not HAS_MBR_LIB:
            print(f"[WARN] {tag}: ZurichNLP mbr library not available — skipping.")
            return
        print(f"\n[{tag}] beam (wrapped)…")
        out_df[f"{tag}_beam_en"] = mtr.beam(inputs, args.batch_size, args.max_new_tokens, args.beam)
        compute_metrics(f"{tag}_beam_en")

        print(f"[{tag}] MBR decoding (metric={args.mbr_dec_metric}, samples={args.mbr_dec_samples})…")
        mtr.mbr_config = build_mbr_config(args.mbr_dec_metric, args.mbr_dec_samples)
        out_df[f"{tag}_mbr_dec_{args.mbr_dec_metric}_en"] = mtr.mbr_dec(
            inputs, args.batch_size, args.max_new_tokens, args.top_p, args.temperature
        )
        compute_metrics(f"{tag}_mbr_dec_{args.mbr_dec_metric}_en")

        print(f"[{tag}] sampling {args.mbr_samples} for external MBR (wrapped)…")
        cands = mtr.sample(inputs, args.mbr_samples, batch_size=1,
                           max_new_tokens=args.max_new_tokens,
                           top_p=args.top_p, temperature=args.temperature)
        out_df[f"{tag}_mbr_chrf_en"] = mbr_select_chrF(cands)
        compute_metrics(f"{tag}_mbr_chrf_en")

    # ------ run requested models ------
    if args.mbart_seq:
        run_plain("mbart_seq", make_mbart_seq_plain(hf_token=args.hf_token))

    if args.mbart_cg:
        run_plain("mbart_cg", make_mbart_cg_plain(hf_token=args.hf_token))

    if args.mbart_seq_mbr:
        run_mbr("mbart_seq_mbr", make_mbart_seq_mbr(hf_token=args.hf_token))

    if args.mbart_cg_mbr:
        run_mbr("mbart_cg_mbr", make_mbart_cg_mbr(hf_token=args.hf_token))

    out_df.to_csv(args.out_path, index=False)
    print(f"\n✅ Saved → {args.out_path}")


if __name__ == "__main__":
    main()

"""
Created on Tue Aug 12 2025 — mBART-only edition
"""

"""
Created on Tue Aug 12 18:12:44 2025

@author: niran
"""

