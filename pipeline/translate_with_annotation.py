# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
translate_with_annotation.py — Reference-free PL→EN evaluation pipeline with:
- Families: Marian, mT5 (sdadas & google/mt5-base), mBART (m2m/o2m/m2o incl. facebook/mbart-large-50-many-to-many-mmt)
- FT/RT alignment per architecture+variant, or explicit override via --ft-model / --rt-model
- Objectives: seq, cg, mbr (external), zmbr (Zurich decode)
- Reference-free scoring by RT-as-gold (choose EN pseudo-gold using --ft-gold-from)
- Semantic similarity backends: --sim-backend {xlmr,labse}
- Optional Google baseline
- Optional COMET-QE (reference-free)
- Optional annotation (--annotate): alignment/slots/determinacy/order/branching
- Pooled candidates into runs.parquet; selected into selected_summary.parquet

CLI (new, systematized):
  --architecture    mt5|mbart|marian|all    (repeatable)
  --variant   (per architecture)            mt5:{o2o,m2m}; mbart:{m2m,o2m,m2o}; marian:{o2o,o2m}
  --objective seq|cg|mbr|zmbr         (repeatable)
  --runs-all  run all valid combos

Back-compat flags are accepted and mapped:
  --mbart-seq, --mbart-cg, --mt5-seq, --mt5-cg,
  --marian-mbr, --mbart-seq-mbr, --mbart-cg-mbr, --mt5-seq-mbr, --mt5-cg-mbr
"""

import os, sys, json, argparse, platform, gc, subprocess
from typing import List, Optional, Tuple, Dict, Any

import pandas as pd
import sacrebleu
import torch
from tqdm import tqdm
import transformers
from huggingface_hub import snapshot_download

# Optional Google baseline
try:
    from google.cloud import translate_v2 as translate
    HAS_GOOGLE = True
except Exception:
    HAS_GOOGLE = False

# Optional ZurichNLP MBR
HAS_MBR_LIB = True
try:
    from mbr import MBR, MBRConfig
except Exception:
    HAS_MBR_LIB = False

try:
    from transformers import MBart50TokenizerFast as MBartTok
except Exception:
    from transformers import MBartTokenizer as MBartTok

try:
    from transformers import set_seed as hf_set_seed
except Exception:
    hf_set_seed = None


# ============================== Device & env ==============================
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================== HF helpers ==============================
def hf_auth_kwargs(hf_token: str):
    return {"use_auth_token": hf_token, "token": hf_token} if hf_token else {}

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
        return transformers.MBartForConditionalGeneration.from_pretrained(model_id, **kw)
    except Exception:
        local = snapshot_download(repo_id=model_id, token=hf_token, local_files_only=False)
        return transformers.MBartForConditionalGeneration.from_pretrained(local)

def load_mt5_tokenizer(model_id: str, hf_token: str):
    kw = hf_auth_kwargs(hf_token)
    try:
        return transformers.MT5Tokenizer.from_pretrained(model_id, **kw)
    except Exception:
        local = snapshot_download(repo_id=model_id, token=hf_token, local_files_only=False)
        return transformers.MT5Tokenizer.from_pretrained(local)

def load_mt5_cg(model_id: str, hf_token: str):
    kw = hf_auth_kwargs(hf_token)
    try:
        return transformers.MT5ForConditionalGeneration.from_pretrained(model_id, **kw)
    except Exception:
        local = snapshot_download(repo_id=model_id, token=hf_token, local_files_only=False)
        return transformers.MT5ForConditionalGeneration.from_pretrained(local)

# ============================== Embedding backends ==============================
_LABSE = None
_XLMR = None

def maybe_load_labse():
    global _LABSE
    if _LABSE is None:
        try:
            from sentence_transformers import SentenceTransformer
            _LABSE = SentenceTransformer("sentence-transformers/LaBSE")
        except Exception:
            _LABSE = False
    return _LABSE

def maybe_load_xlmr():
    global _XLMR
    if _XLMR is None:
        try:
            from sentence_transformers import SentenceTransformer
            _XLMR = SentenceTransformer("sentence-transformers/paraphrase-xlm-r-multilingual-v1")
        except Exception:
            _XLMR = False
    return _XLMR

def cosine_rows(a, b):
    return (a * b).sum(axis=1)

def labse_scores(a_texts: List[str], b_texts: List[str]) -> List[float]:
    st = maybe_load_labse()
    if not st: return [float("nan")] * len(a_texts)
    import numpy as np
    a = st.encode(a_texts, normalize_embeddings=True, convert_to_numpy=True)
    b = st.encode(b_texts, normalize_embeddings=True, convert_to_numpy=True)
    return cosine_rows(a, b).tolist()

def xlmr_scores(a_texts: List[str], b_texts: List[str]) -> List[float]:
    st = maybe_load_xlmr()
    if not st: return [float("nan")] * len(a_texts)
    import numpy as np
    a = st.encode(a_texts, normalize_embeddings=True, convert_to_numpy=True)
    b = st.encode(b_texts, normalize_embeddings=True, convert_to_numpy=True)
    return cosine_rows(a, b).tolist()

# ============================== Seeding ==============================
def set_global_seed(seed: Optional[int]):
    if seed is None: return
    try:
        import random, numpy as np
        random.seed(seed); np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if hf_set_seed: hf_set_seed(seed)
    except Exception:
        pass

# ============================== Google baseline ==============================
def google_translate_batch(texts: List[str], target_language: str,
                           credentials_path: Optional[str] = None,
                           source_language: Optional[str] = None,
                           chunk_size: int = 100) -> Tuple[List[str], List[Optional[str]]]:
    if not HAS_GOOGLE:
        raise RuntimeError("google-cloud-translate not installed.")
    if credentials_path:
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
    client = translate.Client()
    out_trans, out_detect = [], []
    texts = [t.decode("utf-8") if isinstance(t, (bytes, bytearray)) else str(t) for t in texts]
    for i in range(0, len(texts), chunk_size):
        chunk = texts[i:i+chunk_size]
        resp = client.translate(values=chunk, target_language=target_language,
                                source_language=source_language, format_="text")
        if isinstance(resp, dict): resp = [resp]
        out_trans.extend([r["translatedText"] for r in resp])
        out_detect.extend([r.get("detectedSourceLanguage") for r in resp])
    return out_trans, out_detect

# ============================== Generation helpers ==============================
def batched(lst, bs: int):
    for i in range(0, len(lst), bs):
        yield lst[i:i+bs], i, min(i+bs, len(lst))

def decode(outputs, tok):
    if hasattr(outputs, "sequences"):
        outputs = outputs.sequences
    if isinstance(outputs, (list, tuple)):
        if len(outputs) == 0: return []
        if isinstance(outputs[0], str): return list(outputs)
        if torch.is_tensor(outputs[0]):
            try: outputs = torch.stack(outputs)
            except Exception: pass
    return tok.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True)

def generate_beam_top1(model, tok, inputs: List[str], batch_size: int,
                       max_new_tokens=128, num_beams=5, **extra) -> List[str]:
    outs = []
    for batch, _, _ in batched(inputs, batch_size):           
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(DEVICE)
        with torch.inference_mode():
            gen = model.generate(**enc, max_new_tokens=max_new_tokens,
                                 num_beams=max(1, num_beams),
                                 num_return_sequences=1, early_stopping=True, do_sample=False, **extra)
        outs.extend(decode(gen, tok))
    return outs

def generate_beam_k(model, tok, inputs: List[str], *, k: int, max_new_tokens=128, num_beams=5) -> List[List[str]]:
    pools: List[List[str]] = []
    for text in tqdm(inputs, desc="Beam search (top-K)"):
        enc = tok([text], return_tensors="pt", padding=True, truncation=True, max_length=512).to(DEVICE)
        with torch.inference_mode():
            gen = model.generate(**enc, max_new_tokens=max_new_tokens, num_beams=max(num_beams, k),
                                 num_return_sequences=k, early_stopping=True, do_sample=False)
        pools.append(decode(gen, tok))
    return pools

def generate_samples(model, tok, inputs: List[str], *, num_samples=6, batch_size: int = 1,
                     max_new_tokens=128, top_p=0.95, temperature=0.7, **extra) -> List[List[str]]:
    all_cands: List[List[str]] = []
    for text in tqdm(inputs, desc="Sampling candidates"):
        enc = tok([text], return_tensors="pt", padding=True, truncation=True, max_length=512).to(DEVICE)
        cands = []
        with torch.inference_mode():
            for _ in range(num_samples):
                gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=True,
                                     top_p=top_p, temperature=temperature, num_beams=1, **extra)
                cands.append(decode(gen, tok)[0])
        all_cands.append(cands)
    return all_cands

# ============================== MBR consensus ==============================
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

# ============================== Translators (FT & matched RT) ==============================
class BaseTranslator:
    def __init__(self, model, tokenizer, forced_bos_token_id: Optional[int] = None):
        self.model = model.to(DEVICE)
        self.tok = tokenizer
        self.extra = {}
        if forced_bos_token_id is not None:
            self.extra["forced_bos_token_id"] = forced_bos_token_id
    def mbr_generate(self, inputs: List[str], **gen_kwargs):
        outs = []
        bs = int(gen_kwargs.pop("batch_size", 8))
        for batch, _, _ in batched(inputs, bs):
            enc = self.tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(DEVICE)
            with torch.inference_mode():
                gen = self.model.generate(**enc, **self.extra, **gen_kwargs)
            outs.append(gen)
        return outs
    
    def zmbr_generate(self, inputs: List[str], **gen_kwargs):
        return self.mbr_generate(inputs, **gen_kwargs)

    def beam_top1(self, inputs: List[str], batch_size: int, max_new_tokens: int, num_beams: int, **kwargs):
        return generate_beam_top1(self.model, self.tok, inputs, batch_size, max_new_tokens, num_beams, **self.extra)

    def beam_k(self, inputs: List[str], k: int, max_new_tokens: int, num_beams: int, **kwargs):
        return generate_beam_k(self.model, self.tok, inputs, k=k, max_new_tokens=max_new_tokens, num_beams=num_beams)

    def sample(self, inputs: List[str], num_samples: int, batch_size: int, max_new_tokens: int,
               top_p: float, temperature: float, **kwargs):
        return generate_samples(self.model, self.tok, inputs,
                                num_samples=num_samples, batch_size=batch_size,
                                max_new_tokens=max_new_tokens, top_p=top_p,
                                temperature=temperature, **self.extra)

class PrefixTranslator(BaseTranslator):
    """
    Injects a natural-language task prefix into the encoder input.
    Ensures ALL generation paths (beam, sample, mbr, zmbr) receive the same conditioning.
    """
    def __init__(self, model, tokenizer, prefix: str, forced_bos_token_id: Optional[int] = None):
        super().__init__(model, tokenizer, forced_bos_token_id)
        self.prefix = prefix

    def _add_prefix(self, inputs: List[str]) -> List[str]:
        # keep it simple; avoid double spaces; allow empty strings to pass through unchanged
        return [f"{self.prefix} {t}".strip() if t.strip() else t for t in inputs]

    # ----- Unified entrypoints (NEW) -----
    def mbr_generate(self, inputs: List[str], **gen_kwargs):
        return super().mbr_generate(self._add_prefix(inputs), **gen_kwargs)

    def zmbr_generate(self, inputs: List[str], **gen_kwargs):
        # identical to mbr_generate; separated for clarity if you diverge later
        return super().mbr_generate(self._add_prefix(inputs), **gen_kwargs)

    # ----- Beam / sample APIs (keyword-safe) -----
    def beam_top1(self, inputs: List[str], batch_size: int, max_new_tokens: int, num_beams: int, **kwargs):
        return super().beam_top1(self._add_prefix(inputs), batch_size, max_new_tokens, num_beams, **kwargs)

    def beam_k(self, inputs: List[str], k: int, max_new_tokens: int, num_beams: int, **kwargs):
        return super().beam_k(self._add_prefix(inputs), k=k, max_new_tokens=max_new_tokens, num_beams=num_beams, **kwargs)

    def sample(self, inputs: List[str], num_samples: int, batch_size: int, max_new_tokens: int,
               top_p: float, temperature: float, **kwargs):
        return super().sample(self._add_prefix(inputs), num_samples, batch_size, max_new_tokens, top_p, temperature, **kwargs)

def needs_mt5_base_prefix(model_name: str) -> bool:
    return model_name.lower().startswith("google/mt5")

# --- Marian ---


def make_marian_pl_en_seq(hf_token="") -> BaseTranslator:
    name = "Helsinki-NLP/opus-mt-pl-en"
    tok = load_tokenizer(name, hf_token)
    mdl = load_seq2seq(name, hf_token)
    return BaseTranslator(mdl, tok)

def make_marian_en_pl_seq(hf_token="", model_name="Helsinki-NLP/opus-mt-en-mul") -> BaseTranslator:
    tok = load_tokenizer(model_name, hf_token)
    mdl = load_seq2seq(model_name, hf_token)
    return PrefixTranslator(mdl, tok, ">>pl<<")

# --- mBART ---
MBART_ID_M2M = "facebook/mbart-large-50-many-to-many-mmt"
MBART_ID_O2M = "facebook/mbart-large-50-one-to-many-mmt"
MBART_ID_M2O = "facebook/mbart-large-50-many-to-one-mmt"

def make_mbart_seq(src_lang_code: str, tgt_lang_code: str, model_id: str, hf_token="") -> BaseTranslator:
    tok = load_tokenizer(model_id, hf_token)
    forced_bos = tok.lang_code_to_id.get(tgt_lang_code, None) if hasattr(tok, "lang_code_to_id") else None
    if hasattr(tok, "src_lang"): tok.src_lang = src_lang_code
    mdl = load_seq2seq(model_id, hf_token)
    return BaseTranslator(mdl, tok, forced_bos)

def make_mbart_cg(src_lang_code: str, tgt_lang_code: str, model_id: str, hf_token="") -> BaseTranslator:
    tok = load_mbart_tokenizer(model_id, hf_token)
    forced_bos = tok.lang_code_to_id.get(tgt_lang_code, None) if hasattr(tok, "lang_code_to_id") else None
    if hasattr(tok, "src_lang"): tok.src_lang = src_lang_code
    mdl = load_mbart_cg(model_id, hf_token)
    return BaseTranslator(mdl, tok, forced_bos)

# --- mT5 ---
def make_mt5_seq(model_name: str, hf_token="") -> BaseTranslator:
    tok = load_tokenizer(model_name, hf_token)
    mdl = load_seq2seq(model_name, hf_token)
    if needs_mt5_base_prefix(model_name):
        return PrefixTranslator(mdl, tok, "translate Polish to English:")
    return BaseTranslator(mdl, tok)

def make_mt5_cg(model_name: str, hf_token="") -> BaseTranslator:
    tok = load_mt5_tokenizer(model_name, hf_token)
    mdl = load_mt5_cg(model_name, hf_token)
    if needs_mt5_base_prefix(model_name):
        return PrefixTranslator(mdl, tok, "translate Polish to English:")
    return BaseTranslator(mdl, tok)

# --- Back-translator chooser (EN->PL) ---
def make_backtranslator_for(model_key: str, ft_model_name: Optional[str], mbart_variant: str, hf_token="") -> BaseTranslator:
    mk = (model_key or "").lower()
    if mk.startswith("marian"):
        # use en-pl (o2o) OR en-mul caveat if you prefer; here standard en-pl
        return make_marian_en_pl_seq(hf_token=hf_token, model_name="Helsinki-NLP/opus-mt-en-mul")
    if mk.startswith("mt5"):
        if ft_model_name and "sdadas/mt5-base-translator-pl-en" in ft_model_name:
            return make_mt5_cg("sdadas/mt5-base-translator-en-pl", hf_token=hf_token)
        if ft_model_name and "sdadas/mt5-base-translator-en-pl" in ft_model_name:
            return make_mt5_cg("sdadas/mt5-base-translator-pl-en", hf_token=hf_token)
        return PrefixTranslator(load_mt5_cg("google/mt5-base", hf_token=hf_token),
                                load_mt5_tokenizer("google/mt5-base", hf_token=hf_token),
                                "translate English to Polish:")
    if mk.startswith("mbart"):
        var = (mbart_variant or "m2m").lower()
        mid = {"m2m": MBART_ID_M2M, "o2m": MBART_ID_O2M, "m2o": MBART_ID_M2O}[var]
        # back dir EN->PL
        return make_mbart_cg("en_XX", "pl_PL", mid, hf_token=hf_token)
    # default
    return make_marian_en_pl_seq(hf_token=hf_token)

# ============================== Zurich MBR constructors ==============================
def build_mbr_config(metric="chrf", num_samples=8, **out_flags):
    if not HAS_MBR_LIB:
        raise RuntimeError("ZurichNLP mbr is not installed. pip install 'transformers<4.39' mbr")
    return MBRConfig(metric=metric, num_samples=num_samples, **out_flags)

def make_marian_mbr(hf_token=""):
    name = "Helsinki-NLP/opus-mt-pl-en"
    tok = load_tokenizer(name, hf_token)
    MBRMarian = MBR(transformers.MarianMTModel)
    kw = hf_auth_kwargs(hf_token)
    try:
        mdl = MBRMarian.from_pretrained(name, **kw)
    except Exception:
        local = snapshot_download(repo_id=name, token=hf_token, local_files_only=False)
        mdl = MBRMarian.from_pretrained(local)
    return BaseTranslator(mdl, tok)

def make_mt5_mbr(model_name="sdadas/mt5-base-translator-pl-en", hf_token=""):
    tok = load_mt5_tokenizer(model_name, hf_token)
    MBRMT5 = MBR(transformers.MT5ForConditionalGeneration)
    kw = hf_auth_kwargs(hf_token)
    try:
        mdl = MBRMT5.from_pretrained(model_name, **kw)
    except Exception:
        local = snapshot_download(repo_id=model_name, token=hf_token, local_files_only=False)
        mdl = MBRMT5.from_pretrained(local)
    if needs_mt5_base_prefix(model_name):
        return PrefixTranslator(mdl, tok, "translate Polish to English:")
    return BaseTranslator(mdl, tok)

def make_mbart_mbr(src_lang_code="pl_PL", tgt_lang_code="en_XX", variant="m2m", hf_token=""):
    model_id = {"m2m": MBART_ID_M2M, "o2m": MBART_ID_O2M, "m2o": MBART_ID_M2O}[variant]
    tok = load_mbart_tokenizer(model_id, hf_token)
    forced_bos = tok.lang_code_to_id.get(tgt_lang_code, None) if hasattr(tok, "lang_code_to_id") else None
    if hasattr(tok, "src_lang"): tok.src_lang = src_lang_code
    MBRMBart = MBR(transformers.MBartForConditionalGeneration)
    kw = hf_auth_kwargs(hf_token)
    try:
        mdl = MBRMBart.from_pretrained(model_id, **kw)
    except Exception:
        local = snapshot_download(repo_id=model_id, token=hf_token, local_files_only=False)
        mdl = MBRMBart.from_pretrained(local)
    return BaseTranslator(mdl, tok, forced_bos)

# ============================== RT metrics (EN->PL) ==============================
def roundtrip_scores(src_pl: List[str], hyp_en: List[str], backtrans: BaseTranslator,
                     batch_size: int, max_new_tokens: int, num_beams: int = 5
                     ) -> Tuple[List[float], List[float], List[float], List[str]]:
    bt_pl = backtrans.beam_top1(hyp_en, batch_size=batch_size, max_new_tokens=max_new_tokens, num_beams=num_beams)
    chrf = []; bleu = []; ter = []
    for bt, src in zip(bt_pl, src_pl):
        chrf.append(sacrebleu.sentence_chrf(bt, [src]).score)
        bleu.append(sacrebleu.sentence_bleu(bt, [src]).score)
        ter.append(sacrebleu.sentence_ter(bt, [src]).score)
    return chrf, bleu, ter, bt_pl

# Semantic selection via RT cosine (XLM-R / LaBSE)
def select_by_rt_sem(src_pl: List[str], pools_en: List[List[str]], backtrans: BaseTranslator,
                     batch_size: int, max_new_tokens: int, num_beams: int, backend: str) -> List[str]:
    selected = []
    use_xlmr = backend == "xlmr"
    for s, cands in zip(src_pl, pools_en):
        if not cands: selected.append(""); continue
        if len(cands) == 1: selected.append(cands[0]); continue
        bt = backtrans.beam_top1(cands, batch_size=1, max_new_tokens=max_new_tokens, num_beams=num_beams)
        scores = xlmr_scores([s]*len(bt), bt) if use_xlmr else labse_scores([s]*len(bt), bt)
        best_idx = max(range(len(cands)), key=lambda k: (scores[k] if scores[k] == scores[k] else float("-inf")))
        selected.append(cands[best_idx])
    return selected

# ============================== RT-as-gold pseudo-reference ==============================
def _rt_metric_vector(src_pl_list, bt_pl_list, metric: str, sim_backend: str):
    if metric == "rt_chrf":
        return [sacrebleu.sentence_chrf(bt, [src]).score for bt, src in zip(bt_pl_list, src_pl_list)]
    if metric == "rt_bleu":
        return [sacrebleu.sentence_bleu(bt, [src]).score for bt, src in zip(bt_pl_list, src_pl_list)]
    if metric == "rt_ter":
        return [sacrebleu.sentence_ter(bt, [src]).score for bt, src in zip(bt_pl_list, src_pl_list)]
    if metric == "rt_sem_xlmr":
        return xlmr_scores(src_pl_list, bt_pl_list)
    if metric == "rt_sem_labse":
        return labse_scores(src_pl_list, bt_pl_list)
    raise ValueError(f"Unknown rt metric: {metric}")

def pick_en_gold_for_sentence(src_pl: str, cands_en: List[str], backtrans: BaseTranslator,
                              max_new_tokens: int, num_beams: int,
                              metric: str, sim_backend: str) -> Tuple[int, str, str, float]:
    if not cands_en:
        return -1, "", "", float("nan")
    bt_pl = backtrans.beam_top1(cands_en, batch_size=1, max_new_tokens=max_new_tokens, num_beams=num_beams)
    scores = _rt_metric_vector([src_pl]*len(bt_pl), bt_pl, metric, sim_backend)
    if metric == "rt_ter":
        best_idx = min(range(len(scores)), key=lambda i: scores[i] if scores[i] == scores[i] else float("inf"))
    else:
        best_idx = max(range(len(scores)), key=lambda i: scores[i] if scores[i] == scores[i] else float("-inf"))
    return best_idx, cands_en[best_idx], bt_pl[best_idx], scores[best_idx]

def forward_vs_gold_metrics(cands_en: List[str], en_gold: str):
    if not en_gold:
        n = len(cands_en)
        return [float("nan")]*n, [float("nan")]*n, [float("nan")]*n
    bleu = [sacrebleu.sentence_bleu(h, [en_gold]).score for h in cands_en]
    chrf = [sacrebleu.sentence_chrf(h, [en_gold]).score for h in cands_en]
    ter  = [sacrebleu.sentence_ter(h,  [en_gold]).score for h in cands_en]
    return bleu, chrf, ter

# ============================== Pool utils ==============================
def flatten_pools(srcs: List[str], pools: List[List[str]]) -> Tuple[List[int], List[int], List[str], List[str]]:
    sent_ids, cand_ids, src_flat, en_flat  = [], [], [], []
    for s_id, (src, cands) in enumerate(zip(srcs, pools)):
        for c_id, en in enumerate(cands):
            sent_ids.append(s_id); cand_ids.append(c_id); src_flat.append(src); en_flat.append(en)
    return sent_ids, cand_ids, src_flat, en_flat

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def write_parquet_or_csv(df: pd.DataFrame, path: str):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".parquet":
        try:
            df.to_parquet(path, index=False); return
        except Exception:
            path = os.path.splitext(path)[0] + ".csv"
    df.to_csv(path, index=False)

def free_model(*objs):
    for o in objs:
        try: del o
        except Exception: pass
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try: torch.cuda.ipc_collect()
        except Exception: pass
    gc.collect()

# ============================== Annotation ==============================
def _parse_order(order_str: str | None):
    if not order_str: return None
    s = str(order_str).lower().strip().replace("-", "_").replace(" ", "")
    s = s.replace(">", "_before_").replace("<", "_before_")
    if "_before_" in s:
        l, r = s.split("_before_", 1)
        def canon(x):
            if "subj" in x: return "SUBJ"
            if "root" in x: return "ROOT"
            return x.upper()
        return (canon(l), canon(r))
    if "subj" in s and "root" in s:
        return ("SUBJ","ROOT") if s.index("subj") < s.index("root") else ("ROOT","SUBJ")
    return None

def order_binary_pl_from_order(order_str: str | None):
    pr = _parse_order(order_str)
    if not pr: return None
    left, right = pr
    if left == "SUBJ" and right == "ROOT": return "Det"
    if left == "ROOT" and right == "SUBJ": return "Indef"
    return None

def order_binary_en_from_det(subj_det_label: str | None):
    if not isinstance(subj_det_label, str): return None
    low = subj_det_label.strip().lower()
    if low.startswith("def"):   return "Det"
    if low.startswith("indef"): return "Indef"
    return None

def _kendall_tau_from_pairs(pairs):
    if not pairs or len(pairs) < 2: return None
    pairs2 = [(int(s), int(t)) for (s, t) in pairs]
    pairs2.sort(key=lambda x: x[0])
    seq = [t for _, t in pairs2]
    try:
        from scipy.stats import kendalltau
        return kendalltau(range(len(seq)), seq, nan_policy='omit').correlation
    except Exception:
        return None

def _branching_for_targets_spacy_fallback(doc, positions):
    out = []
    if doc is None: return [None for _ in (positions or [])]
    def _num_children(tok):
        n_lefts = getattr(tok, "n_lefts", None)
        n_rights = getattr(tok, "n_rights", None)
        if n_lefts is not None and n_rights is not None:
            return int(n_lefts) + int(n_rights)
        return sum(1 for _ in tok.children)
    n = len(doc)
    for p in (positions or []):
        if p is None or p < 0 or p >= n:
            out.append(None); continue
        tok = doc[p]
        out.append({
            "i": p,
            "text": tok.text,
            "dep": tok.dep_,
            "head_i": tok.head.i if tok.head is not None else None,
            "head_text": tok.head.text if tok.head is not None else None,
            "num_children": _num_children(tok),
        })
    return out

def load_annotation_modules():
    import sys
    if "" not in sys.path:
        sys.path.insert(0, "")

    mods = {}

    try:
        from alignment_wrappers import AwesomeAligner
        mods["AwesomeAligner"] = AwesomeAligner
    except Exception as e:
        print(f"[annotate][WARN] AwesomeAligner not available: {e}")
        mods["AwesomeAligner"] = None

    try:
        from syntax_slots import SyntaxSlots
        mods["SyntaxSlots"] = SyntaxSlots
    except Exception as e:
        print(f"[annotate][WARN] SyntaxSlots not available: {e}")
        mods["SyntaxSlots"] = None

    try:
        from determinacy import subject_np_determinacy
        mods["subject_np_determinacy"] = subject_np_determinacy
    except Exception as e:
        print(f"[annotate][WARN] determinacy.subject_np_determinacy not available: {e}")
        mods["subject_np_determinacy"] = None

    try:
        from target_extractor import extract_tuples
        mods["extract_tuples"] = extract_tuples
    except Exception as e:
        print(f"[annotate][WARN] target_extractor.extract_tuples not available: {e}")
        mods["extract_tuples"] = None

    try:
        from features import branching_for_targets_spacy as _branch
        mods["branching_for_targets_spacy"] = _branch
    except Exception:
        mods["branching_for_targets_spacy"] = _branching_for_targets_spacy_fallback

    if mods.get("SyntaxSlots") is None:
        print("[annotate][WARN] Annotation disabled (SyntaxSlots missing).")
        return None

    return mods

def annotate_source_list(src_list: List[str], mods, lang="pl"):
    if not mods or mods.get("SyntaxSlots") is None:
        n = len(src_list)
        return [None]*n, [None]*n, [None]*n, [None]*n

    slots = mods["SyntaxSlots"](lang)
    detf  = mods.get("subject_np_determinacy")

    # Batch process all texts at once
    batch_results = slots.analyze_batch(src_list)
    
    res_order, res_obin, res_subj, res_det = [], [], [], []
    for a in batch_results:
        ordv = a.get("order")
        res_order.append(ordv)
        res_obin.append(order_binary_pl_from_order(ordv))
        res_subj.append(a.get("subj_span"))
        try:
            if detf is not None:
                res_det.append(detf(a.get("doc"), a.get("subj_span"), lang, mode="general"))
            else:
                res_det.append(None)
        except Exception:
            res_det.append(None)

    return res_order, res_obin, res_subj, res_det

def annotate_pair(src: str, en: str, mods, tgt_lang="en", target_words=None):
    if not mods or mods.get("SyntaxSlots") is None:
        return dict(order=None, order_binary=None, subj_span=None, det_general=None,
                    align_kendall_tau=None, align_pairs=None, target_positions=None, target_branching=None)

    slots  = mods["SyntaxSlots"](tgt_lang)
    detf   = mods.get("subject_np_determinacy")
    branch = mods.get("branching_for_targets_spacy")

    a = slots.analyze(en or "")
    order = a.get("order")
    subj_span = a.get("subj_span")

    try:
        det = detf(a.get("doc"), subj_span, tgt_lang, mode="general") if detf is not None else None
    except Exception:
        det = None
    order_b = order_binary_en_from_det(det)

    align_pairs = None
    kt = None
    pairs = []
    try:
        AA = mods["AwesomeAligner"]() if mods.get("AwesomeAligner") else None
        if AA:
            r = AA.align(src or "", en or "")
            pairs = r.get("word_align", []) or []
            align_pairs = str(pairs)
            kt = _kendall_tau_from_pairs(pairs)
    except Exception:
        pass

    positions = []
    branching = None
    try:
        doc = a.get("doc")
        toks = [t.text for t in doc] if doc is not None else []
        used = set()
        for w in (target_words or []):
            wlow = str(w).strip().lower()
            pos = -1
            for j, tok in enumerate(toks):
                if j in used: continue
                if tok.lower() == wlow:
                    pos = j; used.add(j); break
            positions.append(pos)
        if any(p == -1 for p in positions) and pairs:
            src_tokens = (src or "").split()
            tgt2src = {}
            for (si, ti) in pairs:
                si, ti = int(si), int(ti)
                tgt2src.setdefault(ti, set()).add(si)
            for k, (tw, pos) in enumerate(zip((target_words or []), positions)):
                if pos != -1: continue
                twlow = str(tw).strip().lower()
                src_hits = [i for i, st in enumerate(src_tokens) if str(st).lower() == twlow]
                cand_t = []
                for si in src_hits:
                    for ti, src_set in tgt2src.items():
                        if si in src_set:
                            cand_t.append(ti)
                chosen = -1
                for ti in sorted(set(cand_t)):
                    if 0 <= ti < len(toks) and ti not in used:
                        chosen = ti; used.add(ti); break
                positions[k] = chosen

        if doc is not None and branch is not None:
            branching = branch(doc, positions)
    except Exception:
        pass

    return dict(
        order=order,
        order_binary=order_b,
        subj_span=subj_span,
        det_general=det,
        align_kendall_tau=kt,
        align_pairs=align_pairs,
        target_positions=positions,
        target_branching=branching
    )

# ============================== Architecture/variant/objective plan ==============================
def paired_ids_for_architecture_variant(architecture: str, variant: str) -> dict:
    arch = architecture.lower()
    var = variant.lower()

    if arch == "mt5":
        if var == "o2o":
            return {
                "ft_id": "sdadas/mt5-base-translator-pl-en",
                "rt_id": "sdadas/mt5-base-translator-en-pl",
                "ft_variant": "o2o", "rt_variant": "o2o",
                "ft_needs_prefix": False, "rt_needs_prefix": False,
                "notes": "mT5 sdadas o2o pair"
            }
        if var == "m2m":
            return {
                "ft_id": "google/mt5-base",
                "rt_id": "google/mt5-base",
                "ft_variant": "m2m", "rt_variant": "m2m",
                "ft_needs_prefix": True, "rt_needs_prefix": True,
                "notes": "mT5 base m2m; use prefixes"
            }
        raise ValueError(f"mt5 variant not recognized: {variant}")

    if arch == "mbart":
        if var == "m2m":
            return {
                "ft_id": "facebook/mbart-large-50-many-to-many-mmt",
                "rt_id": "facebook/mbart-large-50-many-to-many-mmt",
                "ft_variant": "m2m", "rt_variant": "m2m",
                "ft_langs": ("pl_PL","en_XX"), "rt_langs": ("en_XX","pl_PL"),
                "notes": "mBART m2m both directions"
            }
        if var in ("m2o","o2m"):
            return {
                "ft_id": "facebook/mbart-large-50-many-to-one-mmt",
                "rt_id": "facebook/mbart-large-50-one-to-many-mmt",
                "ft_variant": "m2o", "rt_variant": "o2m",
                "ft_langs": ("pl_PL","en_XX"), "rt_langs": ("en_XX","pl_PL"),
                "notes": "mBART paired m2o/o2m"
            }
        raise ValueError(f"mbart variant not recognized: {variant}")

    if arch == "marian":
        if var in ("o2o","o2m"):
            return {
                "ft_id": "Helsinki-NLP/opus-mt-pl-en",
                "rt_id": "Helsinki-NLP/opus-mt-en-mul", 
                "ft_variant": "o2o", "rt_variant": "o2o",
                "notes": "Marian: FT o2o pl-en; RT en-pl (o2o)."
            }
        raise ValueError(f"marian variant not recognized: {variant}")

    raise ValueError(f"architecture not recognized: {architecture}")

def normalize_arch_variants(args) -> list[tuple[str,str]]:
    # decide which architectures
    if args.architecture:
        architectures = [a.lower() for a in args.architecture if a.lower() != "all"]
        if "all" in (a.lower() for a in args.architecture):
            architectures = ["mt5","mbart","marian"]
    else:
        architectures = []
        # back-compat inference
        if args.mbart:  architectures.append("mbart")
        if args.mt5:    architectures.append("mt5")
        if args.marian: architectures.append("marian")
        if not architectures:
            architectures = ["mt5"]  # default

    allowed = {
        "mt5":   ["o2o","m2m"],
        "mbart": ["m2m","o2m","m2o"],
        "marian":["o2o","o2m"],
    }

    req_variants = set(v.lower() for v in (args.variant or []))

    arch_var = []
    for arch in architectures:
        if req_variants:
            vs = [v for v in allowed[arch] if v in req_variants]
            if not vs:
                continue
        else:
            vs = allowed[arch] if (args.runs_all or not args.variant) else allowed[arch]
        for v in vs:
            arch_var.append((arch, v))
    return sorted(set(arch_var))

def normalize_objectives(args) -> list[str]:
    all_obj = ["seq","cg","mbr","zmbr"]
    if args.objective and len(args.objective) > 0:
        return [o.lower() for o in args.objective]
    if args.runs_all or (args.variant and not args.objective):
        return all_obj
    return all_obj

def build_run_plan(args):
    plan = []
    # explicit overrides → single config with chosen objectives
    if args.ft_model or args.rt_model:
        if not (args.ft_model and args.rt_model):
            raise ValueError("If overriding, provide BOTH --ft-model and --rt-model.")
        objectives = normalize_objectives(args)
        for obj in objectives:
            plan.append({
                "architecture": "custom",
                "variant": "custom",
                "objective": obj,
                "ft_id": args.ft_model,
                "rt_id": args.rt_model,
                "ft_meta": {},
                "rt_meta": {},
                "notes": "explicit FT/RT override"
            })
        return plan

    arch_vars = normalize_arch_variants(args)
    objectives = normalize_objectives(args)

    for arch, var in arch_vars:
        ids = paired_ids_for_architecture_variant(arch, var)
        for obj in objectives:
            plan.append({
                "architecture": arch,
                "variant": var,
                "objective": obj,
                "ft_id": ids["ft_id"],
                "rt_id": ids["rt_id"],
                "ft_meta": {k: ids[k] for k in ids if k.startswith("ft_")},
                "rt_meta": {k: ids[k] for k in ids if k.startswith("rt_")},
                "notes": ids.get("notes","")
            })
    return plan

# ============================== Zurich MBR runner ==============================
def run_zmbr_for_family(tag_prefix: str, inputs: List[str], args, backtrans: BaseTranslator,
                        runs_rows: List[Dict[str, Any]], selected_rows: List[Dict[str, Any]],
                        out_df: pd.DataFrame, vers: Dict[str, Any], fam_key_for_print: str):
    if not HAS_MBR_LIB:
        print("[zMBR][WARN] mbr library not available — skipping.")
        return

    # construct zMBR translator according to family/variant
    if tag_prefix.startswith("marian"):
        zmbr_tr = make_marian_mbr(hf_token=args.hf_token)
    
    elif tag_prefix.startswith("mt5"):
        zmbr_tr = make_mt5_mbr(model_name=args.mt5_model, hf_token=args.hf_token)
    
    else:  # mbart
        variant = args.mbart_variant if hasattr(args, "mbart_variant") and args.mbart_variant else "m2m"
        zmbr_tr = make_mbart_mbr("pl_PL", "en_XX", variant, hf_token=args.hf_token)

    zmbr_tr.model.mbr_config = build_mbr_config(metric=args.mbr_dec_metric, num_samples=args.mbr_dec_samples,
                                                return_dict_in_generate=True,
                                                output_all_samples=True,
                                                output_reference_sequences=True,
                                                output_metric_scores=True)

    print(f"[{tag_prefix}] MBR decoding (metric={args.mbr_dec_metric}, samples={args.mbr_dec_samples})…")
    
    outs = zmbr_tr.zmbr_generate(
        inputs,
        batch_size=args.batch_size,
        do_sample=True, num_beams=1,
        top_p=args.top_p, temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        mbr_config=zmbr_tr.model.mbr_config, tokenizer=zmbr_tr.tok
        )
    
    selected_texts: List[str] = []
    s_id_global = 0
    for out, (batch, _, _) in zip(outs, batched(inputs, args.batch_size)):
        sel_texts = decode(out.sequences, zmbr_tr.tok)
        selected_texts.extend(sel_texts)
        # Log selected rows immediately with RT metrics (for selected_summary)
        for i, (src, en) in enumerate(zip(batch, sel_texts)):
            rt_chrf, rt_bleu, rt_ter, bt_pl = roundtrip_scores([src], [en], backtrans,
                                                               batch_size=args.batch_size,
                                                               max_new_tokens=args.max_new_tokens,
                                                               num_beams=args.beam)
            rt_sem = xlmr_scores([src], bt_pl) if args.sim_backend == "xlmr" else labse_scores([src], bt_pl)
            selected_rows.append({
                "sent_id": s_id_global+i, "model": fam_key_for_print, "mode": "mbr", "pool_type": "zmbr",
                "cand_id": "selected", "selected_flag": True,
                "src_pl": src, "text_en": en,
                "rt_chrf": rt_chrf[0], "rt_bleu": rt_bleu[0], "rt_ter": rt_ter[0], "rt_sem": rt_sem[0],
                "decode_params": json.dumps({
                    "batch_size": args.batch_size, "beam": args.beam,
                    "max_new_tokens": args.max_new_tokens,
                    "top_p": args.top_p, "temperature": args.temperature,
                    "mbr_metric": args.mbr_dec_metric, "mbr_num_samples": args.mbr_dec_samples,
                    "sim_backend": args.sim_backend,
                }),
                "versions": json.dumps(vers),
            })
        s_id_global += len(batch)

    out_df[f"{tag_prefix}_en"] = selected_texts
    free_model(zmbr_tr)

ARCH_TO_VARIANTS = {
    "mt5":   {"o2o", "m2m"},
    "mbart": {"m2m", "o2m", "m2o"},
    "marian":{"o2o", "o2m"},
}

# ============================== Main ==============================


def main():
    ap = argparse.ArgumentParser(
        description="Reference-free PL→EN pipeline: architectures/variants/objectives, RT-as-gold, semantic similarity, (z)MBR, COMET-QE, optional annotations.")

    # IO
    ap.add_argument("--in", dest="in_path", nargs="+", required=True,
                help="One or more input file paths. Each will get an 'origin' column in the combined dataframe.")
    ap.add_argument("--out", dest="out_path", required=True,
                    help="Output directory (will contain artifacts and *.parquet results) or a file path for a single CSV/Parquet.")
    ap.add_argument("--col", default="", help="Name of the source-text column (if reading a table); leave empty for line-by-line text files.")
    ap.add_argument("--artifacts-dir", default = "artifacts",
                    help = "Where to save intermediate artifacts (models cache, candidate pools, parquet files).")

    # Auth / seed
    ap.add_argument("--hf-token", default = os.getenv("HF_TOKEN", ""),
                    help = "Hugging Face access token (if required for private models).")
    ap.add_argument("--seed", type = int, default = None,
                    help = "Random seed. If not set, no explicit seeding is done.")

    # Google baseline
    ap.add_argument("--google", action = "store_true",
                    help = "Add Google Translate baseline for PL→EN (requires credentials).")
    ap.add_argument("--google-credentials", default = os.getenv("GOOGLE_APPLICATION_CREDENTIALS",
                    ""), help = "Path to a Google service-account JSON credentials file.")

    # Architecture/variant/objective selectors
    ap.add_argument("--architecture", nargs = "+",
                    help = "Architecture(s) to include. One or more of: mt5, mbart, marian, all. Use 'all' to include all architectures.")
    ap.add_argument("--variant", nargs = "+",
                help="Variant(s) per architecture. mt5:{o2o,m2m}; mbart:{m2m,o2m,m2o}; marian:{o2o,o2m}")
    ap.add_argument("--objective", nargs="+", choices=["seq","cg","mbr","zmbr"],
                    help="Decoding objective(s): seq,cg,mbr,zmbr. If given, only these are run.")
    ap.add_argument("--runs-all", action="store_true",
                    help="Run all valid (architecture × variant × objective) combinations.")

    # Direct overrides: explicit FT/RT model ids (highest precedence)
    ap.add_argument("--ft-model", default="", help="Override FT (PL->EN) model id.")
    ap.add_argument("--rt-model", default="", help="Override RT (EN->PL) model id.")

    # Back-compat toggles (still supported)
    ap.add_argument("--marian", action="store_true", help="Back-compat: include Marian models (use --architecture marian in the new API).")
    ap.add_argument("--mt5", action="store_true", help="Back-compat: include mT5 models (use --architecture mt5 in the new API).")
    ap.add_argument("--mbart", action="store_true", help="Back-compat: include mBART models (use --architecture mbart in the new API).")
    ap.add_argument("--mt5-model", default="sdadas/mt5-base-translator-pl-en", help="Default mT5 PL→EN model to use for zMBR helper.")
    ap.add_argument("--mbart-variant", choices=["m2m","o2m","m2o"], default="m2m", help="mBART translation direction variant for back-compat helpers.")
    ap.add_argument(
    "--include-variants", nargs="+", default=[],
    help="Whitelist variants per architecture, e.g. mt5:o2o  mbart:m2m,m2o  marian:o2o"
    )
    ap.add_argument(
    "--exclude-variants", nargs="+", default=[],
    help="Blacklist variants per architecture, e.g. mt5:m2m  mbart:o2m  marian:o2m"
    )


    # Decoding knobs
    ap.add_argument("--batch-size", type=int, default=8, help="Batch size for generation and scoring.")
    ap.add_argument("--beam", type=int, default=5, help="Num beams for beam search in seq/cg modes.")
    ap.add_argument("--beam-k-cands", type=int, default=0, help="If >0, pool top-K beam candidates per input for later selection.")
    ap.add_argument("--max-new-tokens", type=int, default=128, help="Max new tokens to generate.")
    ap.add_argument("--top-p", type=float, default=0.95, help="Top-p nucleus sampling (for sampling modes).")
    ap.add_argument("--temperature", type=float, default=0.7, help="Temperature for sampling.")

    # External MBR / sampling
    ap.add_argument("--mbr-samples", type=int, default=8, help="Number of samples per sentence for external MBR (if enabled).")
    ap.add_argument("--sample-k-cands", type=int, default=0, help="If >0, pool K sampling candidates per input for later selection.")
    ap.add_argument("--sample-match-beam-k", action="store_true", help="If set, make sampling pool size equal to --beam-k-cands when both are on.")

    # Zurich MBR
    ap.add_argument("--zmbr", action="store_true", help="Enable Zurich MBR (zMBR) decoding helper for certain families.")
    ap.add_argument("--mbr-dec-samples", type=int, default=6, help="zMBR: number of decoder samples per input.")
    ap.add_argument("--mbr-dec-metric", default="chrf", help="zMBR: internal selection metric (chrf|bleu|ter).")

    # COMET-QE
    ap.add_argument("--run-comet", action="store_true", help="Run COMET-QE (reference-free); requires comet-model and comet-score binary.")
    ap.add_argument("--comet-model", default="Unbabel/wmt20-comet-qe-da", help="COMET-QE model id for comet-score.")
    ap.add_argument("--comet-bin", default="comet-score", help="Path to the comet-score CLI binary.")

    # Semantic backend & RT-as-gold
    ap.add_argument("--sim-backend", choices=["xlmr","labse"], default="xlmr", help="Semantic similarity backend for RT-as-gold: XLM-R cosine or LaBSE.")
    ap.add_argument("--ft-gold-from",
                    choices=["rt_chrf","rt_bleu","rt_ter","rt_sem_xlmr","rt_sem_labse"],
                    default="rt_chrf", help="How to select the pseudo-gold from FT outputs for per-sentence selection.")

    # Annotation
    ap.add_argument("--annotate", action="store_true", help="Enable extra linguistic annotations (alignment/slots/determinacy/order/branching).")

    # --------- Legacy flags (mapping only; do not remove) ----------
    ap.add_argument("--mbart-seq", action="store_true", help="Legacy shortcut: run mBART seq.")
    ap.add_argument("--mbart-cg", action="store_true", help="Legacy shortcut: run mBART constrained grammar (cg).")
    ap.add_argument("--mt5-seq", action="store_true", help="Legacy shortcut: run mT5 seq.")
    ap.add_argument("--mt5-cg", action="store_true", help="Legacy shortcut: run mT5 cg.")
    ap.add_argument("--marian-mbr", action="store_true", help="Legacy shortcut: run Marian with MBR.")
    ap.add_argument("--mbart-seq-mbr", action="store_true", help="Legacy shortcut: run mBART seq with external MBR.")
    ap.add_argument("--mbart-cg-mbr", action="store_true", help="Legacy shortcut: run mBART cg with external MBR.")
    ap.add_argument("--mt5-seq-mbr", action="store_true", help="Legacy shortcut: run mT5 seq with external MBR.")
    ap.add_argument("--mt5-cg-mbr", action="store_true", help="Legacy shortcut: run mT5 cg with external MBR.")

    args = ap.parse_args()
    set_global_seed(args.seed)
    ensure_dir(args.artifacts_dir)

    # Map legacy flags → new selectors
    legacy_selected = False
    legacy_objectives = set()
    if args.mbart_seq or args.mbart_cg: args.mbart = True
    if args.mt5_seq  or args.mt5_cg:    args.mt5   = True
    if args.marian_mbr or args.mbart_seq_mbr or args.mbart_cg_mbr or args.mt5_seq_mbr or args.mt5_cg_mbr:
        args.zmbr = True

    if args.mbart_seq: legacy_selected = True; legacy_objectives.add("seq"); args.architecture = (args.architecture or []) + ["mbart"]
    if args.mbart_cg:  legacy_selected = True; legacy_objectives.add("cg");  args.architecture = (args.architecture or []) + ["mbart"]
    if args.mt5_seq:   legacy_selected = True; legacy_objectives.add("seq"); args.architecture = (args.architecture or []) + ["mt5"]
    if args.mt5_cg:    legacy_selected = True; legacy_objectives.add("cg");  args.architecture = (args.architecture or []) + ["mt5"]
    if legacy_selected and not args.objective:
        args.objective = list(legacy_objectives)

    # Load data
    dfs = []
    for path in args.in_path:
        df_part = pd.read_csv(path)
        df_part["origin"] = os.path.basename(path)   # track file of origin

        # add Pairs Idx column if missing (only synth has it by default)
        if "Pairs Idx" not in df_part.columns:
            df_part["Pairs Idx"] = pd.NA

        dfs.append(df_part)

    df = pd.concat(dfs, ignore_index=True)

    col = args.col or ("Edited Sentence" if "Edited Sentence" in df.columns else "Merged Context")
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found. Available: {list(df.columns)}")
    inputs = df[col].fillna("").astype(str).tolist()

    out_df = df.copy()
    out_df["original"] = inputs

    runs_rows: List[Dict[str, Any]] = []
    selected_rows: List[Dict[str, Any]] = []

    vers = {
        "python": platform.python_version(),
        "torch": getattr(torch, "__version__", "n/a"),
        "transformers": getattr(transformers, "__version__", "n/a"),
        "sacrebleu": getattr(sacrebleu, "__version__", "n/a"),
        "mbr": "installed" if HAS_MBR_LIB else "not_installed",
    }
    # ---- helpers: arch/variant planning & metrics (single, canonical copy) ----
    def parse_architecture_flag(items):
        out = {}
        for it in items or []:
            arch, _, vs = it.partition(":")
            arch = arch.strip().lower()
            if not arch or arch not in ARCH_TO_VARIANTS:
                continue
            chosen = set(v.strip().lower() for v in vs.split(",") if v.strip())
            out[arch] = chosen
        return out

    def expand_architectures(args):
        archs = set()
        # new API
        if args.architecture:
            archs.update(a.lower() for a in args.architecture)

        # legacy toggles — must be outside the block above
        if getattr(args, "mbart", False):
            archs.add("mbart")
        if getattr(args, "mt5", False):
            archs.add("mt5")
        if getattr(args, "marian", False):
            archs.add("marian")

        # sensible default (also accepts --architecture all)
        if not archs or ("all" in archs):
            archs = {"mt5", "mbart", "marian"}
        return archs

    def expand_variants_for_arch(arch, args, include_map, exclude_map):
        # start with all legal variants
        base = ARCH_TO_VARIANTS[arch].copy()

        # user-specified --variant narrows globally
        if args.variant:
            base &= set(v.lower() for v in args.variant)

        # per-arch include/exclude maps
        if arch in include_map and include_map[arch]:
            base &= include_map[arch]
        if arch in exclude_map and exclude_map[arch]:
            base -= exclude_map[arch]

        return base

    def build_run_plan(args):
        # per-arch include/exclude from CLI
        include_map = parse_architecture_flag(getattr(args, "include_variants", []))
        exclude_map = parse_architecture_flag(getattr(args, "exclude_variants", []))

        # smart defaults ONLY if --runs-all and user didn't specify any filters
        if args.runs_all and not args.variant and not include_map and not exclude_map:
            exclude_map = {
                # avoid redundant/backward Marian direction & weak mt5 base
                "marian": {"o2m"},
                "mt5": {"m2m"},
                }

        archs = expand_architectures(args)

        # objectives
        if args.objective and len(args.objective) > 0:
            objectives = [o.lower() for o in args.objective]
        else:
            objectives = ["seq", "cg", "mbr", "zmbr"]

        plan = []

        # explicit overrides → single config with chosen objectives
        if args.ft_model or args.rt_model:
            if not (args.ft_model and args.rt_model):
                raise ValueError("If overriding, provide BOTH --ft-model and --rt-model.")
            for obj in objectives:
                plan.append({
                    "architecture": "custom",
                    "variant": "custom",
                    "objective": obj,
                    "ft_id": args.ft_model,
                    "rt_id": args.rt_model,
                    "ft_meta": {},
                    "rt_meta": {},
                    "notes": "explicit FT/RT override"
                })
            
            return plan

        for arch in sorted(archs):
            variants = expand_variants_for_arch(arch, args, include_map, exclude_map)
            for var in sorted(variants):
                ids = paired_ids_for_architecture_variant(arch, var)
                for obj in objectives:
                    plan.append({
                        "architecture": arch,
                        "variant": var,
                        "objective": obj,
                        "ft_id": ids["ft_id"],
                        "rt_id": ids["rt_id"],
                        "ft_meta": {k: ids[k] for k in ids if k.startswith("ft_")},
                        "rt_meta": {k: ids[k] for k in ids if k.startswith("rt_")},
                        "notes": ids.get("notes", "")
                    })
        return plan

    def compute_metrics_on_texts(src_list: List[str], en_list: List[str], backtrans: BaseTranslator) -> Dict[str, List[float]]:
        rt_chrf, rt_bleu, rt_ter, bt_pl = roundtrip_scores(
            src_list, en_list, backtrans,
            batch_size=args.batch_size, max_new_tokens=args.max_new_tokens,
            num_beams=args.beam
            )
        rt_sem = xlmr_scores(src_list, bt_pl) if args.sim_backend == "xlmr" else labse_scores(src_list, bt_pl)
        labse_cross = labse_scores(src_list, en_list)
        return {"rt_chrf": rt_chrf, "rt_bleu": rt_bleu, "rt_ter": rt_ter,
                "rt_sem": rt_sem, "labse_cross": labse_cross, "bt_pl": bt_pl}


    # Annotation modules
    mods = load_annotation_modules() if args.annotate else None
    src_order, src_obin, src_subj, src_det = annotate_source_list(inputs, mods, lang="pl")


    # Sampling parity K
    sample_k_cands = args.sample_k_cands
    if args.sample_match_beam_k and (args.beam_k_cands and args.beam_k_cands > 0):
        sample_k_cands = args.beam_k_cands

    # Plan of (architecture, variant, objective)
    plan = build_run_plan(args)
    if not plan:
        raise ValueError("No (architecture,variant,objective) to run. Check your flags.")

    print("\n=== RUN PLAN ===")
    for cfg in plan:
        print(f"- architecture={cfg['architecture']}, variant={cfg['variant']}, objective={cfg['objective']}, "
          f"FT={cfg['ft_id']}, RT={cfg['rt_id']} | {cfg.get('notes','')}")
    print("================\n")

    # Google baseline once (optional)
    if args.google and HAS_GOOGLE and args.google_credentials:
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = args.google_credentials
        try:
            print("Translating with Google baseline…")
            google_out, google_detected = google_translate_batch(
                inputs, target_language="en", source_language=None
                )
            out_df["google_en"] = google_out
            out_df["google_detected"] = google_detected
        except Exception as e:
            print(f"[WARN] Google failed ({e}). Skipping Google baseline.")

    # helper to add selected
    def add_selected_rows(model: str, mode: str, en_list: List[str], tag: str, backtrans: BaseTranslator):
        mets = compute_metrics_on_texts(inputs, en_list, backtrans)
        for i, en in enumerate(en_list):
            row = {
                "sent_id": i, "model": model, "mode": mode,
                "cand_id": "selected", "selected_flag": True,
                "src_pl": inputs[i], "text_en": en, "tag": tag,
                "rt_chrf": mets["rt_chrf"][i], "rt_bleu": mets["rt_bleu"][i], "rt_ter": mets["rt_ter"][i],
                "rt_sem":  mets["rt_sem"][i], "labse_cross": mets["labse_cross"][i],
                "decode_params": json.dumps({
                    "batch_size": args.batch_size, "beam": args.beam,
                    "max_new_tokens": args.max_new_tokens, "top_p": args.top_p, "temperature": args.temperature,
                    "sim_backend": args.sim_backend,
                }),
                "versions": json.dumps(vers),
            }
            if args.annotate:
                twords = out_df.at[i, 'Target words'] if 'Target words' in out_df.columns else []
                ann = annotate_pair(inputs[i], en, mods, tgt_lang="en", target_words=twords)
                row.update({
                    "ann_order": ann["order"],
                    "ann_order_binary": ann["order_binary"],
                    "ann_subj_span": ann["subj_span"],
                    "ann_det_general": ann["det_general"],
                    "ann_align_kendall_tau": ann["align_kendall_tau"],
                    "ann_align_pairs": ann["align_pairs"],
                    "ann_target_positions": str(ann["target_positions"]),
                    "ann_target_branching": str(ann["target_branching"]),
                    "src_order": src_order[i],
                    "src_order_binary": src_obin[i],
                    "src_subj_span": src_subj[i],
                    "src_det": src_det[i],
                })
            selected_rows.append(row)

    # helper to log pools
    def log_pool_candidates_df(model: str, mode: str, pool_type: str, pools: List[List[str]], backtrans: BaseTranslator) -> pd.DataFrame:
        sent_ids, cand_ids, src_flat, en_flat = flatten_pools(inputs, pools)
        mets = compute_metrics_on_texts(src_flat, en_flat, backtrans)

        # consensus (within-pool) averages, aligned to flattened candidates
        cons_all_chrf: List[float] = []; cons_all_bleu: List[float] = []; cons_all_ter: List[float] = []
        for pool in pools:
            if not pool:
                cons_all_chrf.append(float("nan")); cons_all_bleu.append(float("nan")); cons_all_ter.append(float("nan")); continue
            scores_chrf = []; scores_bleu = []; scores_ter = []
            for i, ci in enumerate(pool):
                chrf_vals = [sacrebleu.sentence_chrf(ci, [cj]).score for j, cj in enumerate(pool) if j != i]
                bleu_vals = [sacrebleu.sentence_bleu(ci, [cj]).score for j, cj in enumerate(pool) if j != i]
                ter_vals  = [sacrebleu.sentence_ter(ci,  [cj]).score for j, cj in enumerate(pool) if j != i]
                scores_chrf.append(sum(chrf_vals)/max(1,len(chrf_vals)))
                scores_bleu.append(sum(bleu_vals)/max(1,len(bleu_vals)))
                scores_ter.append(sum(ter_vals)/max(1,len(ter_vals)))
            cons_all_chrf.extend(scores_chrf); cons_all_bleu.extend(scores_bleu); cons_all_ter.extend(scores_ter)

        df_block = pd.DataFrame({
            "sent_id": sent_ids, "model": model, "mode": mode, "pool_type": pool_type,
            "cand_id": cand_ids, "selected_flag": False,
            "src_pl": src_flat, "text_en": en_flat,
            "origin": [out_df.at[sid, "origin"] for sid in sent_ids],
            "rt_chrf": mets["rt_chrf"], "rt_bleu": mets["rt_bleu"], "rt_ter": mets["rt_ter"],
            "rt_sem":  mets["rt_sem"],  "labse_cross": mets["labse_cross"],
            "decode_params": json.dumps({
                "pool_type": pool_type,
                "batch_size": args.batch_size, "beam": args.beam,
                "max_new_tokens": args.max_new_tokens, "top_p": args.top_p, "temperature": args.temperature,
                "sim_backend": args.sim_backend,
                }),
            "versions": json.dumps(vers),
            })

        df_block["cons_chrf_avg"] = cons_all_chrf[:len(df_block)]
        df_block["cons_bleu_avg"] = cons_all_bleu[:len(df_block)]
        df_block["cons_ter_avg"]  = cons_all_ter[:len(df_block)]

        # ------- Annotation (safe & fast) -------
        if args.annotate:
            twords_col = out_df["Target words"] if "Target words" in out_df.columns else [[] for _ in range(len(out_df))]

            # cache target-side analysis per unique EN text within this df_block
            unique_en = list(dict.fromkeys(df_block["text_en"]))
            en_text_to_result = {}
            try:
                target_slots = mods["SyntaxSlots"]("en") if mods else None
                if target_slots and unique_en:
                    batch_target_results = target_slots.analyze_batch(unique_en)
                    en_text_to_result = dict(zip(unique_en, batch_target_results))
            except Exception:
                en_text_to_result = {}

            a_order=[]; a_order_b=[]; a_subj=[]; a_det=[]; a_kt=[]; a_pairs=[]; a_pos=[]; a_branch=[]
            s_order=[]; s_order_b=[]; s_subj=[]; s_det_list=[]

            detf = mods.get("subject_np_determinacy") if mods else None
            for sid, en_text in zip(df_block["sent_id"], df_block["text_en"]):
                twords = twords_col[sid] if isinstance(twords_col, pd.Series) else []
                if en_text in en_text_to_result:
                    tr = en_text_to_result[en_text]
                    try:
                        det_val = detf(tr.get("doc"), tr.get("subj_span"), "en", mode="general") if detf else None
                    except Exception:
                        det_val = None
                    ann = {
                        "order": tr.get("order"),
                        "order_binary": order_binary_en_from_det(det_val),
                        "subj_span": tr.get("subj_span"),
                        "det_general": det_val,
                        "align_kendall_tau": None,
                        "align_pairs": None,
                        "target_positions": [],
                        "target_branching": None,
                        }   
                else:
                    ann = annotate_pair(inputs[sid], en_text, mods, tgt_lang="en", target_words=twords)

                a_order.append(ann["order"]); a_order_b.append(ann["order_binary"])
                a_subj.append(ann["subj_span"]); a_det.append(ann["det_general"])
                a_kt.append(ann["align_kendall_tau"]); a_pairs.append(ann["align_pairs"])
                a_pos.append(str(ann["target_positions"])); a_branch.append(str(ann["target_branching"]))
                s_order.append(src_order[sid]); s_order_b.append(src_obin[sid])
                s_subj.append(src_subj[sid]); s_det_list.append(src_det[sid])

            df_block["ann_order"] = a_order
            df_block["ann_order_binary"] = a_order_b
            df_block["ann_subj_span"] = a_subj
            df_block["ann_det_general"] = a_det
            df_block["ann_align_kendall_tau"] = a_kt
            df_block["ann_align_pairs"] = a_pairs
            df_block["ann_target_positions"] = a_pos
            df_block["ann_target_branching"] = a_branch
            df_block["src_order"] = s_order
            df_block["src_order_binary"] = s_order_b
            df_block["src_subj_span"] = s_subj
            df_block["src_det"] = s_det_list
        # ----------------------------------------

        return df_block

    # ========= RUN ALL CONFIGS IN PLAN =========
    for cfg in plan:
        arch_key   = cfg["architecture"]
        variant   = cfg["variant"]
        objective = cfg["objective"]
        ft_id     = cfg["ft_id"]
        rt_id     = cfg["rt_id"]

        # Build FT according to objective
        if arch_key == "custom":
            # explicit overrides: try to guess architecture; use CG loaders for safety
            arch_print = f"custom:{ft_id}"
            if "mbart" in ft_id.lower():
                ft_tr = make_mbart_cg("pl_PL", "en_XX", ft_id, hf_token=args.hf_token)
            elif "mt5" in ft_id.lower():
                ft_tr = make_mt5_cg(ft_id, hf_token=args.hf_token)
            else:
                tok = load_tokenizer(ft_id, args.hf_token); mdl = load_seq2seq(ft_id, args.hf_token)
                ft_tr = BaseTranslator(mdl, tok)
        else:
            arch_print = f"{arch_key}:{variant}:{objective}"
            if arch_key == "marian":
                ft_tr = make_marian_pl_en_seq(hf_token=args.hf_token)
            elif arch_key == "mt5":
                model_name = ft_id
                if objective == "seq":
                    ft_tr = make_mt5_seq(model_name, hf_token=args.hf_token)
                else:  # cg/mbr/zmbr share same backbone
                    ft_tr = make_mt5_cg(model_name, hf_token=args.hf_token)
            else:  # mbart
                src_tgt = cfg.get("ft_meta", {}).get("ft_langs", ("pl_PL","en_XX"))
                mid = ft_id
                if objective == "seq":
                    ft_tr = make_mbart_seq(src_tgt[0], src_tgt[1], mid, hf_token=args.hf_token)
                else:
                    ft_tr = make_mbart_cg(src_tgt[0], src_tgt[1], mid, hf_token=args.hf_token)

        # Build RT
        if cfg["architecture"] == "custom" and args.rt_model:
            # explicit override for RT
            if "mbart" in rt_id.lower():
                rt_tr = make_mbart_cg("en_XX", "pl_PL", rt_id, hf_token=args.hf_token)
            elif "mt5" in rt_id.lower():
                # add EN->PL prefix if needed
                if needs_mt5_base_prefix(rt_id):
                    rt_tr = PrefixTranslator(load_mt5_cg(rt_id, args.hf_token),
                                             load_mt5_tokenizer(rt_id, args.hf_token),
                                             "translate English to Polish:")
                else:
                    rt_tr = make_mt5_cg(rt_id, hf_token=args.hf_token)
            else:
                tok = load_tokenizer(rt_id, args.hf_token); mdl = load_seq2seq(rt_id, args.hf_token)
                rt_tr = BaseTranslator(mdl, tok)
        else:
            rt_tr = make_backtranslator_for(arch_key, ft_id, variant if variant!="custom" else "m2m", hf_token=args.hf_token)

        print(f"\n[RUN] architecture={arch_key} variant={variant} objective={objective}")
        print(f"     FT={ft_id}")
        print(f"     RT={rt_id}")

        # 1) Beam top-1 (SEQ & CG & MBR setups all can log this)
        beam_out = ft_tr.beam_top1(inputs, args.batch_size, args.max_new_tokens, args.beam)
        out_df_key = f"{arch_key}_{variant}_{objective}_beam_en"
        out_df[out_df_key] = beam_out
        add_selected_rows(arch_print, "plain", beam_out, tag=f"{arch_print}_beam", backtrans=rt_tr)
        print(f"  -> SELECTED summary will include: {arch_print} [beam top-1]")

        # 2) Beam-K (optional) → runs
        if args.beam_k_cands and args.beam_k_cands > 0:
            beam_pools = ft_tr.beam_k(inputs, k=args.beam_k_cands, max_new_tokens=args.max_new_tokens, num_beams=args.beam)
            runs_df_block = log_pool_candidates_df(arch_print, "beam_k", "beam", beam_pools, backtrans=rt_tr)
            runs_rows.extend(runs_df_block.to_dict("records"))
            print(f"  -> RUNS pool appended: {arch_print} [beam-K={args.beam_k_cands}]")

        # 3) Sample-K parity (optional) → runs + a selected CHRF-MBR
        sample_k_cands = args.beam_k_cands if args.sample_match_beam_k and (args.beam_k_cands and args.beam_k_cands > 0) else args.sample_k_cands
        if sample_k_cands and sample_k_cands > 0:
            sample_k_pools = ft_tr.sample(inputs, num_samples=sample_k_cands, batch_size=1,
                                          max_new_tokens=args.max_new_tokens, top_p=args.top_p, temperature=args.temperature)
            runs_df_block = log_pool_candidates_df(arch_print, "sample_k", "sample_plain_k", sample_k_pools, backtrans=rt_tr)
            runs_rows.extend(runs_df_block.to_dict("records"))
            sel_k = mbr_select_chrF(sample_k_pools)
            out_df[f"{arch_print}_samplek_chrf_en"] = sel_k
            add_selected_rows(arch_print, "sample_k", sel_k, tag=f"{arch_print}_samplek_chrf", backtrans=rt_tr)
            print(f"  -> RUNS pool appended: {arch_print} [sample-K={sample_k_cands}]")
            print(f"  -> SELECTED summary appended: {arch_print} [sample-K MBR(chrf)]")

        # 4) External MBR sampling → runs + selections
        pools = ft_tr.sample(inputs, args.mbr_samples, batch_size=1,
                             max_new_tokens=args.max_new_tokens, top_p=args.top_p, temperature=args.temperature)
        runs_df_block = log_pool_candidates_df(arch_print, "ext_mbr", "sample", pools, backtrans=rt_tr)

        # EN pseudo-gold via RT metric
        en_gold_list, bt_gold_pl_list, gold_idx_list, gold_score_list = [], [], [], []
        for s, cands in zip(inputs, pools):
            best_idx, en_gold, bt_gold_pl, best_score = pick_en_gold_for_sentence(
                s, cands, rt_tr, max_new_tokens=args.max_new_tokens, num_beams=args.beam,
                metric=args.ft_gold_from, sim_backend=args.sim_backend
            )
            en_gold_list.append(en_gold)
            bt_gold_pl_list.append(bt_gold_pl)
            gold_idx_list.append(best_idx)
            gold_score_list.append(best_score)

        # forward vs gold, per-candidate
        sent_ids, _, _, en_flat = flatten_pools(inputs, pools)
        ft_bleu_gold_flat, ft_chrf_gold_flat, ft_ter_gold_flat = [], [], []
        for sid, en_cand in zip(sent_ids, en_flat):
            en_gold = en_gold_list[sid] if sid < len(en_gold_list) else ""
            b, c, t = forward_vs_gold_metrics([en_cand], en_gold)
            ft_bleu_gold_flat.append(b[0]); ft_chrf_gold_flat.append(c[0]); ft_ter_gold_flat.append(t[0])
        runs_df_block["ft_bleu_vs_gold"] = ft_bleu_gold_flat
        runs_df_block["ft_chrf_vs_gold"] = ft_chrf_gold_flat
        runs_df_block["ft_ter_vs_gold"]  = ft_ter_gold_flat
        runs_df_block["ft_gold_en"]      = [en_gold_list[sid] for sid in runs_df_block["sent_id"]]
        runs_df_block["ft_gold_bt_pl"]   = [bt_gold_pl_list[sid] for sid in runs_df_block["sent_id"]]
        runs_df_block["ft_gold_idx"]     = [gold_idx_list[sid] for sid in runs_df_block["sent_id"]]
        runs_df_block["ft_gold_metric"]  = args.ft_gold_from
        runs_df_block["ft_gold_score"]   = [gold_score_list[sid] for sid in runs_df_block["sent_id"]]

        runs_rows.extend(runs_df_block.to_dict("records"))
        print(f"  -> RUNS pool appended: {arch_print} [ext MBR sample={args.mbr_samples}]")

        # classic MBR selections + semantic selection
        sel_chrf = mbr_select_chrF(pools)
        sel_sem  = select_by_rt_sem(inputs, pools, rt_tr, args.batch_size, args.max_new_tokens, args.beam, backend=args.sim_backend)

        out_df[f"{arch_print}_mbr_chrf_en"] = sel_chrf
        out_df[f"{arch_print}_rtsem_{args.sim_backend}_en"] = sel_sem
        add_selected_rows(arch_print, "ext_mbr", sel_chrf, tag=f"{arch_print}_mbr_chrf", backtrans=rt_tr)
        add_selected_rows(arch_print, "ext_mbr", sel_sem,  tag=f"{arch_print}_rtsem_{args.sim_backend}", backtrans=rt_tr)
        print(f"  -> SELECTED summary appended: {arch_print} [MBR(chrf) & RT-sem({args.sim_backend})]")

        # 5) zMBR (if requested or objective is zmbr)
        if objective == "zmbr" or args.zmbr:
            tag = f"{arch_key}_{variant}_zmbr" if arch_key!="custom" else "custom_zmbr"
            run_zmbr_for_family(tag, inputs, args, rt_tr, runs_rows, selected_rows, out_df, vers, fam_key_for_print=arch_print)
            print(f"  -> SELECTED summary appended: {arch_print} [zMBR decode]")

        free_model(ft_tr, rt_tr)

    # ---- COMET-QE (optional) ----
    def add_comet_scores(df: pd.DataFrame, prefix: str):
        if df.empty: return df
        src_path = os.path.join(args.artifacts_dir, f"{prefix}_src.txt")
        mt_path  = os.path.join(args.artifacts_dir, f"{prefix}_mt.txt")
        out_json = os.path.join(args.artifacts_dir, f"{prefix}_comet.json")
        df["src_pl"].to_csv(src_path, index=False, header=False)
        df["text_en"].to_csv(mt_path,  index=False, header=False)
        cmd = [args.comet_bin, "-s", src_path, "-t", mt_path, "--model", args.comet_model, "--to_json", out_json]
        print(f"[COMET] Running: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            with open(out_json, "r", encoding="utf-8") as f:
                comet_json = json.load(f)
            if isinstance(comet_json, dict) and len(comet_json) == 1:
                first_key = list(comet_json.keys())[0]
                entries = comet_json[first_key]
                scores = [float(e.get("COMET", float("nan"))) for e in entries]
            elif isinstance(comet_json, list):
                scores = [float(e.get("COMET", float("nan"))) for e in comet_json]
            else:
                scores = [float("nan")] * len(df)
            if len(scores) != len(df):
                if len(scores) < len(df):
                    scores += [float("nan")] * (len(df) - len(scores))
                else:
                    scores = scores[:len(df)]
            df = df.copy(); df["comet_qe_da"] = scores
        except Exception as e:
            print(f"[COMET][WARN] Failed ({e}); filling NaNs.")
            df = df.copy(); df["comet_qe_da"] = float("nan")
        return df

    runs_df = pd.DataFrame(runs_rows)
    selected_df = pd.DataFrame(selected_rows)

    if args.run_comet:
        if not runs_df.empty:     runs_df = add_comet_scores(runs_df, "runs")
        if not selected_df.empty: selected_df = add_comet_scores(selected_df, "selected")

    # ---- Persist ----
    ensure_dir(args.artifacts_dir)
    if not runs_df.empty:
        write_parquet_or_csv(runs_df, os.path.join(args.artifacts_dir, "runs.parquet"))
    if not selected_df.empty:
        write_parquet_or_csv(selected_df, os.path.join(args.artifacts_dir, "selected_summary.parquet"))
    out_df.to_csv(args.out_path, index=False)

    print("\n✅ Saved wide translations →", args.out_path)
    print("🧪 Sanity: where did things go?")
    if not runs_df.empty:
        print("  • Candidate pools & per-candidate metrics →", os.path.join(args.artifacts_dir, 'runs.parquet'))
        print("    (Includes: rt_* metrics, labse_cross, consensus, ft_*_vs_gold, annotation if --annotate)")
    else:
        print("  • No pools were logged to runs.parquet (did you enable beam-k / sampling / mbr?).")
    if not selected_df.empty:
        print("  • Selected outputs (beam top-1, MBR(chrf), RT-sem, zMBR) →",
              os.path.join(args.artifacts_dir, 'selected_summary.parquet'))
    else:
        print("  • No selected_summary written (unexpected).")

if __name__ == "__main__":
    main()

"""
Created on Thu Aug 21 20:11:33 2025

@author: niran
"""

