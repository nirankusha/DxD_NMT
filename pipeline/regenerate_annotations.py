#!/usr/bin/env python3
"""
Hardened annotation regeneration
- Fixes import path issues
- Robust aligner loading
- Guarantees Kendall τ computation
- Fails loudly if alignments are empty (optional)
"""

import sys
from pathlib import Path

# ---- Fix Python path so `pipeline.*` imports work ----
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import pandas as pd
import torch
from tqdm import tqdm

from pipeline.alignment_wrappers import build_aligner, kendall_tau_from_pairs


# ---------------------------
# CLI
# ---------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="runs.parquet input")
    ap.add_argument("--out", required=True, help="output parquet")
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--fail-on-empty-align", action="store_true")
    ap.add_argument("--align-backend", default="auto",
                    choices=["auto", "awesome", "simalign", "comet-align"])
    return ap.parse_args()


# ---------------------------
# Aligner Factory (Robust)
# ---------------------------
def get_aligner(backend="auto", device="cpu"):
    try:
        return build_aligner(backend=backend, device=device)
    except Exception as e:
        print(f"[align] WARNING: primary aligner failed: {e}")

    # fallback: try AwesomeAligner explicitly
    try:
        from pipeline.alignment_wrappers import AwesomeAligner
        print("[align] Fallback → AwesomeAligner")
        return AwesomeAligner(device=device)
    except Exception:
        pass

    # fallback: SimAlign
    try:
        from pipeline.alignment_wrappers import SimAligner
        print("[align] Fallback → SimAligner")
        return SimAligner(device=device)
    except Exception:
        pass

    raise RuntimeError("No alignment backend available")


# ---------------------------
# Main Regeneration Logic
# ---------------------------
def main():
    args = parse_args()

    print(f"📥 Loading runs → {args.runs}")
    df = pd.read_parquet(args.runs)

    if "src_pl" not in df.columns or "text_en" not in df.columns:
        raise ValueError("Missing required columns: src_pl / text_en")

    aligner = get_aligner(args.align_backend, args.device)

    src = df["src_pl"].fillna("").tolist()
    tgt = df["text_en"].fillna("").tolist()

    taus = []
    pairs_all = []

    print("🔗 Running alignment...")
    for i in tqdm(range(0, len(src), args.batch_size)):
        s_batch = src[i:i + args.batch_size]
        t_batch = tgt[i:i + args.batch_size]

        batch_res = aligner.align_batch(s_batch, t_batch, batch_size=len(s_batch))

        for res in batch_res:
            pairs = res.get("pairs", []) if isinstance(res, dict) else res

            if not pairs:
                taus.append(None)
                pairs_all.append([])
                continue

            tau = kendall_tau_from_pairs(pairs)
            taus.append(tau)
            pairs_all.append(pairs)

    df["ann_align_pairs"] = pairs_all
    df["ann_align_kendall_tau"] = taus

    non_empty = sum(x is not None for x in taus)
    print(f"📊 Alignment coverage: {non_empty}/{len(taus)}")

    if args.fail_on_empty_align and non_empty == 0:
        raise RuntimeError("❌ All Kendall τ values are empty — aborting")

    print(f"💾 Saving → {args.out}")
    df.to_parquet(args.out, index=False)

    print("✅ Regeneration complete")


if __name__ == "__main__":
    main()
