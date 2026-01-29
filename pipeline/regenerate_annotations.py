import argparse
import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from pipeline.alignment_wrappers import AwesomeAligner

def expected_det_from_pl_order(order_val):
    if order_val is None:
        return None
    s = str(order_val).strip().upper()
    if "SV" in s:
        return "definite"
    if "VS" in s:
        return "indefinite"
    return None

def compute_tau(pairs):
    if not pairs or len(pairs) < 2:
        return np.nan
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    try:
        return float(kendalltau(xs, ys).correlation)
    except Exception:
        return np.nan

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    df = pd.read_parquet(args.runs)
    assert "src_pl" in df.columns and "text_en" in df.columns, "Expected src_pl/text_en in runs.parquet"

    if "src_det" not in df.columns:
        df["src_det"] = np.nan
    miss = df["src_det"].isna() | df["src_det"].astype(str).str.lower().isin(["none","nan",""])
    order_col = "src_order_binary" if "src_order_binary" in df.columns else ("src_order" if "src_order" in df.columns else None)
    if order_col is not None and miss.any():
        df.loc[miss, "src_det"] = df.loc[miss, order_col].apply(expected_det_from_pl_order)

    aligner = AwesomeAligner(device=args.device)
    src = df["src_pl"].astype(str).tolist()
    tgt = df["text_en"].astype(str).tolist()
    res = aligner.align_batch(src, tgt, batch_size=args.batch_size)
    pairs = [r.get("word_align", []) for r in res]
    df["ann_align_pairs"] = pairs
    df["ann_align_kendall_tau"] = [compute_tau(p) for p in pairs]
    df.to_parquet(args.out, index=False)
    print("Non-null tau:", int(pd.Series(df["ann_align_kendall_tau"]).notna().sum()), "/", len(df))

if __name__ == "__main__":
    main()
