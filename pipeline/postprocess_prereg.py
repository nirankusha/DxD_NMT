# pipeline/postprocess_prereg.py
import argparse, json, hashlib, time
from pathlib import Path
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

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--selected", required=True)
    ap.add_argument("--outdir", default="artifacts")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--align-batch-size", type=int, default=24)
    ap.add_argument("--fail-on-empty-align", action="store_true")
    ap.add_argument("--export-csv", action="store_true")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    runs_p = Path(args.runs)
    sel_p  = Path(args.selected)
    if not runs_p.exists():
        raise FileNotFoundError(runs_p)
    if not sel_p.exists():
        raise FileNotFoundError(sel_p)

    df = pd.read_parquet(runs_p)
    for col in ["src_pl", "text_en"]:
        if col not in df.columns:
            raise KeyError(f"runs missing {col}")

    # src_det prereg fallback
    if "src_det" not in df.columns:
        df["src_det"] = np.nan
    miss = df["src_det"].isna() | df["src_det"].astype(str).str.lower().isin(["none","nan",""])
    order_col = "src_order_binary" if "src_order_binary" in df.columns else ("src_order" if "src_order" in df.columns else None)
    if order_col is not None and miss.any():
        df.loc[miss, "src_det"] = df.loc[miss, order_col].apply(expected_det_from_pl_order)

    # alignment regeneration
    aligner = AwesomeAligner(device=args.device)
    src = df["src_pl"].astype(str).tolist()
    tgt = df["text_en"].astype(str).tolist()

    print(f"[postprocess] aligning {len(src)} sentence pairs…")
    res = aligner.align_batch(src, tgt, batch_size=args.align_batch_size)
    pairs = [r.get("word_align", []) for r in res]
    df["ann_align_pairs"] = pairs
    df["ann_align_kendall_tau"] = [compute_tau(p) for p in pairs]

    n_tau = int(pd.Series(df["ann_align_kendall_tau"]).notna().sum())
    print(f"[postprocess] non-null tau: {n_tau}/{len(df)}")
    if args.fail_on_empty_align and n_tau == 0:
        raise RuntimeError("Alignment produced zero tau values. Aborting per prereg hardening.")

    df.to_parquet(runs_p, index=False)

    # merge into selected if possible
    sel = pd.read_parquet(sel_p)
    key_cols = [c for c in ["sent_id","model","mode","pool_type","cand_id"] if c in sel.columns and c in df.columns]
    if key_cols:
        sel2 = sel.merge(df[key_cols + ["ann_align_kendall_tau","src_det"]], on=key_cols, how="left")
        sel2.to_parquet(sel_p, index=False)

    manifest = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "runs_parquet": str(runs_p),
        "selected_parquet": str(sel_p),
        "runs_sha256": sha256_file(runs_p),
        "selected_sha256": sha256_file(sel_p),
        "rows_runs": int(len(df)),
        "cols_runs": int(len(df.columns)),
        "non_null_tau": n_tau,
        "device": args.device,
        "align_batch_size": args.align_batch_size,
    }
    (outdir/"prereg_manifest.json").write_text(json.dumps(manifest, indent=2))

    if args.export_csv:
        keep = [c for c in [
            "sent_id","origin","src_pl","text_en","model","mode","pool_type","cand_id",
            "src_order","src_order_binary","ann_order","ann_order_binary",
            "ann_det_general","src_det","ann_align_kendall_tau",
            "rt_chrf","rt_bleu","rt_ter","rt_sem","labse_cross",
            "ft_gold_metric","ft_gold_score","selected_flag"
        ] if c in df.columns]
        df[keep].to_csv(outdir/"analysis_ready_runs.csv", index=False)

    print(f"[postprocess] wrote manifest + optional csv to {outdir}")

if __name__ == "__main__":
    main()
