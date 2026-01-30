# -*- coding: utf-8 -*-
#!/usr/bin/env python3
# analyze_construal_mapping.py
import os, json, argparse, warnings
import numpy as np
import pandas as pd

from ast import literal_eval
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.contingency_tables import Table2x2
from statsmodels.stats.weightstats import DescrStatsW

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

def parse_args():
    ap = argparse.ArgumentParser(description="Construal mapping analysis (χ², ANOVA, LMM, GEE) over runs/selected outputs.")
    ap.add_argument("--artifacts-dir", default="artifacts",
                    help="Directory that holds runs.parquet / selected_summary.parquet")
    ap.add_argument("--input-kind", choices=["runs","selected"], default="runs",
                    help="Use candidate pools (runs) or selected outputs only (selected).")
    ap.add_argument("--runs-path", default="artifacts/runs.parquet",
                    help="Path to runs.parquet (overrides --artifacts-dir).")
    ap.add_argument("--selected-path", default="artifacts/selected_summary.parquet",
                    help="Path to selected_summary.parquet (overrides --artifacts-dir).")
    ap.add_argument("--population-col", default="", 
                    help="Optional column name that marks population (e.g., 'population' with values {corpus,synthetic}). If empty, assumes single population.")
    ap.add_argument("--population-default", default="corpus",
                    help="If population column is missing/empty, fill with this single level.")
    ap.add_argument("--use-selected-flag", action="store_true",
                    help="Filter to rows where selected_flag==True (for runs). Ignored for input-kind=selected.")
    ap.add_argument("--tau-col", default="ann_align_kendall_tau",
                    help="Column with Kendall’s tau (alignment order score).")
    ap.add_argument("--success-col", default="construal_match_bin",
                    help="Binary success (construal matched) column (will be created if missing).")
    ap.add_argument("--sentence-type-col", default="order_cond",
                    help="SV/VS column (will be created from src_order if missing).")
    ap.add_argument("--expected-col", default="src_order_binary",
                    help="Expected construal column (PL mapping).")
    ap.add_argument("--actual-col", default="ann_order_binary",
                    help="Actual construal column (EN mapping).")
    # Factorial ANOVA columns (2x2x2) — user can pick which play here:
    ap.add_argument("--factor-a", default="order_cond", help="Factor A (2 levels), default SV/VS")
    ap.add_argument("--factor-b", default="population", help="Factor B (2 levels), default population")
    ap.add_argument("--factor-c", default="gen2", help="Factor C (2 levels), default objective bin (seq/cg vs mbr/zmbr)")
    ap.add_argument("--save-prefix", default="artifacts/analysis",
                    help="Prefix for saved analysis outputs.")
    ap.add_argument(
    "--metrics", nargs="*", default=[],
    help="List of numeric metric columns to test by SV/VS. If empty, auto-detect common metrics."
    )
    ap.add_argument(
    "--with-mixed", action="store_true",
    help="Also run per-metric mixed model: metric ~ C(order_cond)+C(model_family)+C(gen2) + (1|item_id)"
    )

    return ap.parse_args()

def ensure_dir(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)

def load_input(args):
    runs_path = args.runs_path if args.runs_path else os.path.join(args.artifacts_dir, "runs.parquet")
    selected_path = args.selected_path if args.selected_path else os.path.join(args.artifacts_dir, "selected_summary.parquet")

    if args.input_kind == "runs":
        if not os.path.exists(runs_path):
            raise FileNotFoundError(f"runs file not found: {runs_path}")
        df = pd.read_parquet(runs_path)
        used = runs_path
    else:
        if not os.path.exists(selected_path):
            raise FileNotFoundError(f"selected file not found: {selected_path}")
        df = pd.read_parquet(selected_path)
        used = selected_path

    print("\n=== INPUT SANITY ===")
    print(f"Input kind         : {args.input_kind}")
    print(f"Using file         : {used}")
    print(f"Rows x Cols        : {df.shape[0]} x {df.shape[1]}")
    print("====================\n")
    return df, used

def derive_design(df: pd.DataFrame, args) -> pd.DataFrame:
    df = df.copy()

    # Ensure item_id exists (subject id for mixed models; here we default to sent_id)
    if "item_id" not in df.columns:
        df["item_id"] = df.get("sent_id", pd.factorize(df.index)[0])

    # Parse decode_params JSON safely
    def _safe_json(s):
        if isinstance(s, str) and s.strip().startswith("{"):
            try: 
                return json.loads(s)
            except Exception: 
                return {}
        return {}
    dp = df.get("decode_params", pd.Series([{}]*len(df))).apply(_safe_json)
    df["beam_used"] = dp.apply(lambda d: d.get("beam", np.nan))
    df["top_p_used"] = dp.apply(lambda d: d.get("top_p", np.nan))
    df["temperature_used"] = dp.apply(lambda d: d.get("temperature", np.nan))

    # Sentence type SV/VS from src_order if missing (robust for NumPy>=2.0 + Patsy)
    if args.sentence_type_col not in df.columns:
        src = df.get("src_order")
        if src is None:
            df[args.sentence_type_col] = pd.Series(pd.NA, index=df.index, dtype="string")
        else:
            src = src.astype("string")
            df[args.sentence_type_col] = (
                src.map({"SUBJ_before_ROOT": "SV", "ROOT_before_SUBJ": "VS"})
                   .astype("string")
            )

    # Make Patsy/statsmodels happy
    df[args.sentence_type_col] = df[args.sentence_type_col].astype("category")

    # Expected/actual construal + success bin
    if args.expected_col not in df.columns:
        df[args.expected_col] = df.get("src_order_binary", np.nan)
    if args.actual_col not in df.columns:
        df[args.actual_col] = df.get("ann_order_binary", np.nan)
    if args.success_col not in df.columns:
        df[args.success_col] = (df[args.expected_col] == df[args.actual_col]).astype(int)

    # Population
    if args.population_col and args.population_col in df.columns:
        df["population"] = df[args.population_col].astype(str)
    else:
        df["population"] = args.population_default

    # Parse model string into family / variant / objective
    def _parse_model(m):
        if not isinstance(m, str): 
            return ("unknown","unknown","unknown")
        parts = m.split(":")
        fam = parts[0] if len(parts) > 0 else "unknown"
        var = parts[1] if len(parts) > 1 else "unknown"
        obj = parts[2] if len(parts) > 2 else "unknown"
        return (fam, var, obj)
    fam_var_obj = df.get("model", "").apply(_parse_model)
    df["model_family"]  = fam_var_obj.apply(lambda t: t[0])
    df["model_variant"] = fam_var_obj.apply(lambda t: t[1])
    df["objective"]     = fam_var_obj.apply(lambda t: t[2])

    # Binary generation factor for 2×2×2
    def _gen2(obj):
        if str(obj).lower() in {"mbr","zmbr"}: 
            return "mbrish"
        return "plain"
    df["gen2"] = df["objective"].apply(_gen2)

    # Optionally filter to selected_flag if using runs
    if args.input_kind == "runs" and args.use_selected_flag:
        before = len(df)
        df = df[df.get("selected_flag", False) == True].copy()
        print(f"[filter] kept selected_flag==True rows: {len(df)}/{before}")

    # Numeric τ
    if args.tau_col not in df.columns:
        raise ValueError(f"Tau column '{args.tau_col}' not found in DF.")
    df[args.tau_col] = pd.to_numeric(df[args.tau_col], errors="coerce")

    # Keep only valid SV/VS levels
    df = df[df[args.sentence_type_col].isin(["SV","VS"])].copy()

    return df

def save_table(df, path):
    ensure_dir(path)
    if path.endswith(".parquet"):
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)

def assumption_checks_tau(df, tau_col, group_col, prefix):
    """
    Check normality of residuals (via groupwise residuals) and Levene for variance.
    If violations, recommend non-parametric.
    """
    out_lines = []
    out_lines.append(f"# Assumption checks for ANOVA on {tau_col} by {group_col}")

    # Residuals from 1-way ANOVA model: tau ~ C(group)
    try:
        model = smf.ols(f"{tau_col} ~ C({group_col})", data=df).fit()
        resid = model.resid.dropna()
        # Shapiro-Wilk
        if len(resid) >= 3 and len(resid) <= 5000:
            sw = stats.shapiro(resid)
            out_lines.append(f"Shapiro-Wilk on residuals: W={sw.statistic:.4f}, p={sw.pvalue:.4g}")
        else:
            out_lines.append("Shapiro-Wilk skipped (n outside [3,5000]).")
        # QQ: we can’t draw plots here; just compute normaltest (D’Agostino-Pearson)
        nt = stats.normaltest(resid, nan_policy='omit') if len(resid) >= 8 else None
        if nt:
            out_lines.append(f"Normaltest (D’Agostino-Pearson): stat={nt.statistic:.4f}, p={nt.pvalue:.4g}")
    except Exception as e:
        out_lines.append(f"[WARN] OLS residual computation failed: {e}")

    # Levene’s test across groups
    try:
        groups = [g[tau_col].dropna().values for _, g in df.groupby(group_col)]
        lv = stats.levene(*groups, center='median')
        out_lines.append(f"Levene (center=median): W={lv.statistic:.4f}, p={lv.pvalue:.4g}")
    except Exception as e:
        out_lines.append(f"[WARN] Levene failed: {e}")

    # Save
    txt_path = f"{prefix}_assumptions.txt"
    ensure_dir(txt_path)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(out_lines))
    print(f"[assumptions] wrote → {txt_path}")

def chi_square_sanity(df, sentence_type_col, success_col, prefix):
    """
    2x2 Chi-square (SV/VS × success).
    """
    sub = df[[sentence_type_col, success_col]].dropna().copy()
    tab = pd.crosstab(sub[sentence_type_col], sub[success_col])
    res = stats.chi2_contingency(tab)
    summary = {
        "test": "Chi-square 2x2 (sentence_type × success)",
        "chi2": res[0], "p": res[1], "dof": res[2],
        "table": tab.to_dict()
    }
    out = pd.DataFrame([summary])
    out_path = f"{prefix}_chi2.csv"
    save_table(out, out_path)
    print(f"[chi²] {summary}")
    print(f"[chi²] wrote → {out_path}")
    return out

def one_way_anova_tau(df, tau_col, sentence_type_col, prefix):
    """
    One-way ANOVA on Kendall τ by sentence type (e.g., SV vs VS).
    Also runs Mann–Whitney U as a non-parametric fallback.

    Writes:
      - {prefix}_anova_oneway.csv
      - {prefix}_anova_oneway_mw.csv  (if both groups present)
      - {prefix}_anova_oneway_SKIPPED.txt (if skipped)
    """
    sub = df[[tau_col, sentence_type_col]].dropna().copy()

    if sub.empty:
        note_path = f"{prefix}_anova_oneway_SKIPPED.txt"
        with open(note_path, "w") as f:
            f.write(
                f"SKIPPED: No non-null values for tau_col={tau_col}. "
                "Alignment likely missing or filtered out."
            )
        print(f"[ANOVA 1-way] skipped (no tau). wrote → {note_path}")
        return

    sub[sentence_type_col] = sub[sentence_type_col].astype("category")

    if sub[sentence_type_col].nunique() < 2:
        note_path = f"{prefix}_anova_oneway_SKIPPED.txt"
        levels = list(sub[sentence_type_col].unique())
        with open(note_path, "w") as f:
            f.write(
                f"SKIPPED: Only one level present in {sentence_type_col}. "
                f"Levels={levels}"
            )
        print(f"[ANOVA 1-way] skipped (single group). wrote → {note_path}")
        return

    # Parametric ANOVA
    try:
        model = smf.ols(f"{tau_col} ~ C({sentence_type_col})", data=sub).fit()
        aov = anova_lm(model, typ=2)
        aov_path = f"{prefix}_anova_oneway.csv"
        save_table(aov.reset_index().rename(columns={"index": "term"}), aov_path)
        print(f"[ANOVA 1-way] wrote → {aov_path}")
    except Exception as e:
        print(f"[ANOVA 1-way][WARN] failed: {e}")

    # Non-parametric fallback
    try:
        sv = sub[sub[sentence_type_col] == "SV"][tau_col].dropna()
        vs = sub[sub[sentence_type_col] == "VS"][tau_col].dropna()

        if len(sv) > 0 and len(vs) > 0:
            mw = stats.mannwhitneyu(sv, vs, alternative="two-sided")
            out = pd.DataFrame([{
                "test": "Mann-Whitney U (SV vs VS)",
                "U": mw.statistic,
                "p": mw.pvalue,
                "n_SV": len(sv),
                "n_VS": len(vs),
                "mean_SV": float(sv.mean()),
                "mean_VS": float(vs.mean()),
                "median_SV": float(sv.median()),
                "median_VS": float(vs.median()),
            }])
            mw_path = f"{prefix}_anova_oneway_mw.csv"
            save_table(out, mw_path)
            print(f"[Mann-Whitney] wrote → {mw_path}")
    except Exception as e:
        print(f"[Mann-Whitney][WARN] {e}")

def factorial_2x2x2_tau(df, tau_col, factor_a, factor_b, factor_c, prefix):
    """
    2x2x2 factorial ANOVA on tau. Requires each factor to have 2 levels present.
    We’ll fit a full-factorial OLS with interactions.
    """
    # Ensure 2 levels each
    for fac in [factor_a, factor_b, factor_c]:
        levels = df[fac].dropna().unique().tolist()
        if len(levels) != 2:
            print(f"[2x2x2][WARN] Factor {fac} has {len(levels)} level(s): {levels}. Attempting to coerce to 2 levels…")
    sub = df[[tau_col, factor_a, factor_b, factor_c]].dropna().copy()

    # Make sure factors are treated as categorical
    for fac in [factor_a, factor_b, factor_c]:
        sub[fac] = sub[fac].astype("category")

    formula = f"{tau_col} ~ C({factor_a})*C({factor_b})*C({factor_c})"
    model = smf.ols(formula, data=sub).fit()
    aov = anova_lm(model, typ=2)
    aov_path = f"{prefix}_anova_2x2x2.csv"
    save_table(aov.reset_index().rename(columns={"index":"term"}), aov_path)
    print(f"[ANOVA 2x2x2] wrote → {aov_path}")

def mixed_lmm_tau(df, tau_col, sentence_type_col, prefix):
    """
    Mixed model on tau: random intercept by item_id; fixed effects: sentence_type + model_family + gen2.
    Works on unbalanced data.
    """
    sub = df[[tau_col, "item_id", sentence_type_col, "model_family", "gen2"]].dropna().copy()
    sub["item_id"] = sub["item_id"].astype(str)
    # Build formula
    formula = f"{tau_col} ~ C({sentence_type_col}) + C(model_family) + C(gen2)"
    try:
        md = smf.mixedlm(formula, sub, groups=sub["item_id"])
        mdf = md.fit(method='lbfgs', maxiter=500, disp=False)
        summ = mdf.summary().as_text()
    except Exception as e:
        summ = f"[LMM][ERROR] {e}"
    txt_path = f"{prefix}_lmm_tau.txt"
    ensure_dir(txt_path)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(summ)
    print(f"[LMM] wrote → {txt_path}")

def gee_success(df, success_col, sentence_type_col, prefix):
    """
    Binomial GEE on construal success with clustering by item_id.
    Predictors: sentence_type + model_family + gen2 (+ population if available).
    """
    sub = df[[success_col, "item_id", sentence_type_col, "model_family", "gen2", "population"]].dropna().copy()
    sub["item_id"] = sub["item_id"].astype(str)
    sub[success_col] = sub[success_col].astype(int)

    # Build design matrix
    formula = f"{success_col} ~ C({sentence_type_col}) + C(model_family) + C(gen2) + C(population)"
    fam = sm.families.Binomial()
    try:
        model = smf.gee(formula, "item_id", data=sub, family=fam)
        res = model.fit()
        summ = res.summary().as_text()
    except Exception as e:
        summ = f"[GEE][ERROR] {e}"

    txt_path = f"{prefix}_gee_success.txt"
    ensure_dir(txt_path)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(summ)
    print(f"[GEE] wrote → {txt_path}")

def autodetect_metrics(df: pd.DataFrame) -> list[str]:
    """Pick likely numeric metric columns (exclude ids/text/bins)."""
    num = df.select_dtypes(include=["number", "float", "int"]).columns.tolist()
    drop = {
        "sent_id","item_id","beam_used","top_p_used","temperature_used",
        "construal_match_bin","src_order_binary","ann_order_binary"
    }
    # keep common RT/similarity metrics if present
    prefer = ["rt_chrf","rt_bleu","rt_ter","rt_sem","labse_cross","ann_align_kendall_tau","fluency"]
    cand = [c for c in num if c not in drop]
    # stable order: prefer known, then the rest
    ordered = [c for c in prefer if c in cand] + [c for c in cand if c not in prefer]
    return ordered

def test_metric_by_order(df: pd.DataFrame, metric: str,
                         sentence_type_col: str, prefix: str) -> None:
    """One-way ANOVA (metric ~ SV/VS) + MW fallback, saved to CSV."""
    from statsmodels.stats.anova import anova_lm
    sub = df[[metric, sentence_type_col]].dropna().copy()
    if sub.empty:
        print(f"[metric][SKIP] {metric}: no data after dropna.")
        return
    sub[sentence_type_col] = sub[sentence_type_col].astype("category")
    # ANOVA
    model = smf.ols(f"{metric} ~ C({sentence_type_col})", data=sub).fit()
    aov = anova_lm(model, typ=2)
    save_table(aov.reset_index().rename(columns={"index":"term"}),
               f"{prefix}_{metric}_anova.csv")
    print(f"[metric][ANOVA] {metric} → {prefix}_{metric}_anova.csv")
    # Mann–Whitney
    sv = sub[sub[sentence_type_col]=="SV"][metric].dropna()
    vs = sub[sub[sentence_type_col]=="VS"][metric].dropna()
    if len(sv) and len(vs):
        mw = stats.mannwhitneyu(sv, vs, alternative="two-sided")
        eff = (mw.statistic/(len(sv)*len(vs))*2) - 1  # rank-biserial r
        out = pd.DataFrame([{
            "test":"Mann-Whitney U (SV vs VS)",
            "U":float(mw.statistic), "p":float(mw.pvalue),
            "n_SV":len(sv), "n_VS":len(vs), "rank_biserial_r":eff
        }])
        save_table(out, f"{prefix}_{metric}_mw.csv")
        print(f"[metric][MW] {metric} → {prefix}_{metric}_mw.csv")
    else:
        print(f"[metric][MW][SKIP] {metric}: insufficient group sizes.")

def mixed_metric_by_order(df: pd.DataFrame, metric: str,
                          sentence_type_col: str, prefix: str) -> None:
    """
    Per-metric model with robust fallback:
      1) MixedLM: metric ~ C(order_cond) + C(model_family) + C(gen2) + (1|item_id)
      2) If MixedLM fails (e.g., singular RE cov): OLS with item-clustered SEs
      3) If clustering not viable: OLS with HC3 robust SEs

    Always writes a single text report to: {prefix}_{metric}_lmm.txt
    The header of the file states which estimator was used.
    """
    path = f"{prefix}_{metric}_lmm.txt"

    # Build subset
    need = [metric, "item_id", sentence_type_col, "model_family", "gen2"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        ensure_dir(path)
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"[SKIP] Missing columns for {metric}: {missing}\n")
        print(f"[metric][LMM][SKIP] {metric}: missing {missing}")
        return

    sub = df[need].dropna().copy()
    if sub.empty:
        ensure_dir(path)
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"[SKIP] No data for {metric} after dropna.\n")
        print(f"[metric][LMM][SKIP] {metric}: no data after dropna.")
        return

    # Types
    sub["item_id"] = sub["item_id"].astype(str)
    sub[sentence_type_col] = sub[sentence_type_col].astype("category")
    formula = f"{metric} ~ C({sentence_type_col}) + C(model_family) + C(gen2)"

    # 1) Try MixedLM
    try:
        md = smf.mixedlm(formula, sub, groups=sub["item_id"])
        mdf = md.fit(method="lbfgs", maxiter=500, disp=False)
        summary_txt = mdf.summary().as_text()
        ensure_dir(path)
        with open(path, "w", encoding="utf-8") as f:
            f.write("[ESTIMATOR] MixedLM (random intercept by item_id)\n\n")
            f.write(summary_txt)
        print(f"[metric][LMM] {metric} → {path}")
        return
    except Exception as e:
        err_msg = str(e)

    # 2) Fallback: OLS with item-clustered SEs (if clustering viable)
    n_groups = sub["item_id"].nunique()
    min_group_n = sub.groupby("item_id").size().min() if n_groups > 0 else 0
    try_cluster = n_groups > 1 and min_group_n >= 1  # allow unequal cluster sizes

    if try_cluster:
        try:
            ols = smf.ols(formula, data=sub).fit(
                cov_type="cluster", cov_kwds={"groups": sub["item_id"]}
            )
            ensure_dir(path)
            with open(path, "w", encoding="utf-8") as f:
                f.write("[ESTIMATOR] OLS with item-clustered robust SEs "
                        "(MixedLM failed; reason: " + err_msg + ")\n")
                f.write(f"[INFO] Clusters: {n_groups}, min cluster size: {min_group_n}\n\n")
                f.write(ols.summary().as_text())
            print(f"[metric][OLS-cluster] {metric} → {path}")
            return
        except Exception as e2:
            err_msg += f" | OLS-cluster failed: {e2}"

    # 3) Final fallback: OLS with HC3 robust SEs
    try:
        ols = smf.ols(formula, data=sub).fit(cov_type="HC3")
        ensure_dir(path)
        with open(path, "w", encoding="utf-8") as f:
            f.write("[ESTIMATOR] OLS with HC3 robust SEs "
                    "(MixedLM/cluster fallback failed)\n")
            f.write(f"[INFO] Fallback reasons: {err_msg}\n\n")
            f.write(ols.summary().as_text())
        print(f"[metric][OLS-HC3] {metric} → {path}")
    except Exception as e3:
        ensure_dir(path)
        with open(path, "w", encoding="utf-8") as f:
            f.write("[ERROR] All estimators failed.\n")
            f.write(f"MixedLM/Cluster/HC3 errors: {err_msg} | {e3}\n")
        print(f"[metric][LMM][ERROR] {metric}: {e3}")

from statsmodels.stats.proportion import proportions_ztest, binom_test as sm_binom_test
from scipy.stats import pointbiserialr, mannwhitneyu

def _above_chance(df, success_col, prefix):
    """
    Test if construal success is above chance (0.5).
    Writes both z-test and exact binomial results.
    """
    # ensure ints
    n = int(df[success_col].notna().sum())
    k = int(df[success_col].sum())

    # z-test vs 0.5
    stat, pval_z = proportions_ztest(k, n, value=0.5)

    # exact binomial: statsmodels uses 'prop' (not 'p')
    try:
        pval_exact = sm_binom_test(k, n, prop=0.5, alternative="two-sided")
    except TypeError:
        # fallback to SciPy if available
        from scipy.stats import binomtest
        pval_exact = binomtest(k, n, p=0.5, alternative="two-sided").pvalue

    out = pd.DataFrame([{
        "test": "above_chance (0.5)",
        "successes": k,
        "n": n,
        "prop": k / n if n else float("nan"),
        "z_stat": float(stat),
        "pval_z": float(pval_z),
        "pval_exact": float(pval_exact),
    }])
    path = f"{prefix}_stage1_above_chance.csv"
    save_table(out, path)
    print(f"[stage1] wrote → {path}")

def _tau_vs_success(df, tau_col, success_col, prefix):
    """
    Test correlation between τ and binary success.
    Saves point-biserial r and logistic regression.
    """
    sub = df[[tau_col, success_col]].dropna().copy()
    r, p = pointbiserialr(sub[success_col], sub[tau_col])

    # Logistic regression
    model = smf.logit(f"{success_col} ~ {tau_col}", data=sub).fit(disp=False)

    # Save correlation
    corr_out = pd.DataFrame([{"test":"pointbiserialr",
                              "r": float(r), "p": float(p),
                              "n": len(sub)}])
    save_table(corr_out, f"{prefix}_stage2_corr.csv")

    # Save logistic summary
    txt_path = f"{prefix}_stage2_logit.txt"
    ensure_dir(txt_path)
    with open(txt_path,"w") as f:
        f.write(model.summary().as_text())
    print(f"[stage2] wrote → {prefix}_stage2_corr.csv, {txt_path}")

def _tau_metrics_interactions(df, tau_col, success_col,
                                    metrics, prefix, stype_col="order_cond"):
    """
    (a) logistic regression with τ*metrics interactions,
    (b) ANOVA-style split (success vs failure).
    """
    sub = df[[success_col, tau_col]+metrics+[stype_col]].dropna().copy()
    sub[success_col] = sub[success_col].astype(int)

    # --- Logistic with interactions ---
    terms = " + ".join([f"{tau_col}*{m}" for m in metrics])
    formula = f"{success_col} ~ {tau_col} + " + terms
    try:
        model = smf.logit(formula, data=sub).fit(disp=False)
        txt_path = f"{prefix}_stage3_logit.txt"
        ensure_dir(txt_path)
        with open(txt_path,"w") as f:
            f.write(model.summary().as_text())
        print(f"[stage3][logit] wrote → {txt_path}")
    except Exception as e:
        print(f"[stage3][logit][ERROR] {e}")

    # --- ANOVA-style split ---
    rows = []
    for m in [tau_col] + metrics:
        s_vals = sub.loc[sub[success_col]==1, m].dropna()
        f_vals = sub.loc[sub[success_col]==0, m].dropna()
        if len(s_vals) and len(f_vals):
            u, p = mannwhitneyu(s_vals, f_vals, alternative="two-sided")
            rows.append({
                "metric": m, "test":"Mann-Whitney U",
                "U": float(u), "p": float(p),
                "n_success": len(s_vals), "n_fail": len(f_vals),
                "mean_success": s_vals.mean(), "mean_fail": f_vals.mean()
            })
    if rows:
        out = pd.DataFrame(rows)
        csv_path = f"{prefix}_stage3_anova_split.csv"
        save_table(out, csv_path)
        print(f"[stage3][anova-split] wrote → {csv_path}")



def main():
    args = parse_args()

    # 1) Load source
    loaded = load_input(args)

    # Allow load_input to return df OR (df, used_file, meta...)
    if isinstance(loaded, tuple):
      df = loaded[0]
      inferred_used_file = loaded[1] if len(loaded) > 1 else None
      extra_meta = loaded[2:] if len(loaded) > 2 else None
    else:
      df = loaded
      inferred_used_file = None
      extra_meta = None

    # provenance for logs
    # provenance for logs
    if getattr(args, 'auto_discover', False):
      used_file = f"AUTO:{getattr(args,'prereg_root', '')}"
    else:
      used_file = inferred_used_file or (
          args.runs_path if args.input_kind == 'runs' else args.selected_path
      )

      if not used_file:
          used_file = str(Path(args.artifacts_dir) / (
              'runs.parquet' if args.input_kind == 'runs' else 'selected_summary.parquet'
          ))

    # 2) Derive design columns
    df = derive_design(df, args)

    # 3) Assumption checks for τ ANOVA
    as_prefix = f"{args.save_prefix}_assumptions_{args.input_kind}"
    assumption_checks_tau(df, args.tau_col, args.sentence_type_col, as_prefix)

    # 4) χ² sanity: SV/VS × success
    chi_prefix = f"{args.save_prefix}_chi_{args.input_kind}"
    chi_square_sanity(df, args.sentence_type_col, args.success_col, chi_prefix)

    # 5a) One-way ANOVA on τ (SV vs VS) with MW fallback
    a1_prefix = f"{args.save_prefix}_a1_{args.input_kind}"
    one_way_anova_tau(df, args.tau_col, args.sentence_type_col, a1_prefix)

    # 5b) Metrics: SV vs VS across multiple numeric metrics
    metrics = args.metrics if args.metrics else autodetect_metrics(df)
    metrics = [m for m in metrics if m != args.tau_col]
    print(f"[metrics] Testing SV/VS for: {metrics}")
    mprefix = f"{args.save_prefix}_metrics_{args.input_kind}"
    for m in metrics:
        test_metric_by_order(df, m, args.sentence_type_col, mprefix)
        if args.with_mixed:
            mixed_metric_by_order(df, m, args.sentence_type_col, mprefix)


    # 6) 2×2×2 factorial ANOVA on τ
    a3_prefix = f"{args.save_prefix}_a3_{args.input_kind}"
    needed = [args.factor_a, args.factor_b, args.factor_c]

    missing = [c for c in needed if c not in df.columns]
    if missing:
        print(f"[2x2x2][SKIP] Missing factors: {', '.join(missing)}")
    else:
        # check each factor has exactly 2 non-null levels
        level_info = {c: sorted(pd.Series(df[c]).dropna().unique().tolist()) for c in needed}
        bad = {c: levs for c, levs in level_info.items() if len(levs) != 2}
        if bad:
            for c, levs in bad.items():
                print(f"[2x2x2][SKIP] Factor '{c}' has {len(levs)} level(s): {levs}. Need exactly 2.")
            print("[2x2x2][SKIP] Skipping factorial ANOVA due to collapsed factor(s).")
        else:
            factorial_2x2x2_tau(df, args.tau_col, args.factor_a, args.factor_b, args.factor_c, a3_prefix)

    # 7) Mixed LMM on τ
    lmm_prefix = f"{args.save_prefix}_lmm_{args.input_kind}"
    mixed_lmm_tau(df, args.tau_col, args.sentence_type_col, lmm_prefix)

    # 8) GEE on success
    gee_prefix = f"{args.save_prefix}_gee_{args.input_kind}"
    gee_success(df, args.success_col, args.sentence_type_col, gee_prefix)
    
    # 9) Stage 1–3 analyses
    _prefix = f"{args.save_prefix}_level_{args.input_kind}"
    _above_chance(df, args.success_col, _prefix)
    _tau_vs_success(df, args.tau_col, args.success_col, _prefix)
    metrics = args.metrics if args.metrics else ["rt_chrf","rt_bleu","rt_ter"]
    _tau_metrics_interactions(df, args.tau_col, args.success_col,
                                metrics, _prefix,
                                stype_col=args.sentence_type_col)

    
    # 10) Sanity message: which steps used which outputs
    print("\n=== OUTPUT SANITY ===")
    print(f"Data used                 : {used_file}")

    want = {
        "item_id","model","model_family","model_variant","objective","gen2","population",
        "src_order","src_order_binary","ann_order_binary",
        args.tau_col, args.success_col, args.sentence_type_col
    }
    have = sorted(set(df.columns) & want)
    print(f"Design columns available  : {have}")

    print("Saved core analyses:")
    print(f"  Assumptions (τ) → {as_prefix}_assumptions.txt")
    print(f"  Chi-square      → {chi_prefix}_chi2.csv")
    print(f"  ANOVA 1-way (τ) → {a1_prefix}_anova_oneway.csv (+ MW CSV)")
    print(f"  ANOVA 2x2x2     → {a3_prefix}_anova_2x2x2.csv (if factors present)")
    print(f"  LMM (τ)         → {lmm_prefix}_lmm_tau.txt")
    print(f"  GEE (success)   → {gee_prefix}_gee_success.txt")

    # echo metrics actually tested
    print("\nMetrics tested (SV vs VS):", metrics)
    print(f"  Metrics ANOVA/MW prefix → {mprefix}_<metric>_anova.csv / _mw.csv")
    if args.with_mixed:
        print(f"  Metrics LMM prefix      → {mprefix}_{metrics}_lmm.txt")

    # stage outputs
    stage_prefix = f"{args.save_prefix}_level_{args.input_kind}"
    print("\nStage analyses:")
    print(f"  Stage1 (above chance)   → {stage_prefix}_stage1_above_chance.csv")
    print(f"  Stage2 (τ↔success)      → {stage_prefix}_stage2_corr.csv, {stage_prefix}_stage2_logit.txt")
    print(f"  Stage3 (logit τ×metrics)→ {stage_prefix}_stage3_logit.txt")
    print(f"  Stage3 (ANOVA-style)    → {stage_prefix}_stage3_anova_split.csv")
    print("=====================================\n")

if __name__ == "__main__":
    main()
"""
Created on Thu Aug 21 20:38:07 2025

@author: niran
"""

