
"""
Driver to enrich translations CSV with:
- Target words & POS (from 'Match')
- Syntax slots (ROOT, SUBJ) for source & targets
- Order-binary features (PL rule)
- Determinacy (EN strict/general; PL lexica)
- Cross-lingual construal (strict & general) with special-case
- Awesome-Align word/subword alignments + Kendall's tau
- Target-word positions in target & branching (target+source)
- Unique combo stats are left to viz_uniques.py for plotting
"""
import argparse, pandas as pd
from scipy.stats import kendalltau

from alignment_wrappers import AwesomeAligner
from syntax_slots import SyntaxSlots
from determinacy import subject_np_determinacy, EN_DEF_STRICT, EN_INDEF_STRICT, EN_DEF_GENERAL, EN_INDEF_GENERAL, PL_DEF, PL_INDEF
from features import order_binary, branching_for_targets_spacy, source_branching_from_align, construal_match_crosslingual
from target_extractor import extract_tuples, map_targets_to_positions

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in', dest='in_path', required=True)
    ap.add_argument('--out', dest='out_path', required=True)
    ap.add_argument('--src-col', default='original')
    ap.add_argument('--src-lang', default='pl')
    ap.add_argument('--tgt-lang', default='en')
    ap.add_argument('--cand-suffix', default='_en', help="Candidate columns ending with this suffix will be processed")
    args = ap.parse_args()

    df = pd.read_csv(args.in_path)

    # Extract targets from 'Match' if needed
    if 'Target words' not in df.columns and 'Match' in df.columns:
        tw_tp = df['Match'].fillna("").astype(str).apply(extract_tuples)
        df['Target words'] = tw_tp.apply(lambda x: x[0])
        df['Target POS']   = tw_tp.apply(lambda x: x[1])

    # Source analysis
    src_slots = SyntaxSlots(args.src_lang)
    trg_slots = SyntaxSlots(args.tgt_lang)
    aligner  = AwesomeAligner()

    src_an = df[args.src_col].fillna("").astype(str).apply(src_slots.analyze)
    df['src_order'] = src_an.apply(lambda d: d['order'])
    df['src_order_binary'] = src_an.apply(lambda d: order_binary(d['order']))
    df['src_subj_span'] = src_an.apply(lambda d: d['subj_span'])
    df['src_det_PL'] = [subject_np_determinacy(d['doc'], d['subj_span'], args.src_lang, mode="general") for d in src_an]
    df['src_has_PL_DEF'] = df['src_det_PL'].eq('Def')
    df['src_has_PL_INDEF'] = df['src_det_PL'].eq('Indef')

    # Candidate columns
    cand_cols = [c for c in df.columns if c.endswith(args.cand_suffix)]
    for col in cand_cols:
        trg_an = df[col].fillna("").astype(str).apply(trg_slots.analyze)
        df[f'{col}_order'] = trg_an.apply(lambda d: d['order'])
        df[f'{col}_order_binary'] = trg_an.apply(lambda d: order_binary(d['order']))
        df[f'{col}_subj_span'] = trg_an.apply(lambda d: d['subj_span'])

        # Determinacy in EN (strict and general)
        en_det_strict = [subject_np_determinacy(d['doc'], d['subj_span'], args.tgt_lang, mode="strict") for d in trg_an]
        en_det_general = [subject_np_determinacy(d['doc'], d['subj_span'], args.tgt_lang, mode="general") for d in trg_an]
        df[f'{col}_det_strict']  = en_det_strict
        df[f'{col}_det_general'] = en_det_general

        # Construal: strict and general with special-case
        df[f'{col}_construal_strict'] = [
            construal_match_crosslingual(pl_ob, pl_def, pl_indef, ed, mode="strict")
            for pl_ob, pl_def, pl_indef, ed in zip(df['src_order_binary'], df['src_has_PL_DEF'], df['src_has_PL_INDEF'], en_det_strict)
        ]
        df[f'{col}_construal_general'] = [
            construal_match_crosslingual(pl_ob, pl_def, pl_indef, ed, mode="general")
            for pl_ob, pl_def, pl_indef, ed in zip(df['src_order_binary'], df['src_has_PL_DEF'], df['src_has_PL_INDEF'], en_det_general)
        ]

        # Alignment + Kendall's tau
        kt_scores = []
        tgt_docs = [d['doc'] for d in trg_an]
        tgt_tokens = [[t.text for t in doc] for doc in tgt_docs]
        src_docs = [d['doc'] for d in src_an]

        # Target positions (in spaCy target doc indices)
        tgt_positions_spacy = []
        # For source branching, we need WS alignment positions; keep also WS positions of target words
        tgt_positions_ws = []

        # Build WS tokens once
        def ws_tokens(s: str): return s.strip().split()

        for i, (src_text, tgt_text, target_words) in enumerate(zip(df[args.src_col], df[col], df['Target words'])):
            # Align
            res = aligner.align(str(src_text), str(tgt_text))
            pairs = res['word_align']  # (src_ws_i, tgt_ws_j)

            # Kendall tau (monotonicity of alignment pairs)
            if len(pairs) >= 2:
                s_idx, t_idx = zip(*pairs)
                kt = kendalltau(s_idx, t_idx, nan_policy='omit').correlation
            else:
                kt = float('nan')
            kt_scores.append(kt)

            # Target positions via spaCy tokens
            tg_tok = tgt_tokens[i]
            positions_spacy = []
            used = set()
            for w in (target_words or []):
                wl = str(w).lower()
                found = -1
                for j, tok in enumerate(tg_tok):
                    if j in used: continue
                    if tok.lower() == wl:
                        found = j; used.add(j); break
                positions_spacy.append(found)
            tgt_positions_spacy.append(positions_spacy)

            # Also record WS positions for the same words (to trace back to source via alignment)
            tg_ws = ws_tokens(str(tgt_text))
            ws_positions = []
            used_ws = set()
            for w in (target_words or []):
                wl = str(w).lower()
                f = -1
                for j, tok in enumerate(tg_ws):
                    if j in used_ws: continue
                    if tok.lower() == wl:
                        f = j; used_ws.add(j); break
                ws_positions.append(f)
            tgt_positions_ws.append(ws_positions)

            # Store alignment pairs for later branching on source; save per-row
            df.loc[i, f'{col}_align_pairs'] = str(pairs)  # stringified list to keep CSV-friendly

        df[f'{col}_align_kendall_tau'] = kt_scores
        df[f'{col}_target_positions'] = tgt_positions_spacy

        # Branching (target side) for target positions
        df[f'{col}_target_branching'] = [
            branching_for_targets_spacy(doc, pos_list)
            for doc, pos_list in zip(tgt_docs, tgt_positions_spacy)
        ]

        # Branching (source side) for aligned source tokens of those target words
        df[f'{col}_source_branching_for_targets'] = [
            source_branching_from_align(src_doc, eval(str(df.loc[i, f'{col}_align_pairs'])), tgt_positions_ws[i])
            for i, src_doc in enumerate(src_docs)
        ]

    df.to_csv(args.out_path, index=False)
    print(f"Saved → {args.out_path}")

if __name__ == "__main__":
    main()
