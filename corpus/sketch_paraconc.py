#!/usr/bin/env python3
"""
SketchEngine Parallel Concordances CLI (multi-query, unique CSV per lemma combo)

Examples:
  # Multiple CQL queries, save to directory
  python sketh_paraconc.py -q 'q[lemma="kobieta"]' 'q[lemma="kobieta"][lemma="wejść"]' --outdir results --save

  # Simple word (auto-wrapped into CQL)
  python sketh_paraconc.py -q kobieta --outdir results --save

  # Inspect corpus alignment
  python sketh_paraconc.py --inspect --corp preloaded/OS_pl

Env:
  export SKETCHENGINE=your_api_key
"""
from __future__ import annotations
import os, re, json, argparse, sys
from typing import Any, Dict, List
import requests
import pandas as pd

def must_ok(r: requests.Response) -> Any:
    r.raise_for_status()
    j = r.json()
    if isinstance(j, dict) and "error" in j and j["error"]:
        raise RuntimeError(str(j["error"]))
    return j

def seg_text(x: Any) -> str:
    if x is None: return ""
    if isinstance(x, str): return x
    if isinstance(x, dict):
        if "str" in x: return x["str"]
        toks = x.get("Tokens")
        if isinstance(toks, list):
            return "".join(t.get("str","") for t in toks if isinstance(t, dict))
        return "".join(seg_text(x.get(k)) for k in ("Left","Kwic","Kw","Right") if k in x)
    if isinstance(x, list): return "".join(seg_text(s) for s in x)
    return ""

def safe_name(s: str) -> str:
    s = s.strip()
    s = s.replace("[","").replace("]","").replace('"','').replace("=","_")
    s = re.sub(r"\s+", "_", s)
    return re.sub(r"[^-_.A-Za-z0-9]+", "_", s)[:120]

def filename_from_query(q: str) -> str:
    """Prefer lemma-based names like 'lemmas_kobieta+wejść'. Fallbacks to word=, then safe query."""
    lemmas = re.findall(r'lemma\s*=\s*"([^"]+)"', q)
    if lemmas:
        return safe_name("lemmas_" + "+".join(lemmas))
    words = re.findall(r'word\s*=\s*"([^"]+)"', q)
    if words:
        return safe_name("words_" + "+".join(words))
    # bare token
    return safe_name(q)

def unique_path(path: str) -> str:
    if not os.path.exists(path):
        return path
    base, ext = os.path.splitext(path)
    k = 1
    while True:
        cand = f"{base}_{k}{ext}"
        if not os.path.exists(cand):
            return cand
        k += 1

def get_parallel_cql(query_word: str, max_results: int, src: str, align: str, viewmode: str="kwic") -> Dict[str, Any]:
    if query_word.startswith('q[') or query_word.startswith('q '):
        cql_query = query_word
    elif query_word.startswith('['):
        cql_query = f'q{query_word}'
    else:
        cql_query = f'q[word="{query_word}"]'
    return {
        "corpname": src, "format": "json", "asyn": 0,
        "q": cql_query, "viewmode": viewmode, "pagesize": max_results,
        "align": align, "kwicleftctx": "-40#", "kwicrightctx": "40#",
        "attrs": "word", "ctxattrs": "word",
    }

def get_parallel_iquery(query_word: str, max_results: int, src: str, align: str, viewmode: str="kwic") -> Dict[str, Any]:
    operations = [{
        "name": "iquery", "arg": query_word, "active": True,
        "query": {
            "queryselector": "iqueryrow", "iquery": query_word,
            "sel_aligned": [align],
            f"queryselector_{align}": "iqueryrow",
            f"iquery_{align}": "",
            f"pcq_pos_neg_{align}": "pos",
            f"filter_nonempty_{align}": "on",
        },
    }]
    return {
        "corpname": src, "format": "json", "asyn": 0,
        "queryselector": "iquery", "iquery": query_word,
        "default_attr": "word", "iquerymode": "simple",
        "viewmode": viewmode, "pagesize": max_results, "align": align,
        "operations": json.dumps(operations),
        "attrs": "word", "ctxattrs": "word",
        "formparts": json.dumps([{
            "corpname": align,
            "formValue": {"queryselector": "iquery", "filter_nonempty": True, "pcq_pos_neg": "pos"},
        }]),
    }

def inspect_corpus_alignment(base: str, headers: Dict[str, str], corpus_name: str):
    print(f"\n=== INSPECTING CORPUS: {corpus_name} ===")
    try:
        params = {"corpname": corpus_name, "format": "json"}
        response = requests.get(f"{base}/corp_info", params=params, headers=headers, timeout=60)
        corp_data = must_ok(response)
        print(f"Corpus name: {corp_data.get('name', 'Unknown')}")
        print(f"Language: {corp_data.get('lang', 'Unknown')}")
        print(f"Size: {corp_data.get('sizes', {})}")
        if 'aligned' in corp_data:
            print(f"Aligned corpora: {corp_data['aligned']}")
        attrs = corp_data.get('attributes', [])
        print(f"Available attributes: {[a.get('name') for a in attrs]}")
        structs = corp_data.get('structs', [])
        print(f"Available structures: {structs}")
    except Exception as e:
        print(f"Failed to inspect corpus: {e}")

def get_parallel_concordances(
    base: str, headers: Dict[str,str], query_word: str, src: str, align: str,
    use_cql: bool=True, max_results: int=5000, debug: bool=False,
    viewmode: str="kwic", stop_after: int|None=None
) -> List[Dict[str,str]]:
    params_base = (
        get_parallel_cql(query_word, max_results, src, align, viewmode)
        if use_cql else
        get_parallel_iquery(query_word, max_results, src, align, viewmode)
    )
    rows: List[Dict[str,str]] = []
    fromp = kept = 0
    while True:
        p = dict(params_base, fromp=fromp)
        try:
            data = must_ok(requests.get(f"{base}/view", params=p, headers=headers, timeout=120))
        except Exception as e:
            print(f"API Error: {e}")
            if not use_cql:
                print("iquery failed, trying CQL method...")
                return get_parallel_concordances(base, headers, query_word, src, align, True,
                                                 max_results, debug, viewmode, stop_after)
            raise
        lines = data.get("Lines", [])
        if not lines: break

        if debug and fromp == 0:
            print("\n=== DEBUG: First line structure ===")
            print(f"Data keys: {list(data.keys())}")
            print(f"Number of lines: {len(lines)}")
            first_line = lines[0]
            print(f"First line keys: {list(first_line.keys())}")
            print(f"Align data present: {'Align' in first_line}")
            print("=== END DEBUG ===\n")

        for i, line in enumerate(lines, start=fromp):
            pl_left  = seg_text(line.get("Left"))
            pl_node  = seg_text(line.get("Kwic")) or seg_text(line.get("Kw"))
            pl_right = seg_text(line.get("Right"))

            en_text = ""
            align_data = line.get("Align", [])
            if isinstance(align_data, list):
                for a in align_data:
                    if isinstance(a, dict):
                        parts = []
                        lp = seg_text(a.get("Left", []));  kp = seg_text(a.get("Kwic", []) or a.get("Kw", [])); rp = seg_text(a.get("Right", []))
                        if lp: parts.append(lp)
                        if kp: parts.append(kp)
                        if rp: parts.append(rp)
                        if parts:
                            en_text = " ".join(parts).strip()
                            break
                    elif isinstance(a, str) and a.strip():
                        en_text = a.strip(); break

            if en_text.strip():
                rows.append({
                    "line_id": i,
                    "pl_left": pl_left.strip(),
                    "pl_node": pl_node.strip(),
                    "pl_right": pl_right.strip(),
                    "en_text": en_text.strip()
                })
                kept += 1
            elif debug and i < fromp + 10:
                print(f"Skipping line {i}: no English text found")

            if stop_after is not None and len(rows) >= stop_after:
                if debug: print(f"Stopping early after {stop_after} rows")
                return rows

        fromp += len(lines)
        concsize = data.get("concsize", 0)
        if fromp >= concsize: break
        if debug:
            print(f"Scanned {fromp}/{concsize}… kept {kept} rows with EN")

    return rows

def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fetch parallel concordances from SketchEngine.")
    p.add_argument("--query", "-q", type=str, nargs="+", required=True,
                   help=("One or more queries (space separated). Bare words or full CQL like "
                         "q[lemma=\"kobieta\"][lemma=\"wejść\"]."))
    p.add_argument("--corp", type=str, default="preloaded/OS_pl", help="Source corpus (corpname)")
    p.add_argument("--align", type=str, default="OS_en", help="Aligned corpus (target)")
    p.add_argument("--base", type=str, default="https://api.sketchengine.eu/bonito/run.cgi", help="API base URL")
    p.add_argument("--api-key", type=str, default=os.environ.get("SKETCHENGINE"), help="API key or SKETCHENGINE env")
    p.add_argument("--iquery", action="store_true", help="Use iquery (experimental). Default: CQL")
    p.add_argument("--max-results", type=int, default=5000, help="Max results to fetch")
    p.add_argument("--viewmode", type=str, default="kwic", choices=["kwic","sen"], help="View mode")
    p.add_argument("--debug", action="store_true", help="Verbose debug prints")
    p.add_argument("--stop-after", type=int, default=None, help="Stop after N kept rows (debug aid)")

    p.add_argument("--save", action="store_true", help="Write CSVs (implied by --outdir)")
    p.add_argument("--output", type=str, default=None, help="Single-file output (ignored if multiple queries)")
    p.add_argument("--outdir", type=str, default=None, help="Directory for one CSV per query (no overwrite)")
    p.add_argument("--inspect", action="store_true", help="Inspect corpus info and exit")
    return p.parse_args(argv)

def main(argv: List[str]) -> int:
    args = parse_args(argv)

    if not args.api_key:
        print("ERROR: Provide API key via --api-key or SKETCHENGINE env var.", file=sys.stderr)
        return 2

    headers = {"Authorization": f"Bearer {args.api_key}"}

    if args.inspect:
        inspect_corpus_alignment(args.base, headers, args.corp)
        return 0

    outdir = args.outdir
    if outdir:
        os.makedirs(outdir, exist_ok=True)

    if args.output and len(args.query) > 1:
        print("WARNING: --output ignored for multiple queries. Use --outdir instead.")
        args.output = None

    total_rows = 0
    for q in args.query:
        rows = get_parallel_concordances(
            base=args.base, headers=headers, query_word=q, src=args.corp, align=args.align,
            use_cql=(not args.iquery), max_results=args.max_results, debug=args.debug,
            viewmode=args.viewmode, stop_after=args.stop_after
        )
        print(f"\n✅ {q} -> {len(rows)} parallel concordances")
        total_rows += len(rows)
        if not rows:
            continue

        df = pd.DataFrame(rows)

        if outdir or args.save or args.output:
            if args.output and len(args.query) == 1:
                out_path = args.output
            else:
                dir_ = outdir or os.getcwd()
                base_name = filename_from_query(q)
                out_path = os.path.join(dir_, f"parallel_{base_name}.csv")
                out_path = unique_path(out_path)
            df.to_csv(out_path, index=False)
            print(f"Saved to: {out_path}")
        else:
            sample = rows[0]
            print("Sample:")
            print(f"  PL: {sample['pl_left']} **{sample['pl_node']}** {sample['pl_right']}")
            print(f"  EN: {sample['en_text']}")

    print(f"\nDone. Processed {len(args.query)} query(ies), total rows: {total_rows}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
