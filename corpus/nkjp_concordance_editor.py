# -*- coding: utf-8 -*-
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
concordance_editor.py

Combine two workflows:
1) Extract concordances from HTML files and merge left + cleaned match + right.
2) Interactively edit and select rows — works in Jupyter/Colab (ipywidgets UI)
   or in a plain terminal (text prompts).

Dependencies:
  pip install beautifulsoup4 pandas ipywidgets

Usage (terminal):
  python concordance_editor.py --html-dir /path/to/SENT_HTML \
      --out-all all_concordances.csv --out-selected selected_concordances.csv

Jupyter/Colab:
  %run concordance_editor.py --html-dir /path/to/SENT_HTML
  # UI will appear automatically if ipywidgets is available.
"""

import os
import re
import sys
import argparse
from typing import List, Dict, Optional

import pandas as pd
from bs4 import BeautifulSoup

# -------------------------
# 1) Extraction utilities
# -------------------------

BRACKET_ANNOT_RE = re.compile(r"\[.*?\]")

def extract_concordances_with_cleaned_merge(directory: str) -> pd.DataFrame:
    """
    Parse all .html files in a directory that contain concordance tables.
    For each row, collect left/match/right and construct a cleaned Merged Context.
    """
    data: List[Dict] = []

    for file_name in sorted(os.listdir(directory)):
        if not file_name.lower().endswith(".html"):
            continue
        file_path = os.path.join(directory, file_name)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                soup = BeautifulSoup(f, "html.parser")
        except Exception as e:
            print(f"[WARN] Could not parse {file_path}: {e}", file=sys.stderr)
            continue

        rows = soup.find_all("tr")
        for row in rows:
            cells = row.find_all("td")
            if len(cells) < 3:
                continue

            left = cells[0].get_text(strip=True)
            match = cells[1].get_text(strip=True)
            right = cells[2].get_text(strip=True)

            # Strip bracketed annotation from match only: [wejść:praet:sg:f:perf]
            match_cleaned = BRACKET_ANNOT_RE.sub(" ", match).strip()

            # Coalesce spaces — ensure single spaces around pieces
            merged_context = " ".join((left, match_cleaned, right)).strip()
            merged_context = re.sub(r"\s+", " ", merged_context)

            data.append({
                "Source File": file_name,
                "Left Context": left,
                "Match": match,
                "Right Context": right,
                "Merged Context": merged_context
            })

    df = pd.DataFrame(data)
    if df.empty:
        print("[INFO] No concordances found. Is the directory correct?")
    return df


# ----------------------------------------
# 2) Notebook UI (ipywidgets) if available
# ----------------------------------------

def in_notebook() -> bool:
    try:
        from IPython import get_ipython  # noqa
        ip = get_ipython()
        return ip is not None and "IPKernelApp" in ip.config
    except Exception:
        return False

def run_notebook_ui(df: pd.DataFrame,
                    start_index: int = 0,
                    default_include: bool = True,
                    out_all: Optional[str] = None,
                    out_selected: Optional[str] = None) -> None:
    """
    Launch an ipywidgets editor for the DataFrame.
    - 'Edited Sentence' starts from 'Merged Context'
    - 'Selected' toggle controls saving to selected CSV
    """
    import ipywidgets as widgets
    from IPython.display import display, clear_output

    if "Edited Sentence" not in df.columns:
        df["Edited Sentence"] = df["Merged Context"].astype(str)
    if "Selected" not in df.columns:
        df["Selected"] = default_include

    n = len(df)
    index = min(max(0, start_index), max(0, n - 1)) if n else 0

    # Widgets
    idx_label = widgets.HTML(value=f"<b>Row:</b> {index+1}/{n}")
    src_label = widgets.HTML(value="")
    text_area = widgets.Textarea(layout=widgets.Layout(width="100%", height="140px"))
    selected_chk = widgets.Checkbox(description="Include this row", value=default_include)
    status = widgets.Output(layout=widgets.Layout(border="1px solid #ddd"))

    btn_prev = widgets.Button(description="◀ Prev", tooltip="Previous row")
    btn_next = widgets.Button(description="Next ▶", button_style="primary", tooltip="Next row")
    btn_save = widgets.Button(description="💾 Save", tooltip="Save CSVs to disk")
    btn_quit = widgets.Button(description="⏹ Save & Exit", tooltip="Save CSVs and stop")

    nav = widgets.HBox([btn_prev, btn_next, btn_save, btn_quit])

    def refresh():
        if n == 0:
            text_area.value = ""
            selected_chk.value = False
            idx_label.value = "No rows."
            src_label.value = ""
            return

        idx_label.value = f"<b>Row:</b> {index+1}/{n}"
        src_label.value = f"<b>Source:</b> {df.at[index, 'Source File']}"
        text_area.value = df.at[index, "Edited Sentence"]
        selected_chk.value = bool(df.at[index, "Selected"])

    def persist_current():
        if n == 0:
            return
        df.at[index, "Edited Sentence"] = text_area.value
        df.at[index, "Selected"] = bool(selected_chk.value)

    @status.capture(clear_output=True)
    def on_prev(_):
        nonlocal index
        persist_current()
        if index > 0:
            index -= 1
        refresh()

    @status.capture(clear_output=True)
    def on_next(_):
        nonlocal index
        persist_current()
        if index < n - 1:
            index += 1
        refresh()

    def _write_outputs():
        msg = []
        if out_all:
            df.to_csv(out_all, index=False, encoding="utf-8")
            msg.append(f"All rows → {out_all}")
        if out_selected:
            sel = df[df["Selected"] == True]  # noqa: E712
            sel.to_csv(out_selected, index=False, encoding="utf-8")
            msg.append(f"Selected rows → {out_selected} ({len(sel)})")
        return "\n".join(msg) if msg else "No output paths provided."

    @status.capture(clear_output=True)
    def on_save(_):
        persist_current()
        print(_write_outputs())

    @status.capture(clear_output=True)
    def on_quit(_):
        persist_current()
        print(_write_outputs())
        # Disable controls
        for w in (btn_prev, btn_next, btn_save, btn_quit, text_area, selected_chk):
            w.disabled = True
        print("Done.")

    btn_prev.on_click(on_prev)
    btn_next.on_click(on_next)
    btn_save.on_click(on_save)
    btn_quit.on_click(on_quit)

    # Layout
    display(idx_label, src_label, text_area, selected_chk, nav, status)
    refresh()


# -------------------------------------
# 3) Terminal (standalone) editor mode
# -------------------------------------

def run_terminal_ui(df: pd.DataFrame,
                    start_index: int = 0,
                    default_include: bool = True,
                    out_all: Optional[str] = None,
                    out_selected: Optional[str] = None) -> None:
    """
    Minimal console UI:
      - shows the current merged context,
      - lets you type an edited sentence (or press Enter to keep),
      - lets you choose whether to include the row (y/N),
      - supports n/p/q to navigate/quit.
    Saves two CSVs at the end if paths are provided.
    """
    if df.empty:
        print("[INFO] Nothing to edit.")
        return

    if "Edited Sentence" not in df.columns:
        df["Edited Sentence"] = df["Merged Context"].astype(str)
    if "Selected" not in df.columns:
        df["Selected"] = default_include

    n = len(df)
    i = min(max(0, start_index), n - 1)

    print("=== Concordance Editor (Terminal) ===")
    print("Commands: [Enter]=keep text | type new text=edit | 'n'=next | 'p'=prev | 'q'=save & quit")
    print("For include prompt: 'y' to include, Enter or 'n' to exclude (default depends on --default-include).")
    print("-------------------------------------")

    while True:
        row = df.iloc[i]
        print(f"\n[{i+1}/{n}] Source: {row['Source File']}")
        print(f"Original: {row['Merged Context']}")
        print(f"Current : {row['Edited Sentence']}")
        new_text = input("Edit (leave blank to keep / 'n' next / 'p' prev / 'q' quit): ").strip()

        if new_text.lower() in ("n", "next"):
            # keep current
            pass
        elif new_text.lower() in ("p", "prev"):
            i = max(0, i - 1)
            continue
        elif new_text.lower() in ("q", "quit"):
            break
        elif new_text != "":
            df.at[i, "Edited Sentence"] = new_text

        incl_default = "y" if bool(df.at[i, "Selected"]) else ("y" if default_include else "n")
        incl = input(f"Include this row? [y/N] (default={incl_default}): ").strip().lower()
        if incl == "":
            incl = incl_default
        df.at[i, "Selected"] = (incl == "y")

        # Next row
        if i < n - 1:
            i += 1
        else:
            print("\n[End of list]")
            break

    # Save
    if out_all:
        df.to_csv(out_all, index=False, encoding="utf-8")
        print(f"All rows saved → {out_all}")
    if out_selected:
        sel = df[df["Selected"] == True]  # noqa: E712
        sel.to_csv(out_selected, index=False, encoding="utf-8")
        print(f"Selected rows saved → {out_selected} ({len(sel)})")


# -------------
# Entry point
# -------------

def main():
    parser = argparse.ArgumentParser(description="Extract NKJP concordances from HTML, edit & select rows.")
    parser.add_argument("--html-dir", required=True, help="Directory containing concordance HTML files.")
    parser.add_argument("--start-index", type=int, default=0, help="Row index to start editing from (0-based).")
    parser.add_argument("--default-include", action="store_true",
                        help="Default Selected=True for new sessions (otherwise False).")
    parser.add_argument("--out-all", default="concordances_all.csv", help="Path to save ALL rows CSV.")
    parser.add_argument("--out-selected", default="concordances_selected.csv", help="Path to save SELECTED rows CSV.")
    parser.add_argument("--mode", choices=["auto", "notebook", "terminal"], default="auto",
                        help="Force mode or auto-detect Jupyter.")

    args = parser.parse_args()

    if not os.path.isdir(args.html_dir):
        print(f"[ERROR] No such directory: {args.html_dir}", file=sys.stderr)
        sys.exit(1)

    df = extract_concordances_with_cleaned_merge(args.html_dir)
    if df.empty:
        sys.exit(0)

    if args.mode == "notebook" or (args.mode == "auto" and in_notebook()):
        run_notebook_ui(df,
                        start_index=args.start_index,
                        default_include=args.default_include,
                        out_all=args.out_all,
                        out_selected=args.out_selected)
    else:
        run_terminal_ui(df,
                        start_index=args.start_index,
                        default_include=args.default_include,
                        out_all=args.out_all,
                        out_selected=args.out_selected)

if __name__ == "__main__":
    main()

"""
Created on Tue Aug 12 13:18:09 2025

@author: niran
"""

