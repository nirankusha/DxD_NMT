# -*- coding: utf-8 -*-
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#!/usr/bin/env python3

"""
nkjp_pelcra_client.py - Updated Version
PELCRA NKJP programmatic client (per official reference; servlet params: query, offset, limit<=5000, span,
preserve_order, sort, second_sort, groupBy, groupByLimit, dummystring, sid, and m_* filters).

Updated for new CLARIN-PL infrastructure with endpoint discovery and fallback mechanisms.

- Concordance XML mode: POST to NKJPSpanSearchXML endpoint
  Parses <results><concordance><line>… into rows (left, match, right, + metadata).
- Excel workbook mode (--excel): POST to NKJPSpanSearchExcelXML endpoint
  Saves the returned Excel XML workbook (two sheets) to disk (no parsing).

Robustness:
- Automatic endpoint discovery with fallbacks
- Warm-up GET to index_adv.jsp (cookies/session)
- Detects HTML error/auth pages and dumps to --debug-save
- Cleans malformed XML (unescaped & and junk after </results>)

Examples:
  python nkjp_pelcra_client.py -q 'pleść** bzdura**' --limit 500 --max-total 2000 --out bzdura_kwic.csv
  python nkjp_pelcra_client.py -q 'łza**|łezka**___oko**' --span 2 --preserve-order --out lza_oko.tsv --tsv
  python nkjp_pelcra_client.py -q 'kobieta' --excel --limit-excel 500 --out kobieta_results.xml
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import re
import string
import sys
import time
from typing import Dict, List, Optional, Tuple

import requests
import xml.etree.ElementTree as ET


# ---------------- Config (HTTP hosts with discovery) ----------------
# Primary base URL (new CLARIN-PL infrastructure)
PRIMARY_BASE = "https://pelcra-nkjp.clarin-pl.eu"

# Alternative base URLs to try if primary fails
ALTERNATIVE_BASES = [
    "https://pelcra.clarin-pl.eu/NKJP",
    "http://nkjp.uni.lodz.pl",  # Original URL as last resort
]

# Global variables to store discovered working base
WORKING_BASE = None
XML_ENDPOINT = None
EXCEL_ENDPOINT = None

HEADERS = {
    "User-Agent": "NKJP-Client/0.7 (+python-requests)",
    "Accept": "application/xml,text/xml;q=0.9,*/*;q=0.8",
}


# ---------------- Endpoint Discovery ----------------
def test_endpoint(base_url: str, endpoint_path: str = "/index_adv.jsp", timeout: int = 10) -> bool:
    """Test if a base URL and endpoint are accessible."""
    try:
        test_url = f"{base_url}{endpoint_path}"
        response = requests.head(test_url, timeout=timeout, allow_redirects=True)
        return response.status_code < 400
    except Exception:
        return False

def discover_working_endpoints(debug: bool = False) -> Tuple[str, str, str]:
    """Discover which base URL is currently working and return endpoints."""
    global WORKING_BASE, XML_ENDPOINT, EXCEL_ENDPOINT
    
    if WORKING_BASE:  # Already discovered
        return WORKING_BASE, XML_ENDPOINT, EXCEL_ENDPOINT
    
    bases_to_try = [PRIMARY_BASE] + ALTERNATIVE_BASES
    
    for base_url in bases_to_try:
        if debug:
            print(f"Testing base URL: {base_url}")
        
        if test_endpoint(base_url, "/index_adv.jsp"):
            WORKING_BASE = base_url
            XML_ENDPOINT = f"{base_url}/NKJPSpanSearchXML"
            EXCEL_ENDPOINT = f"{base_url}/NKJPSpanSearchExcelXML"
            
            if debug:
                print(f"✓ Using working base URL: {base_url}")
            elif base_url != PRIMARY_BASE:
                print(f"Note: Using alternative base URL: {base_url}")
            
            return WORKING_BASE, XML_ENDPOINT, EXCEL_ENDPOINT
    
    # Fallback to primary if none work
    if debug:
        print("Warning: Could not verify any base URL, using primary as fallback")
    
    WORKING_BASE = PRIMARY_BASE
    XML_ENDPOINT = f"{PRIMARY_BASE}/NKJPSpanSearchXML"
    EXCEL_ENDPOINT = f"{PRIMARY_BASE}/NKJPSpanSearchExcelXML"
    
    return WORKING_BASE, XML_ENDPOINT, EXCEL_ENDPOINT


# ---------------- XML helpers ----------------
RE_AMP = re.compile(r"&(?!(?:amp|lt|gt|apos|quot|#[0-9]+|#x[0-9A-Fa-f]+);)")

def _clean_xml(text: str) -> str:
    """Trim anything after </results> and escape stray ampersands."""
    end = text.rfind("</results>")
    if end != -1:
        text = text[: end + len("</results>")]
    return RE_AMP.sub("&amp;", text)

def _parse_xml_or_raise(text: str) -> ET.Element:
    try:
        return ET.fromstring(text)
    except ET.ParseError:
        return ET.fromstring(_clean_xml(text))


# ---------------- Payload helpers ----------------
def _rand_sid(n: int = 12) -> str:
    alphabet = string.ascii_letters + string.digits
    return "".join(random.choice(alphabet) for _ in range(n))

def _build_payload(args, offset: int, limit: int, sid: Optional[str] = None) -> Dict[str, str]:
    return {
        "query": args.query,
        "offset": str(max(0, offset)),
        "limit": str(min(max(limit, 1), 5000)),  # server cap
        "span": str(args.span),
        "preserve_order": "true" if args.preserve_order else "false",
        "sort": args.sort,
        "second_sort": args.second_sort,
        "groupBy": args.group_by if args.group_by else "",
        "groupByLimit": str(args.group_limit),
        # metadata filters
        "m_date_from": args.m_date_from or "",
        "m_date_to": args.m_date_to or "",
        "m_styles": args.m_styles or "",
        "m_channels": args.m_channels or "",
        "m_title_mono": args.m_title_mono or "",
        "m_title_mono_NOT": args.m_title_mono_not or "",
        "m_paragraphKWs_MUST": args.m_paragraph_must or "",
        "m_paragraphKWs_MUST_NOT": args.m_paragraph_must_not or "",
        # legacy compatibility (seen in official example)
        "dummystring": "x",
        "sid": sid or _rand_sid(),
    }


# ---------------- Session initialization ----------------
def initialize_session(base_url: str, timeout: int = 30) -> requests.Session:
    """Initialize session with proper headers and warm-up."""
    session = requests.Session()
    
    # Update headers with correct referer
    headers = HEADERS.copy()
    headers["Referer"] = f"{base_url}/index_adv.jsp"
    session.headers.update(headers)
    
    # Session warm-up (cookies, etc.)
    try:
        warmup_url = f"{base_url}/index_adv.jsp"
        session.get(warmup_url, timeout=timeout, allow_redirects=True)
    except Exception:
        pass  # non-fatal
    
    return session


# ---------------- Concordance XML mode ----------------
def fetch_concordance_page(session: requests.Session, args, offset: int, limit: int,
                           debug_dir: Optional[str] = None) -> Tuple[List[Dict], Dict]:
    _, xml_endpoint, _ = discover_working_endpoints()
    
    payload = _build_payload(args, offset, limit)
    r = session.post(xml_endpoint, data=payload, timeout=args.timeout, allow_redirects=True)
    r.raise_for_status()
    body = r.text
    ctype = r.headers.get("Content-Type", "")

    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
        with open(os.path.join(debug_dir, f"xml_off{offset}_lim{limit}.raw.txt"), "w", encoding="utf-8") as f:
            f.write(body)

    if "html" in ctype.lower() or body.lstrip().lower().startswith(("<!doctype html", "<html")):
        if debug_dir:
            with open(os.path.join(debug_dir, "last_error.html"), "w", encoding="utf-8") as f:
                f.write(body)
        raise RuntimeError(f"Server returned HTML (error/auth page) from {xml_endpoint}. See debug dump.")

    try:
        root = _parse_xml_or_raise(body)
    except ET.ParseError as e:
        if debug_dir:
            with open(os.path.join(debug_dir, "last_bad_xml.xml"), "w", encoding="utf-8") as f:
                f.write(body)
        raise RuntimeError(f"Could not parse XML from {xml_endpoint}: {e}")

    meta = {
        "query": (root.findtext("query") or "").strip(),
        "offset": offset,
        "limit": limit,
        "total": -1,
    }
    if "total" in root.attrib:
        try:
            meta["total"] = int(root.attrib.get("total", "-1"))
        except ValueError:
            pass

    rows: List[Dict] = []
    lines_parent = root.find("concordance")
    if lines_parent is not None:
        for line in lines_parent.findall("line"):
            rows.append({
                "left": (line.findtext("left") or "").strip(),
                "match": (line.findtext("match") or "").strip(),
                "right": (line.findtext("right") or "").strip(),
                "pubDate": (line.findtext("pubDate") or "").strip(),
                "channel": (line.findtext("channel") or "").strip(),
                "domain": (line.findtext("domain") or "").strip(),
                "title_mono": (line.findtext("title_mono") or "").strip(),
                "title_a": (line.findtext("title_a") or "").strip(),
            })

    return rows, meta


def fetch_concordance_all(session: requests.Session, args) -> List[Dict]:
    all_rows: List[Dict] = []
    offset = 0
    total_known: Optional[int] = None

    while True:
        rows, meta = fetch_concordance_page(session, args, offset, args.limit, args.debug_save)

        if total_known is None and isinstance(meta.get("total"), int):
            total_known = meta["total"]

        all_rows.extend(rows)
        offset += len(rows)

        # Stop conditions
        if not rows:
            break
        if args.max_total is not None and len(all_rows) >= args.max_total:
            all_rows = all_rows[:args.max_total]
            break
        if isinstance(total_known, int) and total_known >= 0 and offset >= total_known:
            break

        time.sleep(args.pause)

    return all_rows


# ---------------- Excel workbook mode ----------------
def fetch_excel_workbook(session: requests.Session, args) -> bytes:
    _, _, excel_endpoint = discover_working_endpoints()
    
    payload = _build_payload(args, args.offset_excel, args.limit_excel)
    r = session.post(excel_endpoint, data=payload, timeout=args.timeout, allow_redirects=True)
    r.raise_for_status()

    ctype = r.headers.get("Content-Type", "")
    # Some servers return application/vnd.ms-excel or text/xml
    if "html" in ctype.lower() or r.text.lstrip().lower().startswith(("<!doctype html", "<html")):
        if args.debug_save:
            os.makedirs(args.debug_save, exist_ok=True)
            with open(os.path.join(args.debug_save, "excel_last_error.html"), "w", encoding="utf-8") as f:
                f.write(r.text)
        raise RuntimeError(f"Server returned HTML for Excel endpoint {excel_endpoint} (error/auth). See debug dump.")
    return r.content


# ---------------- I/O helpers ----------------
def save_table(rows: List[Dict], path: str, tsv: bool = False) -> None:
    if not rows:
        with open(path, "w", encoding="utf-8") as f:
            f.write("")
        return
    fieldnames = list(rows[0].keys())
    dialect = "excel-tab" if tsv else "excel"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, dialect=dialect)
        w.writeheader()
        w.writerows(rows)


# ---------------- CLI ----------------
def main() -> None:
    ap = argparse.ArgumentParser(description="PELCRA NKJP programmatic client (concordance XML / Excel XML) - Updated for CLARIN-PL")
    ap.add_argument("-q", "--query", required=True, help="Query in PELCRA syntax (e.g., 'pleść** bzdura**', 'łza**|łezka**___oko**').")

    # XML mode paging
    ap.add_argument("--limit", type=int, default=500, help="Page size for XML mode (1..5000).")
    ap.add_argument("--max-total", type=int, default=2000, help="Max total rows to fetch (XML mode).")

    # Query options
    ap.add_argument("--span", type=int, default=2, help="Max gap between query terms.")
    ap.add_argument("--preserve-order", action="store_true", help="Preserve word order in multi-term queries.")
    ap.add_argument("--sort", default="srodek", help="Primary sort: srodek/left/right/source/date/channel.")
    ap.add_argument("--second-sort", default="srodek", help="Secondary sort.")
    ap.add_argument("--group-by", default="", help="Group by: '' | year | source | text.")
    ap.add_argument("--group-limit", type=int, default=0, help="Max results per group (0 = no limit).")

    # Metadata filters
    ap.add_argument("--m-date-from", default="", help="Metadata: date from (YYYY or YYYY-MM-DD).")
    ap.add_argument("--m-date-to", default="", help="Metadata: date to (YYYY or YYYY-MM-DD).")
    ap.add_argument("--m-styles", default="", help="Metadata: styles taxonomy (e.g., 'publ;lit;...').")
    ap.add_argument("--m-channels", default="", help="Metadata: channels.")
    ap.add_argument("--m-title-mono", dest="m_title_mono", default="", help="Metadata: source title MUST contain.")
    ap.add_argument("--m-title-mono-not", dest="m_title_mono_not", default="", help="Metadata: source title MUST NOT contain.")
    ap.add_argument("--m-paragraph-must", dest="m_paragraph_must", default="", help="Metadata: paragraph keywords MUST contain.")
    ap.add_argument("--m-paragraph-must-not", dest="m_paragraph_must_not", default="", help="Metadata: paragraph keywords MUST NOT contain.")

    # Excel mode
    ap.add_argument("--excel", action="store_true", help="Use Excel endpoint and save raw workbook (Excel XML).")
    ap.add_argument("--offset-excel", type=int, default=0, help="Starting offset for Excel export.")
    ap.add_argument("--limit-excel", type=int, default=500, help="Limit for Excel export page (1..5000).")

    # Misc
    ap.add_argument("--pause", type=float, default=0.5, help="Sleep between pages (XML mode).")
    ap.add_argument("--timeout", type=int, default=60, help="HTTP timeout seconds.")
    ap.add_argument("--out", default="", help="Output path: CSV/TSV for XML mode; .xml workbook for --excel.")
    ap.add_argument("--tsv", action="store_true", help="Write TSV instead of CSV (XML mode only).")
    ap.add_argument("--debug-save", default="", help="Directory to dump raw responses and errors.")
    ap.add_argument("--verbose", action="store_true", help="Show endpoint discovery process.")
    args = ap.parse_args()

    # Discover working endpoints
    try:
        base_url, xml_endpoint, excel_endpoint = discover_working_endpoints(debug=args.verbose)
    except Exception as e:
        print(f"[ERROR] Could not discover working endpoints: {e}", file=sys.stderr)
        sys.exit(2)

    # Initialize session
    session = initialize_session(base_url, timeout=30)

    if args.excel:
        if not args.out:
            print("[ERROR] --out is required for --excel mode (e.g., results.xml)", file=sys.stderr)
            sys.exit(2)
        try:
            content = fetch_excel_workbook(session, args)
        except Exception as e:
            print(f"[ERROR] Excel export failed: {e}", file=sys.stderr)
            if "HTML" in str(e):
                print(f"[INFO] This may indicate the API has changed. Consider contacting PELCRA team.", file=sys.stderr)
            sys.exit(2)
        with open(args.out, "wb") as f:
            f.write(content)
        print(f"Wrote Excel XML workbook → {args.out}")
        sys.exit(0)

    # XML concordance mode
    try:
        rows = fetch_concordance_all(session, args)
    except Exception as e:
        print(f"[ERROR] Concordance search failed: {e}", file=sys.stderr)
        if "HTML" in str(e):
            print(f"[INFO] The server returned HTML instead of XML. This may indicate:", file=sys.stderr)
            print(f"       - The API endpoints have changed", file=sys.stderr)
            print(f"       - Authentication is now required", file=sys.stderr)
            print(f"       - The service is temporarily unavailable", file=sys.stderr)
            print(f"       Consider contacting the PELCRA team at piotr.pezik@gmail.com", file=sys.stderr)
        sys.exit(2)

    if not rows:
        print("No rows returned.")
        sys.exit(0)

    if args.out:
        save_table(rows, args.out, tsv=args.tsv)
        print(f"Wrote {len(rows)} rows → {args.out}")
    else:
        # Preview
        preview = min(10, len(rows))
        for i, r in enumerate(rows[:preview]):
            print(f"{i+1:>4}  {r['left']} [ {r['match']} ] {r['right']}  |  {r.get('title_mono','')} ({r.get('pubDate','')})")
        if len(rows) > preview:
            print(f"... and {len(rows)-preview} more rows (use --out to save).")


if __name__ == "__main__":
    main()

"""
Created on Tue Aug 12 11:57:32 2025

@author: niran

Updated on Tue Aug 12 2025 for CLARIN-PL infrastructure
- Added endpoint discovery with fallback mechanisms
- Updated base URLs for new infrastructure
- Enhanced error handling for HTML responses
- Added verbose mode for debugging endpoint discovery
- Improved session initialization with proper headers

Changelog:
v0.7 - Updated for CLARIN-PL infrastructure migration
v0.6 - Original version for nkjp.uni.lodz.pl
"""
"""
Created on Tue Aug 12 11:57:32 2025

@author: niran
"""

