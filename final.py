"""
Cutler Capital — Hedge Fund Letter Scraper
------------------------------------------
Internal Cutler Capital tool to scrape, excerpt, and compile hedge-fund letters
by fund family and quarter. Uses an external hedge-fund letter database as the
data source; all branding in the UI is Cutler-only.
"""
from __future__ import annotations

# Windows Playwright policy fix
import asyncio, platform
if platform.system() == "Windows":
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    except Exception:
        pass

import re
import os
import sys
import shutil
import traceback
import json
import hashlib
import openai
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
import re as _re
import html as html_lib
import streamlit as st
import requests
import sa_analysis_api as sa_api 

# pypdf compat
try:
    from pypdf import PdfMerger
except Exception:
    from pypdf import PdfWriter, PdfReader
    class PdfMerger:
        def __init__(self): self._w = PdfWriter()
        def append(self, p: str):
            r = PdfReader(p)
            for pg in r.pages: self._w.add_page(pg)
        def write(self, out: str):
            with open(out, 'wb') as f: self._w.write(f)
        def close(self): pass
from pypdf import PdfReader as _PdfReader, PdfWriter as _PdfWriter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError
import sa_news_ai as sa_news
import seekingalpha_excerpts as sa_scraper

# local imports
import importlib.util
HERE = Path(__file__).resolve().parent

def _import(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if not spec or not spec.loader:
        raise ImportError(f"Cannot import {name} from {path}")
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules[name] = mod
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod

excerpt_check = _import("excerpt_check", HERE / "excerpt_check.py")
make_pdf = _import("make_pdf", HERE / "make_pdf.py")

# Seeking Alpha news + AI digest
try:
    sa_news_ai = _import("sa_news_ai", HERE / "sa_news_ai.py")
except Exception:
    sa_news_ai = None  # SA integration is optional

# Ticker dictionary (we reuse for the SA dropdown)
try:
    tickers_mod = _import("tickers", HERE / "tickers.py")
except Exception:
    tickers_mod = None

from sa_analysis_api import (
    fetch_analysis_list,
    fetch_article_details,
    build_sa_analysis_digest,
    analyse_symbol_with_digest,
    AnalysisArticle,
    fetch_analysis_details,
)


# paths (still stored under BSD/ on disk; UI is Cutler-branded only)
BASE = HERE / "BSD"
DL_DIR = BASE / "Downloads"
EX_DIR = BASE / "Excerpts"
CP_DIR = BASE / "Compiled"
MAN_DIR = BASE / "Manifests"   # run manifests (Document Checker + incremental)
DELTA_DIR = BASE / "Delta"     # delta PDFs + JSONs (Document Checker)
for d in (DL_DIR, EX_DIR, CP_DIR, MAN_DIR, DELTA_DIR):
    d.mkdir(parents=True, exist_ok=True)

ai_insights = _import("ai_insights", HERE / "ai_insights.py")

# ---- BIG LIST → 7 batches (no repeats) ----
_BIG_LIST_RAW = r"""
AB (AllianceBernstein)
AQR Funds
AXA Equitable
AXS Investments
Aegis Funds
Alger
Allianz Global Investors
Alps Funds
Amana Mutual Funds
American Century Investments
American Century Investments (ACI)
American Funds
Amundi US
Appleseed Fund
Ariel Investments
Ariel Investments
Artisan Partners
BNY Mellon
Baird Funds
Baron Capital
Baron Funds
BlackRock
Boston Partners
Boston Trust Walden
Brandywine Global
Brown Advisory
Buffalo Funds
Calamos Investments
Calamos Investments
Cambiar Investors
Causeway Capital
Causeway Capital Management
Cavanal Hill Funds
Centerstone Investors
Chiron Investment Management
Clarkston Capital
ClearBridge Investments
Clipper Fund
Cohen & Steers
Columbia Threadneedle
Commerce Funds
Commerce Trust
Crossmark Global
Cullen Funds
Davidson Investment
Delaware Funds
Delaware Investments
Destra Capital
Dodge & Cox
Dreyfus Corporation
Driehaus Capital Management
Eaton Vance
Eventide
Eventide Funds
FPA Funds
Federated Hermes
Federated Hermes
Fidelity Advisor Funds
Fidelity Investments
Fidelity Investments
Fidelity Investments
Fiera Capital
First Eagle
First Eagle Fund
Foundry Partners
Frank Funds
Franklin Templeton
Gabelli Funds
Gerstein Fisher
Goldman Sachs Asset Management
Grandeur Peak
Grandeur Peak Global Advisors
Guardian Capital
Guggenheim Investments
Harbor Capital
Harbor Funds
Heartland Advisors
Hennessy Funds
Homestead Funds
Integrity Viking
Invesco
J.P. Morgan
Janus Fund
Janus Henderson
John Hancock Investments
Keeley Funds
Kinetics Mutual Funds
Lazard Asset Management
Legg Mason
Lincoln Financial
LoCorr Funds
Longleaf Partners
Longleaf Partners
Longleaf Partners
Longleaf Partners
Lord Abbett
Lord Abbett
MFS Investment Management
MFS Investment Management
MFS Investment Management
MP63 Fund
MainStay Investments
MassMutual
MassMutual RetireSmart
Matthews Asia
Mesirow Financial
Morgan Stanley Investment Mgmt
Nationwide Funds
Natixis Investment Managers
Neuberger Berman
New York Life Investments
North Square Investments
Northern Trust
Nuveen
Oakmark Fund
Oakmark Fund
Oakmark Fund
Oakmark Fund
Oakmark Funds
Old Mutual Asset Management
Osterweis Capital
PGIM Investments
PIMCO
Pax World Funds
Polen Capital
Poplar Forest Funds
Principal
Putnam Investments
Queens Road Funds
Reynders McVeigh
Royce Investment Partners
Russell Investments
Scout Investments
Seafarer Capital Partners
Sequoia Funds
Shelton Capital
Smead Capital
State Street Global Advisors
Steward Partners
T. Rowe Price
T. Rowe Price
TCW Group
TIAA
Third Avenue
Thornburg Investment Mgmt
Thrivent Funds
Tocqueville Asset Management
Torray Fund
Torray Resolute
Tortoise Ecofin
Touchstone
Touchstone Investments
Touchstone Sands
Transamerica
Tweedy, Browne Company
Value Line Funds
VanEck
Victory Capital
Virtue Funds
Virtus Investment Partners
WCM Investment Mgmt
Wasatch Global Investors [Check main.py]
Weitz Investment Management
Westwood Holdings
William Blair
Yacktman Funds
Zevenbergen Capital

1290 Funds
1919 Funds
1WS Capital
AFA Funds
AGF Investments
AMG
ARK
Abbey Capital
Absolute Investment Advisers
Adirondack
Adler
AdvisorShares
Advisors Asset Management
Akre Capital
Allspring
Anchor Capital
Angel Oak
Apollo
Aptus
Archer
Ares Capital
Aristotle
Ashmore
Aspiration
Aspiriant
BBH
Baillie Gifford
Belmont Capital
Berkshire
Bexil
Blue Current
Blueprint
Boyar
Boyd Watterson
Brandes
Bretton Capital
Bridges
Bright Rock
CCT Asset Management
CRM Funds
Capital Group
Carillon
Castle Investment Management
Catholic Responsible Investments
Champlain
Chesapeake
Clark Fork
Clifford Capital
Clough Capital
Community Capital Management
Conestoga
Congress Asset Management
Copeland
Core Alternative
CornerCap
Cornerstone
Cove Street Capital
Covered Bridge
Cromwell
CrossingBridge
Cushing
Cutler Investment Group
DWS
Dana Funds
Davenport
Davis
Dean Mutual Funds
Dearborn Partners
Diamond Hill
Dinosaur Group
Direxion
Distillate Capital
Domini
Duff & Phelps
Dupree
E-Valuator
Easterly
Edgar Lomax
Edgewood
Empower
EquityCompass
FMI Funds
FS Investments
Fairholme
Fenimore Asset Management
First Pacific
Firsthand Funds
Flaherty & Crumrine
Forester Funds
Fort Pitt Capital
Frontier Funds
Frost Funds
GMO
GQG Partners
Gator Capital
Geneva
Glenmede
GoodHaven
Grayscale
GuideStone
Guinness Atkinson
Hamlin Funds
Harding Loevner
HartFord
Haverford
Hotchkis & Wiley
Hussman
IDX
IPS Strategic Capital
Impax
Innovator
Intrepid Capital
Jacob
James
Johnson
Kayne Anderson
Kirr, Marbach Partners
Knights of Columbus
Kopernik
LSV
Lawson Kroeker
Leavell
Leuthold
Long Short Advisors
Lyrical
MH Elite
Madison
MainGate
Mairs & Power
Manor Funds
Marsico
Meehan
Meridian
MetLife
Midas
Miller/Howard
Mondrian
Motley Fool
Mundoval
Mutual of America
Muzinich
Needham
NexPoint
Nicholas
North Country
NorthQuest
Northeast Investors
O'Shaughnessy
OCM
OTG
Oberweis
Old Westbury
Otter Creek
Overlay Shares
PIA
PT Asset Management
Pacer
Palm Valley
Panoramic
Papp
Paradigm
Parnassus
Peer Tree
Plan Investment Fund
Plumb
Popular Family of Funds
Potomac
Primark
Provident
Prudential
Quantified Funds
RBC
RMB
Ranger
Reaves
Recurrent
Rice Hall James
RiverPark
SBAuer
Sarofim
Selected Funds
Seven Canyons
Shenkman
Sound Shore
SouthernSun
Spinnaker ETF Trust
Sprott
Standpoint
Summit Global
Summitry
Tanaka
The Private Shares
Tributary
Trillium
U.S. Global Investors
Union Street Partners
VELA
Variant
Villere
Voya
Water Island Capital
Wilshire
Wireless
WisdomTree
Yorktown
abrdn
iMGP Funds

ACCLEVITY
ACR
ALPHCNTRC
Ave Maria Funds
BAHL & GAYNOR
Barrett
Baywood
BLACK OAK
BOSTON PART
Calvert
Centre Global
CIBC ATLAS
Clarity Fund Inc
Clifford Capital
Cohen & Steers
Column
Commonwealth Funds
CS McKee Collective Funds
Dearborn Partners
DF DENT
Diamond Hill
DOMINI IMPACT
DSM Funds
Eagle Energy
Enterprise Funds
EP EMERGING MARKETS
Europac
FAM
FIRST EAGLE
FRONTIER MFG
Galliard Funds
GAMCO
Glenmeade IM
Greenspring
Hundredfold
ICON Funds
ISHARES
Ivy
JamesAdvantage Funds
JH FINANCIAL
JP Morgan
Matrix Asset Advisors Fund
Members Funds
MG TRUST CTF
Muhlenkamp Funds
Oberweiss Funds
PAX World
Payson Funds
PrimeCap Odyssey
RidgeWorth Funds
River Park Funds
Rogue Fudns
RS Investments
Rydex Funds
TransAmerica/Idex
VILLERE FUNDS
ZACKS
"""

def _clean_name(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\[.*?\]", "", s)        # drop bracketed notes
    s = re.sub(r"\s+", " ", s)           # collapse spaces
    return s.strip("-•· ")

def _parse_big_list(raw: str) -> List[str]:
    seen = set(); out: List[str] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith('#'): continue
        name = _clean_name(line)
        key = name.lower()
        if key and key not in seen:
            seen.add(key); out.append(name)
    return out

def _get_all_known_tickers() -> list[str]:
    """Try to infer a ticker list from tickers.py; fall back to a few majors."""
    if tickers_mod is None:
        return ["TSLA", "AAPL", "NVDA"]

    candidates: set[str] = set()

    for name in dir(tickers_mod):
        val = getattr(tickers_mod, name)
        if isinstance(val, dict):
            # keys are often tickers in Vikrant-land
            for k in val.keys():
                candidates.add(str(k).upper())
        elif isinstance(val, (list, tuple, set)):
            # sometimes a flat list of tickers
            for item in val:
                if isinstance(item, str) and item.isalpha() and 1 <= len(item) <= 6:
                    candidates.add(item.upper())

    if not candidates:
        return ["TSLA", "AAPL", "NVDA"]

    return sorted(candidates)

def _chunk_round_robin(items: List[str], k: int) -> List[List[str]]:
    buckets = [[] for _ in range(k)]
    for i, it in enumerate(items): buckets[i % k].append(it)
    return buckets

ALL_FUND_NAMES = _parse_big_list(_BIG_LIST_RAW)
BATCH_COUNT = 7
_batches = _chunk_round_robin(ALL_FUND_NAMES, BATCH_COUNT)
RUNNABLE_BATCHES: Dict[str, List[str]] = {f"Batch {i+1}": b for i, b in enumerate(_batches)}

# External data source URL (kept internal; not shown in UI)
BSD_URL = "https://www.buysidedigest.com/hedge-fund-database/"
FILTERS = {
    "fund": "#md-fund-letter-table-fund-search",
    "quarter": "#md-fund-letter-table-select",
    "search_btn": "input.md-search-btn",
}
TABLE_ROW = "table tbody tr"
COLMAP = {"quarter": 1, "letter_date": 2, "fund_name": 3}

@dataclass
class Hit:
    quarter: str
    letter_date: str
    fund_name: str
    fund_href: str

_DEF_WORD_RE = re.compile(r"^[A-Za-z0-9'&.-]+")

def _first_word(name: str) -> str:
    m = _DEF_WORD_RE.search(name)
    return m.group(0) if m else (name.split()[0] if name.split() else name)

def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._") or "file"

def _set_quarter(page, wanted: str) -> bool:
    """
    Try to select the requested quarter in the site's <select>.
    Returns True if it exists, False otherwise.
    """
    sel = page.locator(FILTERS["quarter"]).first
    try:
        sel.select_option(value=wanted)
    except Exception:
        try:
            sel.select_option(label=wanted)
        except Exception:
            return False
    page.wait_for_timeout(250)
    return True

def _search_by_fund(page, keyword: str, retries: int = 2) -> None:
    """
    Type a fund keyword into the BSD fund search box and trigger the search.

    More robust than the original version:
    - Waits explicitly for the search input to be visible.
    - Uses longer timeouts.
    - Retries a couple of times on timeout (with reload) before giving up.
    """
    for attempt in range(retries + 1):
        try:
            # Wait for the search input to actually be there
            inp = page.locator(FILTERS["fund"]).first
            inp.wait_for(state="visible", timeout=20000)

            # Clear and type the keyword
            inp.fill("")
            inp.type(keyword, delay=10)

            # Click search
            page.locator(FILTERS["search_btn"]).first.click(force=True)

            # Wait for either network idle or at least one row to show
            try:
                page.wait_for_load_state("networkidle", timeout=15000)
            except PlaywrightTimeoutError:
                page.locator(TABLE_ROW).first.wait_for(
                    state="visible",
                    timeout=15000,
                )

            # If we got here without exceptions, search was successful
            return

        except PlaywrightTimeoutError:
            # If we still have retries left, reload and try again
            if attempt < retries:
                page.reload()
                page.wait_for_load_state("domcontentloaded", timeout=20000)
                continue
            # Out of retries: re-raise so caller logs the error for this fund only
            raise

def _parse_rows(page, quarter: str) -> List[Hit]:
    rows = page.locator(TABLE_ROW)
    hits: List[Hit] = []
    for i in range(rows.count()):
        row = rows.nth(i)
        try:
            q = row.locator("td").nth(COLMAP["quarter"]-1).inner_text().strip()
            if q != quarter:
                continue
            letter_date = row.locator("td").nth(COLMAP["letter_date"]-1).inner_text().strip()
            fund_cell = row.locator("td").nth(COLMAP["fund_name"]-1)
            link = fund_cell.locator("a").first
            fund_name = (link.inner_text() or '').strip()
            fund_href = link.get_attribute("href") or ""
            if fund_href:
                hits.append(Hit(q, letter_date, fund_name, fund_href))
        except Exception:
            continue
    return hits

def _download_quarter_pdf_from_fund(page, quarter: str, dest_dir: Path) -> List[Path]:
    dest_dir.mkdir(parents=True, exist_ok=True)
    pdfs: List[Path] = []
    try:
        page.locator("text=Quarterly Letters").first.wait_for(state="visible", timeout=8000)
    except Exception:
        pass
    anchors = page.locator("a").all()
    candidates = []
    for a in anchors:
        try:
            text = (a.inner_text() or '').strip()
            title = a.get_attribute("title") or ""
            href = a.get_attribute("href") or ""
            if not href:
                continue
            if (text == quarter or quarter in title) and ("letters/file" in href or href.lower().endswith('.pdf')):
                candidates.append((a, href))
        except Exception:
            continue
    for a, href in candidates:
        try:
            with page.expect_download(timeout=8000) as dl_info:
                a.click(force=True)
            dl = dl_info.value
            fname = _safe(Path(dl.suggested_filename or Path(href).name or f"{quarter}.pdf").name)
            path = dest_dir / fname
            dl.save_as(str(path))
            pdfs.append(path)
            continue
        except Exception:
            pass
        try:
            r = requests.get(href, timeout=20)
            if r.status_code == 200 and r.content:
                fname = _safe(Path(href).name or f"{quarter}.pdf")
                path = dest_dir / fname
                with open(path, 'wb') as f:
                    f.write(r.content)
                pdfs.append(path)
        except Exception:
            continue
    return pdfs

# excerption + build

def run_excerpt_and_build(pdf_path: Path, out_dir: Path, source_pdf_name: Optional[str] = None, letter_date: Optional[str] = None) -> Optional[Path]:
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        tp = out_dir / "tickers.py"
        if not tp.exists():
            # place a copy so make_pdf can import user tickers
            (HERE / "tickers.py").exists() and shutil.copy2(HERE / "tickers.py", tp)
        excerpt_check.excerpt_pdf_for_tickers(str(pdf_path), debug=False)
        src_json = pdf_path.parent / "excerpts_clean.json"
        if not src_json.exists():
            return None
        dst_json = out_dir / "excerpts_clean.json"
        if src_json != dst_json:
            shutil.copy2(src_json, dst_json)
        out_pdf = out_dir / f"Excerpted_{_safe(pdf_path.stem)}.pdf"
        make_pdf.build_pdf(
            excerpts_json_path=str(dst_json),
            output_pdf_path=str(out_pdf),
            report_title=f"Cutler Capital Excerpts – {pdf_path.stem}",
            source_pdf_name=source_pdf_name or pdf_path.name,
            format_style="legacy",
            letter_date=letter_date
        )
        return out_pdf if out_pdf.exists() else None
    except Exception:
        traceback.print_exc()
        return None

# stamping + compile

def _overlay_single_page(w: float, h: float, left: str, mid: str, right: str) -> BytesIO:
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=(w, h))
    c.setFont("Helvetica", 8.5)
    c.setFillColor(colors.HexColor("#4b2142"))  # Cutler purple
    L = R = 0.75 * 72
    T = 0.75 * 72
    if left:
        c.drawString(L, h - T + 0.35 * 72, left)
    if mid:
        text = (mid[:95] + '…') if len(mid) > 96 else mid
        c.drawCentredString(w / 2.0, h - T + 0.35 * 72, text)
    if right:
        c.drawRightString(w - R, h - T + 0.35 * 72, right)
    c.save()
    buf.seek(0)
    return buf

def _stamp_pdf(src: Path, left: str, mid: str, right: str) -> Path:
    try:
        r = _PdfReader(str(src))
    except Exception:
        return src
    w = _PdfWriter()
    for pg in r.pages:
        W = float(pg.mediabox.width)
        H = float(pg.mediabox.height)
        ov = _PdfReader(_overlay_single_page(W, H, left, mid, right)).pages[0]
        try:
            pg.merge_page(ov)
        except Exception:
            pass
        w.add_page(pg)
    tmp = src.with_suffix('.stamped.tmp.pdf')
    with open(tmp, 'wb') as f:
        w.write(f)
    return tmp

def compile_merged(batch: str, quarter: str, collected: List[Path]) -> Optional[Path]:
    if not collected:
        return None
    out = CP_DIR / f"Compiled_Cutler_{batch.replace(' ', '')}_{quarter}_{datetime.now():%Y%m%d_%H%M%S}.pdf"
    m = PdfMerger()
    added = 0
    for p in collected:
        try:
            title = p.stem.replace('_', ' ').replace('-', ' ')
            stamped = _stamp_pdf(p, left=batch, mid=title, right=f"Run {datetime.now():%Y-%m-%d %H:%M}")
            m.append(str(stamped))
            added += 1
        except Exception:
            continue
    if not added:
        m.close()
        return None
    try:
        m.write(str(out))
    finally:
        m.close()
    return out

# -------------------------------------------------------------------
# Seeking Alpha Analysis API helpers (RapidAPI)
# -------------------------------------------------------------------

SA_ANALYSIS_BASE = "https://seeking-alpha.p.rapidapi.com"


def _get_sa_rapidapi_key() -> str:
    key = os.getenv("SA_RAPIDAPI_KEY")
    if not key:
        raise RuntimeError(
            "SA_RAPIDAPI_KEY env var is not set – add it to your .env for Seeking Alpha Analysis."
        )
    return key

# sa_analysis_api.py

import os
import requests
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

SA_RAPIDAPI_KEY = os.environ.get("SA_RAPIDAPI_KEY")
if not SA_RAPIDAPI_KEY:
    raise SystemExit("SA_RAPIDAPI_KEY env var is not set")

BASE_URL = "https://seeking-alpha.p.rapidapi.com"

HEADERS = {
    "x-rapidapi-key": SA_RAPIDAPI_KEY,
    "x-rapidapi-host": "seeking-alpha.p.rapidapi.com",
}

@dataclass
class AnalysisArticle:
    id: str
    title: str
    published: str
    url: str
    # you already have these fields in your version, keep them if present
    # summary_html: Optional[str] = None
    # body_html: Optional[str] = None


def _call_sa(endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
    url = BASE_URL + endpoint
    resp = requests.get(url, headers=HEADERS, params=params, timeout=20)
    resp.raise_for_status()
    return resp.json()
    

def fetch_analysis_list(symbol: str, size: int = 5) -> List[AnalysisArticle]:
    """
    GET /analysis/v2/list?id={symbol}&size={size}&number=1
    """
    payload = _call_sa("/analysis/v2/list", {"id": symbol.lower(), "size": size, "number": 1})

    items = payload.get("data", [])
    out: List[AnalysisArticle] = []

    for item in items:
        attrs = item.get("attributes", {})
        art_id = str(item.get("id"))
        title = attrs.get("title", "")
        published = attrs.get("publishOn", "")
        link = "https://seekingalpha.com" + item.get("links", {}).get("self", "")

        out.append(AnalysisArticle(
            id=art_id,
            title=title,
            published=published,
            url=link,
        ))

    return out


def fetch_analysis_details(article_id: str) -> Dict[str, Any]:
    """
    GET /analysis/v2/get-details?id={article_id}
    Returns a dict with title, body_html, summary_html and image_url.
    """
    payload = _call_sa("/analysis/v2/get-details", {"id": article_id})

    try:
        main = payload["data"][0]
        attrs = main.get("attributes", {})
    except Exception:
        return {}

    # Different fields exist depending on endpoint version; cover both
    body_html = (
        attrs.get("bodyHtml")    # camelCase
        or attrs.get("body_html")
        or ""
    )
    summary_html = (
        attrs.get("summaryHtml")
        or attrs.get("summary_html")
        or ""
    )
    image_url = attrs.get("gettyImageUrl") or ""

    return {
        "title": attrs.get("title", ""),
        "body_html": body_html,
        "summary_html": summary_html,
        "image_url": image_url,
    }

def fetch_sa_analysis_list(symbol: str, size: int = 10) -> list[dict]:
    """
    Call /analysis/v2/list for a single symbol and return a simplified list
    of articles: id, title, published, primary_tickers, url.
    """
    api_key = _get_sa_rapidapi_key()
    url = f"{SA_ANALYSIS_BASE}/analysis/v2/list"

    params = {
        "id": symbol.lower(),  # API expects lowercase id like 'tsla', 'aapl'
        "size": str(size),
        "number": "1",         # first "page"
    }
    headers = {
        "x-rapidapi-key": api_key,
        "x-rapidapi-host": "seeking-alpha.p.rapidapi.com",
    }

    resp = requests.get(url, headers=headers, params=params, timeout=20)
    resp.raise_for_status()
    payload = resp.json()

    data = payload.get("data", [])
    items: list[dict] = []

    for row in data:
        if not isinstance(row, dict):
            continue
        art_id = row.get("id")
        attrs = row.get("attributes", {}) or {}
        rel = row.get("relationships", {}) or {}

        publish_on = attrs.get("publishOn")
        title = attrs.get("title", "").strip()

        # primary tickers come through as tag ids – we just keep them for debugging;
        # you already know which symbol you asked for.
        pt_data = ((rel.get("primaryTickers") or {}).get("data")) or []
        primary_ids = [
            t.get("id") for t in pt_data
            if isinstance(t, dict) and t.get("id")
        ]

        article_url = f"https://seekingalpha.com/article/{art_id}" if art_id else ""

        items.append(
            {
                "id": art_id,
                "title": title,
                "published": publish_on,
                "primary_tickers": primary_ids,
                "url": article_url,
            }
        )

    return items


def fetch_sa_analysis_body(article_id: str) -> str:
    """
    Call /analysis/v2/get-details for a single article and return the raw HTML body.
    """
    api_key = _get_sa_rapidapi_key()
    url = f"{SA_ANALYSIS_BASE}/analysis/v2/get-details"

    params = {"id": str(article_id)}
    headers = {
        "x-rapidapi-key": api_key,
        "x-rapidapi-host": "seeking-alpha.p.rapidapi.com",
    }

    resp = requests.get(url, headers=headers, params=params, timeout=20)
    resp.raise_for_status()
    payload = resp.json()

    data = payload.get("data") or []
    if not data:
        return ""

    first = data[0]
    attrs = first.get("attributes", {}) or {}

    # In practice this field is usually called "content"
    body_html = attrs.get("content") or ""
    return body_html

def clean_sa_html(html: str, max_len: Optional[int] = None) -> str:
    """
    Convert Seeking Alpha article HTML into clean, readable plain text.

    - Turns block tags (p/div/br/li/h1–h6) into paragraph breaks.
    - Strips all other tags.
    - Normalises whitespace but *keeps* paragraph breaks.
    - Inserts spaces between digits and letters to fix things like
      '13Bofcash' -> '13B of cash'.
    - If max_len is given, truncates on a word boundary.
    """
    import re
    import html as html_lib

    if not html:
        return ""

    # Decode HTML entities (&amp;, &nbsp;, etc.)
    text = html_lib.unescape(html)

    # 1) Turn common block / line-break tags into newlines
    text = re.sub(
        r"(?i)<\s*(br\s*/?|/p|p|/div|div|li|/li|h[1-6]|/h[1-6])[^>]*>",
        "\n",
        text,
    )

    # 2) Remove any remaining tags
    text = re.sub(r"<[^>]+>", " ", text)

    # 3) Normalise whitespace but keep newlines-as-paragraphs
    text = text.replace("\r", "")
    lines = []
    for ln in text.split("\n"):
        # collapse spaces/tabs on each line
        ln = re.sub(r"[ \t]+", " ", ln).strip()
        if ln:
            lines.append(ln)
    # rebuild paragraphs with a blank line between them
    text = "\n\n".join(lines)

    # 4) Fix digit/letter run-ons: 15.7Billion -> 15.7 Billion, EPS1.86 -> EPS 1.86
    text = re.sub(r"(\d)([A-Za-z])", r"\1 \2", text)
    text = re.sub(r"([A-Za-z])(\d)", r"\1 \2", text)

    # 5) Optional truncation for model input
    if max_len and len(text) > max_len:
        cut = text[:max_len]
        cut = cut.rsplit(" ", 1)[0]  # don’t cut in the middle of a word
        text = cut + " ..."

    return text

def _art_attr(art: Any, name: str, default: Any = "") -> Any:
    """Handle both dicts and AnalysisArticle objects."""
    if isinstance(art, dict):
        return art.get(name, default)
    return getattr(art, name, default)

def build_sa_analysis_digest(
    symbol: str,
    articles,
    model: str = "gpt-4o-mini",
    max_chars: int | None = None,
) -> str:
    """
    Build a short bullet-point digest from a list of AnalysisArticle objects.

    `articles` is the list returned by sa_analysis_api.fetch_analysis_list.
    We use only lightweight metadata (date, title, url).
    """
    if not articles:
        return f"No recent Seeking Alpha analysis articles found for {symbol}."

    # Build a compact text summary of the articles we have.
    lines = []
    for art in articles:
        try:
            date_str = art.published.split("T", 1)[0] if art.published else ""
            title = art.title or ""
            url = art.url or ""
            lines.append(f"- {date_str} — {title} ({url})")
        except Exception:
            continue

    context_block = "\n".join(lines)

    system_msg = (
        "You are helping a fundamental portfolio manager at a small buy-side shop.\n"
        "You will receive a list of recent Seeking Alpha ANALYSIS articles for one ticker.\n"
        "Write a concise bullet-point digest (4–7 bullets max) capturing:\n"
        "- Overall stance (bullish / bearish / mixed) across authors\n"
        "- Key fundamental drivers mentioned (earnings, pipeline, margins, cash flow, etc.)\n"
        "- Any repeated risks or points of disagreement\n"
        "- Any notable technical or sentiment comments\n"
        "Keep language plain, professional, and focused on what a PM should know."
    )

    user_msg = (
        f"TICKER: {symbol}\n\n"
        "Recent Seeking Alpha analysis articles:\n"
        f"{context_block}\n\n"
        "Now write the digest."
    )

    try:
        resp = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.3,
            max_tokens=600,
        )
        digest = resp["choices"][0]["message"]["content"].strip()
        return digest
    except Exception as e:
        return f"Error while calling OpenAI: {e}"

def clean_html_to_text(html: str) -> str:
    if not html:
        return ""
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", html)
    # Convert multiple spaces/newlines
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def clean_sa_html_to_markdown(raw_html: str) -> str:
    """
    Convert Seeking Alpha article HTML into clean, readable text for Streamlit.

    - Converts block tags to paragraph breaks.
    - Strips all other tags.
    - Normalises whitespace but keeps paragraph breaks.
    - Fixes digit/letter run-ons like '13Bofcash' -> '13B of cash'.
    """
    import re
    import html as html_lib

    if not raw_html:
        return ""

    # Decode entities (&amp;, &nbsp;, etc.)
    text = html_lib.unescape(raw_html)

    # Turn common block / break tags into newlines
    text = re.sub(
        r"(?i)<\s*(br\s*/?|/p|p|/div|div|/li|li|h[1-6]|/h[1-6])[^>]*>",
        "\n",
        text,
    )

    # Remove remaining tags, leaving a space so words don't glue together
    text = re.sub(r"<[^>]+>", " ", text)

    # Normalise whitespace but keep paragraph structure
    text = text.replace("\r", "")
    lines = []
    for ln in text.split("\n"):
        ln = re.sub(r"[ \t]+", " ", ln).strip()
        if ln:
            lines.append(ln)
    text = "\n\n".join(lines)

    # Fix digit/letter run-ons
    text = re.sub(r"(\d)([A-Za-z])", r"\1 \2", text)
    text = re.sub(r"([A-Za-z])(\d)", r"\1 \2", text)

    return text.strip()

def draw_seeking_alpha_news_section() -> None:
    """
    Seeking Alpha – Analysis digest by ticker.

    Uses:
      - sa_analysis_api.fetch_analysis_list
      - sa_analysis_api.fetch_analysis_details
      - sa_analysis_api.build_sa_analysis_digest (which calls OpenAI)
    """
    import pandas as pd
    import streamlit as st

    st.markdown("### Seeking Alpha – Analysis digest by ticker")

    # ---------- Ticker + controls ----------
    try:
        from tickers import tickers as CUTLER_TICKERS
        universe = sorted(list(CUTLER_TICKERS.keys()))
    except Exception:
        universe = []

    col1, col2 = st.columns([2, 1])
    with col1:
        ticker = st.selectbox(
            "Ticker",
            universe or ["TSLA", "AAPL", "MSFT"],
            index=0,
        )
    with col2:
        max_articles = st.slider(
            "Max analysis pieces to pull",
            min_value=3,
            max_value=10,
            value=5,
            help="How many recent Seeking Alpha *analysis* articles to use.",
        )

    model = st.selectbox(
        "OpenAI model for digest",
        ["gpt-4o-mini", "gpt-4o"],
        index=0,
    )

    if not st.button("Fetch Seeking Alpha analysis & build AI digest"):
        return

    if not ticker:
        st.warning("Please choose a ticker first.")
        return

    # ---------- 1) Fetch list of analysis articles ----------
    try:
        with st.spinner(f"Pulling Seeking Alpha analysis for {ticker} via RapidAPI..."):
            articles = sa_api.fetch_analysis_list(ticker, size=max_articles)
    except Exception as e:
        st.error(f"Error while fetching Seeking Alpha analysis: {e}")
        return

    if not articles:
        st.info(f"No Seeking Alpha analysis articles returned for {ticker}.")
        return

    # ---------- 2) Show table of articles ----------
    rows = []
    for art in articles:
        date_str = art.published.split("T", 1)[0] if getattr(art, "published", "") else ""
        rows.append(
            {
                "Date": date_str,
                "Title": getattr(art, "title", ""),
                "Source": "Seeking Alpha (Analysis)",
                "URL": getattr(art, "url", ""),
            }
        )

    df = pd.DataFrame(rows)
    st.markdown("#### Recent Seeking Alpha analysis articles")
    st.dataframe(df, use_container_width=True)

    # ---------- 3) AI digest (delegates to sa_analysis_api) ----------
    st.markdown("#### AI Analysis Digest")
    try:
        with st.spinner("Asking OpenAI for a short analysis digest..."):
            # IMPORTANT: use the implementation from sa_analysis_api.py
            digest_text = sa_api.build_sa_analysis_digest(
                symbol=ticker,
                articles=articles,
                model=model,
            )
        st.markdown(digest_text)
    except Exception as e:
        st.error(f"Error while calling OpenAI: {e}")

        # -----------------------------------------------------------
    # 4) Pull and display full article bodies (cleaned)
    # -----------------------------------------------------------
    st.markdown("#### Article bodies (full, cleaned)")

    # Helper: normalise any HTML-ish field to a single string
    def _normalize_html(part) -> str:
        if part is None:
            return ""
        if isinstance(part, list):
            return "\n".join(str(x) for x in part if x is not None)
        return str(part)

    # Only fetch bodies for a few articles to avoid hammering the API
    articles_for_bodies = articles[:5]

    for art in articles_for_bodies:
        art_id = art.id
        title = art.title or "Untitled article"

        with st.expander(title):
            st.caption("Full article (cleaned)")

            try:
                # This returns a *flat* dict:
                # {title, summary_html, body_html, images, url}
                details = sa_api.fetch_analysis_details(str(art_id))
            except Exception as e:
                st.write(f"Could not fetch article body: {e}")
                continue

            if not isinstance(details, dict) or not details:
                st.write("No article body text returned.")
                continue

            # In case you ever switch back to raw API JSON, support both shapes:
            if "data" in details:
                # raw RapidAPI payload
                data = details.get("data") or {}
                attrs = data.get("attributes") or {}
                summary_html = attrs.get("summary_html") or attrs.get("summary") or ""
                body_html = (
                    attrs.get("body_html")
                    or attrs.get("content")
                    or attrs.get("body")
                    or ""
                )
                images = attrs.get("images") or []
            else:
                # current helper output from sa_analysis_api.fetch_analysis_details
                summary_html = details.get("summary_html") or details.get("summary") or ""
                body_html = (
                    details.get("body_html")
                    or details.get("content")
                    or details.get("body")
                    or ""
                )
                images = details.get("images") or []

            # -------- Image (if present) --------
            image_url = None
            if isinstance(images, list):
                for img in images:
                    if not isinstance(img, dict):
                        continue
                    image_url = (
                        img.get("url")
                        or img.get("imageUrl")
                        or img.get("src")
                    )
                    if image_url:
                        break

            if not image_url:
                # Fallback if API ever sends a direct field
                image_url = details.get("gettyImageUrl") or details.get("imageUrl")

            if image_url:
                try:
                    st.image(image_url, use_column_width=True)
                except Exception:
                    # If Streamlit can't load it, just skip the image
                    pass

            # -------- Clean and render text --------
            combined_html = (
                _normalize_html(summary_html)
                + "\n\n"
                + _normalize_html(body_html)
            )

            if not combined_html.strip():
                st.write("No article body text returned.")
                continue

            try:
                cleaned_text = clean_sa_html_to_markdown(combined_html)
            except NameError:
                # Very simple fallback if helper is missing
                import re as _re
                tmp = _re.sub(r"<(br|p|div|li)[^>]*>", "\n", combined_html, flags=_re.I)
                tmp = _re.sub(r"<[^>]+>", "", tmp)
                tmp = tmp.replace("\xa0", " ")
                cleaned_text = _re.sub(r"\n{3,}", "\n\n", tmp).strip()

            st.write(cleaned_text)

# ---------- Manifest + Delta helpers ----------

def _write_manifest(
    batch: str,
    quarter: str,
    compiled: Optional[Path],
    items: List[Dict[str, Any]],
    table_rows: Optional[List[Dict[str, Any]]] = None,
) -> Optional[Path]:
    """
    Store a small JSON manifest for a compiled (or incremental) run so the
    Document Checker and incremental updater can compare runs later.
    """
    try:
        qdir = MAN_DIR / quarter
        qdir.mkdir(parents=True, exist_ok=True)
        now = datetime.now()
        payload: Dict[str, Any] = {
            "batch": batch,
            "quarter": quarter,
            "compiled_pdf": str(compiled) if compiled else "",
            "created_at": now.isoformat(timespec="seconds"),
            "items": items,
        }
        if table_rows is not None:
            payload["table_rows"] = table_rows

        fname = qdir / f"manifest_{batch.replace(' ', '')}_{now:%Y%m%d_%H%M%S}.json"
        with open(fname, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        return fname
    except Exception:
        traceback.print_exc()
        return None

def _load_manifests(batch: str, quarter: str) -> List[Dict[str, Any]]:
    """
    Load all manifests for a given batch + quarter, newest first.
    """
    qdir = MAN_DIR / quarter
    if not qdir.exists():
        return []
    out: List[Dict[str, Any]] = []
    for p in qdir.glob("manifest_*.json"):
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            if data.get("batch") != batch or data.get("quarter") != quarter:
                continue
            data["_path"] = str(p)
            out.append(data)
        except Exception:
            continue
    out.sort(key=lambda m: m.get("created_at", ""), reverse=True)
    return out

def _normalize_para_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())

def _collect_keys_from_manifest(manifest: Dict[str, Any]) -> set:
    """
    Build a set of (fund, source_pdf, ticker, text_hash) keys for all narrative
    paragraphs in a manifest. Used to detect whether a paragraph is "new".
    """
    keys = set()
    for meta in manifest.get("items", []):
        ej = meta.get("excerpts_json")
        if not ej:
            continue
        p = Path(ej)
        if not p.exists():
            continue
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        for ticker, lst in data.items():
            if not isinstance(lst, list):
                continue
            for item in lst:
                txt_norm = _normalize_para_text(item.get("text", ""))
                if not txt_norm:
                    continue
                h = hashlib.sha1(txt_norm.encode("utf-8")).hexdigest()
                key = (
                    meta.get("fund_family", ""),
                    meta.get("source_pdf_name", ""),
                    str(ticker),
                    h,
                )
                keys.add(key)
    return keys

def build_delta_pdf(old_manifest: Dict[str, Any], new_manifest: Dict[str, Any]) -> Optional[Path]:
    """
    Compare two manifests (older vs newer) and build a PDF containing ONLY
    paragraphs that are new in the newer manifest.

    Returns the PDF path, or None if no new paragraphs were found.
    """
    old_keys = _collect_keys_from_manifest(old_manifest)
    aggregated: Dict[str, List[Dict[str, Any]]] = {}

    for meta in new_manifest.get("items", []):
        ej = meta.get("excerpts_json")
        if not ej:
            continue
        p = Path(ej)
        if not p.exists():
            continue
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue
        if not isinstance(data, dict):
            continue

        for ticker, lst in data.items():
            if not isinstance(lst, list):
                continue
            for item in lst:
                txt = item.get("text", "")
                norm = _normalize_para_text(txt)
                if not norm:
                    continue
                h = hashlib.sha1(norm.encode("utf-8")).hexdigest()
                key = (
                    meta.get("fund_family", ""),
                    meta.get("source_pdf_name", ""),
                    str(ticker),
                    h,
                )
                if key in old_keys:
                    continue  # already existed in the older run

                pages = item.get("pages") or []
                decorated = f"[{meta.get('fund_family', 'Unknown')} – {meta.get('source_pdf_name', '')}] {txt}"
                aggregated.setdefault(str(ticker), []).append(
                    {"text": decorated, "pages": pages}
                )

    if not aggregated:
        return None

    DELTA_DIR.mkdir(parents=True, exist_ok=True)
    batch = new_manifest.get("batch", "Batch")
    quarter = new_manifest.get("quarter", "Quarter")
    old_id = (old_manifest.get("created_at", "old")
              .replace(":", "").replace("-", "").replace("T", "_"))
    new_id = (new_manifest.get("created_at", "new")
              .replace(":", "").replace("-", "").replace("T", "_"))

    json_path = DELTA_DIR / f"delta_{batch.replace(' ', '')}_{quarter}_{old_id}_to_{new_id}.json"
    pdf_path = DELTA_DIR / f"Delta_Cutler_{batch.replace(' ', '')}_{quarter}_{old_id}_to_{new_id}.pdf"

    try:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(aggregated, f, indent=2, ensure_ascii=False)

        make_pdf.build_pdf(
            excerpts_json_path=str(json_path),
            output_pdf_path=str(pdf_path),
            report_title="New ticker commentary vs prior run",
            source_pdf_name=f"{batch.replace(' ', '')}_{quarter}_Delta",
            format_style="legacy",
            letter_date=None,
        )
    except Exception:
        traceback.print_exc()
        return None

    return pdf_path if pdf_path.exists() else None

# ---------- Quarter helpers ----------

@st.cache_data(show_spinner=False)
def get_available_quarters() -> List[str]:
    """
    Read the available quarters from the site's <select> element.
    Skips 'all' and 'latest_two'. Returns values like:
      ['2025 Q3', '2025 Q2', '2025 Q1', '2024 Q4', ...]
    Cached so we don't hit the site on every rerun.
    """
    vals: List[str] = []
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            ctx = browser.new_context()
            page = ctx.new_page()
            page.set_default_timeout(30000)
            page.goto(BSD_URL)

            sel = page.locator(FILTERS["quarter"]).first
            options = sel.locator("option").all()

            for opt in options:
                val = (opt.get_attribute("value") or "").strip()
                if not val:
                    continue
                if val in ("all", "latest_two"):
                    continue
                vals.append(val)

            browser.close()
    except Exception as e:
        print("WARN: Failed to auto-detect quarters; using fallback list.", e)

    # De-duplicate while preserving order
    seen = set()
    out: List[str] = []
    for v in vals:
        if v not in seen:
            seen.add(v)
            out.append(v)

    if not out:
        # conservative fallback if the site scrape fails
        out = [
            "2025 Q3",
            "2025 Q2",
            "2025 Q1",
            "2024 Q4",
            "2024 Q3",
            "2024 Q2",
            "2024 Q1",
        ]
    return out

def _parse_quarter_label(label: str) -> Optional[Tuple[int, int]]:
    """
    Parse 'YYYY QN' into (YYYY, N). Returns None if it doesn't match.
    """
    m = re.match(r"^(\d{4})\s+Q([1-4])$", label.strip())
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))

def _last_completed_us_quarter(today: Optional[datetime] = None) -> Tuple[int, int]:
    """
    Given today's date, return (year, quarter) for the **last completed US quarter**.
    US quarters:
      Q1: Jan–Mar
      Q2: Apr–Jun
      Q3: Jul–Sep
      Q4: Oct–Dec
    """
    if today is None:
        today = datetime.now()
    y = today.year
    m = today.month

    if 1 <= m <= 3:
        return y - 1, 4
    elif 4 <= m <= 6:
        return y, 1
    elif 7 <= m <= 9:
        return y, 2
    else:
        return y, 3

def choose_default_quarter(available: List[str]) -> Optional[str]:
    """
    Given the list of available quarters from the site, choose the default
    as the **last completed US quarter**, if present. If not present,
    choose the most recent available.
    """
    if not available:
        return None

    parsed: List[Tuple[str, int, int]] = []
    for lab in available:
        pq = _parse_quarter_label(lab)
        if pq:
            parsed.append((lab, pq[0], pq[1]))

    if not parsed:
        return available[0]

    # Sort by (year DESC, quarter DESC) so index 0 is newest
    parsed.sort(key=lambda x: (x[1], x[2]), reverse=True)

    target_year, target_q = _last_completed_us_quarter()

    for lab, year, q in parsed:
        if year < target_year or (year == target_year and q <= target_q):
            return lab

    return parsed[0][0]

# ---------- run one batch (full run, with manifest + table rows) ----------

def run_batch(batch_name: str, quarters: List[str], use_first_word: bool, subset: Optional[List[str]] = None):
    st.markdown(f"### Running {batch_name}")
    brands = RUNNABLE_BATCHES.get(batch_name, [])
    if subset:
        brands = [b for b in brands if b in subset]
    if not brands:
        st.info("No runnable fund families in this batch (after filter).")
        return
    tokens = [(b, _first_word(b) if use_first_word else b) for b in brands]

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        ctx = browser.new_context(accept_downloads=True)
        page = ctx.new_page()
        page.set_default_timeout(30000)
        page.goto(BSD_URL)

        for q in quarters:
            st.write(f"Searching quarter {q} across {len(tokens)} fund families…")

            if not _set_quarter(page, q):
                st.warning(
                    f"Quarter **{q}** is not available on the data source at the moment. "
                    "It may not have any letters yet."
                )
                continue

            outs: List[Path] = []
            manifest_items: List[Dict[str, Any]] = []
            table_rows: List[Dict[str, Any]] = []  # snapshot of table rows

            for i, (brand, token) in enumerate(tokens, 1):
                st.write(f"[{q}] {i}/{len(tokens)} — {brand} (search: {token})")
                try:
                    _search_by_fund(page, token)
                    hits = _parse_rows(page, q)
                    if not hits:
                        continue
                    seen = set()
                    for h in hits:
                        # record table row
                        table_rows.append(
                            {
                                "fund_family": brand,
                                "search_token": token,
                                "quarter": h.quarter,
                                "letter_date": h.letter_date,
                                "fund_name": h.fund_name,
                                "fund_href": h.fund_href,
                            }
                        )

                        if h.fund_href in seen:
                            continue
                        seen.add(h.fund_href)
                        page.goto(h.fund_href)
                        page.wait_for_load_state("domcontentloaded")

                        dest = DL_DIR / q / _safe(brand)
                        pdfs = _download_quarter_pdf_from_fund(page, q, dest)
                        for pdf in pdfs:
                            out_dir = EX_DIR / q / _safe(brand) / _safe(pdf.stem)
                            built = run_excerpt_and_build(
                                pdf,
                                out_dir,
                                source_pdf_name=pdf.name,
                                letter_date=h.letter_date or None,
                            )

                            manifest_items.append(
                                {
                                    "fund_family": brand,
                                    "search_token": token,
                                    "letter_date": h.letter_date or "",
                                    "downloaded_pdf": str(pdf),
                                    "source_pdf_name": pdf.name,
                                    "excerpt_dir": str(out_dir),
                                    "excerpts_json": str(out_dir / "excerpts_clean.json"),
                                    "excerpt_pdf": str(built) if built else "",
                                    "fund_name": h.fund_name,
                                    "fund_href": h.fund_href,
                                }
                            )

                            if built:
                                outs.append(built)

                        page.go_back()
                        page.wait_for_load_state("domcontentloaded")
                except Exception as e:
                    st.error(f"Error on fund family {brand}: {e}")
                    continue

            compiled = compile_merged(batch_name, q, outs)
            if compiled:
                st.success(f"Compiled PDF for {q}: {compiled}")
            else:
                st.info(
                    f"No excerpt PDFs produced for **{q}**. "
                    "The selected fund families may not yet have letters or ticker mentions for this quarter."
                )

            # write manifest regardless (so we capture table_rows snapshot)
            _write_manifest(batch_name, q, compiled, manifest_items, table_rows=table_rows)

        browser.close()

# ---------- NEW: incremental per-batch updater ----------

def _row_key(row: Dict[str, Any]) -> Tuple[str, str, str, str, str]:
    """
    Build a stable key for a table row so we can compare snapshots.
    """
    return (
        row.get("fund_family", ""),
        row.get("fund_name", ""),
        row.get("quarter", ""),
        row.get("letter_date", ""),
        row.get("fund_href", ""),
    )

def run_incremental_update(batch_name: str, quarter: str, use_first_word: bool):
    """
    Fast per-batch incremental mode:
      - Reads the latest manifest for (batch, quarter) to get previous table_rows.
      - Scans the BSD table now (no downloads yet).
      - If table_rows unchanged => nothing to do.
      - If some rows are new/changed => downloads and processes only those.
    """
    st.markdown(f"### Incremental update – {batch_name} / {quarter}")

    manifests = _load_manifests(batch_name, quarter)
    if not manifests:
        st.info(
            "No manifest history found yet for this batch and quarter. "
            "Run a full batch once under 'Run scope' before using incremental mode."
        )
        return

    latest = manifests[0]
    prev_rows = latest.get("table_rows")
    if not prev_rows:
        st.info(
            "Latest manifest for this batch and quarter does not contain table-level "
            "snapshot data (likely created before the incremental feature). "
            "Run a full batch once so the next manifest includes table_rows, "
            "then use incremental mode."
        )
        return

    prev_key_set = { _row_key(r) for r in prev_rows }

    brands = RUNNABLE_BATCHES.get(batch_name, [])
    if not brands:
        st.info("No runnable fund families in this batch.")
        return

    tokens = [(b, _first_word(b) if use_first_word else b) for b in brands]

    current_rows: List[Dict[str, Any]] = []
    row_by_key: Dict[Tuple[str, str, str, str, str], Dict[str, Any]] = {}

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        ctx = browser.new_context(accept_downloads=True)
        page = ctx.new_page()
        page.set_default_timeout(30000)
        page.goto(BSD_URL)

        st.write(f"Scanning BSD table for {batch_name} / {quarter} (no downloads yet)…")

        if not _set_quarter(page, quarter):
            st.warning(
                f"Quarter **{quarter}** is not available on the data source at the moment. "
                "It may not have any letters yet."
            )
            browser.close()
            return

        # scan table rows for all brands
        for i, (brand, token) in enumerate(tokens, 1):
            st.write(f"[{quarter}] {i}/{len(tokens)} — {brand} (search: {token})")
            try:
                _search_by_fund(page, token)
                hits = _parse_rows(page, quarter)
                for h in hits:
                    row = {
                        "fund_family": brand,
                        "search_token": token,
                        "quarter": h.quarter,
                        "letter_date": h.letter_date,
                        "fund_name": h.fund_name,
                        "fund_href": h.fund_href,
                    }
                    key = _row_key(row)
                    current_rows.append(row)
                    row_by_key[key] = row
            except Exception as e:
                st.error(f"Error scanning fund family {brand}: {e}")
                continue

        # Compare snapshots
        current_key_set = set(row_by_key.keys())
        new_keys = current_key_set - prev_key_set

        if not new_keys:
            st.success(
                "No new or changed letters detected for this batch and quarter "
                "compared to the latest manifest. Nothing to download today."
            )
            # Still write a snapshot-only manifest so next comparison is up to date
            _write_manifest(batch_name, quarter, compiled=None, items=[], table_rows=current_rows)
            browser.close()
            return

        st.write(f"Found {len(new_keys)} new or changed table rows. Downloading only those…")

        outs: List[Path] = []
        manifest_items: List[Dict[str, Any]] = []

        processed_hrefs: set = set()

        for key in new_keys:
            row = row_by_key[key]
            href = row.get("fund_href") or ""
            brand = row.get("fund_family") or ""
            token = row.get("search_token") or ""
            letter_date = row.get("letter_date") or ""
            fund_name = row.get("fund_name") or ""

            if not href or not brand:
                continue
            if href in processed_hrefs:
                continue
            processed_hrefs.add(href)

            try:
                st.write(f"Downloading new/updated letter for {brand} – {fund_name} ({letter_date})")
                page.goto(href)
                page.wait_for_load_state("domcontentloaded")

                dest = DL_DIR / quarter / _safe(brand)
                pdfs = _download_quarter_pdf_from_fund(page, quarter, dest)
                for pdf in pdfs:
                    out_dir = EX_DIR / quarter / _safe(brand) / _safe(pdf.stem)
                    built = run_excerpt_and_build(
                        pdf,
                        out_dir,
                        source_pdf_name=pdf.name,
                        letter_date=letter_date or None,
                    )

                    manifest_items.append(
                        {
                            "fund_family": brand,
                            "search_token": token,
                            "letter_date": letter_date,
                            "downloaded_pdf": str(pdf),
                            "source_pdf_name": pdf.name,
                            "excerpt_dir": str(out_dir),
                            "excerpts_json": str(out_dir / "excerpts_clean.json"),
                            "excerpt_pdf": str(built) if built else "",
                            "fund_name": fund_name,
                            "fund_href": href,
                        }
                    )

                    if built:
                        outs.append(built)
            except Exception as e:
                st.error(f"Error downloading for {brand}: {e}")
                continue

        compiled = None
        if outs:
            compiled = compile_merged(batch_name, quarter, outs)
            if compiled:
                st.success(f"Incremental compiled PDF created: {compiled}")
            else:
                st.info(
                    "New letters were found, but no excerpt PDFs were produced. "
                    "They may not contain any tracked tickers."
                )
        else:
            st.info(
                "New letters were detected in the table, but their PDFs could not be "
                "downloaded or yielded no excerpts."
            )

        # Write manifest capturing current snapshot + any new items we processed
        _write_manifest(batch_name, quarter, compiled, manifest_items, table_rows=current_rows)

        browser.close()

# ---------- UI ----------

def main():
    st.set_page_config(page_title="Cutler Capital Scraper", layout="wide")

    # Global styling: Cutler purple theme and modernized controls
    st.markdown(
        """
        <style>
        /* Center all images (logo) */
        .stImage img {
            display: block;
            margin-left: calc(100% - 20px);  /* pushes it ~20px to the right */
            transform: translateX(-50%);
        }
        /* Overall background and font tweaks */
        .stApp {
            background: radial-gradient(circle at top left, #f5f0fb 0, #ffffff 40%, #f7f3fb 100%);
        }
        .block-container {
            padding-top: 4rem;
            max-width: 1100px;
        }
        .app-title {
            text-align: center;
            color: #4b2142;
            font-size: 1.9rem;
            font-weight: 700;
            margin-bottom: 0.3rem;
        }
        .app-subtitle {
            text-align: center;
            color: #6b4f7a;
            font-size: 0.95rem;
            margin-top: 0.1rem;
            margin-bottom: 1.4rem;
        }

        /* Sidebar */
        [data-testid="stSidebar"] > div {
            background: #fbf8ff;
        }
        [data-testid="stSidebar"] h2, [data-testid="stSidebar"] label {
            color: #4b2142;
        }
        
        header[data-testid="stHeader"] {
            background: radial-gradient(circle at top left, #f5f0fb 0, #ffffff 40%, #f7f3fb 100%) !important;
            box-shadow: none !important;
            border-bottom: none !important;
        }
        [data-testid="stToolbar"] {
            background: transparent !important;
        }
        header[data-testid="stHeader"] * {
            color: #4b2142 !important;
        }

        /* Buttons: long, pill-shaped, purple */
        .stButton>button {
            width: 100%;
            border-radius: 999px;
            background: #4b2142;
            color: #ffffff;
            border: 1px solid #4b2142;
            padding: 0.6rem 1.5rem;
            font-weight: 600;
            font-size: 0.95rem;
        }
        .stButton>button:hover {
            background: #612a58;
            border-color: #612a58;
        }

        /* Radio group as pill toggle */
        div[role="radiogroup"] {
            display: flex;
            flex-wrap: nowrap;
            gap: 0.4rem;
        }
        div[role="radiogroup"] > label {
            flex: 1 1 0;
            justify-content: center;
            border-radius: 999px !important;
            padding: 0.35rem 0.95rem !important;
            border: 1px solid #d7c4f3 !important;
            background: #f7f3fb !important;
            color: #4b2142 !important;
            font-weight: 500 !important;
            white-space: nowrap;
        }
        div[role="radiogroup"] > label:hover {
            border-color: #4b2142 !important;
        }
        div[role="radiogroup"] > label[data-baseweb="radio"] > div:first-child > div[aria-checked="true"] + div {
            background: #4b2142 !important;
        }

        /* Card-style containers */
        .cc-card {
            background: #ffffffdd;
            border-radius: 20px;
            padding: 1.3rem 1.4rem;
            border: 1px solid rgba(75,33,66,0.08);
            box-shadow: 0 10px 30px rgba(75,33,66,0.04);
            margin-bottom: 1.1rem;
        }

        /* Fund chips */
        .fund-chip{
            display:inline-block;
            margin:6px 6px 0 0;
            padding:6px 12px;
            border-radius:14px;
            background:#f5effc;
            color:#4b2142;
            font-size:12px;
            font-weight:600;
            border:1px solid rgba(75,33,66,0.35);
            white-space:nowrap;
        }

        /* Gauge (needle) */
        .gauge-wrapper {
            margin-top: 0.5rem;
            margin-bottom: 0.75rem;
        }
        .gauge {
            width: 220px;
            height: 120px;
            margin: 0.2rem auto 0.1rem;
            position: relative;
        }
        .gauge-body {
            width: 100%;
            height: 100%;
            border-radius: 220px 220px 0 0;
            background: #f5effc;
            border: 1px solid rgba(75,33,66,0.25);
            position: relative;
            overflow: hidden;
        }
        .gauge-needle {
            position: absolute;
            width: 2px;
            height: 85%;
            top: 15%;
            left: 50%;
            background: #4b2142;
            transform-origin: bottom center;
            transition: transform 0.25s ease-out;
        }
        .gauge-cover {
            width: 68%;
            height: 68%;
            background: #ffffff;
            border-radius: 50%;
            position: absolute;
            bottom: -10%;
            left: 50%;
            transform: translateX(-50%);
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 700;
            font-size: 0.9rem;
            color: #4b2142;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Header: centered logo and text
        # Header: logo + title in a centered column
    logo_path = HERE / "cutler.png"
    col_left, col_center, col_right = st.columns([1, 2, 1])

    with col_center:
        if logo_path.exists():
            st.image(str(logo_path), width=260)

        st.markdown("<div class='app-title'>Cutler Capital Letter Scraper</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='app-subtitle'>Scrape, excerpt, and compile fund letters by fund family and quarter.</div>",
            unsafe_allow_html=True,
        )


    # Sidebar: run settings
    st.sidebar.header("Run settings")

    quarter_options = get_available_quarters()
    default_q = choose_default_quarter(quarter_options)

    quarters = st.sidebar.multiselect(
        "Quarters",
        quarter_options,
        default=[default_q] if default_q else quarter_options[:1],
    )

    use_first_word = st.sidebar.checkbox(
        "Use first word for search (recommended)",
        value=True,
    )

    batch_names = list(RUNNABLE_BATCHES.keys())

    # Main controls in a card – full run
    with st.container():
        st.markdown("<div class='cc-card'>", unsafe_allow_html=True)
        st.markdown("#### Run scope", unsafe_allow_html=True)
        st.write("Choose whether you want a full run across all batches or a targeted test.")

        run_mode = st.radio(
            "Run mode",
            ["Run all 7 batches", "Run a specific batch"],
            index=1,
        )

        if run_mode == "Run all 7 batches":
            st.info(
                "Runs every fund family in all 7 batches for the selected quarter(s). "
                "Use the specific batch mode below if you are just testing a few names."
            )
            if st.button("Run all 7 batches", use_container_width=True):
                for bn in batch_names:
                    run_batch(bn, quarters, use_first_word, subset=None)
                st.markdown("</div>", unsafe_allow_html=True)
                st.stop()
        else:
            selected_batch = st.selectbox("Choose a batch to run", batch_names)
            if selected_batch:
                names_in_batch = RUNNABLE_BATCHES[selected_batch]
                st.write(f"{selected_batch} contains **{len(names_in_batch)}** fund families.")

                with st.expander("Preview fund families in this batch"):
                    chips = "".join(
                        f"<span class='fund-chip'>{name}</span>"
                        for name in names_in_batch
                    )
                    st.markdown(chips, unsafe_allow_html=True)

                selected_funds = st.multiselect(
                    "Optionally target specific fund families "
                    "(leave empty to run the entire batch):",
                    options=names_in_batch,
                )

                if st.button(f"Run {selected_batch}", use_container_width=True):
                    subset = selected_funds or None
                    run_batch(selected_batch, quarters, use_first_word, subset=subset)

        st.markdown("</div>", unsafe_allow_html=True)

    # Incremental per-batch updater
    st.markdown("<div class='cc-card'>", unsafe_allow_html=True)
    st.markdown("#### Incremental update (per batch)", unsafe_allow_html=True)
    st.write(
        "Use this when interns run the tool daily. It compares the current BSD table "
        "to the latest stored manifest for a batch and quarter, and only downloads "
        "letters that are new or changed."
    )

    inc_quarter = st.selectbox(
        "Quarter for incremental check",
        options=quarter_options,
        index=quarter_options.index(default_q) if default_q in quarter_options else 0,
        key="inc_quarter",
    )
    inc_batch = st.selectbox(
        "Batch for incremental update",
        options=batch_names,
        index=0,
        key="inc_batch",
    )

    if st.button("Check for updates and download new letters", key="inc_btn"):
        run_incremental_update(inc_batch, inc_quarter, use_first_word)

    st.markdown("</div>", unsafe_allow_html=True)

    # Document Checker
    st.markdown("<div class='cc-card'>", unsafe_allow_html=True)
    st.markdown("#### Document Checker", unsafe_allow_html=True)
    st.write(
        "Compare two compiled runs for a given batch and quarter, and generate a PDF "
        "containing only **new** ticker-related paragraphs found in the newer run."
    )

    checker_quarter = st.selectbox(
        "Quarter to inspect",
        options=quarter_options,
        index=quarter_options.index(default_q) if default_q in quarter_options else 0,
        key="checker_quarter",
    )
    checker_batch = st.selectbox(
        "Batch",
        options=batch_names,
        index=0,
        key="checker_batch",
    )

    manifests = _load_manifests(checker_batch, checker_quarter)
    if not manifests:
        st.info(
            "No history found yet for this batch and quarter. "
            "Run the scraper at least twice to compare documents."
        )
    elif len(manifests) == 1:
        only = manifests[0]
        st.info(
            "Only one compiled run is stored so far for this batch and quarter "
            f"(created {only.get('created_at', '')}). Run the scraper again to "
            "create a second run for comparison."
        )
    else:
        labels = [
            f"{i+1}. {m.get('created_at', '')} – {Path(m.get('compiled_pdf', '')).name or '[no compiled PDF]'}"
            for i, m in enumerate(manifests)
        ]
        idx_new = 0
        idx_old = 1 if len(manifests) > 1 else 0

        new_idx = st.selectbox(
            "Newer run",
            options=list(range(len(manifests))),
            format_func=lambda i: labels[i],
            index=idx_new,
            key="checker_new",
        )
        old_idx = st.selectbox(
            "Older run to compare against",
            options=list(range(len(manifests))),
            format_func=lambda i: labels[i],
            index=idx_old,
            key="checker_old",
        )

        if new_idx == old_idx:
            st.warning("Please select two different runs to compare.")
        else:
            if st.button("Generate 'New Since' PDF", key="checker_btn"):
                delta_pdf = build_delta_pdf(
                    old_manifest=manifests[old_idx],
                    new_manifest=manifests[new_idx],
                )
                if delta_pdf:
                    st.success(f"Delta PDF created: {delta_pdf}")
                    try:
                        with open(delta_pdf, "rb") as f:
                            st.download_button(
                                "Download delta PDF",
                                data=f,
                                file_name=delta_pdf.name,
                                mime="application/pdf",
                                key="checker_download",
                            )
                    except Exception:
                        pass
                else:
                    st.info(
                        "No new ticker-related commentary found between these two runs. "
                        "Everything appears to be the same."
                    )

    st.markdown("</div>", unsafe_allow_html=True)

        # ---------- AI Insights: Buy / Hold / Sell ----------
    st.markdown("<div class='cc-card'>", unsafe_allow_html=True)
    st.markdown("#### AI Insights – Buy / Hold / Sell by company", unsafe_allow_html=True)
    st.write(
        "Use OpenAI to classify each ticker in a compiled run as **buy**, **hold**, "
        "**sell**, or **unclear**, with reasoning grounded in the excerpted letters."
    )

    ai_quarter = st.selectbox(
        "Quarter for AI analysis",
        options=quarter_options,
        index=quarter_options.index(default_q) if default_q in quarter_options else 0,
        key="ai_quarter",
    )
    ai_batch = st.selectbox(
        "Batch for AI analysis",
        options=batch_names,
        index=0,
        key="ai_batch",
    )

    ai_manifests = _load_manifests(ai_batch, ai_quarter)
    if not ai_manifests:
        st.info(
            "No manifests found yet for this batch and quarter. "
            "Run this batch at least once (full or incremental) before using AI insights."
        )
    else:
        labels = [
            f"{i+1}. {m.get('created_at', '')} – "
            f"{Path(m.get('compiled_pdf', '')).name or '[no compiled PDF]'}"
            for i, m in enumerate(ai_manifests)
        ]
        ai_manifest_idx = st.selectbox(
            "Which run should the AI analyse?",
            options=list(range(len(ai_manifests))),
            format_func=lambda i: labels[i],
            index=0,
            key="ai_manifest_idx",
        )

        ai_model = st.text_input(
            "OpenAI model name",
            value="gpt-4o-mini",
            help="Any chat-compatible model, e.g. gpt-4o or gpt-4o-mini.",
        )
        ai_use_web = st.checkbox(
            "Allow OpenAI to use web search",
            value=True,
            help="For now this mainly controls how much external context the model "
                 "is encouraged to bring into the `web_check` field.",
        )

        results: List[Dict[str, Any]] = []
        if st.button("Run AI analysis for this run", key="ai_run_btn"):
            manifest = ai_manifests[ai_manifest_idx]
            with st.spinner("Calling OpenAI for ticker-level stances…"):
                try:
                    results = ai_insights.generate_ticker_stances(
                        manifest=manifest,
                        batch=ai_batch,
                        quarter=ai_quarter,
                        model=ai_model,
                        use_web=ai_use_web,
                    )
                except Exception as e:
                    st.error(f"AI analysis failed: {e}")
                    results = []

            st.session_state["ai_results"] = results

        # If we already have results in session, reuse them so we can interact with dropdown
        if "ai_results" in st.session_state and not results:
            results = st.session_state["ai_results"]

        if results:
            # Compact summary table
            summary_rows = []
            for r in results:
                summary_rows.append(
                    {
                        "Ticker": r.get("ticker"),
                        "Company": ", ".join(r.get("company_names") or []),
                        "Stance": r.get("stance"),
                        "Confidence": round(float(r.get("confidence", 0.0)), 2),
                    }
                )
            st.write("**Summary by ticker**")
            st.dataframe(summary_rows, use_container_width=True)

            # Detailed view: dropdown + gauge + reasoning
            ticker_options = [row["Ticker"] for row in summary_rows]
            if ticker_options:
                focus_ticker = st.selectbox(
                    "Detailed view – choose a ticker",
                    options=ticker_options,
                    key="ai_focus_ticker",
                )
                detail = next((r for r in results if r.get("ticker") == focus_ticker), None)

                if detail:
                    stance = (detail.get("stance") or "").lower()
                    conf = float(detail.get("confidence") or 0.0)

                    # Map stance + confidence to a 0–1 position for the gauge
                    if stance == "buy":
                        pos = 0.5 + 0.5 * conf
                    elif stance == "sell":
                        pos = 0.5 - 0.5 * conf
                    elif stance == "hold":
                        pos = 0.5
                    else:  # unclear
                        pos = 0.5
                    pos = max(0.0, min(1.0, pos))
                    angle = -90 + 180 * pos  # -90 (sell) .. 0 (hold) .. +90 (buy)

                    gauge_html = f"""
                    <div class="gauge-wrapper">
                      <div class="gauge">
                        <div class="gauge-body">
                          <div class="gauge-needle" style="transform: rotate({angle:.1f}deg);"></div>
                          <div class="gauge-cover">{stance.upper() if stance else "UNCLEAR"}</div>
                        </div>
                      </div>
                      <div style="display:flex;justify-content:space-between;font-size:0.75rem;color:#6b4f7a;margin-top:0.15rem;">
                        <span>Sell</span><span>Hold</span><span>Buy</span>
                      </div>
                    </div>
                    """
                    st.markdown(gauge_html, unsafe_allow_html=True)

                    company_label = ", ".join(detail.get("company_names") or [])
                    st.markdown(f"**Reasoning for {focus_ticker} ({company_label})**")
                    st.write(detail.get("primary_reasoning", ""))

                    st.markdown("**Evidence from commentaries**")
                    for ev in detail.get("commentary_evidence") or []:
                        st.markdown(f"- {ev}")

                    st.markdown("**Web check**")
                    st.write(detail.get("web_check_summary") or "No additional web context used.")

                    funds = detail.get("fund_families") or []
                    if funds:
                        chips = "".join(
                            f"<span class='fund-chip'>{f}</span>" for f in funds
                        )
                        st.markdown("**Fund sources used in this decision:**", unsafe_allow_html=True)
                        st.markdown(chips, unsafe_allow_html=True)
        else:
            st.info(
                "Run the AI analysis above to see ticker stances, then select a ticker "
                "for a detailed gauge view."
            )

    st.markdown("</div>", unsafe_allow_html=True)

    # ---------- Seeking Alpha news + AI digest ----------
    draw_seeking_alpha_news_section()


    # Output path
    st.markdown("<div class='cc-card'>", unsafe_allow_html=True)
    st.write("**Output root folder (on this machine):**")
    st.code(str(BASE))
    st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
