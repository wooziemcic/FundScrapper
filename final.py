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
import sys
import shutil
import traceback
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import streamlit as st
import requests

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
from playwright.sync_api import sync_playwright

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

# paths (still stored under BSD/ on disk; UI is Cutler-branded only)
BASE = HERE / "BSD"
DL_DIR = BASE / "Downloads"
EX_DIR = BASE / "Excerpts"
CP_DIR = BASE / "Compiled"
for d in (DL_DIR, EX_DIR, CP_DIR): d.mkdir(parents=True, exist_ok=True)

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
Global X
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


def _search_by_fund(page, keyword: str) -> None:
    inp = page.locator(FILTERS["fund"]).first
    inp.fill(""); inp.type(keyword, delay=10)
    page.locator(FILTERS["search_btn"]).first.click(force=True)
    try: page.wait_for_load_state("networkidle", timeout=8000)
    except Exception: page.locator(TABLE_ROW).first.wait_for(state="visible", timeout=8000)

def _parse_rows(page, quarter: str) -> List[Hit]:
    rows = page.locator(TABLE_ROW)
    hits: List[Hit] = []
    for i in range(rows.count()):
        row = rows.nth(i)
        try:
            q = row.locator("td").nth(COLMAP["quarter"]-1).inner_text().strip()
            if q != quarter: continue
            letter_date = row.locator("td").nth(COLMAP["letter_date"]-1).inner_text().strip()
            fund_cell = row.locator("td").nth(COLMAP["fund_name"]-1)
            link = fund_cell.locator("a").first
            fund_name = (link.inner_text() or '').strip()
            fund_href = link.get_attribute("href") or ""
            if fund_href: hits.append(Hit(q, letter_date, fund_name, fund_href))
        except Exception: continue
    return hits

def _download_quarter_pdf_from_fund(page, quarter: str, dest_dir: Path) -> List[Path]:
    dest_dir.mkdir(parents=True, exist_ok=True)
    pdfs: List[Path] = []
    try: page.locator("text=Quarterly Letters").first.wait_for(state="visible", timeout=8000)
    except Exception: pass
    anchors = page.locator("a").all(); candidates = []
    for a in anchors:
        try:
            text = (a.inner_text() or '').strip()
            title = a.get_attribute("title") or ""
            href = a.get_attribute("href") or ""
            if not href: continue
            if (text == quarter or quarter in title) and ("letters/file" in href or href.lower().endswith('.pdf')):
                candidates.append((a, href))
        except Exception: continue
    for a, href in candidates:
        try:
            with page.expect_download(timeout=8000) as dl_info: a.click(force=True)
            dl = dl_info.value
            fname = _safe(Path(dl.suggested_filename or Path(href).name or f"{quarter}.pdf").name)
            path = dest_dir / fname; dl.save_as(str(path)); pdfs.append(path); continue
        except Exception: pass
        try:
            r = requests.get(href, timeout=20)
            if r.status_code == 200 and r.content:
                fname = _safe(Path(href).name or f"{quarter}.pdf"); path = dest_dir / fname
                with open(path, 'wb') as f: f.write(r.content)
                pdfs.append(path)
        except Exception: continue
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
        if not src_json.exists(): return None
        dst_json = out_dir / "excerpts_clean.json"
        if src_json != dst_json: shutil.copy2(src_json, dst_json)
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
        traceback.print_exc(); return None

# stamping + compile

def _overlay_single_page(w: float, h: float, left: str, mid: str, right: str) -> BytesIO:
    buf = BytesIO(); c = canvas.Canvas(buf, pagesize=(w, h))
    c.setFont("Helvetica", 8.5); c.setFillColor(colors.HexColor("#4b2142"))  # Cutler purple
    L = R = 0.75 * 72; T = 0.75 * 72
    if left: c.drawString(L, h - T + 0.35 * 72, left)
    if mid:
        text = (mid[:95] + '…') if len(mid) > 96 else mid
        c.drawCentredString(w / 2.0, h - T + 0.35 * 72, text)
    if right: c.drawRightString(w - R, h - T + 0.35 * 72, right)
    c.save(); buf.seek(0); return buf

def _stamp_pdf(src: Path, left: str, mid: str, right: str) -> Path:
    try: r = _PdfReader(str(src))
    except Exception: return src
    w = _PdfWriter()
    for pg in r.pages:
        W = float(pg.mediabox.width); H = float(pg.mediabox.height)
        ov = _PdfReader(_overlay_single_page(W, H, left, mid, right)).pages[0]
        try: pg.merge_page(ov)
        except Exception: pass
        w.add_page(pg)
    tmp = src.with_suffix('.stamped.tmp.pdf')
    with open(tmp, 'wb') as f: w.write(f)
    return tmp

def compile_merged(batch: str, quarter: str, collected: List[Path]) -> Optional[Path]:
    if not collected: return None
    out = CP_DIR / f"Compiled_Cutler_{batch.replace(' ', '')}_{quarter}_{datetime.now():%Y%m%d_%H%M%S}.pdf"
    m = PdfMerger(); added = 0
    for p in collected:
        try:
            title = p.stem.replace('_', ' ').replace('-', ' ')
            stamped = _stamp_pdf(p, left=batch, mid=title, right=f"Run {datetime.now():%Y-%m-%d %H:%M}")
            m.append(str(stamped)); added += 1
        except Exception: continue
    if not added:
        m.close()
        return None
    try: m.write(str(out))
    finally: m.close()
    return out

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
            page.set_default_timeout(15000)
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

    Example:
      - Today: 2025-11-06 (Nov 6, 2025, in Q4)
        -> last completed = (2025, 3)  i.e., '2025 Q3'
      - Today: 2025-02-10 (Feb, in Q1)
        -> last completed = (2024, 4)  i.e., '2024 Q4'
    """
    if today is None:
        today = datetime.now()
    y = today.year
    m = today.month

    if 1 <= m <= 3:
        # In Q1 -> last completed is previous year's Q4
        return y - 1, 4
    elif 4 <= m <= 6:
        # In Q2 -> last completed is Q1 of the same year
        return y, 1
    elif 7 <= m <= 9:
        # In Q3 -> last completed is Q2 of the same year
        return y, 2
    else:
        # In Q4 -> last completed is Q3 of the same year
        return y, 3


def choose_default_quarter(available: List[str]) -> Optional[str]:
    """
    Given the list of available quarters from the site, choose the default
    as the **last completed US quarter**, if present. If not present,
    choose the most recent available.

    This ensures that on 2025-11-06 the default is '2025 Q3', even if
    '2025 Q4' or future years appear in the dropdown.
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

    # Prefer the latest quarter that is <= the last completed quarter
    for lab, year, q in parsed:
        if year < target_year or (year == target_year and q <= target_q):
            return lab

    # If none <= target (extreme edge case), fall back to newest available
    return parsed[0][0]

# run one batch

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
        page = ctx.new_page(); page.set_default_timeout(15000)
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

            for i, (brand, token) in enumerate(tokens, 1):
                st.write(f"[{q}] {i}/{len(tokens)} — {brand} (search: {token})")
                try:
                    _search_by_fund(page, token)
                    hits = _parse_rows(page, q)
                    if not hits:
                        continue
                    seen = set()
                    for h in hits:
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
        browser.close()

# UI

def main():
    st.set_page_config(page_title="Cutler Capital Scraper", layout="wide")

    # Global styling: Cutler purple theme and modernized controls
    st.markdown(
        """
        <style>
        /* Overall background and font tweaks */
        .stApp {
            background: radial-gradient(circle at top left, #f5f0fb 0, #ffffff 40%, #f7f3fb 100%);
        }
        .block-container {
            /* give enough top space so the logo sits fully below the Streamlit header */
            padding-top: 4rem;         /* was 1.5rem */
            max-width: 1100px;
        }
        .app-title {
            text-align: center;
            color: #4b2142;
            font-size: 1.9rem;
            font-weight: 700;
            margin-bottom: 0.1rem;
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
        
         /* ─────────── HEADER / TOP BAR UNIFICATION ─────────── */
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
            flex: 1 1 0;                    /* make both options same width */
            justify-content: center;        /* center the text */
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
        /* Selected radio */
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
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Header: centered logo and text
    logo_path = HERE / "cutler.png"
    col_left, col_center, col_right = st.columns([1, 2, 1])
    with col_center:
        if logo_path.exists():
            # Fixed width so the whole logo is visible and centered
            st.image(str(logo_path), width=260)
        st.markdown("<div class='app-title'>Cutler Capital Letter Scraper</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='app-subtitle'>Scrape, excerpt, and compile fund letters by fund family and quarter.</div>",
            unsafe_allow_html=True,
        )

    # Sidebar: run settings
    st.sidebar.header("Run settings")

    # Get available quarters dynamically from the site
    quarter_options = get_available_quarters()

    # Choose default as the last completed US quarter (e.g., 2025 Q3 on 2025-11-06)
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

    # Main controls in a card
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

    # Output path
    st.markdown("<div class='cc-card'>", unsafe_allow_html=True)
    st.write("**Output root folder (on this machine):**")
    st.code(str(BASE))
    st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
