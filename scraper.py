"""
Nuremberg Scholar — Yale Avalon IMT Production Scraper
=======================================================
Grounded entirely in confirmed HTML observations from:
  nuremberg_avalon_intelligence_brief_v3_FINAL_Version3.md

What this scrapes:
  Phase 1 — 218 IMT session transcript pages (Volumes 2–22)
  Phase 2 — Judgment sub-pages (27 pages from judcont.asp)
  Phase 3 — Key documents (Charter, Indictment, Wannsee, etc.)
  Phase 4 — Secondary collections (Motions, Orders, Witnesses, etc.)

Output:
  output/sessions/           — one JSON per session transcript
  output/judgment/           — one JSON per judgment sub-page
  output/key_docs/           — one JSON per key document
  output/secondary/          — one JSON per secondary doc
  output/index.csv           — master index of all scraped pages
  output/scrape.log          — full log with timestamps
  output/failed.json         — pages that failed after retries

Usage:
  pip install requests beautifulsoup4 chardet
  python scraper.py

  # Dry run (enumerate URLs, no fetching):
  python scraper.py --dry-run

  # Proceedings only:
  python scraper.py --phase 1

  # Resume interrupted scrape:
  python scraper.py --resume
"""

import re
import csv
import html
import json
import time
import logging
import argparse
import hashlib
from pathlib import Path
from datetime import datetime, timezone
from urllib.parse import urljoin
from collections import Counter

import requests
from bs4 import BeautifulSoup

# ── Config ─────────────────────────────────────────────────────────────────

BASE_URL    = "https://avalon.law.yale.edu"
IMT_BASE    = f"{BASE_URL}/imt"
OUTPUT_DIR  = Path("output")
DELAY       = 1.5      # seconds between requests
TIMEOUT     = 30       # seconds per request
MAX_RETRIES = 3
BACKOFF     = 2.0      # exponential backoff multiplier
USER_AGENT  = "NurembergScholar/1.0 (academic research; github.com/nuremberg-scholar)"

# Confirmed from intelligence brief — all 22 volume menu URLs
VOLUME_MENU_URLS = [
    f"{BASE_URL}/subject_menus/imtproc_v{n}menu.asp"
    for n in range(1, 23)
]

# Confirmed secondary collection menus
SECONDARY_MENUS = {
    "judgment":  f"{BASE_URL}/subject_menus/judcont.asp",      # 27 sub-pages
    "motions":   f"{BASE_URL}/subject_menus/motions.asp",      # 8 docs
    "orders":    f"{BASE_URL}/subject_menus/orders.asp",       # 7 docs
    "cases":     f"{BASE_URL}/subject_menus/cases.asp",        # 14 docs
    "witnesses": f"{BASE_URL}/subject_menus/witness.asp",      # 6 docs
    "jackson":   f"{BASE_URL}/subject_menus/jackson.asp",      # 37 docs
    "pohl":      f"{BASE_URL}/subject_menus/pohl.asp",         # 5 docs
    "nca_v1":    f"{BASE_URL}/subject_menus/nca_vol1.asp",     # 14 docs
    "nca_v2":    f"{BASE_URL}/subject_menus/nca_v2menu.asp",   # 16 docs
    "nca_v3":    f"{BASE_URL}/subject_menus/nca_v3menu.asp",
    "nca_v4":    f"{BASE_URL}/subject_menus/nca_v4menu.asp",
}

# Confirmed key documents (direct URLs — no menu)
KEY_DOCS = {
    "charter":          f"{IMT_BASE}/imtconst.asp",
    "indictment":       f"{IMT_BASE}/count.asp",
    "jackson_report":   f"{IMT_BASE}/imt_jack01.asp",
    "london_agreement": f"{IMT_BASE}/imtchart.asp",
    "nuremberg_code":   f"{IMT_BASE}/nurecode.asp",
    "protocol":         f"{IMT_BASE}/imtprot.asp",
    "royal_warrant":    f"{IMT_BASE}/imtroyal.asp",
    "st_james":         f"{IMT_BASE}/imtjames.asp",
    "jackson_statement":f"{IMT_BASE}/imt_jack02.asp",
    "stroop_report":    f"{IMT_BASE}/1061-ps.asp",
    "wannsee":          f"{IMT_BASE}/wannsee.asp",
    "final_report":     f"{IMT_BASE}/naeve.asp",
}

# Nav links to exclude when extracting content links from menus
NAV_EXCLUDE = re.compile(
    r'default\.asp|subject_menus/(?!imtproc|judcont|motions|orders|cases|witness|'
    r'jackson|pohl|nca)|ancient|medieval|\d{2}th\.asp|21st\.asp|'
    r'accessibility|contact|library|orbis|morris|purpose\.asp|'
    r'major\.asp|lawwar|versailles|blbk|munich|kellogg|nonagres|moscow|triparti',
    re.IGNORECASE
)

SESSION_URL_RE  = re.compile(r'/imt/(\d{2})-(\d{2})-(\d{2})\.asp$')

# Speaker extraction — TWO formats confirmed in the wild
SPEAKER_RE_STRONG = re.compile(r'<(?:strong|b)[^>]*>([^<]+):</(?:strong|b)>', re.IGNORECASE)
SPEAKER_RE_PLAIN  = re.compile(
    r'^([A-Z][A-Z0-9\s\.\-]+(?:\s*\([^)]{1,60}\))?)\s*:\s*(.*)$',
    re.DOTALL
)
NONSPEAKER_RE = re.compile(
    r'^(?:[A-Z\s\-]+DAY|Morning Session|Afternoon Session|Evening Session|'
    r'\d|\[|\(|Nuremberg|Volume|Previous|Next)',
    re.IGNORECASE
)

SESSION_TYPE_RE = re.compile(r'(Morning|Afternoon|Evening)\s+Session', re.IGNORECASE)
DAY_HEADER_RE   = re.compile(
    r'([A-Z][A-Z\s\-]+DAY)\s*(?:<[Bb][Rr]>\s*\w+day,?\s*(\d{1,2}\s+\w+\s+\d{4}))?',
    re.IGNORECASE
)


# ── Page number extraction — all 4 confirmed formats ──────────────────────
#
# F1: <a name="123">              numeric named anchor (NCA docs, some sessions)
# F2: <P CLASS=PAGE ...>1 </P>   explicit PAGE class paragraph
# F3: <P>55 </P>                  bare standalone digit paragraph (session pages)
# F4: <A CLASS="nobold" NAME="p29">29</A>  nobold anchor (vol1 indictment docs)
#
# The original scraper only handled F1 and F2, leaving 163 session pages with
# NO_PAGE_NUMBERS. This unified extractor handles all four.

def _extract_page_numbers_F1(soup) -> list[int]:
    """F1: <a name="123"> where name is a pure integer."""
    results = []
    for tag in soup.find_all("a", attrs={"name": True}):
        name = tag.get("name", "").strip()
        if re.fullmatch(r"\d+", name):
            results.append(int(name))
    return results


def _extract_page_numbers_F2(soup) -> list[int]:
    """F2: <P CLASS=PAGE ...> paragraph whose text is the page number."""
    results = []
    for tag in soup.find_all("p", class_=lambda c: c and "PAGE" in
                             (c if isinstance(c, str) else " ".join(c)).upper()):
        text = tag.get_text(strip=True)
        if re.fullmatch(r"\d+", text):
            results.append(int(text))
    return results


def _extract_page_numbers_F3(soup) -> list[int]:
    """
    F3: bare <P>55 </P> paragraph containing only a page number.
    Guards: not inside a table, no child tags, text length ≤ 6 chars.
    """
    results = []
    for tag in soup.find_all("p"):
        if tag.find_parent("table"):
            continue
        if tag.find():          # has child elements → F4 candidate, skip
            continue
        text = tag.get_text(strip=True)
        if re.fullmatch(r"\d{1,4}", text) and len(str(tag)) <= 30:
            results.append(int(text))
    return results


def _extract_page_numbers_F4(soup) -> list[int]:
    """F4: <A CLASS="nobold" NAME="p29">29</A> — nobold anchor."""
    results = []
    for tag in soup.find_all("a", class_=lambda c: c and "nobold" in
                             (c if isinstance(c, str) else " ".join(c)).lower()):
        name = tag.get("name", "")
        if re.fullmatch(r"p\d+", name):
            page_num = int(name[1:])
            if tag.get_text(strip=True) == str(page_num):
                results.append(page_num)
    return results


def extract_page_numbers(soup) -> list[int]:
    """
    Unified page number extractor. Runs all 4 format detectors and returns a
    sorted, deduplicated list of page numbers.

    Priority for dedup when the same number appears in multiple formats:
    F4 > F2 > F1 > F3  (most explicit markup wins)
    """
    seen: dict[int, int] = {}  # page_num → priority (lower = higher priority)
    priority = {"F4": 0, "F2": 1, "F1": 2, "F3": 3}

    for fmt, nums in [
        ("F1", _extract_page_numbers_F1(soup)),
        ("F2", _extract_page_numbers_F2(soup)),
        ("F3", _extract_page_numbers_F3(soup)),
        ("F4", _extract_page_numbers_F4(soup)),
    ]:
        p = priority[fmt]
        for n in nums:
            if n not in seen or p < seen[n]:
                seen[n] = p

    return sorted(seen.keys())


def detect_page_format(soup) -> str:
    """
    Returns the dominant page-number format found in this document.
    Used for diagnostics and validation flagging only.
    """
    counts = {
        "F1": len(_extract_page_numbers_F1(soup)),
        "F2": len(_extract_page_numbers_F2(soup)),
        "F3": len(_extract_page_numbers_F3(soup)),
        "F4": len(_extract_page_numbers_F4(soup)),
    }
    total = sum(counts.values())
    if total == 0:
        return "NONE"
    top = sorted(counts.items(), key=lambda x: -x[1])
    if len(top) > 1 and top[1][1] / total >= 0.20:
        return "MIXED"
    return top[0][0]


# ── Logging setup ──────────────────────────────────────────────────────────

OUTPUT_DIR.mkdir(exist_ok=True)
for sub in ("sessions", "judgment", "key_docs", "secondary"):
    (OUTPUT_DIR / sub).mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(OUTPUT_DIR / "scrape.log", encoding="utf-8"),
        logging.StreamHandler(),
    ]
)
log = logging.getLogger("scraper")


# ── HTTP session ───────────────────────────────────────────────────────────

http = requests.Session()
http.headers.update({"User-Agent": USER_AGENT})
_last_request = 0.0


def fetch(url: str, is_retry: bool = False) -> tuple[int, str]:
    """
    Fetch a URL with rate limiting and retry logic.
    Returns (status_code, html_text).
    On failure returns (0, error_message).
    """
    global _last_request

    elapsed = time.time() - _last_request
    if elapsed < DELAY:
        time.sleep(DELAY - elapsed)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = http.get(url, timeout=TIMEOUT, allow_redirects=True)
            _last_request = time.time()

            if r.status_code == 200:
                declared = r.encoding or ""
                if declared.lower() in ("iso-8859-1", "latin-1", ""):
                    r.encoding = r.apparent_encoding or "windows-1252"
                return 200, r.text

            elif r.status_code == 404:
                log.debug(f"404 {url}")
                return 404, ""

            elif r.status_code in (429, 503):
                wait = DELAY * (BACKOFF ** attempt)
                log.warning(f"Rate limited ({r.status_code}) on {url} — waiting {wait:.1f}s")
                time.sleep(wait)

            elif r.status_code >= 500:
                if attempt < MAX_RETRIES:
                    wait = DELAY * (BACKOFF ** attempt)
                    log.warning(f"Server error {r.status_code} on {url}, retry {attempt}/{MAX_RETRIES}")
                    time.sleep(wait)
                else:
                    return r.status_code, ""

        except requests.exceptions.Timeout:
            log.warning(f"Timeout on {url}, attempt {attempt}/{MAX_RETRIES}")
            if attempt < MAX_RETRIES:
                time.sleep(DELAY * BACKOFF)
        except requests.exceptions.ConnectionError as e:
            log.warning(f"Connection error on {url}: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(DELAY * BACKOFF)
        except Exception as e:
            log.error(f"Unexpected error fetching {url}: {e}")
            return 0, str(e)

    return 0, "max_retries_exceeded"


# ── URL utilities ──────────────────────────────────────────────────────────

def parse_date_from_url(url: str) -> str | None:
    """Parse ISO date from /imt/MM-DD-YY.asp URL pattern."""
    m = SESSION_URL_RE.search(url)
    if m:
        mm, dd, yy = m.groups()
        return f"19{yy}-{mm}-{dd}"
    return None


def url_to_filename(url: str) -> str:
    """Convert URL to safe filename."""
    slug = re.sub(r'[^\w\-]', '_', url.replace(BASE_URL, '').strip('/'))
    return slug[:120] + ".json"


def is_nav_link(href: str) -> bool:
    return bool(NAV_EXCLUDE.search(href))


def extract_content_links(html_text: str, page_url: str) -> list[str]:
    """Extract content links from a menu page, filtering navigation chrome."""
    soup = BeautifulSoup(html_text, "html.parser")
    links = []
    seen = set()
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if not href or href.startswith("#"):
            continue
        abs_url = urljoin(page_url, href)
        if abs_url in seen:
            continue
        if is_nav_link(abs_url):
            continue
        if not abs_url.startswith(BASE_URL):
            continue
        seen.add(abs_url)
        links.append(abs_url)
    return links


# ── Parsing pipeline ───────────────────────────────────────────────────────

def parse_transcript(url: str, raw_html: str) -> dict:
    """
    Parse a session transcript page into structured JSON.
    """
    soup = BeautifulSoup(raw_html, "html.parser")

    # ── Step 1: Find content container ─────────────────────────────────
    container = soup.find("div", class_="text-properties")
    if not container:
        return {
            "url": url,
            "error": "NO_CONTENT_CONTAINER",
            "validation_flags": ["MISSING_CONTENT_CONTAINER"],
        }

    # ── Step 2: Extract title ───────────────────────────────────────────
    title_div = container.find("div", class_="document-title")
    title = title_div.get_text(strip=True) if title_div else None

    # ── CRITICAL: decompose nav BEFORE str(container) ──────────────────
    # str() snapshots the tree; decomposing after has no effect on the string.
    if title_div:
        title_div.decompose()
    for nav_table in container.find_all("table", class_="site-menu"):
        nav_table.decompose()
    for chrome in container.find_all(class_=["HeaderContainer", "FooterContainer"]):
        chrome.decompose()

    # ── Step 3: Extract page numbers — all 4 formats ───────────────────
    # Run the unified extractor on the container subtree (nav already removed).
    # This correctly handles F1/F2/F3/F4 and MIXED documents.
    page_numbers = extract_page_numbers(container)
    page_format  = detect_page_format(container)

    # NOW snapshot — nav is already removed from the tree
    container_html = str(container)

    # ── Step 4: Extract speaker turns — dual strategy ──────────────────
    turns = []
    preamble = ""

    strong_matches = SPEAKER_RE_STRONG.findall(container_html)

    if strong_matches:
        # Strategy A: <strong>SPEAKER:</strong> markup
        parts = SPEAKER_RE_STRONG.split(container_html)
        preamble = BeautifulSoup(parts[0], "html.parser").get_text(separator=" ", strip=True)
        for i in range(1, len(parts) - 1, 2):
            speaker = parts[i].strip()
            text_html = parts[i + 1] if (i + 1) < len(parts) else ""
            # For page_number within this turn: find nearest page number
            # by scanning page_numbers list rather than regex on text_html
            # (avoids double-counting across formats)
            page_num = page_numbers[0] if page_numbers else None
            clean_text = html.unescape(
                BeautifulSoup(text_html, "html.parser").get_text(separator=" ", strip=True)
            )
            if speaker:
                turns.append({"speaker": speaker, "text": clean_text, "page_number": page_num})
    else:
        # Strategy B: paragraph-level plain-text detection
        current_speaker = None
        current_texts = []
        current_page = page_numbers[0] if page_numbers else None
        pre_parts = []

        for p_tag in container.find_all("p"):
            raw = p_tag.get_text(separator=" ", strip=True)
            if not raw:
                continue

            # F2: CLASS=PAGE paragraph — update current page and skip
            page_attr = " ".join(p_tag.get("class", []))
            if "page" in page_attr.lower():
                digits = re.match(r'\s*(\d+)', raw)
                if digits:
                    current_page = int(digits.group(1))
                continue

            # F3: bare digit paragraph — update current page and skip
            if re.fullmatch(r"\d{1,4}", raw) and not p_tag.find() and len(str(p_tag)) <= 30:
                if not p_tag.find_parent("table"):
                    current_page = int(raw)
                    continue

            # F4: nobold anchor paragraph — update current page and skip
            nobold = p_tag.find("a", class_=lambda c: c and "nobold" in
                                (c if isinstance(c, str) else " ".join(c)).lower())
            if nobold:
                name = nobold.get("name", "")
                if re.fullmatch(r"p\d+", name):
                    current_page = int(name[1:])
                    continue

            # Running date watermarks (e.g. "4 Jan. 46") — skip
            if re.match(r'^\d{1,2}\s+\w+\.?\s+\d{2}\s*$', raw):
                continue

            m = SPEAKER_RE_PLAIN.match(raw)
            if m and not NONSPEAKER_RE.match(raw):
                if current_speaker:
                    turns.append({
                        "speaker": current_speaker,
                        "text": " ".join(current_texts).strip(),
                        "page_number": current_page,
                    })
                current_speaker = m.group(1).strip()
                remainder = m.group(2).strip()
                current_texts = [remainder] if remainder else []
            else:
                if current_speaker:
                    current_texts.append(raw)
                else:
                    pre_parts.append(raw)

        if current_speaker and current_texts:
            turns.append({
                "speaker": current_speaker,
                "text": " ".join(current_texts).strip(),
                "page_number": current_page,
            })

        preamble = " ".join(pre_parts)

    # ── Step 5: Session metadata ────────────────────────────────────────
    full_text = html.unescape(soup.get_text(separator=" "))

    session_types = sorted(set(
        m.group(1).lower()
        for m in SESSION_TYPE_RE.finditer(full_text)
    )) or ["full"]

    day_match = DAY_HEADER_RE.search(full_text[:600])
    day_ordinal = day_match.group(1).strip().title() if day_match else None

    speakers = list(dict.fromkeys(t["speaker"] for t in turns))

    # ── Step 6: Inline content links ───────────────────────────────────
    inline_links = []
    for a in container.find_all("a", href=True):
        href = a["href"].strip()
        if href.startswith("#"):
            continue
        abs_href = urljoin(url, href)
        if is_nav_link(abs_href):
            continue
        inline_links.append({
            "text": a.get_text(strip=True)[:80],
            "href": abs_href,
        })

    # ── Step 7: Validation ──────────────────────────────────────────────
    flags = validate(url, raw_html, turns, page_numbers)

    return {
        "url":              url,
        "title":            title,
        "date_iso":         parse_date_from_url(url),
        "date_source":      "url",
        "day_ordinal":      day_ordinal,
        "session_types":    session_types,
        "page_start":       min(page_numbers) if page_numbers else None,
        "page_end":         max(page_numbers) if page_numbers else None,
        "page_numbers":     page_numbers,
        "page_format":      page_format,         # NEW: which format(s) were found
        "speakers":         speakers,
        "speaker_count":    len(speakers),
        "turns":            turns,
        "turn_count":       len(turns),
        "preamble":         preamble[:500],
        "inline_links":     inline_links,
        "char_count":       len(raw_html),
        "word_count":       len(full_text.split()),
        "scrape_timestamp": datetime.now(timezone.utc).isoformat(),
        "validation_flags": flags,
        "content_hash":     hashlib.md5(raw_html.encode()).hexdigest(),
    }


def parse_document(url: str, raw_html: str, doc_type: str = "document") -> dict:
    """
    Parse a non-session document (key docs, judgment pages, secondary).
    """
    soup = BeautifulSoup(raw_html, "html.parser")

    container = soup.find("div", class_="text-properties")
    content_soup = container if container else soup

    title_div = content_soup.find("div", class_="document-title")
    title = title_div.get_text(strip=True) if title_div else (
        soup.title.get_text(strip=True) if soup.title else None
    )

    for nav in content_soup.find_all("table", class_="site-menu"):
        nav.decompose()

    full_text = html.unescape(content_soup.get_text(separator="\n", strip=True))

    # Use unified extractor on the content subtree
    page_numbers = extract_page_numbers(content_soup)
    page_format  = detect_page_format(content_soup)

    return {
        "url":              url,
        "doc_type":         doc_type,
        "title":            title,
        "full_text":        full_text,
        "page_numbers":     page_numbers,
        "page_format":      page_format,             # NEW
        "char_count":       len(raw_html),
        "word_count":       len(full_text.split()),
        "scrape_timestamp": datetime.now(timezone.utc).isoformat(),
        "validation_flags": validate(url, raw_html, [], page_numbers),
        "content_hash":     hashlib.md5(raw_html.encode()).hexdigest(),
    }


def validate(url: str, raw_html: str, turns: list, pages: list) -> list[str]:
    """Returns list of validation flag strings. Empty = passed all checks."""
    flags = []
    if len(raw_html) < 5_000:
        flags.append("SUSPICIOUSLY_SHORT")
    if not re.search(r'<div[^>]+class=["\']?text-properties', raw_html, re.IGNORECASE):
        flags.append("MISSING_CONTENT_CONTAINER")
    if SESSION_URL_RE.search(url) and not turns:
        flags.append("NO_SPEAKER_TURNS")
    if SESSION_URL_RE.search(url) and not pages:
        flags.append("NO_PAGE_NUMBERS")
    if turns:
        last_text = turns[-1].get("text", "").strip()[-120:]
    else:
        soup_check = BeautifulSoup(raw_html, "html.parser")
        for nav in soup_check.find_all("table", class_="site-menu"):
            nav.decompose()
        for nav in soup_check.find_all(class_=["HeaderContainer", "FooterContainer"]):
            nav.decompose()
        last_text = soup_check.get_text(separator=" ").strip()[-120:]
    if last_text and not re.search(r'[.!?\])]', last_text):
        flags.append("POSSIBLE_TRUNCATION")
    return flags


# ── Index / progress tracking ──────────────────────────────────────────────

class Index:
    """Tracks all scraped URLs. Supports resume."""

    CSV_PATH  = OUTPUT_DIR / "index.csv"
    FAIL_PATH = OUTPUT_DIR / "failed.json"
    FIELDNAMES = [
        "url", "collection", "date_iso", "status", "char_count",
        "turn_count", "speaker_count", "page_format", "validation_flags",
        "scrape_timestamp",
    ]

    def __init__(self):
        self._done: set[str] = set()
        self._rows: list[dict] = []
        self._failed: list[dict] = []

        if self.CSV_PATH.exists():
            with open(self.CSV_PATH, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    if row.get("status") == "ok":
                        self._done.add(row["url"])
            log.info(f"Resuming — {len(self._done)} pages already done")

    def already_done(self, url: str) -> bool:
        return url in self._done

    def record(self, url: str, collection: str, parsed: dict, status: str):
        row = {
            "url":               url,
            "collection":        collection,
            "date_iso":          parsed.get("date_iso", ""),
            "status":            status,
            "char_count":        parsed.get("char_count", 0),
            "turn_count":        parsed.get("turn_count", 0),
            "speaker_count":     parsed.get("speaker_count", 0),
            "page_format":       parsed.get("page_format", "NONE"),  # NEW
            "validation_flags":  "|".join(parsed.get("validation_flags", [])),
            "scrape_timestamp":  parsed.get("scrape_timestamp", ""),
        }
        self._rows.append(row)
        self._done.add(url)

        write_header = not self.CSV_PATH.exists()
        with open(self.CSV_PATH, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=self.FIELDNAMES)
            if write_header:
                w.writeheader()
            w.writerow(row)

    def record_failure(self, url: str, reason: str):
        entry = {"url": url, "reason": reason, "timestamp": datetime.now(timezone.utc).isoformat()}
        self._failed.append(entry)
        with open(self.FAIL_PATH, "w", encoding="utf-8") as f:
            json.dump(self._failed, f, indent=2)


# ── Phase runners ──────────────────────────────────────────────────────────

def run_phase1_sessions(index: Index, dry_run: bool = False) -> list[str]:
    """Phase 1: Enumerate all session URLs from 22 volume menus."""
    log.info("=== PHASE 1: Building session URL list from 22 volume menus ===")
    master_urls = []
    vol_summary = []

    for vol_num, menu_url in enumerate(VOLUME_MENU_URLS, start=1):
        log.info(f"Volume {vol_num:>2}: {menu_url}")
        status, menu_html = fetch(menu_url)

        if status != 200:
            log.warning(f"  Vol {vol_num} menu returned {status} — skipping")
            vol_summary.append({"volume": vol_num, "status": status, "sessions": 0})
            continue

        links = extract_content_links(menu_html, menu_url)

        if vol_num == 1:
            session_links = links
            log.info(f"  Vol 1 (procedural): {len(links)} content links")
        else:
            session_links = [l for l in links if SESSION_URL_RE.search(l)]
            log.info(f"  Vol {vol_num:>2}: {len(session_links)} session links")

        master_urls.extend(session_links)
        vol_summary.append({
            "volume": vol_num,
            "status": status,
            "sessions": len(session_links),
            "first": session_links[0] if session_links else None,
            "last": session_links[-1] if session_links else None,
        })

    seen = set()
    deduped = []
    for u in master_urls:
        if u not in seen:
            seen.add(u)
            deduped.append(u)

    log.info(f"Total session URLs enumerated: {len(deduped)}")

    with open(OUTPUT_DIR / "volume_summary.json", "w", encoding="utf-8") as f:
        json.dump(vol_summary, f, indent=2)

    if dry_run:
        with open(OUTPUT_DIR / "master_urls.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(deduped))
        log.info(f"Dry run — {len(deduped)} URLs written to master_urls.txt")
        return deduped

    log.info(f"=== Scraping {len(deduped)} session pages ===")
    for i, url in enumerate(deduped, start=1):
        if index.already_done(url):
            log.debug(f"Skip (already done): {url}")
            continue

        log.info(f"[{i:>4}/{len(deduped)}] {url}")
        status, page_html = fetch(url)

        if status == 404:
            log.warning(f"  404 — skipping {url}")
            index.record_failure(url, "HTTP_404")
            continue
        elif status != 200:
            log.error(f"  HTTP {status} — failed {url}")
            index.record_failure(url, f"HTTP_{status}")
            continue

        is_session = bool(SESSION_URL_RE.search(url))
        parsed = parse_transcript(url, page_html) if is_session else parse_document(url, page_html, "vol1_doc")

        if parsed.get("validation_flags"):
            log.warning(f"  Flags: {parsed['validation_flags']}")

        collection = "sessions" if is_session else "vol1"
        out_path = OUTPUT_DIR / collection / url_to_filename(url)
        out_path.parent.mkdir(exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(parsed, f, ensure_ascii=False, indent=2)

        index.record(url, collection, parsed, "ok")

    return deduped


def run_phase2_judgment(index: Index):
    """Phase 2: Scrape all 27 Judgment sub-pages from judcont.asp."""
    log.info("=== PHASE 2: Judgment sub-pages ===")

    status, menu_html = fetch(SECONDARY_MENUS["judgment"])
    if status != 200:
        log.error(f"Judgment menu returned {status} — skipping phase 2")
        return

    links = extract_content_links(menu_html, SECONDARY_MENUS["judgment"])
    log.info(f"Judgment menu: {len(links)} sub-pages found")

    for i, url in enumerate(links, start=1):
        if index.already_done(url):
            continue
        log.info(f"  [{i:>2}/{len(links)}] {url}")
        status, page_html = fetch(url)
        if status != 200:
            index.record_failure(url, f"HTTP_{status}")
            continue
        parsed = parse_document(url, page_html, "judgment")
        out_path = OUTPUT_DIR / "judgment" / url_to_filename(url)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(parsed, f, ensure_ascii=False, indent=2)
        index.record(url, "judgment", parsed, "ok")


def run_phase3_key_docs(index: Index):
    """Phase 3: Scrape key documents (Charter, Indictment, Wannsee, etc.)"""
    log.info("=== PHASE 3: Key documents ===")
    for doc_name, url in KEY_DOCS.items():
        if index.already_done(url):
            continue
        log.info(f"  {doc_name}: {url}")
        status, page_html = fetch(url)
        if status != 200:
            index.record_failure(url, f"HTTP_{status}")
            continue
        parsed = parse_document(url, page_html, f"key_doc_{doc_name}")
        out_path = OUTPUT_DIR / "key_docs" / f"{doc_name}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(parsed, f, ensure_ascii=False, indent=2)
        index.record(url, "key_docs", parsed, "ok")


def run_phase4_secondary(index: Index):
    """Phase 4: Scrape secondary collections (excluding judgment — done in phase 2)."""
    log.info("=== PHASE 4: Secondary collections ===")

    skip = {"judgment"}
    for collection_name, menu_url in SECONDARY_MENUS.items():
        if collection_name in skip:
            continue

        log.info(f"  Collection: {collection_name}")
        status, menu_html = fetch(menu_url)
        if status != 200:
            log.warning(f"  {collection_name} menu returned {status} — skipping")
            continue

        links = extract_content_links(menu_html, menu_url)
        log.info(f"  {len(links)} documents in {collection_name}")

        for i, url in enumerate(links, start=1):
            if index.already_done(url):
                continue
            log.info(f"    [{i:>3}] {url}")
            status, page_html = fetch(url)
            if status != 200:
                index.record_failure(url, f"HTTP_{status}")
                continue
            parsed = parse_document(url, page_html, collection_name)
            out_path = OUTPUT_DIR / "secondary" / url_to_filename(url)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(parsed, f, ensure_ascii=False, indent=2)
            index.record(url, collection_name, parsed, "ok")


# ── Pre-flight encoding check ──────────────────────────────────────────────

def check_encoding():
    log.info("Pre-flight: checking server encoding declaration...")
    test_url = f"{IMT_BASE}/11-20-45.asp"
    r = http.get(test_url, timeout=15)
    ct = r.headers.get("Content-Type", "not declared")
    log.info(f"  Content-Type: {ct}")
    log.info(f"  requests.encoding: {r.encoding}")
    log.info(f"  apparent_encoding: {r.apparent_encoding}")

    encoding_info = {
        "content_type_header": ct,
        "requests_detected":   r.encoding,
        "chardet_detected":    r.apparent_encoding,
        "recommendation": (
            "use_apparent_encoding"
            if not r.encoding or r.encoding.lower() in ("iso-8859-1", "latin-1")
            else "use_declared"
        ),
    }
    with open(OUTPUT_DIR / "encoding_check.json", "w") as f:
        json.dump(encoding_info, f, indent=2)
    log.info(f"  Saved to output/encoding_check.json")
    return encoding_info


# ── Final summary ──────────────────────────────────────────────────────────

def print_summary(index: Index):
    if not index.CSV_PATH.exists():
        log.info("No index file found.")
        return

    with open(index.CSV_PATH, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    total   = len(rows)
    ok      = sum(1 for r in rows if r["status"] == "ok")
    flagged = sum(1 for r in rows if r["validation_flags"])
    by_coll = {}
    for r in rows:
        by_coll[r["collection"]] = by_coll.get(r["collection"], 0) + 1

    # Page format distribution across sessions
    fmt_counts = Counter(r.get("page_format", "NONE") for r in rows
                         if r.get("collection") == "sessions")

    print(f"\n{'═'*55}")
    print(f"  SCRAPE COMPLETE")
    print(f"{'═'*55}")
    print(f"  Total pages scraped:  {ok:>5}")
    print(f"  Total attempted:      {total:>5}")
    print(f"  Pages with flags:     {flagged:>5}")
    print(f"\n  By collection:")
    for coll, count in sorted(by_coll.items()):
        print(f"    {coll:<20} {count:>5}")
    if fmt_counts:
        print(f"\n  Session page formats:")
        for fmt, count in sorted(fmt_counts.items(), key=lambda x: -x[1]):
            print(f"    {fmt:<10} {count:>5}")
    print(f"\n  Outputs in: {OUTPUT_DIR.absolute()}")
    print(f"{'═'*55}\n")


# ── Reparse mode ──────────────────────────────────────────────────────────

def run_reparse():
    """
    Reparse all already-downloaded JSON files without re-fetching.
    Reads raw_html from existing JSON files → re-runs parse_transcript/parse_document
    → overwrites JSON and rebuilds index.csv.
    """
    log.info("=== REPARSE MODE: rebuilding from cached HTML on disk ===")

    all_files = list(OUTPUT_DIR.rglob("*.json"))
    content_files = [
        f for f in all_files
        if f.parent.name in ("sessions", "judgment", "key_docs", "secondary", "vol1")
    ]
    log.info(f"Found {len(content_files)} JSON files to reparse")

    index_path = OUTPUT_DIR / "index.csv"
    if index_path.exists():
        index_path.rename(index_path.with_suffix(".csv.bak"))
        log.info("Backed up old index.csv → index.csv.bak")

    index = Index()
    ok = 0
    flagged = 0

    for i, fpath in enumerate(sorted(content_files), 1):
        try:
            old_data = json.loads(fpath.read_text(encoding="utf-8"))
        except Exception as e:
            log.warning(f"Could not read {fpath}: {e}")
            continue

        url      = old_data.get("url", "")
        raw_html = old_data.get("raw_html", "")
        coll     = fpath.parent.name

        if not raw_html:
            log.warning(f"No raw_html in {fpath.name} — skipping (raw_html not stored)")
            continue

        is_session = bool(SESSION_URL_RE.search(url))
        if is_session:
            parsed = parse_transcript(url, raw_html)
        else:
            doc_type = old_data.get("doc_type", coll)
            parsed = parse_document(url, raw_html, doc_type)

        fpath.write_text(json.dumps(parsed, ensure_ascii=False, indent=2), encoding="utf-8")
        flags = parsed.get("validation_flags", [])
        index.record(url, coll, parsed, "ok")

        if flags:
            flagged += 1
            log.debug(f"  [{i}] {fpath.name} flags={flags}")
        ok += 1

        if i % 50 == 0:
            log.info(f"  Reparsed {i}/{len(content_files)} ...")

    log.info(f"Reparse complete: {ok} files, {flagged} with flags")


# ── Entry point ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Nuremberg Avalon IMT Scraper")
    parser.add_argument("--dry-run",  action="store_true", help="Enumerate URLs only, no fetching")
    parser.add_argument("--resume",   action="store_true", help="Skip already-scraped pages")
    parser.add_argument("--phase",    type=int, choices=[1,2,3,4], help="Run a single phase only")
    parser.add_argument("--no-encoding-check", action="store_true", help="Skip pre-flight encoding check")
    parser.add_argument("--reparse",  action="store_true", help="Reparse cached JSON files without re-fetching")
    args = parser.parse_args()

    log.info("Nuremberg Scholar — Yale Avalon IMT Scraper")
    log.info(f"Output directory: {OUTPUT_DIR.absolute()}")

    if args.reparse:
        run_reparse()
        print_summary(Index())
        return

    if not args.dry_run and not args.no_encoding_check:
        check_encoding()

    index = Index()

    if args.dry_run:
        run_phase1_sessions(index, dry_run=True)
        return

    if args.phase == 1 or args.phase is None:
        run_phase1_sessions(index)

    if args.phase == 2 or args.phase is None:
        run_phase2_judgment(index)

    if args.phase == 3 or args.phase is None:
        run_phase3_key_docs(index)

    if args.phase == 4 or args.phase is None:
        run_phase4_secondary(index)

    print_summary(index)


if __name__ == "__main__":
    main()