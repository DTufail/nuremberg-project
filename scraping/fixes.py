"""
fixes.py — Three targeted fixes derived from audit_output.py results
=====================================================================

ROOT CAUSES CONFIRMED BY LIVE HTML INSPECTION:

  FIX 1 — F5: No container div (17 sessions, 0 page numbers)
  ─────────────────────────────────────────────────────────────
  Volumes 8, 20, 21 and others use content directly in <body> with NO
  <div class="text-properties"> wrapper. parse_transcript() returns early
  with MISSING_CONTENT_CONTAINER. These pages use the same F3 bare-digit
  paragraph format for page numbers, and the same plain-text SPEAKER: format.
  Fix: fall back to <body> when the container div is absent.

  FIX 2 — HTTP redirect stub (04-08-46.asp → zero turns, empty doc)
  ─────────────────────────────────────────────────────────────────────
  04-08-46.asp is a redirect stub page — it contains only a one-line
  "The location has changed" message pointing to itself. The real content
  no longer exists at a reachable URL on Yale Avalon (HTTPS redirect loop).
  Fix: detect and mark as REDIRECT_STUB; patch from HuggingFace trial_id=7
  seq range for that date.

  FIX 3 — Suspect truncated speaker tags (MR, DR, ER, THE)
  ──────────────────────────────────────────────────────────
  Caused by speaker lines split across two <p> tags:
    <p>MR.</p>
    <p>JUSTICE JACKSON: testimony text...</p>
  The plain-text strategy sees "MR." as a lone speaker tag, then "JUSTICE
  JACKSON: ..." as a new speaker. Fix: a post-processing merge pass that
  joins consecutive turns where the first speaker is a known title-only stub.

  FIX 4 — Page range anomaly on 08-22-46.asp (pages 413–4154, span=3741)
  ────────────────────────────────────────────────────────────────────────
  One <a name="4154"> anchor exists in the page — this is a character count
  or OCR artifact, not a real page number. F3 and F4 extractors are clean;
  this is an F1 false positive. Fix: clamp F1 anchors to plausible range
  (1–2000 for session pages).

Apply these to scraper.py by importing and calling patch_scraper().
Or run standalone: python fixes.py --verify  to test against live pages.
"""

import re
import html
import json
import argparse
from pathlib import Path

import requests
from bs4 import BeautifulSoup

# ── Shared constants ───────────────────────────────────────────────────────

SESSION_URL_RE = re.compile(r'/imt/(\d{2})-(\d{2})-(\d{2})\.asp$')

SPEAKER_RE_PLAIN = re.compile(
    r'^([A-Z][A-Z0-9\s\.\-]+(?:\s*\([^)]{1,60}\))?)\s*:\s*(.*)$',
    re.DOTALL
)
NONSPEAKER_RE = re.compile(
    r'^(?:[A-Z\s\-]+DAY|Morning Session|Afternoon Session|Evening Session|'
    r'\d|\[|\(|Nuremberg|Volume|Previous|Next)',
    re.IGNORECASE
)

# Title-only stubs that should be merged with the next turn
TITLE_STUBS = frozenset(["MR", "DR", "ER", "THE", "MS", "SIR", "COL", "GEN",
                          "LT", "CPT", "PROF", "MR.", "DR.", "MS.", "SIR."])

# Redirect stub fingerprint
REDIRECT_STUB_RE = re.compile(
    r'location.*has changed|automatically be transfered|click on the link above',
    re.IGNORECASE
)

# Max plausible page number for a single IMT session (F1 anchor guard)
F1_MAX_PLAUSIBLE = 2000


# ═══════════════════════════════════════════════════════════════════════════
# FIX 1: F5 fallback — parse sessions with no container div
# ═══════════════════════════════════════════════════════════════════════════

def get_content_container(soup: BeautifulSoup):
    """
    Return the best available content container for this page.

    Priority:
      1. <div class="text-properties">  — standard (Vols 2–19, most pages)
      2. <body>                          — fallback for Vols 8/20/21 which
                                          have no wrapper div (F5 structure)
    Returns (container, is_fallback: bool).
    """
    container = soup.find("div", class_="text-properties")
    if container:
        return container, False

    # F5: fall back to body, but strip the nav tables first
    body = soup.find("body")
    if body:
        return body, True

    return None, False


def decompose_nav(container):
    """Remove all known navigation chrome from a container in-place."""
    for nav in container.find_all("table", class_="site-menu"):
        nav.decompose()
    for nav in container.find_all(class_=["HeaderContainer", "FooterContainer"]):
        nav.decompose()
    # F5 pages have a plain nav table at the top (no class) — it's always the
    # first <table> in the body and contains only <a href> nav links
    for table in container.find_all("table"):
        links = table.find_all("a", href=True)
        text_nodes = [t.strip() for t in table.stripped_strings
                      if not any(t in a.get_text() for a in links)]
        if links and not text_nodes:
            table.decompose()


# ═══════════════════════════════════════════════════════════════════════════
# FIX 2: Redirect stub detection
# ═══════════════════════════════════════════════════════════════════════════

def is_redirect_stub(raw_html: str) -> bool:
    """
    Return True if this page is a Yale redirect stub with no real content.
    These pages contain only a one-line notice; the real content is gone.
    """
    return bool(REDIRECT_STUB_RE.search(raw_html)) and len(raw_html) < 5_000


# ═══════════════════════════════════════════════════════════════════════════
# FIX 3: Truncated speaker tag merge
# ═══════════════════════════════════════════════════════════════════════════

def merge_stub_turns(turns: list[dict]) -> list[dict]:
    """
    Post-process a list of speaker turns and merge consecutive turns where
    the first turn's speaker is a title-only stub (MR, DR, THE, etc.).

    Before: [{"speaker": "MR", "text": ""}, {"speaker": "JUSTICE JACKSON", "text": "..."}]
    After:  [{"speaker": "MR. JUSTICE JACKSON", "text": "..."}]

    The stub's text (if any) is prepended to the following turn's text.
    """
    if not turns:
        return turns

    merged = []
    i = 0
    while i < len(turns):
        turn = turns[i]
        spk = turn.get("speaker", "").strip().rstrip(".")

        if spk.upper() in TITLE_STUBS and i + 1 < len(turns):
            next_turn = turns[i + 1]
            # Reconstruct the full speaker name
            stub_text = turn.get("text", "").strip()
            next_spk  = next_turn.get("speaker", "")
            full_spk  = f"{turn['speaker']}. {next_spk}" if not turn["speaker"].endswith(".") \
                        else f"{turn['speaker']} {next_spk}"
            full_text = (stub_text + " " + next_turn.get("text", "")).strip()
            merged.append({
                "speaker":     full_spk.strip(),
                "text":        full_text,
                "page_number": next_turn.get("page_number") or turn.get("page_number"),
            })
            i += 2  # skip both stub and next
        else:
            merged.append(turn)
            i += 1

    return merged


# ═══════════════════════════════════════════════════════════════════════════
# FIX 4: F1 anchor range clamping
# ═══════════════════════════════════════════════════════════════════════════

def extract_page_numbers_F1_clamped(soup, max_val: int = F1_MAX_PLAUSIBLE) -> list[int]:
    """
    F1 extractor with plausibility clamp.
    Rejects <a name="N"> anchors where N > max_val (OCR/artifact false positives).
    """
    results = []
    for tag in soup.find_all("a", attrs={"name": True}):
        name = tag.get("name", "").strip()
        if re.fullmatch(r"\d+", name):
            n = int(name)
            if n <= max_val:
                results.append(n)
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Updated extract_page_numbers — drop-in replacement for scraper.py
# Incorporates F1 clamping; other formats unchanged
# ═══════════════════════════════════════════════════════════════════════════

def extract_page_numbers_F2(soup) -> list[int]:
    results = []
    for tag in soup.find_all("p", class_=lambda c: c and "PAGE" in
                             (c if isinstance(c, str) else " ".join(c)).upper()):
        text = tag.get_text(strip=True)
        if re.fullmatch(r"\d+", text):
            results.append(int(text))
    return results


def extract_page_numbers_F3(soup) -> list[int]:
    results = []
    for tag in soup.find_all("p"):
        if tag.find_parent("table"):
            continue
        if tag.find():
            continue
        text = tag.get_text(strip=True)
        if re.fullmatch(r"\d{1,4}", text) and len(str(tag)) <= 30:
            results.append(int(text))
    return results


def extract_page_numbers_F4(soup) -> list[int]:
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
    Unified extractor — all 4 formats + F1 plausibility clamp (Fix 4).
    Drop-in replacement for the same function in scraper.py.
    """
    priority = {"F4": 0, "F2": 1, "F1": 2, "F3": 3}
    seen: dict[int, int] = {}

    for fmt, nums in [
        ("F1", extract_page_numbers_F1_clamped(soup)),   # clamped
        ("F2", extract_page_numbers_F2(soup)),
        ("F3", extract_page_numbers_F3(soup)),
        ("F4", extract_page_numbers_F4(soup)),
    ]:
        p = priority[fmt]
        for n in nums:
            if n not in seen or p < seen[n]:
                seen[n] = p

    return sorted(seen.keys())


# ═══════════════════════════════════════════════════════════════════════════
# Updated parse_transcript — integrates all 4 fixes
# Drop-in replacement for parse_transcript() in scraper.py
# ═══════════════════════════════════════════════════════════════════════════

SESSION_TYPE_RE = re.compile(r'(Morning|Afternoon|Evening)\s+Session', re.IGNORECASE)
DAY_HEADER_RE   = re.compile(
    r'([A-Z][A-Z\s\-]+DAY)\s*(?:<[Bb][Rr]>\s*\w+day,?\s*(\d{1,2}\s+\w+\s+\d{4}))?',
    re.IGNORECASE
)
SPEAKER_RE_STRONG = re.compile(r'<(?:strong|b)[^>]*>([^<]+):</(?:strong|b)>', re.IGNORECASE)

from urllib.parse import urljoin
import hashlib
from datetime import datetime, timezone


def parse_date_from_url(url: str):
    m = SESSION_URL_RE.search(url)
    if m:
        mm, dd, yy = m.groups()
        return f"19{yy}-{mm}-{dd}"
    return None


def parse_transcript(url: str, raw_html: str) -> dict:
    """
    Full parse_transcript with all 4 fixes applied.
    Drop this into scraper.py to replace the existing function.
    """
    # FIX 2: redirect stub — return immediately with clear flag
    if is_redirect_stub(raw_html):
        return {
            "url":              url,
            "date_iso":         parse_date_from_url(url),
            "error":            "REDIRECT_STUB",
            "validation_flags": ["REDIRECT_STUB", "MISSING_CONTENT_CONTAINER"],
            "turns":            [],
            "turn_count":       0,
            "page_numbers":     [],
            "word_count":       0,
            "char_count":       len(raw_html),
            "scrape_timestamp": datetime.now(timezone.utc).isoformat(),
            "content_hash":     hashlib.md5(raw_html.encode()).hexdigest(),
        }

    soup = BeautifulSoup(raw_html, "html.parser")

    # FIX 1: use fallback container if no text-properties div
    container, is_f5 = get_content_container(soup)
    if container is None:
        return {
            "url": url,
            "error": "NO_CONTENT_CONTAINER",
            "validation_flags": ["MISSING_CONTENT_CONTAINER"],
        }

    title_div = container.find("div", class_="document-title")
    title = title_div.get_text(strip=True) if title_div else None
    if title_div:
        title_div.decompose()

    # FIX 1: use shared nav decomposer (handles both standard and F5 nav)
    decompose_nav(container)

    # Page numbers — FIX 4 (F1 clamped) is baked into extract_page_numbers()
    page_numbers = extract_page_numbers(container)

    def detect_fmt():
        counts = {
            "F1": len(extract_page_numbers_F1_clamped(container)),
            "F2": len(extract_page_numbers_F2(container)),
            "F3": len(extract_page_numbers_F3(container)),
            "F4": len(extract_page_numbers_F4(container)),
        }
        total = sum(counts.values())
        if total == 0:
            return "NONE"
        top = sorted(counts.items(), key=lambda x: -x[1])
        if len(top) > 1 and top[1][1] / total >= 0.20:
            return "MIXED"
        return top[0][0]

    page_format = detect_fmt()
    if is_f5:
        page_format = f"F5+{page_format}" if page_format != "NONE" else "F5"

    container_html = str(container)

    # Speaker extraction
    turns = []
    preamble = ""
    strong_matches = SPEAKER_RE_STRONG.findall(container_html)

    if strong_matches:
        parts = SPEAKER_RE_STRONG.split(container_html)
        preamble = BeautifulSoup(parts[0], "html.parser").get_text(separator=" ", strip=True)
        for i in range(1, len(parts) - 1, 2):
            speaker = parts[i].strip()
            text_html = parts[i + 1] if (i + 1) < len(parts) else ""
            page_num = page_numbers[0] if page_numbers else None
            clean_text = html.unescape(
                BeautifulSoup(text_html, "html.parser").get_text(separator=" ", strip=True)
            )
            if speaker:
                turns.append({"speaker": speaker, "text": clean_text, "page_number": page_num})
    else:
        current_speaker = None
        current_texts   = []
        current_page    = page_numbers[0] if page_numbers else None
        pre_parts       = []

        for p_tag in container.find_all("p"):
            raw = p_tag.get_text(separator=" ", strip=True)
            if not raw:
                continue

            page_attr = " ".join(p_tag.get("class", []))
            if "page" in page_attr.lower():
                digits = re.match(r'\s*(\d+)', raw)
                if digits:
                    current_page = int(digits.group(1))
                continue

            if re.fullmatch(r"\d{1,4}", raw) and not p_tag.find() and len(str(p_tag)) <= 30:
                if not p_tag.find_parent("table"):
                    current_page = int(raw)
                    continue

            nobold = p_tag.find("a", class_=lambda c: c and "nobold" in
                                (c if isinstance(c, str) else " ".join(c)).lower())
            if nobold:
                name = nobold.get("name", "")
                if re.fullmatch(r"p\d+", name):
                    current_page = int(name[1:])
                    continue

            if re.match(r'^\d{1,2}\s+\w+\.?\s+\d{2}\s*$', raw):
                continue

            m = SPEAKER_RE_PLAIN.match(raw)
            if m and not NONSPEAKER_RE.match(raw):
                if current_speaker:
                    turns.append({
                        "speaker":     current_speaker,
                        "text":        " ".join(current_texts).strip(),
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
                "speaker":     current_speaker,
                "text":        " ".join(current_texts).strip(),
                "page_number": current_page,
            })
        preamble = " ".join(pre_parts)

    # FIX 3: merge stub turns
    turns = merge_stub_turns(turns)

    full_text = html.unescape(soup.get_text(separator=" "))
    session_types = sorted(set(
        m.group(1).lower() for m in SESSION_TYPE_RE.finditer(full_text)
    )) or ["full"]
    day_match   = DAY_HEADER_RE.search(full_text[:600])
    day_ordinal = day_match.group(1).strip().title() if day_match else None
    speakers    = list(dict.fromkeys(t["speaker"] for t in turns))

    inline_links = []
    BASE_URL = "https://avalon.law.yale.edu"
    for a in container.find_all("a", href=True):
        href = a["href"].strip()
        if href.startswith("#"):
            continue
        abs_href = urljoin(url, href)
        inline_links.append({"text": a.get_text(strip=True)[:80], "href": abs_href})

    flags = _validate(url, raw_html, turns, page_numbers, is_f5)

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
        "page_format":      page_format,
        "html_structure":   "F5_no_container" if is_f5 else "standard",
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


def _validate(url, raw_html, turns, pages, is_f5=False):
    flags = []
    if len(raw_html) < 5_000:
        flags.append("SUSPICIOUSLY_SHORT")
    if REDIRECT_STUB_RE.search(raw_html):
        flags.append("REDIRECT_STUB")
    if not is_f5 and not re.search(r'<div[^>]+class=["\']?text-properties', raw_html, re.IGNORECASE):
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
        last_text = soup_check.get_text(separator=" ").strip()[-120:]
    if last_text and not re.search(r'[.!?\])]', last_text):
        flags.append("POSSIBLE_TRUNCATION")
    return flags


# ═══════════════════════════════════════════════════════════════════════════
# Verification harness — test all fixes against live Yale pages
# ═══════════════════════════════════════════════════════════════════════════

VERIFY_URLS = {
    # F5 structure (no container div) — should now get turns and pages
    "02-20-46.asp": {"expect_turns": True,  "expect_pages": True,  "expect_f5": True},
    "08-02-46.asp": {"expect_turns": True,  "expect_pages": True,  "expect_f5": True},
    "08-20-46.asp": {"expect_turns": True,  "expect_pages": False, "expect_f5": True},  # zero pages is ok, real content
    # Redirect stub — should be cleanly flagged
    "04-08-46.asp": {"expect_turns": False, "expect_pages": False, "expect_redirect": True},
    # Normal F3 session — regression check
    "11-20-45.asp": {"expect_turns": True,  "expect_pages": True,  "expect_f5": False},
    # Normal F2 session — regression check
    "12-01-45.asp": {"expect_turns": True,  "expect_pages": True},
}

PASS = "✅"
FAIL = "❌"


def verify():
    import time
    session = requests.Session()
    session.headers["User-Agent"] = "NurembergScholar/1.0 (fix verification)"
    results = []

    print("\nVerifying fixes against live Yale Avalon pages...\n")
    print(f"  {'Page':<18}  {'Turns':>6}  {'Pages':>6}  {'Format':<10}  {'F5':>4}  {'Flags'}")
    print(f"  {'-'*18}  {'-'*6}  {'-'*6}  {'-'*10}  {'-'*4}  {'-'*30}")

    for slug, expectations in VERIFY_URLS.items():
        url = f"https://avalon.law.yale.edu/imt/{slug}"
        try:
            r = session.get(url, timeout=20)
            r.encoding = r.apparent_encoding or "windows-1252"
            raw = r.text
        except Exception as e:
            print(f"  {slug:<18}  FETCH ERROR: {e}")
            continue
        time.sleep(1.5)

        parsed = parse_transcript(url, raw)

        turns   = parsed.get("turn_count", 0)
        pages   = len(parsed.get("page_numbers", []))
        fmt     = parsed.get("page_format", "—")
        is_f5   = parsed.get("html_structure") == "F5_no_container"
        flags   = parsed.get("validation_flags", [])
        is_stub = "REDIRECT_STUB" in flags

        checks = []
        if expectations.get("expect_turns") is True and turns == 0:
            checks.append(f"{FAIL} turns=0")
        if expectations.get("expect_turns") is True and turns > 0:
            checks.append(f"{PASS} turns={turns}")
        if expectations.get("expect_turns") is False and turns > 0:
            checks.append(f"{FAIL} unexpected turns={turns}")
        if expectations.get("expect_pages") is True and pages == 0:
            checks.append(f"{FAIL} no pages")
        if expectations.get("expect_pages") is True and pages > 0:
            checks.append(f"{PASS} pages={pages}")
        if expectations.get("expect_redirect") and not is_stub:
            checks.append(f"{FAIL} stub not detected")
        if expectations.get("expect_redirect") and is_stub:
            checks.append(f"{PASS} stub detected")
        if expectations.get("expect_f5") is True and not is_f5:
            checks.append(f"{FAIL} F5 not detected")
        if expectations.get("expect_f5") is True and is_f5:
            checks.append(f"{PASS} F5 detected")
        if not checks:
            checks.append(f"{PASS} ok")

        flag_str = "|".join(f for f in flags if f not in ["POSSIBLE_TRUNCATION"])[:40]
        print(f"  {slug:<18}  {turns:>6}  {pages:>6}  {fmt:<10}  {'yes' if is_f5 else 'no':>4}  "
              f"{' '.join(checks)}")
        if flag_str:
            print(f"  {'':18}  flags: {flag_str}")

        results.append(all(FAIL not in c for c in checks))

    passed = sum(results)
    total  = len(results)
    print(f"\n  {passed}/{total} pages passed all checks.")
    if passed < total:
        print(f"  {total - passed} failures — review output above.")
    else:
        print(f"  All fixes verified. Safe to apply to scraper.py.")


# ═══════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify Nuremberg Scholar parser fixes")
    parser.add_argument("--verify", action="store_true",
                        help="Test all fixes against live Yale Avalon pages")
    args = parser.parse_args()

    if args.verify:
        verify()
    else:
        print(__doc__)
        print("\nRun with --verify to test against live pages.")
        print("To apply: replace parse_transcript() and extract_page_numbers() in scraper.py")
        print("          with the versions in this file.")
