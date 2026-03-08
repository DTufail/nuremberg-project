"""
rescrape_missing.py
===================
Re-fetches and re-parses two session pages that were scraped incorrectly:
  - https://avalon.law.yale.edu/imt/08-20-46.asp  (207th Day)
  - https://avalon.law.yale.edu/imt/08-21-46.asp  (208th Day)

These pages have an unusual layout: the text-properties div only contains nav
chrome and the document title, while all <p> transcript paragraphs live as
body-level siblings *outside* the container. scraper.parse_transcript() only
scans inside the container, so it finds zero turns.

This script handles that layout with a fallback parser and overwrites the
broken JSON files in output/sessions/ with corrected data.

Usage:
    python rescrape_missing.py
"""

import re
import html
import json
import hashlib
import sys
from datetime import datetime, timezone
from pathlib import Path

from bs4 import BeautifulSoup

# Re-use constants and utilities from the main scraper — no duplication
import scraper

TARGETS = [
    "https://avalon.law.yale.edu/imt/08-20-46.asp",
    "https://avalon.law.yale.edu/imt/08-21-46.asp",
]

OUTPUT_DIR = Path("output/sessions")


def parse_transcript_body_fallback(url: str, raw_html: str) -> dict:
    """
    Parser for pages where <p> transcript content sits at body level as
    siblings of div.text-properties rather than inside it.

    Uses the same regex constants and page-number extractors as scraper.py.
    """
    soup = BeautifulSoup(raw_html, "html.parser")

    # Title still lives inside the container
    container = soup.find("div", class_="text-properties")
    title_div = container.find("div", class_="document-title") if container else None
    title = title_div.get_text(strip=True) if title_div else None

    # Page numbers are also at body level — run extractor on full soup
    # (F3 format: bare <p>301 </p> paragraphs)
    page_numbers = scraper.extract_page_numbers(soup)
    page_format  = scraper.detect_page_format(soup)

    # Collect all <p> tags that are NOT inside a nav table or the header/footer
    body = soup.body or soup
    turns = []
    current_speaker = None
    current_texts = []
    current_page = page_numbers[0] if page_numbers else None
    pre_parts = []

    for p_tag in body.find_all("p"):
        # Skip anything inside site-menu tables or header/footer chrome
        if p_tag.find_parent("table", class_="site-menu"):
            continue
        if p_tag.find_parent(class_=["HeaderContainer", "FooterContainer"]):
            continue

        raw_text = p_tag.get_text(separator=" ", strip=True)
        if not raw_text:
            continue

        # F2: CLASS=PAGE paragraph — update current page and skip
        page_attr = " ".join(p_tag.get("class", []))
        if "page" in page_attr.lower():
            digits = re.match(r'\s*(\d+)', raw_text)
            if digits:
                current_page = int(digits.group(1))
            continue

        # F3: bare digit paragraph — update current page and skip
        if re.fullmatch(r"\d{1,4}", raw_text) and not p_tag.find() and len(str(p_tag)) <= 30:
            if not p_tag.find_parent("table"):
                current_page = int(raw_text)
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
        if re.match(r'^\d{1,2}\s+\w+\.?\s+\d{2}\s*$', raw_text):
            continue

        m = scraper.SPEAKER_RE_PLAIN.match(raw_text)
        if m and not scraper.NONSPEAKER_RE.match(raw_text):
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
                current_texts.append(raw_text)
            else:
                pre_parts.append(raw_text)

    if current_speaker and (current_texts or current_speaker):
        turns.append({
            "speaker": current_speaker,
            "text": " ".join(current_texts).strip(),
            "page_number": current_page,
        })

    preamble = " ".join(pre_parts)
    full_text = html.unescape(soup.get_text(separator=" "))

    session_types = sorted(set(
        match.group(1).lower()
        for match in scraper.SESSION_TYPE_RE.finditer(full_text)
    )) or ["full"]

    day_match = scraper.DAY_HEADER_RE.search(full_text[:600])
    day_ordinal = day_match.group(1).strip().title() if day_match else None

    speakers = list(dict.fromkeys(t["speaker"] for t in turns))

    # Inline links — from the full body, excluding nav
    inline_links = []
    seen_hrefs = set()
    for a in body.find_all("a", href=True):
        if a.find_parent("table", class_="site-menu"):
            continue
        href = a["href"].strip()
        if href.startswith("#") or href in seen_hrefs:
            continue
        from urllib.parse import urljoin
        abs_href = urljoin(url, href)
        if scraper.is_nav_link(abs_href):
            continue
        seen_hrefs.add(href)
        inline_links.append({"text": a.get_text(strip=True)[:80], "href": abs_href})

    flags = scraper.validate(url, raw_html, turns, page_numbers)

    return {
        "url":              url,
        "title":            title,
        "date_iso":         scraper.parse_date_from_url(url),
        "date_source":      "url",
        "day_ordinal":      day_ordinal,
        "session_types":    session_types,
        "page_start":       min(page_numbers) if page_numbers else None,
        "page_end":         max(page_numbers) if page_numbers else None,
        "page_numbers":     page_numbers,
        "page_format":      page_format,
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


def rescrape(url: str) -> bool:
    filename = scraper.url_to_filename(url)
    out_path = OUTPUT_DIR / filename

    print(f"\n{'='*60}")
    print(f"Fetching: {url}")

    status, raw_html = scraper.fetch(url)
    if status != 200:
        print(f"  ERROR: HTTP {status} — aborting for this file")
        return False

    print(f"  Fetched OK ({len(raw_html):,} chars)")

    # First try the standard parser
    doc = scraper.parse_transcript(url, raw_html)

    # If it found no turns, the <p> tags are probably at body level — use fallback
    if not doc.get("turns"):
        print("  Standard parser found 0 turns — trying body-level fallback parser")
        doc = parse_transcript_body_fallback(url, raw_html)

    doc["raw_html"] = raw_html

    flags = doc.get("validation_flags", [])
    turns = doc.get("turn_count", 0)
    pages = len(doc.get("page_numbers", []))
    print(f"  Turns: {turns}  |  Pages: {pages}  |  Flags: {flags}")

    if "NO_SPEAKER_TURNS" in flags:
        print("  WARNING: still no speaker turns after re-scrape")

    out_path.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  Saved → {out_path}")
    return True


def main():
    ok = 0
    for url in TARGETS:
        if rescrape(url):
            ok += 1

    print(f"\nDone: {ok}/{len(TARGETS)} files re-scraped successfully.")
    if ok < len(TARGETS):
        sys.exit(1)


if __name__ == "__main__":
    main()
