"""
harvard_patch.py
================
Scrapes the Harvard Law Nuremberg Trials Project for the two sessions
missing from Yale Avalon: March 6 and March 7, 1946.

HTML structure (confirmed from live pages):
  - Each transcript page is a <div class="page" data-seq="N" data-page="N" data-date="YYYY-MM-DD">
  - Speaker markup: <span class="speaker">NAME:</span> followed by text in the same <p>
  - Harvard returns ~20 pages per HTTP request (data-from-seq / data-to-seq window)
  - We request seq=5416 (first Mar 6 page) and walk forward in windows of 20

Usage:
    pip install requests beautifulsoup4
    python harvard_patch.py
    python harvard_patch.py --dry-run
    python harvard_patch.py --start 5416
"""

import re
import json
import time
import hashlib
import argparse
from pathlib import Path
from datetime import datetime, timezone

import requests
from bs4 import BeautifulSoup

# ── Config ────────────────────────────────────────────────────────────────────

BASE_URL   = "https://nuremberg.law.harvard.edu"
TRANSCRIPT = "/transcripts/7-transcript-for-imt-trial-of-major-war-criminals"
OUTPUT_DIR = Path("output/sessions")
DELAY      = 2.0
TIMEOUT    = 30
USER_AGENT = "NurembergScholar/1.0 (academic research; gap patch)"

TARGET_DATES = {"1946-03-06", "1946-03-07"}

DEFAULT_START_SEQ = 5416   # first confirmed Mar 6 page
MAX_SEQ           = 5600   # safety ceiling

# ── HTTP ──────────────────────────────────────────────────────────────────────

_session = requests.Session()
_session.headers["User-Agent"] = USER_AGENT
_last = 0.0

def fetch(seq: int) -> tuple[int, str]:
    global _last
    elapsed = time.time() - _last
    if elapsed < DELAY:
        time.sleep(DELAY - elapsed)
    url = f"{BASE_URL}{TRANSCRIPT}?seq={seq}"
    try:
        r = _session.get(url, timeout=TIMEOUT)
        _last = time.time()
        return r.status_code, r.text if r.status_code == 200 else ""
    except Exception as e:
        print(f"  ERROR: {e}")
        return 0, ""

# ── Page parser ───────────────────────────────────────────────────────────────

def parse_page_divs(html: str) -> list[dict]:
    """
    Extract all transcript pages from one Harvard HTTP response.
    Each page is: <div class="page" data-seq="N" data-page="N" data-date="YYYY-MM-DD">
    Speaker markup: <span class="speaker">NAME:</span> rest of text in same <p>
    """
    soup = BeautifulSoup(html, "html.parser")
    results = []

    for div in soup.find_all("div", class_="page"):
        seq_val  = div.get("data-seq")
        page_val = div.get("data-page")
        date_val = div.get("data-date")

        if not seq_val or not date_val:
            continue

        seq      = int(seq_val)
        page_num = int(page_val) if page_val else None
        date_iso = date_val.strip()   # already YYYY-MM-DD

        paragraphs = []
        for p in div.find_all("p"):
            text = p.get_text(separator=" ", strip=True)
            if text:
                paragraphs.append(text)

        results.append({
            "seq":         seq,
            "date_iso":    date_iso,
            "page_number": page_num,
            "paragraphs":  paragraphs,
            "raw_html":    str(div),
        })

    return results

def get_window_bounds(html: str) -> tuple[int, int]:
    """Read data-from-seq and data-to-seq from the transcript-text div."""
    soup = BeautifulSoup(html, "html.parser")
    tt = soup.find("div", class_="transcript-text")
    if tt:
        return int(tt.get("data-from-seq", 0)), int(tt.get("data-to-seq", 0))
    return 0, 0

# ── Speaker turn builder ──────────────────────────────────────────────────────

# Harvard text format: "DR. KRANZBUEHLER: Some text..." or "THE PRESIDENT: ..."
# The span wraps just the name, but get_text() merges it all, so we match
# on the colon separator at the boundary.
SPEAKER_RE = re.compile(
    r'^((?:(?:DR|MR|MRS|SIR|THE|GEN|ADM|COL|MAJ|LT|PROF|SGT|CPT)[\s\.]*)?'
    r'[A-Z][A-Z\s\.\-\(\)]{1,60}?):\s+(.*)',
    re.DOTALL
)

def build_turns(pages: list[dict]) -> list[dict]:
    turns       = []
    cur_speaker = None
    cur_lines   = []
    cur_page    = None

    for page in pages:
        cur_page = page["page_number"] or cur_page
        for para in page["paragraphs"]:
            para = para.strip()
            if not para:
                continue
            m = SPEAKER_RE.match(para)
            if m:
                if cur_speaker and cur_lines:
                    turns.append({
                        "speaker":     cur_speaker,
                        "text":        " ".join(cur_lines).strip(),
                        "page_number": cur_page,
                    })
                cur_speaker = m.group(1).strip()
                rest        = m.group(2).strip()
                cur_lines   = [rest] if rest else []
            else:
                if cur_speaker:
                    cur_lines.append(para)

    if cur_speaker and cur_lines:
        turns.append({
            "speaker":     cur_speaker,
            "text":        " ".join(cur_lines).strip(),
            "page_number": cur_page,
        })
    return turns

# ── Doc builder ───────────────────────────────────────────────────────────────

def build_doc(date_iso: str, pages: list[dict]) -> dict:
    pages        = sorted(pages, key=lambda p: p["seq"])
    page_numbers = sorted(set(p["page_number"] for p in pages if p["page_number"]))
    turns        = build_turns(pages)
    speakers     = list(dict.fromkeys(t["speaker"] for t in turns))
    full_text    = "\n".join("\n".join(p["paragraphs"]) for p in pages)
    raw_html     = "\n<!-- PAGE BREAK -->\n".join(p["raw_html"] for p in pages)
    slug         = date_iso[5:7] + "-" + date_iso[8:10] + "-" + date_iso[2:4]
    url          = f"{BASE_URL}{TRANSCRIPT}?seq={pages[0]['seq']}"

    flags = []
    if not turns:        flags.append("NO_SPEAKER_TURNS")
    if not page_numbers: flags.append("NO_PAGE_NUMBERS")

    return {
        "url":              url,
        "source":           "harvard_law_patch",
        "raw_html":         raw_html,
        "date_iso":         date_iso,
        "date_source":      "harvard_data_attr",
        "slug":             slug,
        "page_start":       min(page_numbers) if page_numbers else None,
        "page_end":         max(page_numbers) if page_numbers else None,
        "page_numbers":     page_numbers,
        "page_format":      "HARVARD",
        "seq_start":        pages[0]["seq"],
        "seq_end":          pages[-1]["seq"],
        "speakers":         speakers,
        "speaker_count":    len(speakers),
        "turns":            turns,
        "turn_count":       len(turns),
        "word_count":       len(full_text.split()),
        "char_count":       len(raw_html),
        "scrape_timestamp": datetime.now(timezone.utc).isoformat(),
        "validation_flags": flags,
        "content_hash":     hashlib.md5(full_text.encode()).hexdigest(),
    }

# ── Main ──────────────────────────────────────────────────────────────────────

def run(start_seq: int, dry_run: bool):
    print(f"\n{'[DRY RUN] ' if dry_run else ''}Harvard patch — Mar 6–7 1946")
    print(f"Start seq={start_seq}  targets={sorted(TARGET_DATES)}\n")

    pages_by_date: dict[str, list[dict]] = {}
    seq               = start_seq
    consecutive_fails = 0
    seen_seqs         = set()

    while seq <= MAX_SEQ:
        if dry_run:
            print(f"  Would fetch seq={seq}")
            if seq >= start_seq + 40:
                print("  ... (dry run truncated)")
                break
            seq += 20
            continue

        print(f"  Fetching seq={seq} ...", end=" ", flush=True)
        status, html = fetch(seq)

        if status != 200:
            print(f"HTTP {status}")
            consecutive_fails += 1
            if consecutive_fails >= 3:
                print("3 consecutive failures — aborting")
                break
            seq += 20
            continue

        consecutive_fails = 0
        from_seq, to_seq  = get_window_bounds(html)
        page_divs         = parse_page_divs(html)
        dates_in_window   = sorted(set(p["date_iso"] for p in page_divs))
        print(f"window={from_seq}–{to_seq}  pages={len(page_divs)}  dates={dates_in_window}")

        found_target = False
        past_target  = False

        for page in page_divs:
            if page["seq"] in seen_seqs:
                continue
            seen_seqs.add(page["seq"])

            d = page["date_iso"]
            if d in TARGET_DATES:
                found_target = True
                pages_by_date.setdefault(d, []).append(page)
                print(f"    ✓ seq={page['seq']}  date={d}  "
                      f"page={page['page_number']}  paras={len(page['paragraphs'])}")
            elif pages_by_date and d not in TARGET_DATES:
                past_target = True

        if past_target and not found_target:
            print("  Moved past target dates — stopping")
            break

        # Advance to next window
        next_seq = (to_seq + 1) if (to_seq and to_seq > seq) else (seq + 20)
        seq = next_seq

    # ── Write output ──────────────────────────────────────────────────────────

    if dry_run:
        return

    print(f"\n{'─'*55}")
    print(f"Collected: { {d: len(p) for d, p in pages_by_date.items()} }")

    if not pages_by_date:
        print("No target pages collected — check start seq or date filter")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for date_iso, pages in sorted(pages_by_date.items()):
        doc = build_doc(date_iso, pages)
        fp  = OUTPUT_DIR / f"{doc['slug']}.json"
        fp.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n  ✅ {date_iso} → {fp}")
        print(f"     pages={len(pages)}  turns={doc['turn_count']}  "
              f"words={doc['word_count']:,}  speakers={doc['speaker_count']}  "
              f"flags={doc['validation_flags']}")

    print("\nDone. Run audit_output.py --sessions to verify.")


def main():
    p = argparse.ArgumentParser(description="Patch missing Yale sessions from Harvard Law")
    p.add_argument("--dry-run", action="store_true", help="Print without fetching")
    p.add_argument("--start",   type=int, default=DEFAULT_START_SEQ,
                   help=f"Starting HLSL seq number (default {DEFAULT_START_SEQ})")
    args = p.parse_args()
    run(args.start, args.dry_run)


if __name__ == "__main__":
    main()