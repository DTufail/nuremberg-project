"""
harvard_april8.py
=================
Fetches the 8 April 1946 IMT session from Harvard Law Nuremberg Trials Project
and replaces the Yale redirect stub at output/sessions/imt_04-08-46_asp.json.

Harvard serves transcript pages as:
  <div class="page" data-seq="N" data-page="N" data-date="YYYY-MM-DD">
The ?date= parameter jumps directly to the window containing that date,
revealing the starting seq number without hardcoding it.

Usage:
    python harvard_april8.py
    python harvard_april8.py --dry-run
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

BASE_URL    = "https://nuremberg.law.harvard.edu"
TRANSCRIPT  = "/transcripts/7-transcript-for-imt-trial-of-major-war-criminals"
OUTPUT_FILE = Path("output/sessions/imt_04-08-46_asp.json")
TARGET_DATE = "1946-04-08"
DELAY       = 2.0
TIMEOUT     = 30
USER_AGENT  = "NurembergScholar/1.0 (academic research; gap patch)"

# ── HTTP ──────────────────────────────────────────────────────────────────────

_session = requests.Session()
_session.headers["User-Agent"] = USER_AGENT
_last = 0.0


def fetch_url(url: str) -> tuple[int, str]:
    global _last
    elapsed = time.time() - _last
    if elapsed < DELAY:
        time.sleep(DELAY - elapsed)
    try:
        r = _session.get(url, timeout=TIMEOUT)
        _last = time.time()
        return r.status_code, r.text if r.status_code == 200 else ""
    except Exception as e:
        print(f"  ERROR: {e}")
        return 0, ""


def fetch_date(date_iso: str) -> tuple[int, str]:
    """Fetch the window that contains the given date using ?date= parameter."""
    url = f"{BASE_URL}{TRANSCRIPT}?date={date_iso}"
    return fetch_url(url)


def fetch_seq(seq: int) -> tuple[int, str]:
    url = f"{BASE_URL}{TRANSCRIPT}?seq={seq}"
    return fetch_url(url)


# ── Page parser ───────────────────────────────────────────────────────────────

def parse_page_divs(html: str) -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")
    results = []
    for div in soup.find_all("div", class_="page"):
        seq_val  = div.get("data-seq")
        page_val = div.get("data-page")
        date_val = div.get("data-date")
        if not seq_val or not date_val:
            continue
        paragraphs = []
        for p in div.find_all("p"):
            text = p.get_text(separator=" ", strip=True)
            if text:
                paragraphs.append(text)
        results.append({
            "seq":         int(seq_val),
            "date_iso":    date_val.strip(),
            "page_number": int(page_val) if page_val else None,
            "paragraphs":  paragraphs,
            "raw_html":    str(div),
        })
    return results


def get_window_bounds(html: str) -> tuple[int, int]:
    soup = BeautifulSoup(html, "html.parser")
    tt = soup.find("div", class_="transcript-text")
    if tt:
        return int(tt.get("data-from-seq", 0)), int(tt.get("data-to-seq", 0))
    return 0, 0


# ── Speaker turn builder ──────────────────────────────────────────────────────

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

def build_doc(pages: list[dict]) -> dict:
    pages        = sorted(pages, key=lambda p: p["seq"])
    page_numbers = sorted(set(p["page_number"] for p in pages if p["page_number"]))
    turns        = build_turns(pages)
    speakers     = list(dict.fromkeys(t["speaker"] for t in turns))
    full_text    = "\n".join("\n".join(p["paragraphs"]) for p in pages)
    raw_html     = "\n<!-- PAGE BREAK -->\n".join(p["raw_html"] for p in pages)
    url          = f"{BASE_URL}{TRANSCRIPT}?seq={pages[0]['seq']}"

    flags = []
    if not turns:        flags.append("NO_SPEAKER_TURNS")
    if not page_numbers: flags.append("NO_PAGE_NUMBERS")

    return {
        "url":              url,
        "source":           "harvard_law_patch",
        "raw_html":         raw_html,
        "date_iso":         TARGET_DATE,
        "date_source":      "harvard_data_attr",
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

def run(dry_run: bool):
    print(f"\n{'[DRY RUN] ' if dry_run else ''}Harvard patch — {TARGET_DATE}")

    # Step 1: discover the starting seq via ?date= parameter
    print(f"\nStep 1: discovering start seq for {TARGET_DATE} ...")
    if dry_run:
        print(f"  Would GET {BASE_URL}{TRANSCRIPT}?date={TARGET_DATE}")
        print("  Would then walk forward collecting pages")
        return

    status, html = fetch_date(TARGET_DATE)
    if status != 200:
        print(f"  Date lookup failed (HTTP {status}) — aborting")
        return

    from_seq, to_seq = get_window_bounds(html)
    page_divs        = parse_page_divs(html)
    dates_found      = sorted(set(p["date_iso"] for p in page_divs))
    print(f"  window={from_seq}–{to_seq}  pages={len(page_divs)}  dates={dates_found}")

    if TARGET_DATE not in dates_found:
        print(f"  {TARGET_DATE} not in first window — check date parameter")
        return

    # Step 2: walk forward collecting all pages for TARGET_DATE
    collected:   dict[int, dict] = {}
    seq                          = from_seq
    consecutive_fails            = 0
    seen_seqs                    = set()

    # Process the first window already fetched
    def absorb_window(divs):
        found = past = False
        for page in divs:
            if page["seq"] in seen_seqs:
                continue
            seen_seqs.add(page["seq"])
            if page["date_iso"] == TARGET_DATE:
                collected[page["seq"]] = page
                print(f"    ✓ seq={page['seq']}  date={page['date_iso']}  "
                      f"page={page['page_number']}  paras={len(page['paragraphs'])}")
                found = True
            elif collected and page["date_iso"] != TARGET_DATE:
                past = True
        return found, past

    found_any, past = absorb_window(page_divs)
    if past and not found_any:
        print("  Already past target in first window — nothing collected")
        return

    seq = to_seq + 1

    while True:
        print(f"  Fetching seq={seq} ...", end=" ", flush=True)
        status, html = fetch_seq(seq)

        if status != 200:
            print(f"HTTP {status}")
            consecutive_fails += 1
            if consecutive_fails >= 3:
                print("3 consecutive failures — aborting")
                break
            seq += 20
            continue

        consecutive_fails        = 0
        from_seq, to_seq         = get_window_bounds(html)
        page_divs                = parse_page_divs(html)
        dates_in_window          = sorted(set(p["date_iso"] for p in page_divs))
        print(f"window={from_seq}–{to_seq}  pages={len(page_divs)}  dates={dates_in_window}")

        found_here, past_here = absorb_window(page_divs)

        if past_here and not found_here:
            print("  Moved past target date — stopping")
            break

        if not to_seq or to_seq <= seq:
            seq += 20
        else:
            seq = to_seq + 1

    # Step 3: write output
    print(f"\n{'─'*55}")
    if not collected:
        print("No pages collected for target date — nothing written")
        return

    pages = list(collected.values())
    doc   = build_doc(pages)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n  ✅ {TARGET_DATE} → {OUTPUT_FILE}")
    print(f"     pages={len(pages)}  turns={doc['turn_count']}  "
          f"words={doc['word_count']:,}  speakers={doc['speaker_count']}  "
          f"flags={doc['validation_flags']}")
    print("\nDone.")


def main():
    p = argparse.ArgumentParser(
        description="Fetch IMT 08 April 1946 session from Harvard Law (replaces Yale stub)"
    )
    p.add_argument("--dry-run", action="store_true", help="Print plan without fetching")
    args = p.parse_args()
    run(args.dry_run)


if __name__ == "__main__":
    main()
