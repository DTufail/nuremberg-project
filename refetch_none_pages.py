"""
refetch_none_pages.py
=====================
Re-fetches the 17 sessions with NO_PAGE_NUMBERS and the 2 sessions with
NO_SPEAKER_TURNS, applies the fixed parser (with all 4 fixes from fixes.py),
and overwrites the JSON files in output/sessions/.

Also stores raw_html in the JSON so future --reparse works.

Usage:
  python refetch_none_pages.py              # re-fetch and fix all 19 problem sessions
  python refetch_none_pages.py --dry-run    # print what would be fetched, no writes
  python refetch_none_pages.py --all        # re-fetch ALL 219 sessions (stores raw_html)

After this runs, re-run audit_output.py to confirm the fixes.
"""

import re
import html
import json
import time
import hashlib
import argparse
from pathlib import Path
from datetime import datetime, timezone
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

# ── Config ─────────────────────────────────────────────────────────────────

OUTPUT_DIR  = Path("output")
DELAY       = 1.5
TIMEOUT     = 30
USER_AGENT  = "NurembergScholar/1.0 (academic research)"
BASE_URL    = "https://avalon.law.yale.edu"
IMT_BASE    = f"{BASE_URL}/imt"

SESSION_URL_RE = re.compile(r'/imt/(\d{2})-(\d{2})-(\d{2})\.asp$')

# The 19 problem sessions identified by audit_output.py
PROBLEM_SLUGS = [
    # 17 NO_PAGE_NUMBERS
    "02-20-46.asp", "07-30-46.asp", "07-31-46.asp",
    "08-02-46.asp", "08-03-46.asp", "08-05-46.asp", "08-06-46.asp",
    "08-12-46.asp", "08-13-46.asp", "08-14-46.asp", "08-15-46.asp",
    "08-16-46.asp", "08-19-46.asp", "08-20-46.asp", "08-21-46.asp",
    "12-01-45.asp",
    # 1 redirect stub (will be flagged, not repaired from Yale)
    "04-08-46.asp",
    # 1 page anomaly
    "08-22-46.asp",
]

# ── Page number extractors (all 4 formats + F1 clamp) ─────────────────────

F1_MAX = 2000

def _F1(soup):
    return [int(t["name"]) for t in soup.find_all("a", attrs={"name": re.compile(r"^\d+$")})
            if int(t["name"]) <= F1_MAX]

def _F2(soup):
    results = []
    for t in soup.find_all("p", class_=lambda c: c and "PAGE" in
                           (c if isinstance(c, str) else " ".join(c)).upper()):
        txt = t.get_text(strip=True)
        if re.fullmatch(r"\d+", txt):
            results.append(int(txt))
    return results

def _F3(soup):
    results = []
    for t in soup.find_all("p"):
        if t.find_parent("table") or t.find():
            continue
        txt = t.get_text(strip=True)
        if re.fullmatch(r"\d{1,4}", txt) and len(str(t)) <= 30:
            results.append(int(txt))
    return results

def _F4(soup):
    results = []
    for t in soup.find_all("a", class_=lambda c: c and "nobold" in
                           (c if isinstance(c, str) else " ".join(c)).lower()):
        name = t.get("name", "")
        if re.fullmatch(r"p\d+", name):
            n = int(name[1:])
            if t.get_text(strip=True) == str(n):
                results.append(n)
    return results

def extract_page_numbers(soup) -> list[int]:
    priority = {"F4": 0, "F2": 1, "F1": 2, "F3": 3}
    seen: dict[int, int] = {}
    for fmt, nums in [("F1", _F1(soup)), ("F2", _F2(soup)),
                      ("F3", _F3(soup)), ("F4", _F4(soup))]:
        p = priority[fmt]
        for n in nums:
            if n not in seen or p < seen[n]:
                seen[n] = p
    return sorted(seen.keys())

def detect_format(soup) -> str:
    counts = {"F1": len(_F1(soup)), "F2": len(_F2(soup)),
              "F3": len(_F3(soup)), "F4": len(_F4(soup))}
    total = sum(counts.values())
    if total == 0:
        return "NONE"
    top = sorted(counts.items(), key=lambda x: -x[1])
    if len(top) > 1 and top[0][1] > 0 and top[1][1] / total >= 0.20:
        return "MIXED"
    return top[0][0] if top[0][1] > 0 else "NONE"

# ── Speaker extraction ─────────────────────────────────────────────────────

SPEAKER_RE_STRONG = re.compile(r'<(?:strong|b)[^>]*>([^<]+):</(?:strong|b)>', re.IGNORECASE)
SPEAKER_RE_PLAIN  = re.compile(
    r'^([A-Z][A-Z0-9\s\.\-]+(?:\s*\([^)]{1,60}\))?)\s*:\s*(.*)$', re.DOTALL)
NONSPEAKER_RE = re.compile(
    r'^(?:[A-Z\s\-]+DAY|Morning Session|Afternoon Session|Evening Session|'
    r'\d|\[|\(|Nuremberg|Volume|Previous|Next)', re.IGNORECASE)
SESSION_TYPE_RE = re.compile(r'(Morning|Afternoon|Evening)\s+Session', re.IGNORECASE)
DAY_HEADER_RE   = re.compile(
    r'([A-Z][A-Z\s\-]+DAY)\s*(?:<[Bb][Rr]>\s*\w+day,?\s*(\d{1,2}\s+\w+\s+\d{4}))?',
    re.IGNORECASE)
TITLE_STUBS = frozenset(["MR", "DR", "ER", "THE", "MS", "SIR", "COL",
                          "GEN", "LT", "CPT", "PROF"])
REDIRECT_RE = re.compile(
    r'location.*has changed|automatically be transfered|click on the link above',
    re.IGNORECASE)

def merge_stub_turns(turns):
    merged, i = [], 0
    while i < len(turns):
        t = turns[i]
        stub = t.get("speaker", "").strip().rstrip(".")
        if stub.upper() in TITLE_STUBS and i + 1 < len(turns):
            nxt = turns[i + 1]
            sep = ". " if not t["speaker"].endswith(".") else " "
            merged.append({
                "speaker":     (t["speaker"] + sep + nxt["speaker"]).strip(),
                "text":        (t.get("text", "") + " " + nxt.get("text", "")).strip(),
                "page_number": nxt.get("page_number") or t.get("page_number"),
            })
            i += 2
        else:
            merged.append(t)
            i += 1
    return merged

def decompose_nav(container):
    for t in container.find_all("table", class_="site-menu"):
        t.decompose()
    for t in container.find_all(class_=["HeaderContainer", "FooterContainer"]):
        t.decompose()
    # Nav-only tables (all links, no prose)
    for table in container.find_all("table"):
        links = table.find_all("a", href=True)
        prose = [s for s in table.stripped_strings
                 if not any(s in a.get_text() for a in links)]
        if links and not prose:
            table.decompose()

def parse_transcript(url: str, raw_html: str) -> dict:
    # Redirect stub
    if REDIRECT_RE.search(raw_html) and len(raw_html) < 5_000:
        return {
            "url": url, "raw_html": raw_html,
            "date_iso": _date(url), "error": "REDIRECT_STUB",
            "validation_flags": ["REDIRECT_STUB", "MISSING_CONTENT_CONTAINER"],
            "turns": [], "turn_count": 0, "page_numbers": [],
            "word_count": 0, "char_count": len(raw_html),
            "scrape_timestamp": datetime.now(timezone.utc).isoformat(),
            "content_hash": hashlib.md5(raw_html.encode()).hexdigest(),
        }

    soup = BeautifulSoup(raw_html, "html.parser")

    # Container: prefer text-properties, fall back to body
    container = soup.find("div", class_="text-properties") or soup.find("body")
    if container is None:
        return {"url": url, "raw_html": raw_html,
                "validation_flags": ["MISSING_CONTENT_CONTAINER"]}

    title_div = container.find("div", class_="document-title")
    title = title_div.get_text(strip=True) if title_div else None
    if title_div:
        title_div.decompose()
    decompose_nav(container)

    page_numbers = extract_page_numbers(container)
    page_format  = detect_format(container)
    container_html = str(container)

    # Speaker extraction
    turns, preamble = [], ""
    strong_matches = SPEAKER_RE_STRONG.findall(container_html)

    if strong_matches:
        parts = SPEAKER_RE_STRONG.split(container_html)
        preamble = BeautifulSoup(parts[0], "html.parser").get_text(separator=" ", strip=True)
        for i in range(1, len(parts) - 1, 2):
            speaker   = parts[i].strip()
            text_html = parts[i + 1] if (i + 1) < len(parts) else ""
            clean     = html.unescape(
                BeautifulSoup(text_html, "html.parser").get_text(separator=" ", strip=True))
            if speaker:
                turns.append({"speaker": speaker, "text": clean,
                               "page_number": page_numbers[0] if page_numbers else None})
    else:
        current_speaker, current_texts = None, []
        current_page = page_numbers[0] if page_numbers else None
        pre_parts = []

        for p in container.find_all("p"):
            raw = p.get_text(separator=" ", strip=True)
            if not raw:
                continue
            # F2 page marker
            cls = " ".join(p.get("class", []))
            if "page" in cls.lower():
                m = re.match(r'\s*(\d+)', raw)
                if m:
                    current_page = int(m.group(1))
                continue
            # F3 bare digit
            if re.fullmatch(r"\d{1,4}", raw) and not p.find() and len(str(p)) <= 30:
                if not p.find_parent("table"):
                    current_page = int(raw)
                    continue
            # F4 nobold anchor
            nb = p.find("a", class_=lambda c: c and "nobold" in
                        (c if isinstance(c, str) else " ".join(c)).lower())
            if nb and re.fullmatch(r"p\d+", nb.get("name", "")):
                current_page = int(nb["name"][1:])
                continue
            # Date watermark
            if re.match(r'^\d{1,2}\s+\w+\.?\s+\d{2}\s*$', raw):
                continue
            # Speaker line
            m = SPEAKER_RE_PLAIN.match(raw)
            if m and not NONSPEAKER_RE.match(raw):
                if current_speaker:
                    turns.append({"speaker": current_speaker,
                                  "text": " ".join(current_texts).strip(),
                                  "page_number": current_page})
                current_speaker = m.group(1).strip()
                rest = m.group(2).strip()
                current_texts = [rest] if rest else []
            else:
                if current_speaker:
                    current_texts.append(raw)
                else:
                    pre_parts.append(raw)

        if current_speaker and current_texts:
            turns.append({"speaker": current_speaker,
                          "text": " ".join(current_texts).strip(),
                          "page_number": current_page})
        preamble = " ".join(pre_parts)

    turns = merge_stub_turns(turns)

    full_text   = html.unescape(soup.get_text(separator=" "))
    session_types = sorted(set(
        m.group(1).lower() for m in SESSION_TYPE_RE.finditer(full_text)
    )) or ["full"]
    day_m       = DAY_HEADER_RE.search(full_text[:600])
    day_ordinal = day_m.group(1).strip().title() if day_m else None
    speakers    = list(dict.fromkeys(t["speaker"] for t in turns))

    inline_links = []
    for a in container.find_all("a", href=True):
        href = a["href"].strip()
        if not href.startswith("#"):
            inline_links.append({"text": a.get_text(strip=True)[:80],
                                  "href": urljoin(url, href)})

    flags = []
    if len(raw_html) < 5_000:
        flags.append("SUSPICIOUSLY_SHORT")
    if not turns:
        flags.append("NO_SPEAKER_TURNS")
    if not page_numbers:
        flags.append("NO_PAGE_NUMBERS")

    return {
        "url":              url,
        "raw_html":         raw_html,          # ← stored for future --reparse
        "title":            title,
        "date_iso":         _date(url),
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

def _date(url):
    m = SESSION_URL_RE.search(url)
    if m:
        mm, dd, yy = m.groups()
        return f"19{yy}-{mm}-{dd}"
    return None

# ── File finder ────────────────────────────────────────────────────────────

def find_json(slug: str) -> Path | None:
    for coll in ("sessions", "vol1"):
        for fp in (OUTPUT_DIR / coll).glob("*.json"):
            try:
                data = json.loads(fp.read_text(encoding="utf-8"))
                if slug in data.get("url", ""):
                    return fp
            except Exception:
                pass
    return None

def all_session_paths() -> list[tuple[Path, str]]:
    """Return (path, url) for every session JSON."""
    results = []
    for fp in sorted((OUTPUT_DIR / "sessions").glob("*.json")):
        try:
            data = json.loads(fp.read_text(encoding="utf-8"))
            url = data.get("url", "")
            if url:
                results.append((fp, url))
        except Exception:
            pass
    return results

# ── HTTP ───────────────────────────────────────────────────────────────────

_http = requests.Session()
_http.headers["User-Agent"] = USER_AGENT
_last = 0.0

def fetch(url: str) -> tuple[int, str]:
    global _last
    elapsed = time.time() - _last
    if elapsed < DELAY:
        time.sleep(DELAY - elapsed)
    try:
        r = _http.get(url, timeout=TIMEOUT)
        _last = time.time()
        if r.status_code == 200:
            r.encoding = r.apparent_encoding or "windows-1252"
            return 200, r.text
        return r.status_code, ""
    except Exception as e:
        return 0, str(e)

# ── Main ───────────────────────────────────────────────────────────────────

def run(slugs: list[str], dry_run: bool):
    print(f"\n{'Dry run — ' if dry_run else ''}Re-fetching {len(slugs)} sessions\n")

    ok = 0
    failed = []

    for i, slug in enumerate(slugs, 1):
        url = f"{IMT_BASE}/{slug}"
        fp  = find_json(slug)

        if fp is None:
            print(f"  [{i:>2}] {slug:<20}  ⚠️  JSON not found — will create new file")
            fp = OUTPUT_DIR / "sessions" / slug.replace("-", "_").replace(".asp", ".json")

        if dry_run:
            print(f"  [{i:>2}] {slug:<20}  would fetch {url}")
            print(f"        → save to {fp}")
            continue

        print(f"  [{i:>2}/{len(slugs)}] {slug:<20}  fetching...", end=" ", flush=True)
        status, raw = fetch(url)

        if status != 200:
            print(f"❌ HTTP {status}")
            failed.append((slug, status))
            continue

        parsed = parse_transcript(url, raw)
        turns  = parsed.get("turn_count", 0)
        pages  = len(parsed.get("page_numbers", []))
        fmt    = parsed.get("page_format", "?")
        flags  = [f for f in parsed.get("validation_flags", [])
                  if f not in ("POSSIBLE_TRUNCATION",)]

        fp.write_text(json.dumps(parsed, ensure_ascii=False, indent=2), encoding="utf-8")
        status_str = "✅" if not flags else f"⚠️  {flags}"
        print(f"turns={turns:<4} pages={pages:<4} fmt={fmt:<6} {status_str}")
        ok += 1

    print(f"\n{'─'*50}")
    print(f"Done: {ok} updated, {len(failed)} failed")
    if failed:
        for slug, code in failed:
            print(f"  ❌ {slug}  HTTP {code}")
    if not dry_run:
        print(f"\nRe-run audit_output.py to verify fixes.")


def main():
    parser = argparse.ArgumentParser(description="Re-fetch and fix problem sessions")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be fetched without doing it")
    parser.add_argument("--all", action="store_true",
                        help="Re-fetch ALL sessions (stores raw_html in every JSON)")
    args = parser.parse_args()

    if args.all:
        paths = all_session_paths()
        slugs = [Path(url).name for _, url in paths]
        print(f"Re-fetching all {len(slugs)} sessions (this stores raw_html in every JSON)")
    else:
        slugs = PROBLEM_SLUGS

    run(slugs, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
