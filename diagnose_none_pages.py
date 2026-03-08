"""
diagnose_none_pages.py
======================
Reads the scraped JSON files for the 17 NONE sessions and prints exactly
what the raw HTML looks like around page numbers and speaker tags.

Run from your project directory:
  python diagnose_none_pages.py
"""

import re
import json
from pathlib import Path
from bs4 import BeautifulSoup

OUTPUT_DIR = Path("output")

NONE_URLS = [
    "02-20-46.asp", "04-08-46.asp", "07-30-46.asp", "07-31-46.asp",
    "08-02-46.asp", "08-03-46.asp", "08-05-46.asp", "08-06-46.asp",
    "08-12-46.asp", "08-13-46.asp", "08-14-46.asp", "08-15-46.asp",
    "08-16-46.asp", "08-19-46.asp", "08-20-46.asp", "08-21-46.asp",
    "12-01-45.asp",
]


def url_to_path(slug: str) -> Path | None:
    """Find the JSON file for a given session slug."""
    # Try sessions/ and vol1/
    for coll in ("sessions", "vol1"):
        for fp in (OUTPUT_DIR / coll).glob("*.json"):
            if slug.replace("-", "_").replace(".asp", "") in fp.stem:
                return fp
            # also try exact match on url field
    # Fallback: search by url field
    for coll in ("sessions", "vol1"):
        for fp in (OUTPUT_DIR / coll).glob("*.json"):
            try:
                data = json.loads(fp.read_text(encoding="utf-8"))
                if slug in data.get("url", ""):
                    return fp
            except Exception:
                pass
    return None


def diagnose(slug: str):
    fp = url_to_path(slug)
    if fp is None:
        print(f"  [{slug}] JSON file not found in output/sessions/ or output/vol1/")
        return

    data = json.loads(fp.read_text(encoding="utf-8"))
    raw_html = data.get("raw_html") or data.get("html") or ""
    url = data.get("url", slug)

    print(f"\n{'='*70}")
    print(f"  {slug}")
    print(f"  {url}")
    print(f"  turns={data.get('turn_count',0)}  pages={len(data.get('page_numbers',[]))}  words={data.get('word_count',0)}")
    print(f"  flags: {data.get('validation_flags', [])}")
    print(f"{'='*70}")

    if not raw_html:
        print(f"  ⚠️  No raw_html stored in JSON — scraper did not save raw HTML.")
        print(f"  Current data keys: {list(data.keys())}")
        print(f"  You need to add  doc['raw_html'] = page_html  in scraper.py before saving.")

        # Show what we DO have
        turns = data.get("turns", [])
        page_numbers = data.get("page_numbers", [])
        print(f"\n  Stored turn_count: {data.get('turn_count', 0)}")
        print(f"  Stored page_numbers: {page_numbers[:10]}")
        if turns:
            print(f"\n  First 3 stored turns:")
            for t in turns[:3]:
                print(f"    [{t.get('speaker','?')}]: {str(t.get('text',''))[:80]}")
        return

    soup = BeautifulSoup(raw_html, "html.parser")

    # 1. Container check
    container = soup.find("div", class_="text-properties")
    print(f"\n  has text-properties div: {container is not None}")
    if container:
        print(f"  container tag count (p): {len(container.find_all('p'))}")

    # 2. Raw HTML around potential page numbers — show first 3000 chars of body content
    body = soup.find("body")
    body_html = str(body)[:4000] if body else raw_html[:4000]

    # Find all <p> tags and show short ones (likely page numbers)
    all_p = soup.find_all("p")
    print(f"\n  Total <p> tags: {len(all_p)}")

    # Short <p> tags — potential page numbers
    short_p = [(str(p), p.get_text(strip=True)) for p in all_p
               if 0 < len(p.get_text(strip=True)) <= 6 and len(str(p)) <= 60
               and not p.find_parent("table")]
    print(f"  Short <p> (≤6 chars, candidate page nums): {len(short_p)}")
    for raw_tag, text in short_p[:10]:
        print(f"    raw: {raw_tag!r}  text: {text!r}")

    # 3. Named anchors
    anchors = soup.find_all("a", attrs={"name": re.compile(r"^\d+$")})
    print(f"\n  <a name=N> numeric anchors: {len(anchors)}")
    for a in anchors[:5]:
        print(f"    {str(a)!r}")

    # 4. CLASS=PAGE
    page_class = soup.find_all("p", class_=lambda c: c and "PAGE" in
                               (c if isinstance(c, str) else " ".join(c)).upper())
    print(f"  <p CLASS=PAGE>: {len(page_class)}")

    # 5. Nobold anchors
    nobold = soup.find_all("a", class_=lambda c: c and "nobold" in
                           (c if isinstance(c, str) else " ".join(c)).lower())
    print(f"  <a class=nobold>: {len(nobold)}")
    for n in nobold[:3]:
        print(f"    {str(n)!r}")

    # 6. Show the first 800 chars of the raw HTML body to see structure
    print(f"\n  First 800 chars of raw HTML (body region):")
    body_start = raw_html.find("<body")
    if body_start == -1:
        body_start = 0
    print("  " + raw_html[body_start:body_start + 800].replace("\n", "\\n").replace("\r", ""))

    # 7. Speaker-line check — find lines matching ALLCAPS: pattern
    lines = raw_html.split("\n")
    speaker_lines = [l.strip() for l in lines
                     if re.match(r'^[A-Z][A-Z0-9\s\.\-]+:', l.strip()) and len(l.strip()) < 200]
    print(f"\n  Lines matching SPEAKER: pattern: {len(speaker_lines)}")
    for line in speaker_lines[:5]:
        print(f"    {line[:100]!r}")

    # 8. If has text-properties div, show the raw HTML of first 600 chars
    if container:
        container_html = str(container)
        print(f"\n  First 600 chars of text-properties container:")
        print("  " + container_html[:600].replace("\n", "\\n"))


def main():
    print("Diagnosing 17 NONE-page sessions from scraped JSON files")
    print(f"Output directory: {OUTPUT_DIR.absolute()}")

    found = 0
    for slug in NONE_URLS:
        fp = url_to_path(slug)
        if fp:
            found += 1

    print(f"JSON files found: {found}/{len(NONE_URLS)}")

    if found == 0:
        print("\n⚠️  No JSON files found.")
        print("Check that OUTPUT_DIR is correct and that scraper.py stored raw_html.")
        print(f"Looking in: {OUTPUT_DIR.absolute()}/sessions/")
        all_sessions = list((OUTPUT_DIR / "sessions").glob("*.json")) if (OUTPUT_DIR / "sessions").exists() else []
        print(f"Files in sessions/: {len(all_sessions)}")
        if all_sessions:
            print(f"Sample filenames: {[f.name for f in all_sessions[:5]]}")
        return

    # Diagnose first 4 — enough to see the pattern
    for slug in ["12-01-45.asp", "02-20-46.asp", "08-02-46.asp", "08-20-46.asp"]:
        diagnose(slug)


if __name__ == "__main__":
    main()
