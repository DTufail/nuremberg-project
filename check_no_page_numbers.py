"""
check_no_page_numbers.py
========================
Report every session (and any other doc) that has no page numbers,
i.e. page_numbers == [] or page_numbers is missing.

Searches all collections: sessions, vol1, secondary, judgment, key_docs.

Usage:
    python check_no_page_numbers.py
    python check_no_page_numbers.py --dir /path/to/output
"""

import argparse
import json
from pathlib import Path

COLLECTIONS = ["sessions", "vol1", "secondary", "judgment", "key_docs"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default="output", help="Output directory (default: output)")
    args = p.parse_args()
    root = Path(args.dir)

    no_pages = []

    for coll in COLLECTIONS:
        coll_dir = root / coll
        if not coll_dir.exists():
            continue
        for fp in sorted(coll_dir.glob("*.json")):
            try:
                doc = json.loads(fp.read_text(encoding="utf-8"))
            except Exception as e:
                print(f"  [WARN] {fp}: {e}")
                continue

            page_numbers = doc.get("page_numbers")
            if not page_numbers:          # [] or None or missing
                no_pages.append({
                    "file":     fp.name,
                    "coll":     coll,
                    "date":     doc.get("date_iso", "?"),
                    "source":   doc.get("source", "?"),
                    "words":    doc.get("word_count", 0),
                    "turns":    doc.get("turn_count", 0),
                    "flags":    doc.get("validation_flags", []),
                })

    if not no_pages:
        print("✅  All docs have page numbers.")
        return

    # Group by collection
    by_coll: dict[str, list] = {}
    for item in no_pages:
        by_coll.setdefault(item["coll"], []).append(item)

    total = len(no_pages)
    print(f"\nDocs with NO page numbers: {total}\n")
    print(f"{'Collection':<12}  {'File':<40}  {'Date':<12}  {'Words':>7}  {'Turns':>5}  Flags")
    print("─" * 105)

    for coll in COLLECTIONS:
        items = by_coll.get(coll, [])
        if not items:
            continue
        print(f"\n  [{coll}]  ({len(items)} doc{'s' if len(items) != 1 else ''})")
        for it in items:
            flags = ", ".join(it["flags"]) if it["flags"] else "—"
            print(f"    {it['file']:<40}  {it['date']:<12}  "
                  f"{it['words']:>7,}  {it['turns']:>5}  {flags}")

    print(f"\n{'─'*105}")
    print(f"Total: {total} docs without page numbers across "
          f"{len(by_coll)} collection(s)")

    # Summary per collection
    print("\nSummary by collection:")
    for coll in COLLECTIONS:
        n = len(by_coll.get(coll, []))
        if n:
            print(f"  {coll:<14} {n}")


if __name__ == "__main__":
    main()
