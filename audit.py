"""
audit_output.py — Nuremberg Scholar Post-Scrape Audit
=======================================================
Reads the output/ directory produced by scraper.py and generates a
comprehensive report covering:

  1. Collection coverage   — doc counts vs expected targets
  2. Session coverage      — calendar completeness, gap detection
  3. Page number health    — format distribution, NONE docs, coverage rates
  4. Speaker quality       — unique speakers, top talkers, suspect truncations
  5. Validation flags      — breakdown of every flag type, worst offenders
  6. Content health        — word counts, zero-turn sessions, empty docs
  7. Date integrity        — ISO date coverage, suspicious dates
  8. Corpus statistics     — total words, chars, estimated tokens

Usage:
  python audit_output.py                          # audit output/ in cwd
  python audit_output.py --dir /path/to/output    # custom output dir
  python audit_output.py --dir output --json      # also write audit.json
  python audit_output.py --dir output --fix-csv   # regenerate index.csv from JSON files
"""

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import date, timedelta
from pathlib import Path


# ── Helpers ────────────────────────────────────────────────────────────────

def load_collection(output_dir: Path, collection: str) -> list[dict]:
    """Load all JSON docs from a collection subdirectory."""
    coll_dir = output_dir / collection
    if not coll_dir.exists():
        return []
    docs = []
    for fp in sorted(coll_dir.glob("*.json")):
        try:
            docs.append(json.loads(fp.read_text(encoding="utf-8")))
        except Exception as e:
            print(f"  [WARN] Could not read {fp.name}: {e}")
    return docs


def load_index(output_dir: Path) -> list[dict]:
    """Load index.csv if present."""
    p = output_dir / "index.csv"
    if not p.exists():
        return []
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def bar(value: int, total: int, width: int = 30) -> str:
    if total == 0:
        return "░" * width
    filled = round(value / total * width)
    return "█" * filled + "░" * (width - filled)


def pct(n: int, d: int) -> str:
    if d == 0:
        return "  n/a"
    return f"{n/d*100:5.1f}%"


def section(title: str):
    print(f"\n{'═'*65}")
    print(f"  {title}")
    print(f"{'═'*65}")


def row(label: str, value, width: int = 42):
    print(f"  {label:<{width}} {value}")


# ── 1. Collection coverage ─────────────────────────────────────────────────

EXPECTED = {
    "sessions":  219,
    "judgment":   68,   # 27 sub-pages each multi-section → 68 files observed
    "key_docs":   12,
    "motions":     8,
    "orders":      7,
    "cases":      14,
    "witnesses":   6,
    "jackson":    68,
    "pohl":        5,
    "nca_v1":     14,
    "nca_v2":     16,
    "vol1":       45,
    # nca_v3, nca_v4 have no firm target
}

def audit_coverage(all_docs: dict[str, list[dict]]) -> dict:
    section("1. COLLECTION COVERAGE")
    print(f"\n  {'Collection':<14} {'Found':>6}  {'Expected':>8}  {'Status'}")
    print(f"  {'-'*14} {'-'*6}  {'-'*8}  {'-'*20}")

    results = {}
    for coll, docs in sorted(all_docs.items()):
        found = len(docs)
        expected = EXPECTED.get(coll, None)
        if expected is None:
            status = f"(no target)"
        elif found >= expected:
            status = "✅ OK"
        elif found >= expected * 0.9:
            status = f"⚠️  {expected - found} short"
        else:
            status = f"❌ {expected - found} SHORT"
        exp_str = str(expected) if expected else "—"
        print(f"  {coll:<14} {found:>6}  {exp_str:>8}  {status}")
        results[coll] = {"found": found, "expected": expected}

    total = sum(len(d) for d in all_docs.values())
    row("\n  Total documents", total)
    return results


# ── 2. Session calendar coverage ──────────────────────────────────────────

SESSION_URL_RE = re.compile(r'/imt/(\d{2})-(\d{2})-(\d{2})\.asp')
IMT_START = date(1945, 11, 14)
IMT_END   = date(1946, 10,  1)

def audit_sessions(sessions: list[dict]) -> dict:
    section("2. SESSION CALENDAR COVERAGE")

    # Parse dates from URLs
    session_dates = set()
    for doc in sessions:
        url = doc.get("url", "")
        m = SESSION_URL_RE.search(url)
        if m:
            mm, dd, yy = m.groups()
            try:
                session_dates.add(date(int(f"19{yy}"), int(mm), int(dd)))
            except ValueError:
                pass

    # All weekdays (Mon–Sat) in the trial range
    all_weekdays = set()
    d = IMT_START
    while d <= IMT_END:
        if d.weekday() < 6:  # Mon=0 … Sat=5
            all_weekdays.add(d)
        d += timedelta(days=1)

    missing = sorted(all_weekdays - session_dates)

    row("Sessions scraped", len(session_dates))
    row("Sessions with valid dates", len(session_dates))
    row("Weekdays in IMT range (Nov 1945–Oct 1946)", len(all_weekdays))
    row("Missing weekdays", len(missing))

    # Month-by-month heatmap
    print(f"\n  Month-by-month coverage:")
    by_month: dict[str, tuple[int, int]] = {}
    for d2 in all_weekdays:
        key = d2.strftime("%Y-%m")
        found_c, total_c = by_month.get(key, (0, 0))
        by_month[key] = (found_c, total_c + 1)
    for d2 in session_dates:
        key = d2.strftime("%Y-%m")
        found_c, total_c = by_month.get(key, (0, 0))
        by_month[key] = (found_c + 1, total_c)

    for month in sorted(by_month):
        found_c, total_c = by_month[month]
        b = bar(found_c, total_c, 24)
        print(f"    {month}  {found_c:>3}/{total_c:<3}  {b}")

    # Multi-day gaps
    print(f"\n  Multi-day gaps (3+ consecutive missing weekdays):")
    gap_start = None
    gap_count = 0
    gaps = []
    for d2 in sorted(all_weekdays):
        if d2 in missing:
            if gap_start is None:
                gap_start = d2
            gap_count += 1
        else:
            if gap_count >= 3:
                gaps.append((gap_start, d2 - timedelta(days=1), gap_count))
            gap_start = None
            gap_count = 0
    if gap_count >= 3:
        gaps.append((gap_start, IMT_END, gap_count))

    if gaps:
        for g_start, g_end, g_len in gaps:
            tag = "⚠️  SEPT GAP" if g_start.month == 9 else ""
            print(f"    {g_start} → {g_end}  ({g_len} days)  {tag}")
    else:
        print(f"    No significant gaps found")

    return {"scraped": len(session_dates), "missing": len(missing), "gaps": len(gaps)}


# ── 3. Page number health ──────────────────────────────────────────────────

def audit_page_numbers(sessions: list[dict]) -> dict:
    section("3. PAGE NUMBER HEALTH")

    fmt_counter: Counter = Counter()
    no_pages = []
    page_counts = []
    range_issues = []

    for doc in sessions:
        fmt = doc.get("page_format", "NONE")
        fmt_counter[fmt] += 1
        pages = doc.get("page_numbers", [])
        page_counts.append(len(pages))

        url = doc.get("url", "")
        if not pages:
            no_pages.append(url)
        else:
            # Sanity: page range should be reasonable (1–30000)
            lo, hi = min(pages), max(pages)
            span = hi - lo
            if hi > 30_000 or span > 2000:
                range_issues.append((url, lo, hi, span))

    total = len(sessions)

    print(f"\n  Format distribution across {total} session pages:")
    for fmt, count in sorted(fmt_counter.items(), key=lambda x: -x[1]):
        b = bar(count, total, 24)
        ok = "✅" if fmt not in ("NONE",) else "❌"
        print(f"    {ok} {fmt:<8} {count:>4}  {b}  {pct(count, total)}")

    coverage_rate = (total - fmt_counter["NONE"]) / total * 100 if total else 0
    print(f"\n  Page number coverage rate:  {coverage_rate:.1f}%")

    if page_counts:
        avg = sum(page_counts) / len(page_counts)
        non_zero = [p for p in page_counts if p > 0]
        avg_non_zero = sum(non_zero) / len(non_zero) if non_zero else 0
        print(f"  Avg pages per session:      {avg_non_zero:.1f}  (excl. NONE docs)")

    if no_pages:
        print(f"\n  Sessions with NO page numbers ({len(no_pages)}):")
        for u in no_pages[:20]:
            print(f"    {u}")
        if len(no_pages) > 20:
            print(f"    ... and {len(no_pages) - 20} more")

    if range_issues:
        print(f"\n  Page range anomalies ({len(range_issues)}):")
        for u, lo, hi, span in range_issues[:10]:
            print(f"    {u}  pages {lo}–{hi}  (span={span})")

    return {
        "format_dist": dict(fmt_counter),
        "coverage_pct": round(coverage_rate, 1),
        "none_count": fmt_counter["NONE"],
    }


# ── 4. Speaker quality ────────────────────────────────────────────────────

SUSPECT_SPEAKERS_RE = re.compile(r'^(MR|DR|ER|THE|MS|SIR|COL|GEN|LT|CPT|PROF)\s*$', re.IGNORECASE)

def audit_speakers(sessions: list[dict]) -> dict:
    section("4. SPEAKER QUALITY")

    all_speakers: Counter = Counter()
    zero_turn_sessions = []
    turns_per_session = []
    suspect_tags = Counter()
    name_variants: dict[str, list[str]] = defaultdict(list)

    for doc in sessions:
        turns = doc.get("turns", [])
        url = doc.get("url", "")
        turns_per_session.append(len(turns))

        if not turns:
            zero_turn_sessions.append(url)

        for t in turns:
            spk = t.get("speaker", "").strip()
            if spk:
                all_speakers[spk] += 1
                if SUSPECT_SPEAKERS_RE.match(spk):
                    suspect_tags[spk] += 1

    # Name variant detection — group by first token
    for spk in all_speakers:
        first = spk.split()[0] if spk.split() else spk
        name_variants[first].append(spk)

    variant_groups = {
        k: sorted(v) for k, v in name_variants.items()
        if len(v) > 1 and any(c > 5 for c in [all_speakers[s] for s in v])
    }

    total_turns = sum(all_speakers.values())
    row("Unique speaker tags", len(all_speakers))
    row("Total speaker turns", total_turns)
    row("Zero-turn sessions", len(zero_turn_sessions))
    if turns_per_session:
        avg_turns = sum(turns_per_session) / len(turns_per_session)
        non_zero_turns = [t for t in turns_per_session if t > 0]
        median_t = sorted(non_zero_turns)[len(non_zero_turns)//2] if non_zero_turns else 0
        row("Avg turns/session (all)", f"{avg_turns:.0f}")
        row("Median turns/session (non-zero)", median_t)

    print(f"\n  Top 20 speakers:")
    for spk, count in all_speakers.most_common(20):
        b = bar(count, total_turns, 20)
        print(f"    {count:>6}  {b}  {spk}")

    if zero_turn_sessions:
        print(f"\n  Zero-turn sessions ({len(zero_turn_sessions)}) — need investigation:")
        for u in zero_turn_sessions:
            print(f"    {u}")

    if suspect_tags:
        print(f"\n  Suspect truncated speaker tags ({len(suspect_tags)} types):")
        for tag, count in suspect_tags.most_common():
            print(f"    {tag!r:<12}  {count} occurrences")

    if variant_groups:
        print(f"\n  Speaker name variant groups (sample — top 10):")
        for prefix, variants in list(variant_groups.items())[:10]:
            counts = [f"{s} ({all_speakers[s]})" for s in variants[:4]]
            print(f"    {prefix}: {', '.join(counts)}")

    return {
        "unique_speakers": len(all_speakers),
        "total_turns": total_turns,
        "zero_turn_sessions": len(zero_turn_sessions),
        "suspect_tags": dict(suspect_tags),
    }


# ── 5. Validation flags ────────────────────────────────────────────────────

def audit_flags(all_docs: dict[str, list[dict]]) -> dict:
    section("5. VALIDATION FLAGS")

    flag_counter: Counter = Counter()
    flag_by_collection: dict[str, Counter] = defaultdict(Counter)
    worst: list[tuple[str, list[str]]] = []

    for coll, docs in all_docs.items():
        for doc in docs:
            flags = doc.get("validation_flags", [])
            for f in flags:
                flag_counter[f] += 1
                flag_by_collection[coll][f] += 1
            if flags:
                worst.append((doc.get("url", "?"), flags))

    total_docs = sum(len(d) for d in all_docs.values())
    total_flagged = len(worst)

    row("Total documents", total_docs)
    row("Documents with at least one flag", f"{total_flagged}  ({pct(total_flagged, total_docs)})")

    print(f"\n  Flag type breakdown:")
    for flag, count in flag_counter.most_common():
        b = bar(count, total_docs, 24)
        print(f"    {flag:<35} {count:>4}  {b}")

    print(f"\n  Flags by collection:")
    print(f"  {'Collection':<14}", end="")
    all_flag_types = [f for f, _ in flag_counter.most_common()]
    for ft in all_flag_types[:5]:
        print(f"  {ft[:18]:<18}", end="")
    print()
    print(f"  {'-'*14}", end="")
    for _ in all_flag_types[:5]:
        print(f"  {'-'*18}", end="")
    print()
    for coll in sorted(flag_by_collection):
        print(f"  {coll:<14}", end="")
        for ft in all_flag_types[:5]:
            c = flag_by_collection[coll].get(ft, 0)
            print(f"  {c:>18}", end="")
        print()

    # POSSIBLE_TRUNCATION deep-dive — how many are false positives?
    trunc_docs = [url for url, flags in worst if "POSSIBLE_TRUNCATION" in flags]
    if trunc_docs:
        print(f"\n  POSSIBLE_TRUNCATION sample ({min(5, len(trunc_docs))} of {len(trunc_docs)}):")
        print(f"  (These are mostly false positives from nav footer — verify manually)")
        for u in trunc_docs[:5]:
            print(f"    {u}")

    return {
        "total_flagged": total_flagged,
        "flag_counts": dict(flag_counter),
    }


# ── 6. Content health ─────────────────────────────────────────────────────

def audit_content(all_docs: dict[str, list[dict]]) -> dict:
    section("6. CONTENT HEALTH")

    word_counts_by_coll: dict[str, list[int]] = defaultdict(list)
    empty_docs = []
    short_docs = []

    for coll, docs in all_docs.items():
        for doc in docs:
            wc = doc.get("word_count", 0)
            word_counts_by_coll[coll].append(wc)
            url = doc.get("url", "?")
            if wc == 0:
                empty_docs.append((coll, url))
            elif wc < 100:
                short_docs.append((coll, url, wc))

    total_words = sum(w for wcs in word_counts_by_coll.values() for w in wcs)
    total_chars = sum(
        doc.get("char_count", 0)
        for docs in all_docs.values()
        for doc in docs
    )
    # Rough token estimate: ~1.3 words per token for English legal text
    est_tokens = int(total_words / 1.3)

    row("Total words (all collections)", f"{total_words:,}")
    row("Total HTML chars (all collections)", f"{total_chars:,}")
    row("Estimated tokens (~÷1.3)", f"{est_tokens:,}")
    row("Empty documents (0 words)", len(empty_docs))
    row("Very short documents (<100 words)", len(short_docs))

    print(f"\n  Word count by collection:")
    print(f"  {'Collection':<14}  {'Docs':>5}  {'Total words':>12}  {'Avg':>7}  {'Min':>7}  {'Max':>7}")
    print(f"  {'-'*14}  {'-'*5}  {'-'*12}  {'-'*7}  {'-'*7}  {'-'*7}")
    for coll in sorted(word_counts_by_coll):
        wcs = word_counts_by_coll[coll]
        if not wcs:
            continue
        total_w = sum(wcs)
        avg_w = total_w // len(wcs)
        print(f"  {coll:<14}  {len(wcs):>5}  {total_w:>12,}  {avg_w:>7,}  {min(wcs):>7,}  {max(wcs):>7,}")

    if empty_docs:
        print(f"\n  Empty documents ({len(empty_docs)}):")
        for coll, url in empty_docs[:10]:
            print(f"    [{coll}]  {url}")

    if short_docs:
        print(f"\n  Very short documents (<100 words, sample):")
        for coll, url, wc in short_docs[:10]:
            print(f"    [{coll}]  {wc:>4} words  {url}")

    return {
        "total_words": total_words,
        "total_chars": total_chars,
        "est_tokens": est_tokens,
        "empty_docs": len(empty_docs),
    }


# ── 7. Date integrity ─────────────────────────────────────────────────────

def audit_dates(all_docs: dict[str, list[dict]]) -> dict:
    section("7. DATE INTEGRITY")

    missing_date = []
    bad_date = []
    date_dist: Counter = Counter()

    for coll, docs in all_docs.items():
        for doc in docs:
            d = doc.get("date_iso", "") or ""
            if not d:
                missing_date.append((coll, doc.get("url", "?")))
                continue
            # Validate format
            if not re.fullmatch(r'\d{4}-\d{2}-\d{2}', d):
                bad_date.append((coll, doc.get("url", "?"), d))
                continue
            # Year bucketing
            year = d[:4]
            date_dist[year] += 1
            # Plausibility check for sessions
            if coll == "sessions":
                try:
                    parsed = date.fromisoformat(d)
                    if not (IMT_START <= parsed <= IMT_END + timedelta(days=30)):
                        bad_date.append((coll, doc.get("url", "?"), d))
                except ValueError:
                    bad_date.append((coll, doc.get("url", "?"), d))

    sessions_with_date = sum(1 for doc in all_docs.get("sessions", []) if doc.get("date_iso"))
    total_sessions = len(all_docs.get("sessions", []))

    row("Sessions with valid date_iso", f"{sessions_with_date}/{total_sessions}")
    row("Documents missing date_iso (all collections)", len(missing_date))
    row("Documents with malformed/implausible date", len(bad_date))

    if date_dist:
        print(f"\n  Date distribution by year:")
        for year in sorted(date_dist):
            print(f"    {year}: {date_dist[year]:>4} docs")

    if bad_date:
        print(f"\n  Malformed/implausible dates (sample):")
        for coll, url, d in bad_date[:10]:
            print(f"    [{coll}]  {d}  {url}")

    return {
        "missing_date": len(missing_date),
        "bad_date": len(bad_date),
        "date_dist": dict(date_dist),
    }


# ── 8. NCA volume deep-dive ───────────────────────────────────────────────

def audit_nca(all_docs: dict[str, list[dict]]) -> dict:
    section("8. NCA VOLUMES DEEP-DIVE")

    nca_colls = {k: v for k, v in all_docs.items() if k.startswith("nca")}
    if not nca_colls:
        print("  No NCA collections found.")
        return {}

    total_nca_words = 0
    for coll, docs in sorted(nca_colls.items()):
        words = sum(d.get("word_count", 0) for d in docs)
        total_nca_words += words
        flagged = sum(1 for d in docs if d.get("validation_flags"))
        no_pages = sum(1 for d in docs if not d.get("page_numbers"))
        print(f"  {coll:<10}  {len(docs):>4} docs  {words:>9,} words  "
              f"{flagged:>3} flagged  {no_pages:>3} no-pages")

    row("\n  Total NCA words", f"{total_nca_words:,}")

    # nca_v4 is by far the biggest — spot check
    nca_v4 = all_docs.get("nca_v4", [])
    if nca_v4:
        page_fmt_dist = Counter(d.get("page_format", "NONE") for d in nca_v4)
        print(f"\n  nca_v4 page format distribution ({len(nca_v4)} docs):")
        for fmt, count in page_fmt_dist.most_common():
            b = bar(count, len(nca_v4), 20)
            print(f"    {fmt:<8} {count:>4}  {b}")

    return {"total_nca_words": total_nca_words}


# ── 9. Priority fix list ───────────────────────────────────────────────────

def print_priorities(results: dict):
    section("9. PRIORITY FIX LIST")

    issues = []

    # NONE page numbers
    none_count = results.get("pages", {}).get("none_count", 0)
    if none_count > 0:
        issues.append((
            "HIGH" if none_count > 10 else "MED",
            f"{none_count} sessions still have NO page numbers",
            "Investigate HTML of those specific pages — may be a 5th format or truly absent"
        ))

    # Zero-turn sessions
    zero_turns = results.get("speakers", {}).get("zero_turn_sessions", 0)
    if zero_turns > 0:
        issues.append((
            "HIGH",
            f"{zero_turns} sessions have zero speaker turns",
            "Check if HTML structure differs; may need a 3rd speaker extraction strategy"
        ))

    # Suspect speaker tags
    suspects = results.get("speakers", {}).get("suspect_tags", {})
    if suspects:
        issues.append((
            "MED",
            f"Suspect speaker tags: {list(suspects.keys())}",
            "Caused by split <p> tags — fix with stateful span-joining in parser"
        ))

    # Missing key docs
    key_found = results.get("coverage", {}).get("key_docs", {}).get("found", 0)
    if key_found < 12:
        issues.append((
            "MED",
            f"key_docs: {key_found}/12 — {12 - key_found} missing",
            "Check KEY_DOCS dict in scraper for correct URLs"
        ))

    # September gap
    session_res = results.get("sessions", {})
    if session_res.get("gaps", 0) > 0:
        issues.append((
            "INFO",
            "September 1946 gap (~24 days) confirmed — patch from HuggingFace trial_id=7",
            "Use HF records where date_iso BETWEEN 1946-09-02 AND 1946-09-28"
        ))

    # POSSIBLE_TRUNCATION volume
    flag_counts = results.get("flags", {}).get("flag_counts", {})
    trunc_count = flag_counts.get("POSSIBLE_TRUNCATION", 0)
    if trunc_count > 50:
        issues.append((
            "LOW",
            f"POSSIBLE_TRUNCATION on {trunc_count} docs — mostly false positives",
            "Nav footer lacks terminal punctuation; add footer-aware check to validate()"
        ))

    if not issues:
        print("\n  ✅ No critical issues found.")
    else:
        for priority, title, action in issues:
            icon = {"HIGH": "🔴", "MED": "🟡", "INFO": "🔵", "LOW": "⚪"}.get(priority, "•")
            print(f"\n  {icon} [{priority}] {title}")
            print(f"       → {action}")


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Audit Nuremberg Scholar scrape output")
    parser.add_argument("--dir", default="output", help="Path to scraper output directory")
    parser.add_argument("--json", action="store_true", help="Write audit results to audit_output.json")
    parser.add_argument("--fix-csv", action="store_true", help="Regenerate index.csv from JSON files")
    args = parser.parse_args()

    output_dir = Path(args.dir)
    if not output_dir.exists():
        print(f"Error: output directory not found: {output_dir}")
        sys.exit(1)

    print(f"\nNuremberg Scholar — Post-Scrape Audit")
    print(f"Output directory: {output_dir.absolute()}")

    # Load all collections
    collections = [
        "sessions", "judgment", "key_docs", "secondary",
        "vol1", "motions", "orders", "cases", "witnesses",
        "jackson", "pohl", "nca_v1", "nca_v2", "nca_v3", "nca_v4",
    ]

    # Also check for any collections not in the list above
    for d in output_dir.iterdir():
        if d.is_dir() and d.name not in collections and not d.name.startswith("_"):
            collections.append(d.name)

    all_docs: dict[str, list[dict]] = {}
    for coll in collections:
        docs = load_collection(output_dir, coll)
        if docs:
            all_docs[coll] = docs

    if not all_docs:
        print("No JSON files found in output directory.")
        sys.exit(1)

    sessions = all_docs.get("sessions", [])

    # Run all audits
    results = {}
    results["coverage"] = audit_coverage(all_docs)
    results["sessions"] = audit_sessions(sessions)
    results["pages"]    = audit_page_numbers(sessions)
    results["speakers"] = audit_speakers(sessions)
    results["flags"]    = audit_flags(all_docs)
    results["content"]  = audit_content(all_docs)
    results["dates"]    = audit_dates(all_docs)
    results["nca"]      = audit_nca(all_docs)
    print_priorities(results)

    # Summary line
    section("SUMMARY")
    total_words = results["content"]["total_words"]
    est_tokens  = results["content"]["est_tokens"]
    total_docs  = sum(r["found"] for r in results["coverage"].values() if isinstance(r, dict))
    none_pages  = results["pages"]["none_count"]
    zero_turns  = results["speakers"]["zero_turn_sessions"]
    total_turns = results["speakers"]["total_turns"]
    unique_spk  = results["speakers"]["unique_speakers"]

    row("Total documents", f"{total_docs:,}")
    row("Total words", f"{total_words:,}")
    row("Estimated tokens", f"{est_tokens:,}")
    row("Total speaker turns", f"{total_turns:,}")
    row("Unique speakers", unique_spk)
    row("Sessions missing page numbers", none_pages)
    row("Sessions missing speaker turns", zero_turns)
    print()

    if args.json:
        out_path = output_dir / "audit_output.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"  Audit JSON → {out_path}")

    if args.fix_csv:
        _rebuild_csv(output_dir, all_docs)


def _rebuild_csv(output_dir: Path, all_docs: dict[str, list[dict]]):
    """Regenerate index.csv from the JSON files on disk."""
    section("REBUILDING index.csv")
    fieldnames = [
        "url", "collection", "date_iso", "char_count", "word_count",
        "turn_count", "speaker_count", "page_format", "validation_flags",
    ]
    rows = []
    for coll, docs in all_docs.items():
        for doc in docs:
            rows.append({
                "url":              doc.get("url", ""),
                "collection":       coll,
                "date_iso":         doc.get("date_iso", ""),
                "char_count":       doc.get("char_count", 0),
                "word_count":       doc.get("word_count", 0),
                "turn_count":       doc.get("turn_count", 0),
                "speaker_count":    doc.get("speaker_count", 0),
                "page_format":      doc.get("page_format", "NONE"),
                "validation_flags": "|".join(doc.get("validation_flags", [])),
            })
    out_path = output_dir / "index.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"  Wrote {len(rows)} rows → {out_path}")


if __name__ == "__main__":
    main()