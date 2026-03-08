"""
audit2.py — Nuremberg Scholar Output Audit v2
==============================================
Comprehensive audit of output/  covering:

  1.  Collection coverage & doc counts
  2.  Session calendar coverage (with Harvard patch awareness)
  3.  Page number health per format
  4.  Speaker quality & variant analysis
  5.  Validation flags breakdown
  6.  Content health (words, tokens, empty docs)
  7.  Date integrity
  8.  Chunk file health (chunks.jsonl)
  9.  Source distribution (yale vs harvard_law_patch)
  10. Adjournment chain integrity (crosscheck summary)
  11. Top-level readiness score

Usage:
    python audit2.py
    python audit2.py --output-dir /path/to/output
    python audit2.py --json audit2_report.json
"""

import re
import json
import argparse
import sys
from pathlib import Path
from datetime import date, timedelta
from collections import defaultdict, Counter

# ── Config ────────────────────────────────────────────────────────────────────

DEFAULT_OUTPUT = Path("output")

COLLECTION_TARGETS = {
    "judgment":  68,
    "key_docs":  12,
    "vol1":      45,
    "sessions":  221,   # updated: 219 Yale + 2 Harvard
}

IMT_START = date(1945, 11, 20)
IMT_END   = date(1946, 10, 1)

KNOWN_RECESSES = set()
for d in [date(1945, 12, 21) + timedelta(i) for i in range(12)]:
    KNOWN_RECESSES.add(d.isoformat())
for d in [date(1946, 4, 19) + timedelta(i) for i in range(4)]:
    KNOWN_RECESSES.add(d.isoformat())
for d in [date(1946, 9, 2) + timedelta(i) for i in range(26)]:
    KNOWN_RECESSES.add(d.isoformat())

BAR_WIDTH = 24

# ── Helpers ───────────────────────────────────────────────────────────────────

def bar(n, total, width=BAR_WIDTH) -> str:
    if total == 0:
        return "░" * width
    filled = int(round(width * n / total))
    return "█" * filled + "░" * (width - filled)

def tok(words: int) -> int:
    return int(words / 1.3)

def pct(n, total) -> str:
    return f"{100*n/total:.1f}%" if total else "—"

def section(title: str):
    print(f"\n{'═'*70}")
    print(f"  {title}")
    print(f"{'═'*70}")

def load_collection(path: Path) -> list[dict]:
    docs = []
    if not path.exists():
        return docs
    for fp in sorted(path.glob("*.json")):
        try:
            docs.append(json.loads(fp.read_text(encoding="utf-8")))
        except Exception as e:
            print(f"  LOAD ERROR {fp.name}: {e}")
    return docs

def load_jsonl(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
    return rows

def weekdays_in_range(start: date, end: date) -> list[date]:
    out, d = [], start
    while d <= end:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(1)
    return out

# ── Sections ──────────────────────────────────────────────────────────────────

def audit_coverage(collections: dict[str, list]) -> dict:
    section("1. COLLECTION COVERAGE")
    total = 0
    results = {}
    print(f"\n  {'Collection':<16} {'Found':>6}  {'Target':>7}  Status")
    print(f"  {'-'*16} {'-'*6}  {'-'*7}  {'-'*25}")
    for name, target in COLLECTION_TARGETS.items():
        found = len(collections.get(name, []))
        total += found
        ok    = found >= target
        status = "✅ OK" if ok else f"❌ {target - found} SHORT"
        results[name] = {"found": found, "target": target, "ok": ok}
        print(f"  {name:<16} {found:>6}  {target:>7}  {status}")
    # secondary has no fixed target
    sec = len(collections.get("secondary", []))
    total += sec
    print(f"  {'secondary':<16} {sec:>6}  {'—':>7}  (no target)")
    print(f"\n  Total documents  {total:>30}")
    results["total"] = total
    return results

def audit_calendar(sessions: list[dict]) -> dict:
    section("2. SESSION CALENDAR COVERAGE")
    session_dates = set()
    for s in sessions:
        d = s.get("date_iso")
        if d:
            session_dates.add(d)

    all_weekdays = weekdays_in_range(IMT_START, IMT_END)
    missing      = [d for d in all_weekdays
                    if d.isoformat() not in session_dates
                    and d.isoformat() not in KNOWN_RECESSES]

    print(f"\n  Sessions in corpus              {len(sessions):>6}")
    print(f"  Sessions with valid date_iso    {len(session_dates):>6}")
    print(f"  Weekdays in IMT range           {len(all_weekdays):>6}")
    print(f"  Known recess days (excluded)    {len(KNOWN_RECESSES):>6}")
    print(f"  Unexplained missing weekdays    {len(missing):>6}")

    # Month breakdown
    print(f"\n  Month-by-month coverage:")
    months: dict[str, tuple[int,int]] = {}
    for d in all_weekdays:
        key = d.strftime("%Y-%m")
        have, total2 = months.get(key, (0, 0))
        if d.isoformat() in session_dates:
            have += 1
        months[key] = (have, total2 + 1)

    for ym, (have, total2) in sorted(months.items()):
        b = bar(have, total2)
        print(f"    {ym}   {have:>2}/{total2:<2}   {b}")

    # Source breakdown
    sources = Counter(s.get("source", "yale") for s in sessions)
    print(f"\n  Source breakdown:")
    for src, cnt in sorted(sources.items()):
        print(f"    {src:<25} {cnt:>4} sessions")

    # Multi-day gaps
    gaps = []
    missing_set = set(d.isoformat() for d in missing)
    i = 0
    while i < len(missing):
        j = i
        while j + 1 < len(missing) and (missing[j+1] - missing[j]).days <= 3:
            j += 1
        if j > i:
            gaps.append((missing[i], missing[j]))
        i = j + 1

    if gaps:
        print(f"\n  Multi-day gaps (unexplained):")
        for g_start, g_end in gaps:
            days = (g_end - g_start).days + 1
            print(f"    {g_start.isoformat()} → {g_end.isoformat()}  ({days} days)")
    else:
        print(f"\n  ✅ No unexplained multi-day gaps")

    return {"session_dates": list(session_dates), "missing": [d.isoformat() for d in missing]}

def audit_pages(sessions: list[dict]) -> dict:
    section("3. PAGE NUMBER HEALTH")
    fmt_counts: Counter = Counter()
    no_pages = []
    anomalies = []

    for s in sessions:
        fmt = s.get("page_format", "UNKNOWN")
        fmt_counts[fmt] += 1
        if not s.get("page_start"):
            no_pages.append(s.get("url", s.get("slug", "?")))
        span = 0
        if s.get("page_start") and s.get("page_end"):
            span = s["page_end"] - s["page_start"]
            if span > 500:
                anomalies.append((s.get("url", "?"), s["page_start"], s["page_end"], span))

    total_sess = len(sessions)
    has_pages  = total_sess - len(no_pages)
    avg_pages  = sum(s.get("page_end", 0) - s.get("page_start", 0)
                     for s in sessions
                     if s.get("page_start") and s.get("page_end")) / max(has_pages, 1)

    print(f"\n  Format distribution across {total_sess} sessions:")
    good_formats = {"F1","F2","F3","F4","MIXED","HARVARD"}
    for fmt, cnt in fmt_counts.most_common():
        icon = "✅" if fmt in good_formats else "❌"
        b    = bar(cnt, total_sess)
        print(f"    {icon} {fmt:<10} {cnt:>4}  {b}  {pct(cnt, total_sess):>6}")

    print(f"\n  Page number coverage rate:  {pct(has_pages, total_sess)}")
    print(f"  Avg pages/session (excl NONE): {avg_pages:.1f}")

    if no_pages:
        print(f"\n  Sessions with NO page numbers ({len(no_pages)}):")
        for u in no_pages[:20]:
            print(f"    {u}")

    if anomalies:
        print(f"\n  Page range anomalies (span > 500):")
        for url, ps, pe, span in anomalies:
            print(f"    {url}  pages {ps}–{pe}  (span={span})")

    return {"no_pages": len(no_pages), "formats": dict(fmt_counts)}

def audit_speakers(all_docs: list[dict]) -> dict:
    section("4. SPEAKER QUALITY")
    all_speakers:  Counter = Counter()
    zero_turn_docs = []
    per_session_turns = []

    for doc in all_docs:
        turns = doc.get("turns", [])
        if not turns:
            zero_turn_docs.append(doc.get("url", doc.get("slug", "?")))
            continue
        per_session_turns.append(len(turns))
        for t in turns:
            spk = t.get("speaker", "").strip()
            if spk:
                all_speakers[spk] += 1

    total_turns  = sum(per_session_turns)
    unique_spk   = len(all_speakers)
    avg_turns    = total_turns / max(len(per_session_turns), 1)
    median_turns = sorted(per_session_turns)[len(per_session_turns)//2] if per_session_turns else 0

    print(f"\n  Unique speaker tags         {unique_spk:>8}")
    print(f"  Total speaker turns         {total_turns:>8,}")
    print(f"  Zero-turn documents         {len(zero_turn_docs):>8}")
    print(f"  Avg turns/doc               {avg_turns:>8.0f}")
    print(f"  Median turns/doc (non-zero) {median_turns:>8}")

    print(f"\n  Top 20 speakers:")
    max_cnt = all_speakers.most_common(1)[0][1] if all_speakers else 1
    for spk, cnt in all_speakers.most_common(20):
        b = bar(cnt, max_cnt, 20)
        print(f"  {cnt:>7,}  {b}  {spk}")

    # Variant groups
    prefix_groups: dict[str, list] = defaultdict(list)
    for spk in all_speakers:
        prefix = spk.split()[0] if spk.split() else spk
        prefix_groups[prefix].append(spk)

    noisy = {k: v for k, v in prefix_groups.items() if len(v) > 1}
    print(f"\n  Speaker prefix groups with variants ({len(noisy)} groups, sample top 8):")
    for prefix, variants in sorted(noisy.items(), key=lambda x: -len(x[1]))[:8]:
        sample = variants[:4]
        print(f"    {prefix}: {', '.join(sample)}{' ...' if len(variants)>4 else ''}")

    if zero_turn_docs:
        print(f"\n  Zero-turn documents ({len(zero_turn_docs)}):")
        for u in zero_turn_docs:
            print(f"    {u}")

    return {"unique_speakers": unique_spk, "total_turns": total_turns,
            "zero_turn_docs": len(zero_turn_docs)}

def audit_flags(all_docs: list[dict], collections: dict[str, list]) -> dict:
    section("5. VALIDATION FLAGS")
    flag_counter: Counter = Counter()
    flag_by_coll: dict[str, Counter] = defaultdict(Counter)

    for coll_name, docs in collections.items():
        for doc in docs:
            for flag in doc.get("validation_flags", []):
                flag_counter[flag] += 1
                flag_by_coll[coll_name][flag] += 1

    total_docs    = sum(len(d) for d in collections.values())
    flagged_docs  = sum(1 for doc in all_docs if doc.get("validation_flags"))

    print(f"\n  Total documents             {total_docs:>8}")
    print(f"  Documents with any flag     {flagged_docs:>8}  ({pct(flagged_docs, total_docs)})")

    print(f"\n  Flag type breakdown:")
    for flag, cnt in flag_counter.most_common():
        b = bar(cnt, total_docs)
        print(f"    {flag:<35} {cnt:>5}  {b}")

    print(f"\n  Flags by collection:")
    all_flags = list(flag_counter.keys())
    col_w = 16
    flag_w = 18
    header = f"  {'Collection':<{col_w}}" + "".join(f"  {f[:flag_w]:<{flag_w}}" for f in all_flags[:5])
    print(header)
    print("  " + "-"*col_w + ("  " + "-"*flag_w) * min(len(all_flags), 5))
    for coll_name in collections:
        row = f"  {coll_name:<{col_w}}"
        for flag in all_flags[:5]:
            cnt = flag_by_coll[coll_name].get(flag, 0)
            row += f"  {cnt:<{flag_w}}"
        print(row)

    return {"total_flagged": flagged_docs, "flags": dict(flag_counter)}

def audit_content(collections: dict[str, list]) -> dict:
    section("6. CONTENT HEALTH")
    all_docs   = [d for docs in collections.values() for d in docs]
    total_words = sum(d.get("word_count", 0) for d in all_docs)
    total_chars = sum(d.get("char_count", 0) for d in all_docs)
    empty       = [d for d in all_docs if d.get("word_count", 0) == 0]
    short       = [d for d in all_docs if 0 < d.get("word_count", 0) < 100]

    print(f"\n  Total words (all collections)   {total_words:>12,}")
    print(f"  Total HTML chars                {total_chars:>12,}")
    print(f"  Estimated tokens (~÷1.3)        {tok(total_words):>12,}")
    print(f"  Empty documents (0 words)       {len(empty):>12}")
    print(f"  Very short (<100 words)         {len(short):>12}")

    print(f"\n  Word count by collection:")
    print(f"  {'Collection':<16} {'Docs':>6}  {'Total words':>12}  {'Avg':>7}  {'Min':>7}  {'Max':>7}")
    print(f"  {'-'*16} {'-'*6}  {'-'*12}  {'-'*7}  {'-'*7}  {'-'*7}")
    for name, docs in sorted(collections.items()):
        if not docs:
            continue
        words  = [d.get("word_count", 0) for d in docs]
        total  = sum(words)
        avg    = total // len(words) if words else 0
        mn, mx = min(words), max(words)
        print(f"  {name:<16} {len(docs):>6}  {total:>12,}  {avg:>7,}  {mn:>7,}  {mx:>7,}")

    if empty:
        print(f"\n  Empty documents:")
        for d in empty:
            coll = d.get("collection", "?")
            print(f"    [{coll}]  {d.get('url', d.get('slug', '?'))}")

    return {"total_words": total_words, "total_tokens": tok(total_words),
            "empty": len(empty), "short": len(short)}

def audit_dates(all_docs: list[dict]) -> dict:
    section("7. DATE INTEGRITY")
    sessions = [d for d in all_docs if d.get("date_iso")]
    missing  = [d for d in all_docs if not d.get("date_iso")]
    bad      = []
    year_counts: Counter = Counter()

    for d in sessions:
        iso = d["date_iso"]
        try:
            parsed = date.fromisoformat(iso)
            year_counts[parsed.year] += 1
            if parsed.year < 1945 or parsed.year > 1947:
                bad.append((iso, d.get("url", "?")))
        except ValueError:
            bad.append((iso, d.get("url", "?")))

    print(f"\n  Documents with date_iso         {len(sessions):>6}")
    print(f"  Documents missing date_iso      {len(missing):>6}")
    print(f"  Documents with implausible date {len(bad):>6}")

    print(f"\n  Date distribution by year:")
    for yr, cnt in sorted(year_counts.items()):
        print(f"    {yr}:  {cnt:>4} docs")

    if bad:
        print(f"\n  Implausible dates:")
        for iso, url in bad[:10]:
            print(f"    {iso}  {url}")

    return {"with_date": len(sessions), "missing_date": len(missing), "bad_date": len(bad)}

def audit_chunks(output_dir: Path) -> dict:
    section("8. CHUNK FILE HEALTH")
    chunks_path = output_dir / "chunks.jsonl"
    if not chunks_path.exists():
        print(f"\n  ❌ chunks.jsonl not found at {chunks_path}")
        return {}

    chunks = load_jsonl(chunks_path)
    if not chunks:
        print(f"\n  ❌ chunks.jsonl is empty")
        return {}

    total       = len(chunks)
    coll_counts: Counter = Counter(c.get("collection", "?") for c in chunks)
    tok_counts  = [c.get("token_count", 0) for c in chunks]
    empty_chunks= sum(1 for t in tok_counts if t == 0)
    avg_tok     = sum(tok_counts) / max(total, 1)
    min_tok     = min(tok_counts) if tok_counts else 0
    max_tok     = max(tok_counts) if tok_counts else 0
    total_tok   = sum(tok_counts)

    # Check all mandatory fields
    required = {"chunk_id","collection","body","token_count"}
    missing_fields = sum(1 for c in chunks if not required.issubset(c.keys()))

    print(f"\n  Total chunks                    {total:>10,}")
    print(f"  Total tokens                    {total_tok:>10,}")
    print(f"  Avg tokens/chunk                {avg_tok:>10.1f}")
    print(f"  Min / Max tokens                {min_tok:>6} / {max_tok}")
    print(f"  Empty chunks (0 tokens)         {empty_chunks:>10}")
    print(f"  Chunks missing required fields  {missing_fields:>10}")

    print(f"\n  Chunks by collection:")
    max_c = max(coll_counts.values()) if coll_counts else 1
    for coll, cnt in coll_counts.most_common():
        b = bar(cnt, total)
        print(f"    {coll:<16} {cnt:>7,}  {b}  {pct(cnt, total):>6}")

    # Token distribution buckets
    buckets = Counter()
    for t in tok_counts:
        if   t == 0:         buckets["0"] += 1
        elif t < 128:        buckets["1-127"] += 1
        elif t < 256:        buckets["128-255"] += 1
        elif t < 512:        buckets["256-511"] += 1
        elif t < 768:        buckets["512-767"] += 1
        else:                buckets["768+"] += 1

    print(f"\n  Token size distribution:")
    for label in ["0","1-127","128-255","256-511","512-767","768+"]:
        cnt = buckets.get(label, 0)
        b   = bar(cnt, total)
        print(f"    {label:<10} {cnt:>7,}  {b}")

    return {"total_chunks": total, "total_tokens": total_tok,
            "avg_tokens": avg_tok, "empty": empty_chunks,
            "by_collection": dict(coll_counts)}

def audit_sources(sessions: list[dict]) -> dict:
    section("9. SOURCE INTEGRITY")
    sources  = Counter(s.get("source", "yale") for s in sessions)
    formats  = Counter(s.get("page_format", "?") for s in sessions)
    has_html = sum(1 for s in sessions if s.get("raw_html"))
    has_hash = sum(1 for s in sessions if s.get("content_hash"))

    print(f"\n  Sessions by source:")
    for src, cnt in sources.most_common():
        b = bar(cnt, len(sessions))
        print(f"    {src:<28} {cnt:>4}  {b}")

    print(f"\n  Sessions with raw_html stored   {has_html:>6} / {len(sessions)}")
    print(f"  Sessions with content_hash      {has_hash:>6} / {len(sessions)}")

    # Duplicate content hashes
    hashes = [s["content_hash"] for s in sessions if s.get("content_hash")]
    dupes  = len(hashes) - len(set(hashes))
    if dupes:
        print(f"\n  ⚠️  Duplicate content hashes: {dupes}")
    else:
        print(f"\n  ✅ No duplicate content (all hashes unique)")

    return {"sources": dict(sources), "duplicates": dupes}

def audit_readiness(cov: dict, content: dict, chunks: dict, flags: dict) -> dict:
    section("10. RAG READINESS SCORE")

    checks = [
        ("Sessions complete (221)",        cov.get("sessions", {}).get("ok", False)),
        ("Judgment complete (68)",          cov.get("judgment", {}).get("ok", False)),
        ("Vol1 complete (45)",              cov.get("vol1", {}).get("ok", False)),
        ("No empty documents",              content.get("empty", 1) == 0),
        ("chunks.jsonl exists & non-empty", chunks.get("total_chunks", 0) > 0),
        ("Chunk avg tokens 350-550",        350 <= chunks.get("avg_tokens", 0) <= 550),
        ("Total tokens > 5M",               content.get("total_tokens", 0) > 5_000_000),
        ("Zero MISSING_CONTENT flags",      flags.get("flags", {}).get("MISSING_CONTENT_CONTAINER", 0) == 0),
    ]

    passed = sum(1 for _, ok in checks if ok)
    total  = len(checks)

    print()
    for label, ok in checks:
        icon = "✅" if ok else "❌"
        print(f"  {icon}  {label}")

    score = int(100 * passed / total)
    print(f"\n  Score: {passed}/{total}  ({score}%)")

    if score == 100:
        print("  🟢 Corpus ready for embedding & retrieval pipeline")
    elif score >= 75:
        print("  🟡 Corpus mostly ready — fix flagged items before eval")
    else:
        print("  🔴 Corpus needs attention before moving to RAG")

    return {"score": score, "passed": passed, "total": total}

# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Nuremberg Scholar Audit v2")
    p.add_argument("--output-dir", default="output", help="Path to output/ directory")
    p.add_argument("--json",       default=None,     help="Write JSON report to this path")
    args = p.parse_args()

    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        print(f"Output directory not found: {output_dir}")
        sys.exit(1)

    print(f"\n{'═'*70}")
    print(f"  Nuremberg Scholar — Output Audit v2")
    print(f"  Output directory: {output_dir.resolve()}")
    print(f"{'═'*70}")

    # Load all collections
    coll_names = ["judgment", "key_docs", "secondary", "sessions", "vol1"]
    collections: dict[str, list] = {}
    for name in coll_names:
        path = output_dir / name
        collections[name] = load_collection(path)

    all_docs = [d for docs in collections.values() for d in docs]
    sessions = collections.get("sessions", [])

    # Run sections
    cov     = audit_coverage(collections)
    cal     = audit_calendar(sessions)
    pages   = audit_pages(sessions)
    spk     = audit_speakers(all_docs)
    flags   = audit_flags(all_docs, collections)
    content = audit_content(collections)
    dates   = audit_dates(all_docs)
    chunks  = audit_chunks(output_dir)
    src     = audit_sources(sessions)
    ready   = audit_readiness(cov, content, chunks, flags)

    # Priority fix list
    section("11. PRIORITY FIX LIST")
    items = []
    if not cov.get("sessions", {}).get("ok"):
        items.append(("🔴 HIGH", f"Sessions: {cov['sessions']['found']}/221"))
    if not cov.get("judgment", {}).get("ok"):
        items.append(("🔴 HIGH", f"Judgment incomplete"))
    if pages.get("no_pages", 0) > 0:
        items.append(("🟡 MED",  f"{pages['no_pages']} sessions still have NO page numbers (Yale digitisation gap)"))
    if spk.get("zero_turn_docs", 0) > 0:
        items.append(("🟡 MED",  f"{spk['zero_turn_docs']} docs have zero speaker turns"))
    if not cov.get("key_docs", {}).get("ok"):
        short = cov["key_docs"]["target"] - cov["key_docs"]["found"]
        items.append(("🟡 MED",  f"key_docs: {cov['key_docs']['found']}/12 — {short} missing"))
    if content.get("empty", 0) > 0:
        items.append(("🟡 MED",  f"{content['empty']} empty documents"))
    if chunks.get("empty", 0) > 0:
        items.append(("🟡 MED",  f"{chunks['empty']} empty chunks in chunks.jsonl"))
    if src.get("duplicates", 0) > 0:
        items.append(("🟡 MED",  f"{src['duplicates']} duplicate content hashes"))
    if flags.get("flags", {}).get("POSSIBLE_TRUNCATION", 0) > 0:
        n = flags["flags"]["POSSIBLE_TRUNCATION"]
        items.append(("⚪ LOW",  f"POSSIBLE_TRUNCATION on {n} docs (mostly false positives from nav footer)"))

    if not items:
        print("\n  ✅ No issues found")
    else:
        print()
        for severity, msg in items:
            print(f"  {severity}  {msg}")

    # JSON report
    if args.json:
        report = {
            "coverage": cov, "calendar": {"missing": cal["missing"]},
            "pages": pages, "speakers": spk, "flags": flags,
            "content": content, "dates": dates, "chunks": chunks,
            "sources": src, "readiness": ready,
            "priority_fixes": [{"severity": s, "message": m} for s, m in items],
        }
        Path(args.json).write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\n  JSON report written to: {args.json}")

    print(f"\n{'═'*70}\n")


if __name__ == "__main__":
    main()
