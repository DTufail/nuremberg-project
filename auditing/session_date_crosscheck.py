"""
session_date_crosscheck.py
==========================
For every session JSON in output/sessions/, extracts:
  1. The declared date_iso from the JSON metadata
  2. The internal date header from the transcript text
     e.g. "sitting at Nurnberg, Germany, on 6 March 1946"
  3. The adjournment line at the end of the session
     e.g. "The Tribunal adjourned until 7 March 1946 at 1000 hours"
  4. The NEXT session's date_iso

Then cross-checks:
  A. metadata date == internal header date
  B. adjournment target == next session date  (or known recess)
  C. flags sessions where adjournment target is missing from corpus

Usage:
    python session_date_crosscheck.py
    python session_date_crosscheck.py --verbose        # show all sessions
    python session_date_crosscheck.py --problems-only  # only show mismatches
    python session_date_crosscheck.py --output report.txt
"""

import re
import json
import argparse
from pathlib import Path
from datetime import datetime, date, timedelta

SESSIONS_DIR = Path("output/sessions")

MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12
}

# Known legitimate gaps — adjourn target not in corpus because no session existed
KNOWN_RECESSES = {
    # Christmas/New Year 1945-46
    *[str((date(1945, 12, 21) + timedelta(days=i)).isoformat())
      for i in range(12)],
    # Easter 1946
    "1946-04-19", "1946-04-20", "1946-04-21", "1946-04-22",
    # September deliberation period
    *[str((date(1946, 9, 2) + timedelta(days=i)).isoformat())
      for i in range(26)],
    # Weekends — we only flag weekdays, but include common ones
}

# ── Date extraction helpers ───────────────────────────────────────────────────

DATE_WRITTEN_RE = re.compile(
    r'\b(\d{1,2})\s+(January|February|March|April|May|June|July|'
    r'August|September|October|November|December)\s+(19\d{2})\b',
    re.IGNORECASE
)

HEADER_RE = re.compile(
    r'sitting at\s+N[uü]rnberg.*?on\s+(\d{1,2}\s+\w+\s+19\d{2})',
    re.IGNORECASE | re.DOTALL
)

OFFICIAL_TRANSCRIPT_RE = re.compile(
    r'official\s+transcript.*?on\s+(\d{1,2}\s+\w+\s+19\d{2})',
    re.IGNORECASE | re.DOTALL
)

ADJOURN_RE = re.compile(
    r'(?:tribunal\s+)?adjourned?\s+until\s+(\d{1,2}\s+\w+\s+19\d{2})',
    re.IGNORECASE
)

ADJOURN_OPEN_RE = re.compile(
    r'(?:will\s+now\s+adjourn|tribunal\s+will\s+(?:now\s+)?adjourn'
    r'|we\s+will\s+adjourn\s+now'
    r'|court\s+(?:will\s+)?adjourn)',
    re.IGNORECASE
)

def parse_written_date(s: str) -> str | None:
    """Convert '6 March 1946' → '1946-03-06'"""
    m = DATE_WRITTEN_RE.search(s)
    if not m:
        return None
    day   = int(m.group(1))
    month = MONTHS.get(m.group(2).lower())
    year  = int(m.group(3))
    if not month:
        return None
    try:
        return date(year, month, day).isoformat()
    except ValueError:
        return None

DAY_HEADER_RE = re.compile(
    r'^(?:MONDAY|TUESDAY|WEDNESDAY|THURSDAY|FRIDAY|SATURDAY|SUNDAY)'
    r',?\s+(\d{1,2}\s+\w+\s+19\d{2})',
    re.IGNORECASE | re.MULTILINE
)

def extract_internal_date(text: str) -> str | None:
    """
    Extract session date from the official transcript header ONLY.
    Does NOT fall back to arbitrary date mentions (those are exhibit dates).

    Recognised formats:
      "sitting at Nurnberg, Germany, on 6 March 1946"
      "WEDNESDAY, 26 NOVEMBER 1945"  (Yale all-caps day header)
      "official transcript ... on 7 March 1946, 1000-1300"
    """
    header = text[:2000]

    # Format 1: "sitting at Nurnberg ... on DD Month YYYY"
    for pattern in (HEADER_RE, OFFICIAL_TRANSCRIPT_RE):
        m = pattern.search(header)
        if m:
            d = parse_written_date(m.group(1))
            if d:
                return d

    # Format 2: "WEDNESDAY, 26 NOVEMBER 1945" day-of-week header line
    m = DAY_HEADER_RE.search(header)
    if m:
        d = parse_written_date(m.group(1))
        if d:
            return d

    # No match — return None rather than a wrong date from an exhibit
    return None

def extract_adjournment(text: str) -> tuple[str | None, str | None]:
    """
    Returns (adjourn_target_iso, adjourn_raw_text)
    Searches the last 3000 chars for adjournment lines.
    """
    tail = text[-3000:]

    # "adjourned until DD Month YYYY"
    matches = list(ADJOURN_RE.finditer(tail))
    if matches:
        last = matches[-1]
        raw  = last.group(0)
        d    = parse_written_date(last.group(1))
        return d, raw

    # "The Tribunal will now adjourn" (no date — open-ended)
    if ADJOURN_OPEN_RE.search(tail):
        return "OPEN", ADJOURN_OPEN_RE.search(tail).group(0)

    return None, None

def get_full_text(doc: dict) -> str:
    """Reconstruct full text from turns or raw paragraphs."""
    turns = doc.get("turns", [])
    if turns:
        return " ".join(t.get("text", "") for t in turns)
    return doc.get("raw_text", "")

# ── Load sessions ─────────────────────────────────────────────────────────────

def load_sessions() -> list[dict]:
    sessions = []
    for fp in sorted(SESSIONS_DIR.glob("*.json")):
        try:
            doc = json.loads(fp.read_text(encoding="utf-8"))
            doc["_filepath"] = str(fp)
            sessions.append(doc)
        except Exception as e:
            print(f"  LOAD ERROR {fp.name}: {e}")
    return sessions

# ── Main check ────────────────────────────────────────────────────────────────

def run(verbose: bool, problems_only: bool, output_path: str | None):
    sessions = load_sessions()
    if not sessions:
        print(f"No sessions found in {SESSIONS_DIR}")
        return

    # Sort by date_iso
    sessions.sort(key=lambda d: d.get("date_iso", ""))

    # Build date → session map
    date_map = {s["date_iso"]: s for s in sessions if s.get("date_iso")}

    lines      = []
    problems   = []
    ok_count   = 0
    warn_count = 0

    def log(s):
        lines.append(s)
        if not problems_only or "[" in s:
            print(s)

    log("=" * 70)
    log("  Nuremberg Scholar — Session Date Cross-Check")
    log(f"  {len(sessions)} sessions loaded from {SESSIONS_DIR}")
    log("=" * 70)

    for i, session in enumerate(sessions):
        meta_date   = session.get("date_iso", "MISSING")
        source      = session.get("source", "yale")
        slug        = session.get("slug", Path(session["_filepath"]).stem)
        text        = get_full_text(session)

        internal_date          = extract_internal_date(text)
        adjourn_target, adj_raw = extract_adjournment(text)

        # Next session in sorted list
        next_session  = sessions[i + 1] if i + 1 < len(sessions) else None
        next_date     = next_session.get("date_iso") if next_session else None

        # ── Checks ────────────────────────────────────────────────────────────

        issues = []

        # A: metadata vs internal header
        if internal_date and internal_date != meta_date:
            issues.append(f"DATE MISMATCH: metadata={meta_date} but header says {internal_date}")

        # B: adjournment target vs next session
        if adjourn_target and adjourn_target not in ("OPEN", None):
            if next_date and adjourn_target != next_date:
                if adjourn_target in KNOWN_RECESSES:
                    # adjourn into a known recess — next session after recess is fine
                    pass
                elif adjourn_target in date_map:
                    # target exists but isn't the immediate next — skip in corpus
                    issues.append(
                        f"SESSION SKIPPED: adjourned to {adjourn_target} "
                        f"but next session is {next_date}"
                    )
                else:
                    # target date not in corpus at all
                    issues.append(
                        f"MISSING SESSION: adjourned to {adjourn_target} "
                        f"but {adjourn_target} not in corpus"
                    )
            elif not next_date:
                # last session
                pass

        elif adjourn_target == "OPEN":
            # open adjournment — just note it
            if verbose:
                issues.append(f"INFO: open adjournment (no target date)")

        elif adjourn_target is None:
            if verbose:
                issues.append("INFO: no adjournment line found")

        # ── Output ────────────────────────────────────────────────────────────

        has_problem = any("MISMATCH" in x or "MISSING" in x or "SKIPPED" in x
                          for x in issues)

        if has_problem:
            warn_count += 1
            problems.append({
                "date":            meta_date,
                "slug":            slug,
                "internal_date":   internal_date,
                "adjourn_target":  adjourn_target,
                "adjourn_raw":     adj_raw,
                "next_session":    next_date,
                "issues":          issues,
            })
        else:
            ok_count += 1

        if verbose or has_problem:
            prefix = "❌" if has_problem else "✅"
            log(f"\n{prefix} {meta_date}  [{slug}]  source={source}")
            log(f"   header date : {internal_date or '(not found)'}")
            log(f"   adjourned to: {adjourn_target or '(not found)'}  ← {adj_raw or ''}")
            log(f"   next session: {next_date or '(last)'}")
            for issue in issues:
                log(f"   ⚠️  {issue}")

    # ── Summary ───────────────────────────────────────────────────────────────

    log("\n" + "=" * 70)
    log("  SUMMARY")
    log("=" * 70)
    log(f"  Sessions checked   : {len(sessions)}")
    log(f"  ✅ Clean           : {ok_count}")
    log(f"  ❌ Problems        : {warn_count}")

    if problems:
        log(f"\n  Problem sessions:")
        for p in problems:
            log(f"    {p['date']}  {p['slug']}")
            for issue in p["issues"]:
                log(f"      → {issue}")

    # ── Write JSON report ─────────────────────────────────────────────────────

    report_path = output_path or "session_crosscheck_report.json"
    report = {
        "total":    len(sessions),
        "ok":       ok_count,
        "problems": warn_count,
        "issues":   problems,
    }
    Path(report_path).write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log(f"\n  Full report written to: {report_path}")


def main():
    p = argparse.ArgumentParser(description="Cross-check session dates vs adjournment lines")
    p.add_argument("--verbose",       action="store_true", help="Print every session")
    p.add_argument("--problems-only", action="store_true", help="Only print sessions with issues")
    p.add_argument("--output",        type=str, default=None, help="JSON report output path")
    args = p.parse_args()
    run(args.verbose, args.problems_only, args.output)


if __name__ == "__main__":
    main()