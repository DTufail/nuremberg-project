"""
adjournment_check.py
====================
For each session that precedes a missing weekday gap,
extracts the adjournment line from the transcript text.

Pattern: [The Tribunal adjourned until ...]

Usage:
    python adjournment_check.py
"""

import json
import re
from pathlib import Path
from datetime import date, timedelta

OUTPUT_DIR = Path("output/sessions")
ADJOURN_RE = re.compile(
    r'\[The Tribunal adjourned[^\]]{0,120}\]',
    re.IGNORECASE
)

# ── Load all scraped sessions ──────────────────────────────────────────────────

sessions = {}
for fp in OUTPUT_DIR.glob("*.json"):
    try:
        d = json.loads(fp.read_text(encoding="utf-8"))
        iso = d.get("date_iso")
        if iso:
            sessions[iso] = d
    except Exception:
        pass

# ── Find missing weekdays ──────────────────────────────────────────────────────

start = date(1945, 11, 14)
end   = date(1946, 10, 1)

scraped = set(sessions.keys())
missing = set()
cur = start
while cur <= end:
    if cur.weekday() < 5:
        if cur.isoformat() not in scraped:
            missing.add(cur)
    cur += timedelta(days=1)

# ── For each missing day, find the last session before it ─────────────────────

checked = set()
results = []

for m in sorted(missing):
    # walk back to find the most recent scraped session
    prev = m - timedelta(days=1)
    for _ in range(10):
        if prev.isoformat() in scraped:
            break
        prev -= timedelta(days=1)
    else:
        continue

    if prev in checked:
        continue
    checked.add(prev)

    doc = sessions[prev.isoformat()]

    # Extract adjournment lines from turns text
    full_text = " ".join(
        t.get("text", "") for t in doc.get("turns", [])
    )
    # Also check preamble
    full_text += " " + doc.get("preamble", "")

    matches = ADJOURN_RE.findall(full_text)

    # Take the last one (final adjournment of the day)
    adjourn = matches[-1].strip() if matches else "— no adjournment line found —"

    results.append((prev, m, adjourn))

# ── Print ──────────────────────────────────────────────────────────────────────

print(f"{'Last session':<14}  {'First missing day':<18}  Adjournment line")
print("─" * 90)
for prev, first_missing, line in results:
    print(f"{prev.isoformat():<14}  {first_missing.isoformat():<18}  {line}")
