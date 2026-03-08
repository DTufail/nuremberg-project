"""
missing_days.py
===============
Lists all weekdays in the IMT range (Nov 14, 1945 – Oct 1, 1946)
that are NOT covered by a scraped session JSON.
"""

import json
from pathlib import Path
from datetime import date, timedelta

OUTPUT_DIR = Path("output/sessions")

# Collect all scraped dates
scraped = set()
for fp in OUTPUT_DIR.glob("*.json"):
    try:
        d = json.loads(fp.read_text(encoding="utf-8"))
        iso = d.get("date_iso")
        if iso:
            scraped.add(iso)
    except Exception:
        pass

# Walk every weekday in the IMT range
start = date(1945, 11, 14)
end   = date(1946, 10, 1)

missing = []
cur = start
while cur <= end:
    if cur.weekday() < 5:  # Mon–Fri
        iso = cur.isoformat()
        if iso not in scraped:
            missing.append(cur)
    cur += timedelta(days=1)

print(f"Scraped sessions : {len(scraped)}")
print(f"Missing weekdays : {len(missing)}")
print()

# Group into consecutive runs
runs = []
run_start = None
prev = None
for d in missing:
    if prev is None or (d - prev).days > 3:
        if run_start:
            runs.append((run_start, prev))
        run_start = d
    prev = d
if run_start:
    runs.append((run_start, prev))

print(f"{'Date':<14} {'Day':<10} {'Gap note'}")
print("─" * 50)
prev_d = None
for d in missing:
    gap = f"  (+{(d - prev_d).days}d gap)" if prev_d and (d - prev_d).days > 7 else ""
    print(f"{d.isoformat():<14} {d.strftime('%A'):<10}{gap}")
    prev_d = d

print()
print("─" * 50)
print(f"Consecutive runs of 3+ missing weekdays:")
for s, e in runs:
    days = sum(1 for d in missing if s <= d <= e)
    if days >= 3:
        print(f"  {s} → {e}  ({days} days)")
