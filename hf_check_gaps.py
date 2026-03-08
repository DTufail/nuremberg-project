"""
hf_check_gaps.py
================
Checks the HuggingFace Adherence/nuremberg-trials-complete dataset
for records on the genuinely missing dates (currently just Mar 6-7 1946).

Prints record counts, page ranges, speaker counts, and sample text
so you can confirm whether HF has usable content to patch from.

Usage:
    pip install datasets
    python hf_check_gaps.py
"""

from datasets import load_dataset

# Dates to check — confirmed Yale gaps (not recesses)
TARGET_DATES = [
    "1946-03-06",
    "1946-03-07",
]

print("Loading HuggingFace dataset (trial_id=7, IMT)...")
ds = load_dataset(
    "Adherence/nuremberg-trials-complete",
    split="test",
    streaming=True,  # don't download 153k records
)

# Collect matching records
hits = {d: [] for d in TARGET_DATES}

for record in ds:
    iso = record.get("date_iso") or record.get("date") or ""
    # HF has year-off-by-one bug on some trials but trial_id=7 (IMT) should be correct
    if iso[:10] in hits:
        hits[iso[:10]].append(record)

# ── Report ────────────────────────────────────────────────────────────────────

for date_str in TARGET_DATES:
    records = hits[date_str]
    print(f"\n{'─'*60}")
    print(f"Date: {date_str}  —  {len(records)} records found")

    if not records:
        print("  ❌ No records in HF dataset for this date")
        continue

    # Show field names from first record
    print(f"  Fields: {list(records[0].keys())}")

    # Page range
    pages = [r.get("page_start") or r.get("seq") for r in records if r.get("page_start") or r.get("seq")]
    if pages:
        print(f"  Page/seq range: {min(pages)} → {max(pages)}")

    # Speakers
    speakers = list(dict.fromkeys(
        r.get("speaker", "").strip() for r in records if r.get("speaker", "").strip()
    ))
    print(f"  Unique speakers ({len(speakers)}): {speakers[:10]}")

    # Word count estimate
    total_words = sum(len((r.get("text") or r.get("content") or "").split()) for r in records)
    print(f"  Total words: {total_words:,}")

    # Sample — first 3 records
    print(f"  Sample records:")
    for r in records[:3]:
        text = (r.get("text") or r.get("content") or "")[:200]
        speaker = r.get("speaker", "?")
        page = r.get("page_start") or r.get("seq") or "?"
        print(f"    [{page}] {speaker}: {text!r}")