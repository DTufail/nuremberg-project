"""
investigate_secondary.py  —  Diagnose secondary collection chunk quality
=========================================================================
Run from nuremberg-project/:
    python investigate_secondary.py

Checks:
  1. All chunks from the 'secondary' collection — metadata completeness
  2. Chunks starting mid-sentence (bad boundary)
  3. The specific chap16_part01 slug — what document is it, full body
  4. All secondary chunks with missing date/speaker
  5. Token count distribution for secondary chunks
"""

import json
from pathlib import Path
from collections import defaultdict

META_FILE   = Path("output/index/metadata.jsonl")
CHUNKS_FILE = Path("output/chunks.jsonl")

# ── Load all secondary chunks from metadata ───────────────────────────────────

print("=" * 60)
print("SECONDARY COLLECTION AUDIT")
print("=" * 60)

secondary = []
with META_FILE.open(encoding="utf-8") as f:
    for i, line in enumerate(f):
        line = line.strip()
        if not line:
            continue
        m = json.loads(line)
        if m.get("collection") == "secondary":
            m["_row_idx"] = i   # FAISS row index
            secondary.append(m)

print(f"\nTotal secondary chunks : {len(secondary)}")

# ── 1. Metadata completeness ─────────────────────────────────────────────────

missing_date    = [c for c in secondary if not c.get("date_iso")]
missing_speaker = [c for c in secondary if not c.get("speaker")]
missing_both    = [c for c in secondary if not c.get("date_iso") and not c.get("speaker")]

print(f"\nMissing date_iso       : {len(missing_date)} / {len(secondary)}")
print(f"Missing speaker        : {len(missing_speaker)} / {len(secondary)}")
print(f"Missing both           : {len(missing_both)} / {len(secondary)}")

# ── 2. Bad chunk boundaries (starts mid-sentence) ────────────────────────────

import re
mid_sentence_starts = []
for c in secondary:
    body = c.get("body", "").strip()
    if body and body[0].islower():
        mid_sentence_starts.append(c)
    # Also check for starts with lowercase after common mid-sentence patterns
    elif body and re.match(r'^[a-z]|^(these|those|such|this|that|which|who|and|but|or|of|to|in|for|with|by)\b', body, re.IGNORECASE) and body[0].islower():
        mid_sentence_starts.append(c)

print(f"\nChunks starting mid-sentence: {len(mid_sentence_starts)}")
for c in mid_sentence_starts[:5]:
    print(f"  slug={c.get('slug')}  body[:80]={repr(c.get('body','')[:80])}")

# ── 3. Investigate chap16_part01 specifically ────────────────────────────────

print(f"\n{'─'*60}")
print("SPECIFIC CHUNK: chap16_part01")
print(f"{'─'*60}")

chap16 = [c for c in secondary if c.get("slug") == "chap16_part01"]
if chap16:
    for c in chap16:
        print(f"  chunk_id    : {c.get('chunk_id')}")
        print(f"  row_idx     : {c.get('_row_idx')}")
        print(f"  date_iso    : {c.get('date_iso')}")
        print(f"  speaker     : {c.get('speaker')}")
        print(f"  page_number : {c.get('page_number')}")
        print(f"  token_count : {c.get('token_count')}")
        print(f"  chunk_index : {c.get('chunk_index')} / {c.get('total_chunks')}")
        print(f"  source_url  : {c.get('source_url')}")
        print(f"\n  FULL BODY:\n")
        print(c.get("body", ""))
        print()
else:
    print("  Not found in metadata.jsonl — checking chunks.jsonl...")
    with CHUNKS_FILE.open(encoding="utf-8") as f:
        for line in f:
            c = json.loads(line.strip())
            if c.get("slug") == "chap16_part01":
                print(json.dumps(c, indent=2, ensure_ascii=False))

# ── 4. All secondary slugs and their metadata completeness ───────────────────

print(f"\n{'─'*60}")
print("ALL SECONDARY SLUGS")
print(f"{'─'*60}")

slug_groups = defaultdict(list)
for c in secondary:
    slug_groups[c.get("slug", "NO_SLUG")].append(c)

print(f"\n{'Slug':<30} {'Chunks':>6} {'Has date':>9} {'Has speaker':>12} {'Source URL'}")
print("-" * 80)
for slug, chunks in sorted(slug_groups.items()):
    has_date    = sum(1 for c in chunks if c.get("date_iso"))
    has_speaker = sum(1 for c in chunks if c.get("speaker"))
    url         = chunks[0].get("source_url", "")[:40] if chunks else ""
    print(f"  {slug:<28} {len(chunks):>6} {has_date:>9} {has_speaker:>12}  {url}")

# ── 5. Token count distribution ──────────────────────────────────────────────

print(f"\n{'─'*60}")
print("TOKEN COUNT DISTRIBUTION (secondary)")
print(f"{'─'*60}")

token_counts = sorted(c.get("token_count", 0) for c in secondary)
if token_counts:
    n = len(token_counts)
    print(f"  min    : {token_counts[0]}")
    print(f"  p25    : {token_counts[n//4]}")
    print(f"  median : {token_counts[n//2]}")
    print(f"  p75    : {token_counts[3*n//4]}")
    print(f"  p95    : {token_counts[int(n*0.95)]}")
    print(f"  max    : {token_counts[-1]}")

# ── 6. Check what the previous chunk looks like (context for mid-sentence) ───

print(f"\n{'─'*60}")
print("CHUNK BEFORE chap16_part01 (to see what sentence was cut)")
print(f"{'─'*60}")

if chap16:
    target_idx = chap16[0].get("chunk_index", 1)
    target_slug_base = "chap16"
    prev_chunks = [
        c for c in secondary
        if c.get("slug", "").startswith(target_slug_base)
        and c.get("chunk_index") == target_idx - 1
    ]
    if prev_chunks:
        prev = prev_chunks[0]
        print(f"  slug={prev.get('slug')}  chunk_index={prev.get('chunk_index')}")
        print(f"  Last 200 chars of body:")
        print(f"  ...{prev.get('body','')[-200:]}")
    else:
        # Show all chap16 chunks in order
        all_chap16 = sorted(
            [c for c in secondary if "chap16" in c.get("slug","")],
            key=lambda x: x.get("chunk_index", 0)
        )
        print(f"  Found {len(all_chap16)} chap16 chunks:")
        for c in all_chap16:
            print(f"    chunk_index={c.get('chunk_index')}  slug={c.get('slug')}  "
                  f"body[:60]={repr(c.get('body','')[:60])}")

print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
print(f"  Total secondary chunks   : {len(secondary)}")
print(f"  Missing date             : {len(missing_date)} ({100*len(missing_date)//max(len(secondary),1)}%)")
print(f"  Missing speaker          : {len(missing_speaker)} ({100*len(missing_speaker)//max(len(secondary),1)}%)")
print(f"  Bad chunk boundaries     : {len(mid_sentence_starts)}")
print(f"  Unique slugs             : {len(slug_groups)}")
