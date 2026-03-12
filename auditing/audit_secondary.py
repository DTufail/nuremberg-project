"""
audit_secondary.py
==================
Full audit of output/secondary/*.json files.

Reports:
  1. Per-doc-type counts and metadata coverage
  2. Date extraction feasibility (filename vs full_text vs hopeless)
  3. Speaker feasibility
  4. Chunk boundary quality (token distribution)
  5. Sample rows for manual inspection
  6. Actionable fix plan with per-file decisions

Usage:
  python audit_secondary.py
  python audit_secondary.py --verbose        # print every file
  python audit_secondary.py --sample N       # N samples per doc_type (default 3)
"""

import re
import json
import argparse
from pathlib import Path
from collections import Counter, defaultdict

try:
    import tiktoken
    _enc = tiktoken.get_encoding("cl100k_base")
    def count_tokens(text: str) -> int:
        return len(_enc.encode(text))
except ImportError:
    def count_tokens(text: str) -> int:
        return max(1, len(text) // 4)

SECONDARY_DIR = Path("output/secondary")
CHUNKS_FILE   = Path("output/chunks.jsonl")

# ── Date extraction patterns ────────────────────────────────────────────────

# Filename fragment: imt_01-02-46_asp_kalt  →  01-02-46  →  1946-01-02
_FNAME_DATE_RE = re.compile(r'_(\d{2})-(\d{2})-(\d{2})_')

# Full-text: "Wednesday, 2 January 1946" or "2 Jan. 46" or "January 2, 1946"
_TEXT_DATE_LONG = re.compile(
    r'\b(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday),?\s+'
    r'(\d{1,2})\s+(January|February|March|April|May|June|July|August|'
    r'September|October|November|December)\s+(19\d{2})',
    re.IGNORECASE
)
_TEXT_DATE_SHORT = re.compile(
    r'\b(\d{1,2})\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\.?\s+'
    r"'?(\d{2})\b",
    re.IGNORECASE
)
_TEXT_DATE_US = re.compile(
    r'\b(January|February|March|April|May|June|July|August|'
    r'September|October|November|December)\s+(\d{1,2}),?\s+(19\d{2})',
    re.IGNORECASE
)

MONTH_MAP = {
    'january':1,'february':2,'march':3,'april':4,'may':5,'june':6,
    'july':7,'august':8,'september':9,'october':10,'november':11,'december':12,
    'jan':1,'feb':2,'mar':3,'apr':4,'jun':6,'jul':7,'aug':8,
    'sep':9,'oct':10,'nov':11,'dec':12,
}

# Speaker extraction from filename fragment (after last underscore)
# e.g. imt_01-03-46_asp_ohlendorf  →  OHLENDORF
_FNAME_SPEAKER_RE = re.compile(r'_asp_([a-z]+)$', re.IGNORECASE)

# Known witness/defendant names to validate speaker extraction
KNOWN_NAMES = {
    'ohlendorf', 'wisliceny', 'hollriegel', 'schellenberg', 'zelewski',
    'kalt', 'kaltmotion', 'staff', 'lahousen', 'dieckmann', 'hoess',
    'speer', 'sauckel', 'streicher', 'funk', 'frank', 'frick', 'hess',
    'goering', 'ribbentrop', 'keitel', 'rosenberg', 'jodl', 'seyss',
    'neurath', 'raeder', 'doenitz', 'papen', 'schacht', 'fritzsche',
}

# NCA approximate dates by collection
NCA_DATES = {
    'nca_v1': '1946-01-01',   # Vol 1 published Jan 1946
    'nca_v2': '1946-01-01',
    'nca_v3': '1946-01-01',
    'nca_v4': '1946-01-01',
}

JACKSON_DATE  = '1946-06-01'  # Jackson papers span 1945-46; approximate
POHL_DATE     = '1947-11-03'  # Pohl case verdict Nov 1947
WITNESSES_DATE = None          # witness statements vary — need per-file
CASES_DATE    = None
MOTIONS_DATE  = None

# ── Helpers ─────────────────────────────────────────────────────────────────

def extract_date_from_filename(stem: str):
    """Try to extract YYYY-MM-DD from filename stem."""
    m = _FNAME_DATE_RE.search(stem)
    if m:
        mm, dd, yy = m.group(1), m.group(2), m.group(3)
        year = int(yy) + (1900 if int(yy) > 30 else 2000)
        return f"{year}-{mm}-{dd}"
    return None

def extract_date_from_text(text: str):
    """Try to extract first date from full_text."""
    # Long form: Wednesday, 2 January 1946
    m = _TEXT_DATE_LONG.search(text[:2000])
    if m:
        day, month, year = m.group(1), m.group(2).lower(), m.group(3)
        return f"{year}-{MONTH_MAP.get(month, 0):02d}-{int(day):02d}"
    # US form: January 2, 1946
    m = _TEXT_DATE_US.search(text[:2000])
    if m:
        month, day, year = m.group(1).lower(), m.group(2), m.group(3)
        return f"{year}-{MONTH_MAP.get(month, 0):02d}-{int(day):02d}"
    # Short form: 2 Jan. 46
    m = _TEXT_DATE_SHORT.search(text[:2000])
    if m:
        day, month, yy = m.group(1), m.group(2).lower(), m.group(3)
        year = int(yy) + (1900 if int(yy) > 30 else 2000)
        return f"{year}-{MONTH_MAP.get(month, 0):02d}-{int(day):02d}"
    return None

def extract_speaker_from_filename(stem: str):
    """Try to get witness name from filename fragment."""
    m = _FNAME_SPEAKER_RE.search(stem)
    if m:
        name = m.group(1).lower()
        # Filter out non-name suffixes
        if name in ('asp', 'motion', 'inter', 'annex1', 'annex2', 'staff',
                    'part01', 'part02', 'part03', 'preface', 'v2', 'v3'):
            return None
        return name.upper()
    return None

def infer_date_for_doc(stem: str, doc_type: str, full_text: str):
    """Best-effort date inference. Returns (date_iso, source) or (None, 'hopeless')."""
    # 1. Filename date (session witness docs)
    d = extract_date_from_filename(stem)
    if d:
        return d, 'filename'
    # 2. Doc-type known approximate dates
    if doc_type in NCA_DATES:
        return NCA_DATES[doc_type], 'doc_type_approx'
    if doc_type == 'jackson':
        return JACKSON_DATE, 'doc_type_approx'
    if doc_type == 'pohl':
        return POHL_DATE, 'doc_type_approx'
    # 3. Full-text scan (witnesses, cases, motions)
    d = extract_date_from_text(full_text)
    if d:
        return d, 'full_text'
    return None, 'hopeless'

def classify_boundary_quality(token_count: int) -> str:
    if token_count < 90:
        return 'SHORT'
    elif token_count <= 512:
        return 'OK'
    else:
        return 'OVERLONG'

# ── Main audit ───────────────────────────────────────────────────────────────

def run(verbose: bool, sample_n: int):
    files = sorted(SECONDARY_DIR.glob("*.json"))
    if not files:
        print(f"ERROR: No JSON files found in {SECONDARY_DIR}")
        return

    print(f"\n{'='*70}")
    print(f"  SECONDARY COLLECTION AUDIT")
    print(f"  {len(files)} documents in {SECONDARY_DIR}")
    print(f"{'='*70}\n")

    # Per-file records
    records = []
    for fp in files:
        try:
            doc = json.loads(fp.read_text(encoding='utf-8'))
        except Exception as e:
            print(f"  ✗ {fp.name}: {e}")
            continue

        stem     = fp.stem
        doc_type = doc.get('doc_type', '?')
        url      = doc.get('url', '')
        full_text = doc.get('full_text', '') or ''
        word_count = doc.get('word_count', 0)

        date_iso, date_src = infer_date_for_doc(stem, doc_type, full_text)
        speaker = extract_speaker_from_filename(stem)
        # Only trust speaker if it looks like an actual name
        if speaker and speaker.lower() not in KNOWN_NAMES:
            speaker_src = 'filename_unverified'
        elif speaker:
            speaker_src = 'filename_known'
        else:
            speaker_src = 'none'

        # Token distribution for this doc
        tok = count_tokens(full_text) if full_text else 0

        records.append({
            'file':       fp.name,
            'stem':       stem,
            'doc_type':   doc_type,
            'url':        url,
            'word_count': word_count,
            'tok_total':  tok,
            'date_iso':   date_iso,
            'date_src':   date_src,
            'speaker':    speaker,
            'speaker_src': speaker_src,
            'has_text':   bool(full_text.strip()),
        })

    # ── Section 1: doc_type summary ─────────────────────────────────────────
    print("─── 1. DOC_TYPE BREAKDOWN ─────────────────────────────────────────")
    by_type = defaultdict(list)
    for r in records:
        by_type[r['doc_type']].append(r)

    print(f"  {'doc_type':<14} {'count':>6}  {'has_date%':>10}  {'has_speaker%':>13}  {'has_text%':>10}")
    print(f"  {'─'*14} {'─'*6}  {'─'*10}  {'─'*13}  {'─'*10}")
    for dt in sorted(by_type):
        grp = by_type[dt]
        n = len(grp)
        n_date    = sum(1 for r in grp if r['date_iso'])
        n_speaker = sum(1 for r in grp if r['speaker'])
        n_text    = sum(1 for r in grp if r['has_text'])
        print(f"  {dt:<14} {n:>6}  {n_date/n*100:>9.0f}%  {n_speaker/n*100:>12.0f}%  {n_text/n*100:>9.0f}%")
    print()

    # ── Section 2: Date source breakdown ────────────────────────────────────
    print("─── 2. DATE INFERENCE SOURCES ─────────────────────────────────────")
    date_srcs = Counter(r['date_src'] for r in records)
    for src, cnt in date_srcs.most_common():
        pct = cnt / len(records) * 100
        print(f"  {src:<22}  {cnt:>4}  ({pct:.0f}%)")
    print()

    hopeless = [r for r in records if r['date_src'] == 'hopeless']
    if hopeless:
        print(f"  ⚠  {len(hopeless)} docs with no inferable date:")
        for r in hopeless[:10]:
            print(f"       {r['file']}  [{r['doc_type']}]  words={r['word_count']}")
        if len(hopeless) > 10:
            print(f"       ... and {len(hopeless)-10} more")
    print()

    # ── Section 3: Speaker breakdown ────────────────────────────────────────
    print("─── 3. SPEAKER INFERENCE ──────────────────────────────────────────")
    spk_srcs = Counter(r['speaker_src'] for r in records)
    for src, cnt in spk_srcs.most_common():
        print(f"  {src:<24}  {cnt:>4}")
    print()
    known_spk = [r for r in records if r['speaker_src'] == 'filename_known']
    if known_spk:
        print(f"  Verified speakers ({len(known_spk)} docs):")
        for r in known_spk[:20]:
            print(f"    {r['speaker']:<20}  {r['file']}")
        if len(known_spk) > 20:
            print(f"    ... and {len(known_spk)-20} more")
    print()

    # ── Section 4: Token / chunk quality ────────────────────────────────────
    print("─── 4. TOKEN DISTRIBUTION (full_text before chunking) ─────────────")
    toks = [r['tok_total'] for r in records if r['tok_total'] > 0]
    if toks:
        toks_s = sorted(toks)
        n = len(toks_s)
        print(f"  count   : {n}")
        print(f"  min     : {toks_s[0]:,}")
        print(f"  p25     : {toks_s[n//4]:,}")
        print(f"  median  : {toks_s[n//2]:,}")
        print(f"  p75     : {toks_s[3*n//4]:,}")
        print(f"  p90     : {toks_s[int(n*0.9)]:,}")
        print(f"  max     : {toks_s[-1]:,}")
        print(f"  empty   : {sum(1 for r in records if not r['has_text'])}")
    print()

    # ── Section 5: Existing chunks.jsonl audit ───────────────────────────────
    if CHUNKS_FILE.exists():
        print("─── 5. EXISTING CHUNKS.JSONL — SECONDARY CHUNKS ───────────────────")
        sec_chunks = []
        with CHUNKS_FILE.open(encoding='utf-8') as f:
            for line in f:
                try:
                    c = json.loads(line)
                    if c.get('collection') == 'secondary':
                        sec_chunks.append(c)
                except Exception:
                    pass

        print(f"  Total secondary chunks : {len(sec_chunks)}")
        n_no_date    = sum(1 for c in sec_chunks if not c.get('date_iso'))
        n_no_speaker = sum(1 for c in sec_chunks if not c.get('speaker'))
        tok_counts   = [c.get('token_count', 0) for c in sec_chunks]
        bq = Counter(classify_boundary_quality(t) for t in tok_counts)

        print(f"  Missing date_iso       : {n_no_date}  ({n_no_date/len(sec_chunks)*100:.0f}%)" if sec_chunks else "")
        print(f"  Missing speaker        : {n_no_speaker}  ({n_no_speaker/len(sec_chunks)*100:.0f}%)" if sec_chunks else "")
        print(f"  Boundary quality:")
        for label in ('OK', 'SHORT', 'OVERLONG'):
            cnt = bq.get(label, 0)
            pct = cnt / len(sec_chunks) * 100 if sec_chunks else 0
            print(f"    {label:<10} {cnt:>5}  ({pct:.0f}%)")

        # Patchable vs not
        # For each chunk, check if we can infer date from the source file
        chunk_slug_to_doc = {}
        for r in records:
            slug = r['stem']
            chunk_slug_to_doc[slug] = r

        patchable = 0
        not_patchable = 0
        for c in sec_chunks:
            slug = c.get('slug', '')
            doc  = chunk_slug_to_doc.get(slug)
            if doc and doc['date_iso']:
                patchable += 1
            else:
                not_patchable += 1
        print(f"\n  Chunks patchable (date inferable)  : {patchable}")
        print(f"  Chunks not patchable (hopeless)    : {not_patchable}")
        print()
    else:
        print("─── 5. EXISTING CHUNKS.JSONL — NOT FOUND ──────────────────────────\n")

    # ── Section 6: Fix plan ──────────────────────────────────────────────────
    print("─── 6. FIX PLAN ────────────────────────────────────────────────────")
    print()
    print("  Strategy: patch chunks.jsonl in-place (no re-scrape, no re-embed)")
    print()
    print("  Per doc_type actions:")
    actions = {
        'nca_v1':   ('date_iso=1946-01-01 (approx)', 'speaker=None (legal doc)'),
        'nca_v2':   ('date_iso=1946-01-01 (approx)', 'speaker=None (legal doc)'),
        'nca_v3':   ('date_iso=1946-01-01 (approx)', 'speaker=None (legal doc)'),
        'nca_v4':   ('date_iso=1946-01-01 (approx)', 'speaker=None (legal doc)'),
        'jackson':  ('date_iso=1946-06-01 (approx)', 'speaker=JACKSON (inferred)'),
        'pohl':     ('date_iso=1947-11-03',           'speaker=None'),
        'witnesses':('date_iso from full_text or filename', 'speaker from filename if known'),
        'cases':    ('date_iso from full_text',       'speaker=None'),
        'motions':  ('date_iso from filename or full_text', 'speaker from filename if known'),
    }
    for dt, (date_act, spk_act) in actions.items():
        cnt = len(by_type.get(dt, []))
        print(f"  [{dt:<10}] ({cnt:>3} docs)  date: {date_act}")
        print(f"  {'':<13}              speaker: {spk_act}")
    print()

    # ── Section 7: Sample rows ───────────────────────────────────────────────
    print("─── 7. SAMPLE ROWS ─────────────────────────────────────────────────")
    for dt in sorted(by_type):
        grp = by_type[dt]
        samples = grp[:sample_n]
        print(f"\n  [{dt}]")
        for r in samples:
            print(f"    file     : {r['file']}")
            print(f"    url      : {r['url']}")
            print(f"    date_iso : {r['date_iso']}  (src: {r['date_src']})")
            print(f"    speaker  : {r['speaker']}  (src: {r['speaker_src']})")
            print(f"    words    : {r['word_count']}  tokens: {r['tok_total']:,}")
            print()

    # ── Verbose: every file ──────────────────────────────────────────────────
    if verbose:
        print("─── VERBOSE: ALL FILES ─────────────────────────────────────────────")
        print(f"  {'file':<45} {'doc_type':<12} {'date_iso':<12} {'date_src':<20} {'speaker'}")
        print(f"  {'─'*45} {'─'*12} {'─'*12} {'─'*20} {'─'*20}")
        for r in records:
            print(f"  {r['file']:<45} {r['doc_type']:<12} {str(r['date_iso']):<12} {r['date_src']:<20} {r['speaker']}")

    print(f"\n{'='*70}")
    print(f"  AUDIT COMPLETE — {len(records)} documents audited")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description="Nuremberg secondary collection audit")
    parser.add_argument('--verbose', action='store_true',
                        help='Print every file in a table')
    parser.add_argument('--sample', type=int, default=3, metavar='N',
                        help='Number of sample rows per doc_type (default: 3)')
    args = parser.parse_args()
    run(args.verbose, args.sample)


if __name__ == '__main__':
    main()
