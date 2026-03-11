"""
patch_secondary.py  (v2)
========================
Fixes three issues found in dry-run v1:

  FIX 1 — Slug matching uses source_url (not slug field) to look up source docs.
           Chunk slugs may contain #fragments; source docs are keyed by base URL.

  FIX 2 — Duplicate detection: date-slug #fragment chunks whose base URL matches
           a primary session are deleted. Exhibit annexes (007-ps#annex1) are kept.

  FIX 3 — Rechunking reads from source doc full_text (not from existing chunk text).
           Overlong threshold raised to 600 to avoid re-chunking near-boundary docs.

Usage:
  python patch_secondary.py              # dry run
  python patch_secondary.py --apply      # write patched chunks.jsonl
  python patch_secondary.py --apply --no-rechunk  # metadata patch only
"""

import re
import json
import shutil
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

SECONDARY_DIR   = Path("output/secondary")
SESSIONS_DIR    = Path("output/sessions")
CHUNKS_FILE     = Path("output/chunks.jsonl")
BACKUP_FILE     = Path("output/chunks.jsonl.bak")

SECONDARY_MAX   = 512
OVERLAP         = 64
OVERLONG_THRESH = 600   # only rechunk if token_count > this

# ── Date inference ─────────────────────────────────────────────────────────────

_FNAME_DATE_RE = re.compile(r'_(\d{2})-(\d{2})-(\d{2})(?:_|$)')
_TEXT_DATE_LONG = re.compile(
    r'\b(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday),?\s+'
    r'(\d{1,2})\s+(January|February|March|April|May|June|July|August|'
    r'September|October|November|December)\s+(19\d{2})', re.IGNORECASE)
_TEXT_DATE_SHORT = re.compile(
    r'\b(\d{1,2})\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\.?\s+'
    r"'?(\d{2})\b", re.IGNORECASE)
_TEXT_DATE_US = re.compile(
    r'\b(January|February|March|April|May|June|July|August|'
    r'September|October|November|December)\s+(\d{1,2}),?\s+(19\d{2})',
    re.IGNORECASE)

MONTH_MAP = {
    'january':1,'february':2,'march':3,'april':4,'may':5,'june':6,
    'july':7,'august':8,'september':9,'october':10,'november':11,'december':12,
    'jan':1,'feb':2,'mar':3,'apr':4,'jun':6,'jul':7,'aug':8,
    'sep':9,'oct':10,'nov':11,'dec':12,
}

_FNAME_SPEAKER_RE = re.compile(r'_asp_([a-z]+)$', re.IGNORECASE)

KNOWN_NAMES = {
    'ohlendorf','wisliceny','hollriegel','schellenberg','zelewski',
    'lahousen','dieckmann','hoess','speer','sauckel','streicher',
    'funk','frank','frick','hess','goering','ribbentrop','keitel',
    'rosenberg','jodl','seyss','neurath','raeder','doenitz',
    'papen','schacht','fritzsche','griffith',
}

NON_SPEAKER_SUFFIXES = {
    'kalt','kaltmotion','staff','inter','annex1','annex2',
    'generaltreaties','norway1','part01','part02','part03',
    'part04','part05','preface','v2','v3',
}

DOCTYPE_DATES = {
    'nca_v1': '1946-01-01',
    'nca_v2': '1946-01-01',
    'nca_v3': '1946-01-01',
    'nca_v4': '1946-01-01',
    'jackson': '1945-11-01',
    'pohl':    '1947-11-03',
}

DOCTYPE_SPEAKERS = {
    'jackson': 'ROBERT H. JACKSON',
}

_SESSION_DATE_URL_RE = re.compile(r'/imt/(\d{2}-\d{2}-\d{2})\.asp', re.IGNORECASE)


def _base_url(url: str) -> str:
    return url.split('#')[0]


def slug_from_url(url: str) -> str:
    name = url.rstrip('/').split('/')[-1]
    name = name.replace('.asp', '').replace('.html', '')
    return name


def base_slug(url: str) -> str:
    return slug_from_url(_base_url(url))


def extract_date_from_filename(stem: str):
    m = _FNAME_DATE_RE.search(stem)
    if m:
        mm, dd, yy = m.group(1), m.group(2), m.group(3)
        year = int(yy) + (1900 if int(yy) > 30 else 2000)
        return f"{year}-{mm}-{dd}"
    return None


def extract_date_from_text(text: str):
    m = _TEXT_DATE_LONG.search(text[:3000])
    if m:
        day, month, year = m.group(1), m.group(2).lower(), m.group(3)
        return f"{year}-{MONTH_MAP.get(month,0):02d}-{int(day):02d}"
    m = _TEXT_DATE_US.search(text[:3000])
    if m:
        month, day, year = m.group(1).lower(), m.group(2), m.group(3)
        return f"{year}-{MONTH_MAP.get(month,0):02d}-{int(day):02d}"
    m = _TEXT_DATE_SHORT.search(text[:3000])
    if m:
        day, month, yy = m.group(1), m.group(2).lower(), m.group(3)
        year = int(yy) + (1900 if int(yy) > 30 else 2000)
        return f"{year}-{MONTH_MAP.get(month,0):02d}-{int(day):02d}"
    return None


def extract_speaker_from_filename(stem: str):
    m = _FNAME_SPEAKER_RE.search(stem)
    if not m:
        return None
    name = m.group(1).lower()
    if name in NON_SPEAKER_SUFFIXES:
        return None
    if name in KNOWN_NAMES:
        return name.upper()
    return None


def infer_metadata(fp: Path, doc: dict) -> dict:
    stem      = fp.stem
    doc_type  = doc.get('doc_type', '')
    full_text = doc.get('full_text', '') or ''

    date_iso = extract_date_from_filename(stem)
    if not date_iso:
        date_iso = DOCTYPE_DATES.get(doc_type)
    if not date_iso:
        date_iso = extract_date_from_text(full_text)

    speaker = DOCTYPE_SPEAKERS.get(doc_type)
    if not speaker:
        speaker = extract_speaker_from_filename(stem)

    return {'date_iso': date_iso, 'speaker': speaker, 'doc_type': doc_type}


# ── Duplicate detection ────────────────────────────────────────────────────────

def is_session_duplicate(chunk_url: str, primary_session_urls: set) -> bool:
    if '#' not in chunk_url:
        return False
    base = _base_url(chunk_url)
    if not _SESSION_DATE_URL_RE.search(base):
        return False
    return base in primary_session_urls


# ── Chunker ───────────────────────────────────────────────────────────────────

def _last_n_tokens(text: str, n: int) -> str:
    words = text.split()
    result, tok = [], 0
    for w in reversed(words):
        tok += count_tokens(w)
        if tok > n:
            break
        result.insert(0, w)
    return " ".join(result)


def sliding_window_split(text: str, max_tokens: int, overlap: int) -> list:
    if count_tokens(text) <= max_tokens:
        return [text]
    separators = ["\n\n", "\n", ". ", " "]
    for sep in separators:
        parts = text.split(sep)
        if len(parts) < 2:
            continue
        chunks, current, current_tok = [], [], 0
        for part in parts:
            t = count_tokens(part + sep)
            if current_tok + t > max_tokens and current:
                chunk_text = sep.join(current)
                chunks.append(chunk_text)
                overlap_text = _last_n_tokens(chunk_text, overlap)
                current = [overlap_text, part] if overlap_text else [part]
                current_tok = count_tokens(sep.join(current))
            else:
                current.append(part)
                current_tok += t
        if current:
            chunks.append(sep.join(current))
        if all(count_tokens(c) <= max_tokens for c in chunks):
            return chunks
    words = text.split()
    chunks, current, current_tok = [], [], 0
    for word in words:
        t = count_tokens(word + " ")
        if current_tok + t > max_tokens and current:
            chunks.append(" ".join(current))
            ov, ov_tok = [], 0
            for w in reversed(current):
                wt = count_tokens(w)
                if ov_tok + wt > overlap:
                    break
                ov.insert(0, w)
                ov_tok += wt
            current = ov + [word]
            current_tok = count_tokens(" ".join(current))
        else:
            current.append(word)
            current_tok += t
    if current:
        chunks.append(" ".join(current))
    return chunks


def strip_nav(text: str) -> str:
    nav_tokens = {
        "previous document","next document","judgment contents","general",
        "schacht","von papen","fritzsche","hess","reich cabinet",
        "general staff and okw","contents","volume iii menu","volume iv menu",
        "volume v menu","nuremberg trials page",
    }
    lines = text.split("\n")
    start = 0
    for i, line in enumerate(lines):
        s = line.strip().lower()
        if s in nav_tokens or (len(s) < 50 and not any(c in s for c in '.,:;?!)')):
            start = i + 1
        else:
            break
    end = len(lines)
    for i in range(len(lines)-1, max(start, len(lines)-15), -1):
        s = lines[i].strip().lower()
        if s in nav_tokens or not s:
            end = i
        else:
            break
    return "\n".join(lines[start:end]).strip()


def build_context_header(date_iso, slug, speaker, collection, page):
    parts = []
    if date_iso:
        parts.append(f"Date: {date_iso}")
    parts.append(f"Source: {slug}")
    if speaker:
        parts.append(f"Speaker: {speaker}")
    parts.append(f"Collection: {collection}")
    if page is not None:
        parts.append(f"Page: {page}")
    return "[" + " | ".join(parts) + "]"


def rechunk_from_source(doc: dict, meta: dict) -> list:
    url       = doc.get('url', '')
    slug      = base_slug(url)
    full_text = doc.get('full_text', '') or ''
    clean     = strip_nav(full_text)
    if not clean:
        return []
    raw_chunks = sliding_window_split(clean, SECONDARY_MAX, OVERLAP)
    result = []
    total  = len(raw_chunks)
    for i, body in enumerate(raw_chunks):
        body = body.strip()
        if not body:
            continue
        header = build_context_header(
            meta['date_iso'], slug, meta['speaker'], 'secondary', None)
        full = f"{header}\n{body}"
        result.append({
            "chunk_id":     f"secondary::{slug}::{i:04d}",
            "text":         full,
            "body":         body,
            "collection":   "secondary",
            "source_url":   url,
            "date_iso":     meta['date_iso'],
            "speaker":      meta['speaker'],
            "page_number":  None,
            "token_count":  count_tokens(full),
            "chunk_index":  i,
            "total_chunks": total,
            "slug":         slug,
        })
    return result


def patch_chunk_metadata(chunk: dict, meta: dict) -> dict:
    c    = dict(chunk)
    c['date_iso'] = meta['date_iso']
    c['speaker']  = meta['speaker']
    slug = c.get('slug', base_slug(c.get('source_url', '')))
    header = build_context_header(
        meta['date_iso'], slug, meta['speaker'], 'secondary', c.get('page_number'))
    body = c.get('body', '')
    c['text'] = f"{header}\n{body}"
    c['token_count'] = count_tokens(c['text'])
    return c


# ── Main ──────────────────────────────────────────────────────────────────────

def run(apply: bool, no_rechunk: bool):
    print(f"\n{'='*70}")
    print(f"  SECONDARY CHUNKS PATCH v2 {'(DRY RUN)' if not apply else '(APPLYING)'}")
    print(f"{'='*70}\n")

    # Primary session URLs
    print("Building primary session URL index...")
    primary_session_urls = set()
    if SESSIONS_DIR.exists():
        for fp in SESSIONS_DIR.glob("*.json"):
            try:
                doc = json.loads(fp.read_text(encoding='utf-8'))
                u = doc.get('url', '')
                if u:
                    primary_session_urls.add(_base_url(u))
            except Exception:
                pass
    print(f"  {len(primary_session_urls)} primary session URLs indexed")

    # Secondary source docs keyed by base URL
    print("Loading secondary source documents...")
    source_by_base_url = {}
    for fp in sorted(SECONDARY_DIR.glob("*.json")):
        try:
            doc = json.loads(fp.read_text(encoding='utf-8'))
        except Exception as e:
            print(f"  x {fp.name}: {e}")
            continue
        url  = doc.get('url', '')
        base = _base_url(url)
        meta = infer_metadata(fp, doc)
        source_by_base_url[base] = (fp, doc, meta)
    print(f"  {len(source_by_base_url)} source docs loaded\n")

    # Load chunks
    print("Loading chunks.jsonl...")
    all_chunks = []
    with CHUNKS_FILE.open(encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    all_chunks.append(json.loads(line))
                except Exception:
                    pass
    print(f"  {len(all_chunks):,} total chunks")

    non_secondary = [c for c in all_chunks if c.get('collection') != 'secondary']
    secondary     = [c for c in all_chunks if c.get('collection') == 'secondary']
    print(f"  {len(non_secondary):,} non-secondary (untouched)")
    print(f"  {len(secondary):,} secondary to process\n")

    # Group by base URL
    by_base_url = defaultdict(list)
    for c in secondary:
        base = _base_url(c.get('source_url', ''))
        by_base_url[base].append(c)

    stats = Counter()
    deleted_chunks   = []
    patched_chunks   = []
    rechunked_new    = []
    orphaned_chunks  = []
    rechunked_bases  = set()

    for base_url, chunks in by_base_url.items():
        chunk_url = chunks[0].get('source_url', '')

        if is_session_duplicate(chunk_url, primary_session_urls):
            deleted_chunks.extend(chunks)
            stats['deleted'] += len(chunks)
            continue

        entry = source_by_base_url.get(base_url)
        if entry is None:
            orphaned_chunks.extend(chunks)
            stats['orphaned'] += len(chunks)
            continue

        fp, doc, meta = entry

        tok_counts = [c.get('token_count', 0) for c in chunks]
        n_overlong = sum(1 for t in tok_counts if t > OVERLONG_THRESH)
        needs_rechunk = (not no_rechunk) and (n_overlong > 0)

        if needs_rechunk and base_url not in rechunked_bases:
            rechunked_bases.add(base_url)
            new_chunks = rechunk_from_source(doc, meta)
            if new_chunks:
                rechunked_new.extend(new_chunks)
                stats['rechunked_docs']       += 1
                stats['rechunked_chunks_old'] += len(chunks)
                stats['rechunked_chunks_new'] += len(new_chunks)
                continue

        for c in chunks:
            patched_chunks.append(patch_chunk_metadata(c, meta))
            stats['patched'] += 1

    all_new_secondary = patched_chunks + rechunked_new
    n_total   = len(all_new_secondary)
    n_date    = sum(1 for c in all_new_secondary if c.get('date_iso'))
    n_speaker = sum(1 for c in all_new_secondary if c.get('speaker'))
    bq_ok  = sum(1 for c in all_new_secondary if 90 <= c.get('token_count',0) <= 512)
    bq_sh  = sum(1 for c in all_new_secondary if c.get('token_count',0) < 90)
    bq_ol  = sum(1 for c in all_new_secondary if c.get('token_count',0) > 512)

    print("─── PATCH SUMMARY ──────────────────────────────────────────────────")
    print(f"  Chunks deleted (session dupes)      : {stats.get('deleted',0):>6}")
    print(f"  Chunks patched (metadata only)      : {stats.get('patched',0):>6}")
    print(f"  Chunks rechunked:")
    print(f"    Source docs rechunked             : {stats.get('rechunked_docs',0):>6}")
    print(f"    Old chunks removed                : {stats.get('rechunked_chunks_old',0):>6}")
    print(f"    New chunks produced               : {stats.get('rechunked_chunks_new',0):>6}")
    print(f"  Orphaned (kept as-is)               : {stats.get('orphaned',0):>6}")
    print()
    print(f"  New secondary total                 : {n_total:>6}")
    if n_total:
        print(f"  date_iso coverage                   : {n_date/n_total*100:.1f}%")
        print(f"  speaker coverage                    : {n_speaker/n_total*100:.1f}%")
        print(f"  Boundary quality:")
        print(f"    OK  (90-512 tok)  {bq_ok:>6}  ({bq_ok/n_total*100:.0f}%)")
        print(f"    SHORT (<90)       {bq_sh:>6}  ({bq_sh/n_total*100:.0f}%)")
        print(f"    OVERLONG (>512)   {bq_ol:>6}  ({bq_ol/n_total*100:.0f}%)")
    print()

    if deleted_chunks:
        dup_urls = sorted(set(c.get('source_url','') for c in deleted_chunks))
        print(f"  Deleted URLs ({len(dup_urls)}):")
        for u in dup_urls:
            print(f"    {u}")
        print()

    if orphaned_chunks:
        orph_urls = sorted(set(c.get('source_url','') for c in orphaned_chunks))
        print(f"  Orphaned URLs ({len(orph_urls)}) - kept as-is:")
        for u in orph_urls[:15]:
            print(f"    {u}")
        if len(orph_urls) > 15:
            print(f"    ... and {len(orph_urls)-15} more")
        print()

    total_out = len(non_secondary) + n_total + len(orphaned_chunks)
    print(f"─── FINAL COUNTS ───────────────────────────────────────────────────")
    print(f"  Non-secondary (unchanged)  : {len(non_secondary):>7,}")
    print(f"  New secondary              : {n_total:>7,}")
    print(f"  Orphaned secondary (kept)  : {len(orphaned_chunks):>7,}")
    print(f"  Total                      : {total_out:>7,}")
    print(f"  Delta vs original          : {total_out - len(all_chunks):>+7,}")
    print()

    if not apply:
        print("  DRY RUN - no files written. Re-run with --apply to commit.")
        return

    shutil.copy2(CHUNKS_FILE, BACKUP_FILE)
    print(f"  Backup -> {BACKUP_FILE}")
    final = non_secondary + all_new_secondary + orphaned_chunks
    with CHUNKS_FILE.open('w', encoding='utf-8') as f:
        for c in final:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"  Written -> {CHUNKS_FILE}  ({len(final):,} lines)")
    print()
    print("  Next steps:")
    print("  1. Re-embed secondary: python embedder.py --collection secondary")
    print("  2. Spot-check: python rag.py --query 'Ohlendorf Einsatzgruppe' --no-ui")
    print("  3. Re-audit: python audit_secondary.py")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--apply',      action='store_true')
    parser.add_argument('--no-rechunk', action='store_true')
    args = parser.parse_args()
    run(args.apply, args.no_rechunk)


if __name__ == '__main__':
    main()