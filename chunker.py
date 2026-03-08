"""
chunker.py
==========
Nuremberg Scholar — RAG chunking pipeline.

Reads all scraped JSON documents from output/ and produces chunks.jsonl,
ready for BGE-M3 embedding.

Three chunking strategies, matched to document type:

  sessions   → Speaker-turn grouping with rolling merge (~400 tokens target)
               Short turns (< 30 tok) merged forward. Oversized turns split
               at sentence boundaries. Every chunk carries speaker in text.

  judgment   → Paragraph-recursive chunking (400–600 tokens, 64 overlap).
               Section headers detected and used as hard split points.

  secondary  → Sliding window recursive split (512 tokens, 64 overlap).
  key_docs     Same. No speaker structure, treat as flat prose.

  vol1       → Same as secondary.

Context header prepended to every chunk (free metadata injection):
  [Date: {date_iso} | Session: {slug} | Speaker: {speaker} | Collection: {coll} | Page: {page}]

Output schema (one JSON object per line in chunks.jsonl):
  {
    "chunk_id":    "sessions::01-02-46::0042",   # collection::slug::seq
    "text":        "...",                          # context_header + body
    "body":        "...",                          # body only (no header)
    "collection":  "sessions",
    "source_url":  "https://avalon.law.yale.edu/imt/01-02-46.asp",
    "date_iso":    "1946-01-02",
    "speaker":     "MR. JUSTICE JACKSON",         # null for non-session docs
    "page_number": 213,                            # null if absent
    "token_count": 387,
    "chunk_index": 42,
    "total_chunks": 180,
    "slug":        "01-02-46"
  }

Usage:
  python chunker.py                    # chunk everything → output/chunks.jsonl
  python chunker.py --collection sessions
  python chunker.py --dry-run          # print stats, write nothing
  python chunker.py --stats            # print per-collection stats after chunking

Requirements: pip install tiktoken
"""

import re
import json
import argparse
from pathlib import Path

try:
    import tiktoken
    _enc = tiktoken.get_encoding("cl100k_base")
    def count_tokens(text: str) -> int:
        return len(_enc.encode(text))
except ImportError:
    # Fallback: ~4 chars per token
    def count_tokens(text: str) -> int:
        return max(1, len(text) // 4)

# ── Config ────────────────────────────────────────────────────────────────────

OUTPUT_DIR  = Path("output")
CHUNKS_FILE = OUTPUT_DIR / "chunks.jsonl"

# Token budgets
SESSION_TARGET   = 400   # target tokens per session chunk
SESSION_MAX      = 512   # hard max before forced split
SESSION_MIN      = 30    # min tokens for a standalone turn (else merge forward)
JUDGMENT_TARGET  = 500
JUDGMENT_MAX     = 600
OVERLAP_TOKENS   = 64
SECONDARY_MAX    = 512

# Collections to process and their subdirectories
COLLECTIONS = {
    "sessions":  OUTPUT_DIR / "sessions",
    "judgment":  OUTPUT_DIR / "judgment",
    "key_docs":  OUTPUT_DIR / "key_docs",
    "secondary": OUTPUT_DIR / "secondary",
    "vol1":      OUTPUT_DIR / "vol1",
}

# Collections that have speaker turns
TRANSCRIPT_COLLECTIONS = {"sessions"}

# Sentence boundary split pattern
SENTENCE_END = re.compile(r'(?<=[.!?])\s+')

# ── Helpers ───────────────────────────────────────────────────────────────────

def slug_from_url(url: str) -> str:
    """Extract slug from Yale URL, e.g. '01-02-46' from .../imt/01-02-46.asp"""
    if not url:
        return "unknown"
    name = url.rstrip("/").split("/")[-1]
    return name.replace(".asp", "").replace(".html", "")


def build_context_header(
    date_iso: str | None,
    slug: str,
    speaker: str | None,
    collection: str,
    page: int | None,
) -> str:
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


def split_at_sentences(text: str, max_tokens: int) -> list[str]:
    """Split long text at sentence boundaries to stay under max_tokens."""
    sentences = SENTENCE_END.split(text)
    chunks, current, current_tok = [], [], 0
    for sent in sentences:
        t = count_tokens(sent)
        if current_tok + t > max_tokens and current:
            chunks.append(" ".join(current))
            current, current_tok = [], 0
        current.append(sent)
        current_tok += t
    if current:
        chunks.append(" ".join(current))
    return chunks or [text]


def sliding_window_split(text: str, max_tokens: int, overlap: int) -> list[str]:
    """
    Recursive character-based split with token-aware sizing and overlap.
    Splits preferentially on paragraph breaks, then sentence ends, then words.
    """
    if count_tokens(text) <= max_tokens:
        return [text]

    # Try to split on paragraph boundary first
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
                # Overlap: carry last N tokens forward
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

    # Last resort: split on token boundaries
    words = text.split()
    chunks, current, current_tok = [], [], 0
    for word in words:
        t = count_tokens(word + " ")
        if current_tok + t > max_tokens and current:
            chunks.append(" ".join(current))
            # Overlap
            ov = []
            ov_tok = 0
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


def _last_n_tokens(text: str, n: int) -> str:
    """Return the last ~n tokens of text as a string."""
    words = text.split()
    result, tok = [], 0
    for w in reversed(words):
        tok += count_tokens(w)
        if tok > n:
            break
        result.insert(0, w)
    return " ".join(result)


# ── Session chunker (speaker-turn grouping) ───────────────────────────────────

def chunk_session(doc: dict, collection: str) -> list[dict]:
    """
    Group speaker turns into ~400-token chunks.
    - Merge short turns (< SESSION_MIN tokens) forward.
    - Split oversized turns at sentence boundaries.
    - Each chunk carries speaker label + context header.
    """
    url      = doc.get("url", "")
    date_iso = doc.get("date_iso")
    slug     = slug_from_url(url)
    turns    = doc.get("turns", [])

    if not turns:
        return []

    # Flatten turns: split oversized, tag each unit with speaker + page
    units: list[dict] = []
    for turn in turns:
        speaker = turn.get("speaker", "UNKNOWN")
        text    = (turn.get("text") or "").strip()
        page    = turn.get("page_number")
        if not text:
            continue
        tok = count_tokens(text)
        if tok > SESSION_MAX:
            parts = split_at_sentences(text, SESSION_MAX)
            for part in parts:
                if part.strip():
                    units.append({"speaker": speaker, "text": part.strip(), "page": page})
        else:
            units.append({"speaker": speaker, "text": text, "page": page})

    if not units:
        return []

    # Merge short units forward greedily up to SESSION_TARGET
    merged: list[dict] = []
    buffer_text   = ""
    buffer_speaker = units[0]["speaker"]
    buffer_page    = units[0]["page"]
    buffer_tok     = 0

    def flush(buf_text, buf_speaker, buf_page):
        if buf_text.strip():
            merged.append({"speaker": buf_speaker, "text": buf_text.strip(), "page": buf_page})

    for unit in units:
        unit_text = f"{unit['speaker']}: {unit['text']}"
        unit_tok  = count_tokens(unit_text)

        if buffer_tok + unit_tok <= SESSION_TARGET:
            # Merge into current buffer; keep earliest speaker label prominent
            if buffer_text:
                buffer_text += "\n" + unit_text
            else:
                buffer_text   = unit_text
                buffer_speaker = unit["speaker"]
                buffer_page    = unit["page"]
            buffer_tok += unit_tok
        else:
            flush(buffer_text, buffer_speaker, buffer_page)
            buffer_text    = unit_text
            buffer_speaker = unit["speaker"]
            buffer_page    = unit["page"]
            buffer_tok     = unit_tok

    flush(buffer_text, buffer_speaker, buffer_page)

    # Build final chunk objects
    chunks = []
    for i, m in enumerate(merged):
        header = build_context_header(date_iso, slug, m["speaker"], collection, m["page"])
        body   = m["text"]
        full   = f"{header}\n{body}"
        chunks.append({
            "body":        body,
            "text":        full,
            "speaker":     m["speaker"],
            "page_number": m["page"],
            "token_count": count_tokens(full),
        })

    return _finalise(chunks, doc, collection, slug)


# ── Judgment chunker (paragraph-recursive) ────────────────────────────────────

# Yale nav boilerplate patterns to strip from full_text
_NAV_STRIP_RE = re.compile(
    r'^(?:Previous Document|Next Document|Judgment Contents|'
    r'General|Schacht|von Papen|Fritzsche|Hess|Reich Cabinet|'
    r'General Staff and OKW|Contents|Table of Contents|'
    r'Source:\s*Trial of War Criminals.*?(?=\n[A-Z]))\s*\n',
    re.MULTILINE | re.DOTALL
)

def strip_nav(text: str) -> str:
    """Remove Yale navigation boilerplate from scraped full_text."""
    # Remove leading nav lines (short lines before actual prose begins)
    lines = text.split("\n")
    # Skip leading lines that are pure nav (short, no sentence punctuation,
    # match known nav strings)
    nav_tokens = {
        "previous document", "next document", "judgment contents",
        "general", "schacht", "von papen", "fritzsche", "hess",
        "reich cabinet", "general staff and okw", "contents",
    }
    start = 0
    for i, line in enumerate(lines):
        stripped = line.strip().lower()
        if stripped in nav_tokens or (len(stripped) < 40 and not any(c in stripped for c in '.,:;?!')):
            start = i + 1
        else:
            break
    # Also strip trailing nav
    end = len(lines)
    for i in range(len(lines) - 1, max(start, len(lines) - 10), -1):
        stripped = lines[i].strip().lower()
        if stripped in nav_tokens or stripped == "":
            end = i
        else:
            break
    return "\n".join(lines[start:end]).strip()


JUDGMENT_SECTION_RE = re.compile(
    r'^(?:PART\s+[IVXLC]+|CHAPTER\s+\d+|SECTION\s+\d+|[A-Z][A-Z\s]{4,}:?\s*$)',
    re.MULTILINE
)

def chunk_judgment(doc: dict, collection: str) -> list[dict]:
    """
    Paragraph-recursive chunking with section-header hard splits.
    """
    url      = doc.get("url", "")
    date_iso = doc.get("date_iso")
    slug     = slug_from_url(url)

    # Judgments store content in full_text (not turns)
    turns = doc.get("turns", [])
    if turns:
        full_text = "\n\n".join(
            f"{t.get('speaker','')}: {t.get('text','')}" if t.get('speaker')
            else (t.get('text') or '')
            for t in turns
            if t.get('text')
        )
    else:
        full_text = doc.get("full_text", "") or doc.get("preamble", "") or ""

    full_text = strip_nav(full_text)
    if not full_text.strip():
        return []

    # Split on section headers first (hard boundaries)
    sections = JUDGMENT_SECTION_RE.split(full_text)
    header_matches = JUDGMENT_SECTION_RE.findall(full_text)

    # Reassemble with section labels
    labelled = []
    labelled.append(("", sections[0]))
    for i, match in enumerate(header_matches):
        labelled.append((match.strip(), sections[i + 1] if i + 1 < len(sections) else ""))

    raw_chunks = []
    for section_label, section_text in labelled:
        if not section_text.strip():
            continue
        prefix = f"{section_label}\n" if section_label else ""
        pieces = sliding_window_split(section_text.strip(), JUDGMENT_MAX, OVERLAP_TOKENS)
        for piece in pieces:
            if piece.strip():
                raw_chunks.append(prefix + piece.strip())

    chunks = []
    for body in raw_chunks:
        header = build_context_header(date_iso, slug, None, collection, None)
        full   = f"{header}\n{body}"
        chunks.append({
            "body":        body,
            "text":        full,
            "speaker":     None,
            "page_number": None,
            "token_count": count_tokens(full),
        })

    return _finalise(chunks, doc, collection, slug)


# ── Secondary / vol1 / key_docs chunker (sliding window) ─────────────────────

def chunk_flat(doc: dict, collection: str) -> list[dict]:
    """
    Sliding window recursive split for flat prose documents.
    """
    url      = doc.get("url", "")
    date_iso = doc.get("date_iso")
    slug     = slug_from_url(url)

    turns = doc.get("turns", [])
    if turns:
        full_text = "\n\n".join(
            f"{t.get('speaker','')}: {t.get('text','')}" if t.get('speaker')
            else (t.get('text') or '')
            for t in turns
            if t.get('text')
        )
    else:
        full_text = doc.get("full_text", "") or doc.get("preamble", "") or ""

    full_text = strip_nav(full_text)
    if not full_text.strip():
        return []

    raw_chunks = sliding_window_split(full_text.strip(), SECONDARY_MAX, OVERLAP_TOKENS)

    chunks = []
    for body in raw_chunks:
        if not body.strip():
            continue
        header = build_context_header(date_iso, slug, None, collection, None)
        full   = f"{header}\n{body}"
        chunks.append({
            "body":        body,
            "text":        full,
            "speaker":     None,
            "page_number": None,
            "token_count": count_tokens(full),
        })

    return _finalise(chunks, doc, collection, slug)


# ── Finalise: add IDs and source metadata ─────────────────────────────────────

def _finalise(chunks: list[dict], doc: dict, collection: str, slug: str) -> list[dict]:
    url      = doc.get("url", "")
    date_iso = doc.get("date_iso")
    total    = len(chunks)
    result   = []
    for i, ch in enumerate(chunks):
        result.append({
            "chunk_id":    f"{collection}::{slug}::{i:04d}",
            "text":        ch["text"],
            "body":        ch["body"],
            "collection":  collection,
            "source_url":  url,
            "date_iso":    date_iso,
            "speaker":     ch.get("speaker"),
            "page_number": ch.get("page_number"),
            "token_count": ch["token_count"],
            "chunk_index": i,
            "total_chunks": total,
            "slug":        slug,
        })
    return result


# ── Dispatch ──────────────────────────────────────────────────────────────────

def chunk_doc(doc: dict, collection: str) -> list[dict]:
    # Skip flagged/empty documents
    flags = doc.get("validation_flags", [])
    if "REDIRECT_STUB" in flags:
        return []
    if doc.get("word_count", 0) == 0:
        return []

    if collection == "sessions":
        return chunk_session(doc, collection)
    elif collection == "judgment":
        return chunk_judgment(doc, collection)
    else:
        return chunk_flat(doc, collection)


# ── Main ──────────────────────────────────────────────────────────────────────

def run(collections_filter: list[str] | None, dry_run: bool, show_stats: bool):
    stats: dict[str, dict] = {}
    all_chunks: list[dict] = []

    colls = collections_filter or list(COLLECTIONS.keys())

    for coll in colls:
        coll_dir = COLLECTIONS.get(coll)
        if coll_dir is None or not coll_dir.exists():
            print(f"  ⚠️  {coll}: directory not found ({coll_dir})")
            continue

        json_files = sorted(coll_dir.glob("*.json"))
        if not json_files:
            print(f"  ⚠️  {coll}: no JSON files found")
            continue

        coll_chunks = 0
        coll_tokens = 0
        coll_docs   = 0
        skipped     = 0

        print(f"\n  [{coll}]  {len(json_files)} documents")

        for fp in json_files:
            try:
                doc = json.loads(fp.read_text(encoding="utf-8"))
            except Exception as e:
                print(f"    ✗ {fp.name}: {e}")
                continue

            chunks = chunk_doc(doc, coll)

            if not chunks:
                skipped += 1
                continue

            coll_docs   += 1
            coll_chunks += len(chunks)
            coll_tokens += sum(c["token_count"] for c in chunks)
            all_chunks.extend(chunks)

        stats[coll] = {
            "docs":    coll_docs,
            "skipped": skipped,
            "chunks":  coll_chunks,
            "tokens":  coll_tokens,
            "avg_tok": round(coll_tokens / coll_chunks, 1) if coll_chunks else 0,
        }
        print(f"    docs={coll_docs}  skipped={skipped}  chunks={coll_chunks}  "
              f"tokens={coll_tokens:,}  avg={stats[coll]['avg_tok']} tok/chunk")

    total_chunks = len(all_chunks)
    total_tokens = sum(c["token_count"] for c in all_chunks)

    print(f"\n{'─'*60}")
    print(f"  Total chunks : {total_chunks:,}")
    print(f"  Total tokens : {total_tokens:,}")
    print(f"  Est. embed time (BGE-M3 @ 1k tok/s): ~{total_tokens//1000}s")

    if dry_run:
        print("\n  [dry-run] Nothing written.")
        return

    CHUNKS_FILE.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with CHUNKS_FILE.open("w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
            written += 1

    print(f"\n  ✅  Written → {CHUNKS_FILE}  ({written:,} lines)")

    if show_stats:
        print(f"\n{'═'*60}")
        print("  Per-collection summary:")
        print(f"  {'Collection':<12} {'Docs':>6} {'Chunks':>8} {'Tokens':>10} {'Avg tok':>8}")
        print(f"  {'─'*12} {'─'*6} {'─'*8} {'─'*10} {'─'*8}")
        for coll, s in stats.items():
            print(f"  {coll:<12} {s['docs']:>6} {s['chunks']:>8} {s['tokens']:>10,} {s['avg_tok']:>8}")


def main():
    parser = argparse.ArgumentParser(description="Nuremberg Scholar chunker")
    parser.add_argument("--collection", nargs="+",
                        choices=list(COLLECTIONS.keys()),
                        help="Only chunk these collections")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print stats without writing chunks.jsonl")
    parser.add_argument("--stats", action="store_true",
                        help="Print per-collection stats table after chunking")
    args = parser.parse_args()

    print("\nNuremberg Scholar — Chunker")
    print("=" * 60)
    run(args.collection, args.dry_run, args.stats)


if __name__ == "__main__":
    main()