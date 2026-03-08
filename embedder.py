"""
embedder.py
===========
Nuremberg Scholar — BGE-M3 embedding pipeline.

Reads output/chunks.jsonl and produces:
  output/index/dense.faiss       — FAISS flat L2 index (1024-dim dense vectors)
  output/index/sparse.jsonl      — BGE-M3 lexical weights per chunk (BM25-style)
  output/index/metadata.jsonl    — chunk metadata parallel to FAISS rows
  output/index/stats.json        — index build summary

Model: BAAI/bge-m3
  - Single model, outputs dense (1024d) + sparse (lexical weights)
  - No instruction prefix needed
  - MIT license, HF Spaces compatible
  - 8192 token context (handles long judgment chunks without truncation)

Usage:
  pip install FlagEmbedding faiss-cpu

  python embedder.py                        # embed everything
  python embedder.py --batch-size 32        # tune for your VRAM
  python embedder.py --device cpu           # force CPU
  python embedder.py --dry-run              # validate chunks.jsonl, print stats
  python embedder.py --resume               # skip already-embedded chunks

Hardware guide:
  CPU only         → batch_size=4,  ~6-8 hrs for 44k chunks
  T4 (16GB VRAM)   → batch_size=32, ~25-35 min
  A10G (24GB VRAM) → batch_size=64, ~15-20 min
  HF Spaces (free) → batch_size=16, CPU tier ~4-6 hrs
"""

import os
import json
import time
import argparse
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────

CHUNKS_FILE  = Path("output/chunks.jsonl")
INDEX_DIR    = Path("output/index")
DENSE_FILE   = INDEX_DIR / "dense.faiss"
SPARSE_FILE  = INDEX_DIR / "sparse.jsonl"
META_FILE    = INDEX_DIR / "metadata.jsonl"
STATS_FILE   = INDEX_DIR / "stats.json"
PROGRESS_FILE = INDEX_DIR / ".progress"  # for --resume

MODEL_NAME   = "BAAI/bge-m3"
EMBED_DIM    = 1024
DEFAULT_BATCH = 32
MAX_TOKENS   = 8192  # BGE-M3 context window

# ── Lazy imports (fail fast with clear messages) ───────────────────────────────

def _import_deps():
    try:
        from FlagEmbedding import BGEM3FlagModel
    except ImportError:
        raise SystemExit(
            "\n❌  FlagEmbedding not installed.\n"
            "    pip install FlagEmbedding\n"
        )
    try:
        import faiss
    except ImportError:
        raise SystemExit(
            "\n❌  faiss not installed.\n"
            "    pip install faiss-cpu   (CPU)\n"
            "    pip install faiss-gpu   (GPU, requires CUDA)\n"
        )
    import numpy as np
    return BGEM3FlagModel, faiss, np


# ── Chunk loader ──────────────────────────────────────────────────────────────

def load_chunks(path: Path) -> list[dict]:
    chunks = []
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                chunks.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  ⚠️  Line {i+1}: JSON parse error — {e}")
    return chunks


def load_progress(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return set(path.read_text().splitlines())


def save_progress(path: Path, chunk_ids: list[str]):
    with path.open("a") as f:
        for cid in chunk_ids:
            f.write(cid + "\n")


# ── Batch iterator ────────────────────────────────────────────────────────────

def batched(lst: list, n: int):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


# ── Main embedding loop ───────────────────────────────────────────────────────

def embed(batch_size: int, device: str, resume: bool, dry_run: bool):
    # Validate input
    if not CHUNKS_FILE.exists():
        raise SystemExit(f"\n❌  {CHUNKS_FILE} not found. Run chunker.py first.\n")

    print(f"\nNuremberg Scholar — Embedder")
    print("=" * 60)
    print(f"  Model      : {MODEL_NAME}")
    print(f"  Chunks     : {CHUNKS_FILE}")
    print(f"  Index dir  : {INDEX_DIR}")
    print(f"  Batch size : {batch_size}")
    print(f"  Device     : {device}")

    chunks = load_chunks(CHUNKS_FILE)
    print(f"  Loaded     : {len(chunks):,} chunks")

    if dry_run:
        # Stats only
        collections = {}
        token_counts = []
        for c in chunks:
            coll = c.get("collection", "unknown")
            collections[coll] = collections.get(coll, 0) + 1
            token_counts.append(c.get("token_count", 0))
        print(f"\n  [dry-run] Chunk distribution:")
        for coll, n in sorted(collections.items()):
            print(f"    {coll:<12}  {n:>6} chunks")
        print(f"\n  Token stats:")
        token_counts.sort()
        n = len(token_counts)
        print(f"    min={token_counts[0]}  "
              f"p50={token_counts[n//2]}  "
              f"p95={token_counts[int(n*0.95)]}  "
              f"p99={token_counts[int(n*0.99)]}  "
              f"max={token_counts[-1]}")
        over_limit = sum(1 for t in token_counts if t > MAX_TOKENS)
        print(f"    Over {MAX_TOKENS} tokens: {over_limit}")
        print(f"\n  Est. embed time @ batch={batch_size}:")
        # Rough estimate: T4 ~32 chunks/s, CPU ~2 chunks/s
        t4_est  = len(chunks) / (32 * (batch_size / 32))
        cpu_est = len(chunks) / 2
        print(f"    T4 GPU  : ~{t4_est/60:.0f} min")
        print(f"    CPU     : ~{cpu_est/60:.0f} min")
        print(f"\n  [dry-run] Nothing written.\n")
        return

    BGEM3FlagModel, faiss, np = _import_deps()

    # Resume: skip already done
    done_ids = load_progress(PROGRESS_FILE) if resume else set()
    if done_ids:
        print(f"  Resuming   : {len(done_ids):,} already embedded, skipping")
        chunks = [c for c in chunks if c["chunk_id"] not in done_ids]
        print(f"  Remaining  : {len(chunks):,} chunks")

    if not chunks:
        print("  Nothing to embed. Index already complete.")
        return

    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"\n  Loading {MODEL_NAME}...")
    t0 = time.time()
    use_fp16 = (device != "cpu")
    model = BGEM3FlagModel(
        MODEL_NAME,
        use_fp16=use_fp16,
        device=device,
    )
    print(f"  Model loaded in {time.time()-t0:.1f}s")

    # Prepare output files
    # FAISS: build incrementally, write at end
    # sparse + metadata: stream to disk
    dense_vectors = []

    # If resuming, we need to append to existing sparse/meta files
    sparse_mode = "a" if (resume and SPARSE_FILE.exists()) else "w"
    meta_mode   = "a" if (resume and META_FILE.exists())   else "w"

    sparse_f = SPARSE_FILE.open(sparse_mode, encoding="utf-8")
    meta_f   = META_FILE.open(meta_mode, encoding="utf-8")

    total     = len(chunks)
    embedded  = 0
    t_start   = time.time()
    batch_ids = []

    print(f"\n  Embedding {total:,} chunks...\n")

    for batch_num, batch in enumerate(batched(chunks, batch_size)):
        texts = [c["text"] for c in batch]

        # BGE-M3: encode with both dense and sparse
        output = model.encode(
            texts,
            batch_size=batch_size,
            max_length=MAX_TOKENS,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,  # skip ColBERT to save memory
        )

        dense  = output["dense_vecs"]    # np.ndarray (batch, 1024)
        sparse = output["lexical_weights"]  # list of dicts {token: weight}

        for i, chunk in enumerate(batch):
            # Dense vector
            vec = dense[i].astype("float32")
            dense_vectors.append(vec)

            # Sparse weights — serialise as {token_str: weight}
            lex = sparse[i]
            # BGE-M3 returns token_id keys; decode to strings for readability
            # and BM25-style retrieval
            sparse_f.write(json.dumps({
                "chunk_id": chunk["chunk_id"],
                "weights":  {str(k): float(v) for k, v in lex.items()},
            }, ensure_ascii=False) + "\n")

            # Metadata (everything except the full text body — keep index lean)
            meta_f.write(json.dumps({
                "chunk_id":    chunk["chunk_id"],
                "collection":  chunk["collection"],
                "source_url":  chunk["source_url"],
                "date_iso":    chunk["date_iso"],
                "speaker":     chunk["speaker"],
                "page_number": chunk["page_number"],
                "token_count": chunk["token_count"],
                "chunk_index": chunk["chunk_index"],
                "total_chunks": chunk["total_chunks"],
                "slug":        chunk["slug"],
                "body":        chunk["body"],   # keep body for retrieval display
            }, ensure_ascii=False) + "\n")

            batch_ids.append(chunk["chunk_id"])

        embedded += len(batch)

        # Progress
        elapsed  = time.time() - t_start
        rate     = embedded / elapsed if elapsed > 0 else 0
        eta      = (total - embedded) / rate if rate > 0 else 0
        print(
            f"  [{embedded:>6}/{total}]  "
            f"{rate:.1f} chunks/s  "
            f"ETA {eta/60:.1f}min  "
            f"batch {batch_num+1}",
            end="\r", flush=True
        )

        # Save progress every 10 batches
        if resume and batch_num % 10 == 0:
            save_progress(PROGRESS_FILE, batch_ids)
            batch_ids = []

    sparse_f.close()
    meta_f.close()

    if resume and batch_ids:
        save_progress(PROGRESS_FILE, batch_ids)

    print(f"\n\n  Building FAISS index ({len(dense_vectors):,} vectors × {EMBED_DIM}d)...")

    # If resuming, load existing vectors first
    if resume and DENSE_FILE.exists():
        existing = faiss.read_index(str(DENSE_FILE))
        existing_vecs = faiss.rev_swig_ptr(existing.get_xb(), existing.ntotal * EMBED_DIM)
        existing_vecs = existing_vecs.reshape(existing.ntotal, EMBED_DIM).copy()
        all_vecs = np.vstack([existing_vecs, np.array(dense_vectors, dtype="float32")])
    else:
        all_vecs = np.array(dense_vectors, dtype="float32")

    # Normalize for cosine similarity (BGE-M3 dense vecs benefit from L2 norm)
    faiss.normalize_L2(all_vecs)

    # Flat index — exact search, no approximation loss
    # For 44k vectors this is ~180MB and sub-10ms query time
    index = faiss.IndexFlatIP(EMBED_DIM)  # Inner Product after L2 norm = cosine
    index.add(all_vecs)
    faiss.write_index(index, str(DENSE_FILE))

    total_time = time.time() - t_start

    # Stats
    stats = {
        "model":          MODEL_NAME,
        "embed_dim":      EMBED_DIM,
        "total_chunks":   index.ntotal,
        "collections":    {},
        "total_time_s":   round(total_time, 1),
        "chunks_per_sec": round(embedded / total_time, 1),
        "dense_index":    str(DENSE_FILE),
        "sparse_index":   str(SPARSE_FILE),
        "metadata":       str(META_FILE),
        "device":         device,
        "batch_size":     batch_size,
    }

    # Count per collection from metadata
    with META_FILE.open(encoding="utf-8") as f:
        for line in f:
            m = json.loads(line)
            coll = m.get("collection", "unknown")
            stats["collections"][coll] = stats["collections"].get(coll, 0) + 1

    STATS_FILE.write_text(json.dumps(stats, indent=2))

    print(f"\n  ✅  Index complete")
    print(f"  ─────────────────────────────────────────")
    print(f"  Dense FAISS  : {DENSE_FILE}  ({index.ntotal:,} vectors)")
    print(f"  Sparse       : {SPARSE_FILE}")
    print(f"  Metadata     : {META_FILE}")
    print(f"  Total time   : {total_time/60:.1f} min")
    print(f"  Throughput   : {embedded/total_time:.1f} chunks/s")
    print(f"\n  Per collection:")
    for coll, n in sorted(stats["collections"].items()):
        print(f"    {coll:<12}  {n:>6}")
    print()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Nuremberg Scholar — BGE-M3 embedder")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH,
                        help=f"Batch size (default {DEFAULT_BATCH}). "
                             "Reduce if OOM. T4=32, A10G=64, CPU=4.")
    parser.add_argument("--device", default="cuda",
                        help="Device: cuda / cpu / mps (default: cuda)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate chunks.jsonl and print stats, no embedding")
    parser.add_argument("--resume", action="store_true",
                        help="Skip already-embedded chunks (uses .progress file)")
    args = parser.parse_args()

    # Auto-detect device
    if args.device == "cuda":
        try:
            import torch
            if not torch.cuda.is_available():
                print("  ⚠️  CUDA not available, falling back to CPU")
                args.device = "cpu"
                args.batch_size = min(args.batch_size, 4)
        except ImportError:
            args.device = "cpu"
            args.batch_size = min(args.batch_size, 4)

    embed(
        batch_size=args.batch_size,
        device=args.device,
        resume=args.resume,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
