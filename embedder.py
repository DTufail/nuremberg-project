"""
embedder.py  —  Nuremberg Scholar BGE-M3 Embedder (HuggingFace-only)
=====================================================================
pip install transformers torch faiss-cpu huggingface_hub

BGE-M3 sparse head (verified from FlagEmbedding source):
  sparse_linear = Linear(1024, 1)  — scalar weight per token position
  Flow: relu(linear(hidden)) -> scatter_reduce onto input_ids -> vocab sparse vector
  NOT a Linear(1024, vocab_size) projection.
  sparse_linear.pt is ~3.52 kB = 1025 float32 params = Linear(1024,1) ✓

Usage (Colab T4):
    !pip install transformers torch faiss-cpu huggingface_hub
    %run embedder.py --batch-size 32
"""

import json
import time
import argparse
from pathlib import Path

CHUNKS_FILE   = Path("output/chunks.jsonl")
INDEX_DIR     = Path("output/index")
DENSE_FILE    = INDEX_DIR / "dense.faiss"
SPARSE_FILE   = INDEX_DIR / "sparse.jsonl"
META_FILE     = INDEX_DIR / "metadata.jsonl"
STATS_FILE    = INDEX_DIR / "stats.json"
PROGRESS_FILE = INDEX_DIR / ".progress"

MODEL_NAME    = "BAAI/bge-m3"
EMBED_DIM     = 1024
DEFAULT_BATCH = 32
MAX_TOKENS    = 8192
UNUSED_TOKENS = [0, 1, 2]   # <s>, <pad>, </s> — always zeroed in sparse output


class BGEM3:
    """
    BGE-M3 using only transformers + torch + huggingface_hub.

    Dense  : CLS token, L2-normalised → (1024,)
    Sparse : Linear(1024,1) scalar weights per token, scatter onto input_ids
             vocab positions, max-pool → decoded {token_str: weight} dict
    """

    def __init__(self, model_name, device, use_fp16=True):
        import torch
        import torch.nn as nn
        from transformers import AutoTokenizer, AutoModel
        from huggingface_hub import hf_hub_download

        self.device = torch.device(device)
        self.torch  = torch
        self.fp16   = use_fp16 and device != "cpu"

        print(f"  Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.vocab_size = self.tokenizer.vocab_size   # 250002

        print(f"  Loading backbone ({model_name})...")
        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.fp16 else torch.float32,
        )
        self.model.to(self.device)
        self.model.eval()

        print(f"  Loading sparse_linear.pt...")
        sparse_path = hf_hub_download(repo_id=model_name, filename="sparse_linear.pt")
        raw = torch.load(sparse_path, map_location="cpu", weights_only=True)

        # sparse_linear is Linear(1024, 1) — scalar weight per token position
        out_features = raw["weight"].shape[0]   # should be 1
        in_features  = raw["weight"].shape[1]   # should be 1024
        print(f"  sparse_linear: Linear({in_features}, {out_features}) ✓")

        self.sparse_linear = nn.Linear(in_features, out_features, bias=True)
        self.sparse_linear.load_state_dict(raw, strict=True)
        if self.fp16:
            self.sparse_linear = self.sparse_linear.half()
        self.sparse_linear.to(self.device)
        self.sparse_linear.eval()

        print(f"  Ready  device={device}  fp16={self.fp16}  vocab={self.vocab_size}")

    def encode(self, texts, max_length=MAX_TOKENS):
        import torch
        import torch.nn.functional as F
        import numpy as np

        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}

        with self.torch.no_grad():
            out = self.model(**enc, return_dict=True)

        last_hidden = out.last_hidden_state     # (N, seq_len, 1024)

        # ── Dense: CLS token, L2-normalised ──────────────────────────────────
        dense    = F.normalize(last_hidden[:, 0, :].float(), p=2, dim=-1)
        dense_np = dense.cpu().numpy().astype("float32")

        # ── Sparse: scalar weight per token → scatter onto vocab positions ────
        with self.torch.no_grad():
            # (N, seq_len, 1) — one scalar weight per token position
            token_weights = torch.relu(
                self.sparse_linear(last_hidden)
            ).squeeze(-1).float()             # (N, seq_len)

        # scatter_reduce: place each token's weight at its vocab ID position
        # result: (N, vocab_size) sparse vector
        sparse_emb = torch.zeros(
            len(texts), self.vocab_size,
            dtype=torch.float32,
            device=self.device
        )
        sparse_emb = sparse_emb.scatter_reduce(
            dim    = 1,
            index  = enc["input_ids"],
            src    = token_weights,
            reduce = "amax",
            include_self=False,
        )

        # Zero out special tokens
        for uid in UNUSED_TOKENS:
            if uid < self.vocab_size:
                sparse_emb[:, uid] = 0.0

        # Decode nonzero positions to {token_string: weight}
        lexical_list = []
        for i in range(len(texts)):
            nonzero = sparse_emb[i].nonzero(as_tuple=True)[0].tolist()
            if not nonzero:
                lexical_list.append({})
                continue
            scores  = sparse_emb[i][nonzero].cpu().tolist()
            decoded = {}
            for tid, score in zip(nonzero, scores):
                if score <= 0:
                    continue
                tok = self.tokenizer.decode([tid]).strip()
                if tok:
                    decoded[tok] = round(float(score), 4)
            lexical_list.append(decoded)

        return {"dense_vecs": dense_np, "lexical_weights": lexical_list}


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_chunks(path):
    chunks = []
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                chunks.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  WARNING line {i+1}: {e}")
    return chunks

def load_progress(path):
    if not path.exists():
        return set()
    return set(path.read_text().splitlines())

def save_progress(path, ids):
    with path.open("a") as f:
        for cid in ids:
            f.write(cid + "\n")

def batched(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


# ── Main ──────────────────────────────────────────────────────────────────────

def embed(batch_size, device, resume, dry_run):
    if not CHUNKS_FILE.exists():
        raise SystemExit(f"\nERROR: {CHUNKS_FILE} not found. Run chunker.py first.\n")

    print(f"\nNuremberg Scholar — Embedder (HuggingFace-only)")
    print("=" * 60)
    print(f"  Model      : {MODEL_NAME}")
    print(f"  Chunks     : {CHUNKS_FILE}")
    print(f"  Batch size : {batch_size}")
    print(f"  Device     : {device}")

    chunks = load_chunks(CHUNKS_FILE)
    print(f"  Loaded     : {len(chunks):,} chunks")

    if dry_run:
        from collections import Counter
        colls  = Counter(c.get("collection", "?") for c in chunks)
        tokens = sorted(c.get("token_count", 0) for c in chunks)
        n      = len(tokens)
        print(f"\n  [dry-run] Collections:")
        for coll, cnt in colls.most_common():
            print(f"    {coll:<14} {cnt:>6,}")
        print(f"\n  Token stats:  min={tokens[0]}  p50={tokens[n//2]}  "
              f"p95={tokens[int(n*.95)]}  max={tokens[-1]}")
        print(f"  Est. T4 time: ~{len(chunks)/32/60:.0f} min")
        print(f"\n  [dry-run] Nothing written.\n")
        return

    done_ids = load_progress(PROGRESS_FILE) if resume else set()
    if done_ids:
        print(f"  Resuming   : {len(done_ids):,} already done")
        chunks = [c for c in chunks if c["chunk_id"] not in done_ids]
        print(f"  Remaining  : {len(chunks):,}")

    if not chunks:
        print("  Nothing to embed — already complete.")
        return

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    model = BGEM3(MODEL_NAME, device=device, use_fp16=(device != "cpu"))

    import faiss
    import numpy as np

    dense_vecs  = []
    sparse_mode = "a" if (resume and SPARSE_FILE.exists()) else "w"
    meta_mode   = "a" if (resume and META_FILE.exists())   else "w"

    sparse_f = SPARSE_FILE.open(sparse_mode, encoding="utf-8")
    meta_f   = META_FILE.open(meta_mode,     encoding="utf-8")

    total    = len(chunks)
    embedded = 0
    t_start  = time.time()
    prog_ids = []

    print(f"\n  Embedding {total:,} chunks...\n")

    for b_num, batch in enumerate(batched(chunks, batch_size)):
        texts  = [c["body"] for c in batch]
        out    = model.encode(texts, max_length=MAX_TOKENS)
        dense  = out["dense_vecs"]
        sparse = out["lexical_weights"]

        for i, chunk in enumerate(batch):
            dense_vecs.append(dense[i])

            sparse_f.write(json.dumps({
                "chunk_id": chunk["chunk_id"],
                "weights":  sparse[i],
            }, ensure_ascii=False) + "\n")

            meta_f.write(json.dumps({
                "chunk_id":     chunk["chunk_id"],
                "collection":   chunk.get("collection"),
                "source_url":   chunk.get("source_url"),
                "date_iso":     chunk.get("date_iso"),
                "speaker":      chunk.get("speaker"),
                "page_number":  chunk.get("page_number"),
                "token_count":  chunk.get("token_count"),
                "chunk_index":  chunk.get("chunk_index"),
                "total_chunks": chunk.get("total_chunks"),
                "slug":         chunk.get("slug"),
                "body":         chunk.get("body"),
            }, ensure_ascii=False) + "\n")

            prog_ids.append(chunk["chunk_id"])

        embedded += len(batch)
        elapsed   = time.time() - t_start
        rate      = embedded / elapsed if elapsed else 0
        eta       = (total - embedded) / rate if rate else 0
        print(f"  [{embedded:>6}/{total}]  {rate:.1f} ch/s  ETA {eta/60:.1f}min",
              end="\r", flush=True)

        if resume and b_num % 10 == 0:
            save_progress(PROGRESS_FILE, prog_ids)
            prog_ids = []

    sparse_f.close()
    meta_f.close()
    if resume and prog_ids:
        save_progress(PROGRESS_FILE, prog_ids)

    print(f"\n\n  Building FAISS index ({len(dense_vecs):,} x {EMBED_DIM}d)...")

    if resume and DENSE_FILE.exists():
        existing  = faiss.read_index(str(DENSE_FILE))
        n_e       = existing.ntotal
        exist_arr = faiss.rev_swig_ptr(
            existing.get_xb(), n_e * EMBED_DIM
        ).reshape(n_e, EMBED_DIM).copy()
        all_vecs  = np.vstack([exist_arr, np.array(dense_vecs, dtype="float32")])
    else:
        all_vecs = np.array(dense_vecs, dtype="float32")

    faiss.normalize_L2(all_vecs)
    index = faiss.IndexFlatIP(EMBED_DIM)
    index.add(all_vecs)
    faiss.write_index(index, str(DENSE_FILE))

    total_time = time.time() - t_start

    coll_counts = {}
    with META_FILE.open(encoding="utf-8") as f:
        for line in f:
            m    = json.loads(line)
            coll = m.get("collection", "?")
            coll_counts[coll] = coll_counts.get(coll, 0) + 1

    stats = {
        "model":          MODEL_NAME,
        "embed_dim":      EMBED_DIM,
        "total_vectors":  index.ntotal,
        "collections":    coll_counts,
        "total_time_s":   round(total_time, 1),
        "chunks_per_sec": round(embedded / total_time, 1),
        "device":         device,
        "batch_size":     batch_size,
        "backend":        "huggingface_transformers",
    }
    STATS_FILE.write_text(json.dumps(stats, indent=2))

    print(f"\n  Index complete")
    print(f"  dense.faiss    : {index.ntotal:,} vectors")
    print(f"  sparse.jsonl   : {SPARSE_FILE}")
    print(f"  metadata.jsonl : {META_FILE}")
    print(f"  Time           : {total_time/60:.1f} min")
    print(f"  Throughput     : {embedded/total_time:.1f} ch/s")
    print(f"\n  Per collection:")
    for coll, n in sorted(coll_counts.items()):
        print(f"    {coll:<14} {n:>6,}")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=DEFAULT_BATCH)
    ap.add_argument("--device",     default="cuda")
    ap.add_argument("--dry-run",    action="store_true")
    ap.add_argument("--resume",     action="store_true")
    args = ap.parse_args()

    if args.device == "cuda":
        try:
            import torch
            if not torch.cuda.is_available():
                print("  CUDA not available — falling back to CPU")
                args.device     = "cpu"
                args.batch_size = min(args.batch_size, 4)
        except ImportError:
            args.device     = "cpu"
            args.batch_size = min(args.batch_size, 4)

    embed(args.batch_size, args.device, args.resume, args.dry_run)


if __name__ == "__main__":
    main()
