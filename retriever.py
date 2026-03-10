"""
retriever.py  —  Nuremberg Scholar Hybrid Retriever
=====================================================
Pipeline (paper-backed):
  1. Query encoding  : BGE-M3 dense (1024d) + sparse (lexical weights)
                       Uses same HF-only BGEM3 class as embedder.py
  2. Dense retrieval : FAISS FlatIP top-N  (cosine via L2-norm + inner product)
  3. Sparse retrieval: dot-product over inverted sparse index, top-N
  4. RRF fusion      : k=60, merge dense+sparse ranked lists → top-K candidates
  5. Reranking       : bge-reranker-v2-m3 cross-encoder → sigmoid scores → top-K_final
  6. Return          : list of ranked Result objects with metadata + scores

Design decisions from literature:
  - RRF k=60: industry standard, robust across domains (Cormack et al. 2009)
  - Dense N=100, Sparse N=100 → RRF top-100 → rerank to top-5
    (two-stage funnel: high recall first, high precision second)
  - BGE-M3 paper recommends dense+sparse hybrid for long-document corpus;
    sparse alone outperforms dense by ~10 NDCG points on long docs (MLDR)
  - bge-reranker-v2-m3 is the official reranker pairing for bge-m3 embeddings
  - Scores sigmoid-mapped to [0,1] for interpretability at generation time
  - No query instruction prefix needed for BGE-M3 (unlike BGE v1.5)

Usage:
    from retriever import Retriever
    r = Retriever()
    results = r.retrieve("What did Göring say about the Luftwaffe?", top_k=5)
    for res in results:
        print(res)

    # CLI smoke test
    python retriever.py --query "crimes against humanity Article 6c"
    python retriever.py --query "Ohlendorf Einsatzgruppen" --top-k 3 --no-rerank
    python retriever.py --query "London Agreement 1945" --dense-only
"""

import json
import time
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

# ── Paths ─────────────────────────────────────────────────────────────────────

INDEX_DIR     = Path("output/index")
DENSE_FILE    = INDEX_DIR / "dense.faiss"
SPARSE_FILE   = INDEX_DIR / "sparse.jsonl"
META_FILE     = INDEX_DIR / "metadata.jsonl"

EMBED_MODEL   = "BAAI/bge-m3"
RERANK_MODEL  = "BAAI/bge-reranker-v2-m3"

EMBED_DIM     = 1024
RRF_K         = 60       # Cormack et al. 2009 — robust standard
DENSE_N       = 100      # candidates from dense retrieval
SPARSE_N      = 100      # candidates from sparse retrieval
RERANK_INPUT  = 100      # max chunks sent to reranker (post-RRF)
DEFAULT_TOP_K = 5        # final chunks returned to generator
MAX_Q_TOKENS  = 512      # query max tokens (queries are short)


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class Result:
    chunk_id:     str
    body:         str
    collection:   str
    date_iso:     Optional[str]
    speaker:      Optional[str]
    source_url:   Optional[str]
    page_number:  Optional[int]
    slug:         Optional[str]
    # Scores
    dense_rank:   Optional[int]   = None
    sparse_rank:  Optional[int]   = None
    rrf_score:    float           = 0.0
    rerank_score: Optional[float] = None   # sigmoid [0,1], None if skipped

    def __str__(self):
        rerank = f"  rerank={self.rerank_score:.4f}" if self.rerank_score is not None else ""
        return (
            f"[{self.collection}] {self.date_iso or '?'}  {self.slug or ''}\n"
            f"  speaker={self.speaker or '—'}  page={self.page_number or '?'}\n"
            f"  rrf={self.rrf_score:.5f}{rerank}\n"
            f"  {self.body[:200]}..."
        )


# ── BGE-M3 query encoder (reuses embedder logic, query-side only) ─────────────

UNUSED_TOKENS = [0, 1, 2]   # <s>, <pad>, </s>


class QueryEncoder:
    """
    Encodes a query into:
      dense_vec      : np.ndarray (1024,)  L2-normalised
      sparse_weights : dict {token_str: score}

    sparse_linear = Linear(1024, 1) — scalar weight per token position.
    Scatter onto input_ids vocab positions via scatter_reduce("amax").
    """

    def __init__(self, model_name: str, device: str):
        import torch
        import torch.nn as nn
        from transformers import AutoTokenizer, AutoModel
        from huggingface_hub import hf_hub_download

        self.device    = torch.device(device)
        self.torch     = torch
        self.fp16      = device != "cpu"

        self.tokenizer  = AutoTokenizer.from_pretrained(model_name)
        self.vocab_size = self.tokenizer.vocab_size   # 250002

        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.fp16 else torch.float32,
        )
        self.model.to(self.device)
        self.model.eval()

        # sparse_linear: Linear(1024, 1) — ~3.52 kB
        sparse_path        = hf_hub_download(repo_id=model_name, filename="sparse_linear.pt")
        raw                = torch.load(sparse_path, map_location="cpu", weights_only=True)
        in_f, out_f        = raw["weight"].shape[1], raw["weight"].shape[0]
        self.sparse_linear = nn.Linear(in_f, out_f, bias=True)
        self.sparse_linear.load_state_dict(raw, strict=True)
        if self.fp16:
            self.sparse_linear = self.sparse_linear.half()
        self.sparse_linear.to(self.device)
        self.sparse_linear.eval()

    def encode(self, query: str) -> dict:
        import torch
        import numpy as np
        import torch.nn.functional as F

        enc = self.tokenizer(
            [query],
            padding=True,
            truncation=True,
            max_length=MAX_Q_TOKENS,
            return_tensors="pt",
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}

        with self.torch.no_grad():
            out         = self.model(**enc, return_dict=True)
            last_hidden = out.last_hidden_state              # (1, seq, 1024)

            # Dense: CLS token, L2-normalised
            dense    = F.normalize(last_hidden[:, 0, :].float(), p=2, dim=-1)
            dense_np = dense.cpu().numpy().astype("float32")[0]  # (1024,)

            # Sparse: Linear(1024,1) -> relu -> scalar per token -> scatter
            token_weights = torch.relu(
                self.sparse_linear(last_hidden)
            ).squeeze(-1).float()                            # (1, seq_len)

        sparse_emb = torch.zeros(
            1, self.vocab_size, dtype=torch.float32, device=self.device
        )
        sparse_emb = sparse_emb.scatter_reduce(
            dim=1,
            index=enc["input_ids"],
            src=token_weights,
            reduce="amax",
            include_self=False,
        )
        for uid in UNUSED_TOKENS:
            if uid < self.vocab_size:
                sparse_emb[0, uid] = 0.0

        nonzero = sparse_emb[0].nonzero(as_tuple=True)[0].tolist()
        scores  = sparse_emb[0][nonzero].cpu().tolist()
        sparse  = {}
        for tid, score in zip(nonzero, scores):
            if score <= 0:
                continue
            tok = self.tokenizer.decode([tid]).strip()
            if tok:
                sparse[tok] = round(float(score), 4)

        return {"dense_vec": dense_np, "sparse_weights": sparse}


# ── Reranker ──────────────────────────────────────────────────────────────────

class Reranker:
    """
    bge-reranker-v2-m3 cross-encoder.
    Pure HuggingFace: AutoModelForSequenceClassification.
    Scores sigmoid-mapped to [0,1] per HF model card recommendation.
    """

    def __init__(self, model_name: str, device: str):
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        self.device    = torch.device(device)
        self.torch     = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model     = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device != "cpu" else torch.float32,
        )
        self.model.to(self.device)
        self.model.eval()

    def rerank(self, query: str, candidates: list[Result],
               batch_size: int = 32) -> list[Result]:
        """
        Score all candidates, return sorted by rerank_score descending.
        Input pairs: [query, chunk_body].
        Output scores: sigmoid logits in [0, 1].
        """
        import torch

        pairs = [[query, c.body] for c in candidates]
        all_scores = []

        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i + batch_size]
            enc   = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            with torch.no_grad():
                logits = self.model(**enc, return_dict=True).logits.view(-1).float()
            scores = torch.sigmoid(logits).cpu().tolist()
            all_scores.extend(scores)

        for candidate, score in zip(candidates, all_scores):
            candidate.rerank_score = round(score, 6)

        return sorted(candidates, key=lambda x: x.rerank_score, reverse=True)


# ── Sparse index (in-memory inverted index over sparse.jsonl) ─────────────────

class SparseIndex:
    """
    Loads sparse.jsonl into an inverted index: {token -> {chunk_idx: weight}}.
    Query: dot-product over query token weights → ranked list.
    chunk_idx corresponds to the row order in metadata.jsonl (= FAISS row).
    """

    def __init__(self, sparse_path: Path):
        # inverted: token_str -> list of (chunk_idx, weight)
        self.inverted: dict[str, list[tuple[int, float]]] = {}
        self.chunk_ids: list[str] = []

        print(f"  Loading sparse index from {sparse_path}...")
        t0 = time.time()
        with sparse_path.open(encoding="utf-8") as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                self.chunk_ids.append(obj["chunk_id"])
                for token, weight in obj.get("weights", {}).items():
                    if token not in self.inverted:
                        self.inverted[token] = []
                    self.inverted[token].append((idx, weight))

        print(f"  Sparse index: {len(self.chunk_ids):,} chunks, "
              f"{len(self.inverted):,} unique tokens  "
              f"({time.time()-t0:.1f}s)")

    def query(self, sparse_weights: dict[str, float], top_n: int) -> list[tuple[int, float]]:
        """
        Returns list of (chunk_idx, dot_product_score) sorted descending.
        """
        scores: dict[int, float] = {}
        for token, q_weight in sparse_weights.items():
            if token in self.inverted:
                for chunk_idx, d_weight in self.inverted[token]:
                    scores[chunk_idx] = scores.get(chunk_idx, 0.0) + q_weight * d_weight

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_n]


# ── RRF fusion ────────────────────────────────────────────────────────────────

def reciprocal_rank_fusion(
    dense_ranked:  list[tuple[int, float]],   # [(chunk_idx, score), ...]
    sparse_ranked: list[tuple[int, float]],   # [(chunk_idx, score), ...]
    k: int = RRF_K,
) -> list[tuple[int, float]]:
    """
    RRF(d) = Σ  1 / (k + rank_r(d))
    Documents absent from a list contribute 0 from that list.
    Returns list of (chunk_idx, rrf_score) sorted descending.
    """
    rrf: dict[int, float] = {}

    for rank, (chunk_idx, _) in enumerate(dense_ranked, start=1):
        rrf[chunk_idx] = rrf.get(chunk_idx, 0.0) + 1.0 / (k + rank)

    for rank, (chunk_idx, _) in enumerate(sparse_ranked, start=1):
        rrf[chunk_idx] = rrf.get(chunk_idx, 0.0) + 1.0 / (k + rank)

    return sorted(rrf.items(), key=lambda x: x[1], reverse=True)


# ── Main Retriever ────────────────────────────────────────────────────────────

class Retriever:
    """
    Full hybrid retrieval pipeline.

    Parameters
    ----------
    device        : "cuda" / "cpu"
    dense_n       : candidates from FAISS  (default 100)
    sparse_n      : candidates from sparse index (default 100)
    rerank_input  : max chunks sent to reranker (default 100)
    top_k         : final results returned (default 5)
    use_reranker  : bool (default True)
    dense_only    : skip sparse + RRF, just return FAISS top-k (baseline mode)
    """

    def __init__(
        self,
        device:       str  = "cuda",
        dense_n:      int  = DENSE_N,
        sparse_n:     int  = SPARSE_N,
        rerank_input: int  = RERANK_INPUT,
        top_k:        int  = DEFAULT_TOP_K,
        use_reranker: bool = True,
        dense_only:   bool = False,
    ):
        import faiss

        self.device       = device
        self.dense_n      = dense_n
        self.sparse_n     = sparse_n
        self.rerank_input = rerank_input
        self.top_k        = top_k
        self.use_reranker = use_reranker
        self.dense_only   = dense_only

        # ── Load FAISS index ──────────────────────────────────────────────────
        if not DENSE_FILE.exists():
            raise FileNotFoundError(f"Dense index not found: {DENSE_FILE}")
        print(f"  Loading FAISS index...")
        self.faiss_index = faiss.read_index(str(DENSE_FILE))
        print(f"  FAISS: {self.faiss_index.ntotal:,} vectors")

        # ── Load metadata ─────────────────────────────────────────────────────
        print(f"  Loading metadata...")
        self.metadata: list[dict] = []
        with META_FILE.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.metadata.append(json.loads(line))
        print(f"  Metadata: {len(self.metadata):,} records")

        # ── Chunk_id → row index map ──────────────────────────────────────────
        self.chunk_id_to_idx = {m["chunk_id"]: i for i, m in enumerate(self.metadata)}

        # ── Sparse index ──────────────────────────────────────────────────────
        if not dense_only:
            self.sparse_index = SparseIndex(SPARSE_FILE)
        else:
            self.sparse_index = None

        # ── Query encoder ─────────────────────────────────────────────────────
        print(f"  Loading query encoder ({EMBED_MODEL})...")
        self.encoder = QueryEncoder(EMBED_MODEL, device)

        # ── Reranker ──────────────────────────────────────────────────────────
        self.reranker = None
        if use_reranker:
            print(f"  Loading reranker ({RERANK_MODEL})...")
            self.reranker = Reranker(RERANK_MODEL, device)

        print(f"\n  Retriever ready  "
              f"dense_n={dense_n}  sparse_n={sparse_n}  "
              f"rerank={use_reranker}  top_k={top_k}\n")

    def retrieve(self, query: str, top_k: Optional[int] = None) -> list[Result]:
        import numpy as np

        top_k = top_k or self.top_k
        t0    = time.time()

        # ── 1. Encode query ───────────────────────────────────────────────────
        encoded     = self.encoder.encode(query)
        dense_vec   = encoded["dense_vec"]          # (1024,)
        sparse_w    = encoded["sparse_weights"]     # {token: score}

        # ── 2. Dense retrieval (FAISS) ────────────────────────────────────────
        q_vec   = dense_vec.reshape(1, -1).astype("float32")
        scores, indices = self.faiss_index.search(q_vec, self.dense_n)
        dense_ranked = [
            (int(idx), float(score))
            for idx, score in zip(indices[0], scores[0])
            if idx >= 0
        ]

        if self.dense_only:
            results = self._build_results(
                dense_ranked[:top_k],
                dense_ranked=dense_ranked,
                sparse_ranked=[],
            )
            if self.use_reranker and self.reranker:
                results = self.reranker.rerank(query, results)
            return results[:top_k]

        # ── 3. Sparse retrieval ───────────────────────────────────────────────
        sparse_ranked = self.sparse_index.query(sparse_w, self.sparse_n)

        # ── 4. RRF fusion ─────────────────────────────────────────────────────
        fused = reciprocal_rank_fusion(dense_ranked, sparse_ranked, k=RRF_K)
        fused = fused[:self.rerank_input]

        # ── 5. Build Result objects ───────────────────────────────────────────
        # Build rank lookup for annotation
        dense_rank_map  = {idx: r+1 for r, (idx, _) in enumerate(dense_ranked)}
        sparse_rank_map = {idx: r+1 for r, (idx, _) in enumerate(sparse_ranked)}

        candidates = []
        for chunk_idx, rrf_score in fused:
            if chunk_idx >= len(self.metadata):
                continue
            m = self.metadata[chunk_idx]
            candidates.append(Result(
                chunk_id    = m.get("chunk_id", ""),
                body        = m.get("body", ""),
                collection  = m.get("collection", ""),
                date_iso    = m.get("date_iso"),
                speaker     = m.get("speaker"),
                source_url  = m.get("source_url"),
                page_number = m.get("page_number"),
                slug        = m.get("slug"),
                dense_rank  = dense_rank_map.get(chunk_idx),
                sparse_rank = sparse_rank_map.get(chunk_idx),
                rrf_score   = rrf_score,
            ))

        # ── 6. Rerank ─────────────────────────────────────────────────────────
        if self.use_reranker and self.reranker and candidates:
            candidates = self.reranker.rerank(query, candidates)

        elapsed = time.time() - t0
        print(f"  Retrieved {len(candidates[:top_k])} results in {elapsed:.2f}s  "
              f"(dense={len(dense_ranked)} sparse={len(sparse_ranked)} "
              f"fused={len(fused)} reranked={self.use_reranker})")

        return candidates[:top_k]

    def _build_results(self, ranked, dense_ranked, sparse_ranked) -> list[Result]:
        dense_rank_map  = {idx: r+1 for r, (idx, _) in enumerate(dense_ranked)}
        sparse_rank_map = {idx: r+1 for r, (idx, _) in enumerate(sparse_ranked)}
        results = []
        for chunk_idx, rrf_score in ranked:
            if chunk_idx >= len(self.metadata):
                continue
            m = self.metadata[chunk_idx]
            results.append(Result(
                chunk_id    = m.get("chunk_id", ""),
                body        = m.get("body", ""),
                collection  = m.get("collection", ""),
                date_iso    = m.get("date_iso"),
                speaker     = m.get("speaker"),
                source_url  = m.get("source_url"),
                page_number = m.get("page_number"),
                slug        = m.get("slug"),
                dense_rank  = dense_rank_map.get(chunk_idx),
                sparse_rank = sparse_rank_map.get(chunk_idx),
                rrf_score   = rrf_score,
            ))
        return results


# ── CLI smoke test ────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Nuremberg Scholar — Retriever smoke test")
    ap.add_argument("--query",      required=True, help="Query string")
    ap.add_argument("--top-k",      type=int, default=DEFAULT_TOP_K)
    ap.add_argument("--device",     default="cuda")
    ap.add_argument("--no-rerank",  action="store_true")
    ap.add_argument("--dense-only", action="store_true")
    ap.add_argument("--dense-n",    type=int, default=DENSE_N)
    ap.add_argument("--sparse-n",   type=int, default=SPARSE_N)
    args = ap.parse_args()

    if args.device == "cuda":
        try:
            import torch
            if not torch.cuda.is_available():
                args.device = "cpu"
        except ImportError:
            args.device = "cpu"

    print(f"\nNuremberg Scholar — Retriever")
    print("=" * 60)

    retriever = Retriever(
        device       = args.device,
        dense_n      = args.dense_n,
        sparse_n     = args.sparse_n,
        top_k        = args.top_k,
        use_reranker = not args.no_rerank,
        dense_only   = args.dense_only,
    )

    print(f"\nQuery: {args.query}\n")
    results = retriever.retrieve(args.query, top_k=args.top_k)

    print(f"\n{'='*60}")
    print(f"Top {len(results)} results:")
    print(f"{'='*60}\n")
    for i, r in enumerate(results, 1):
        print(f"  ── Result {i} ──")
        print(f"  {r}\n")


if __name__ == "__main__":
    main()
