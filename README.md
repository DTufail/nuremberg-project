---
license: cc-by-4.0
language:
- en
tags:
- rag
- legal
- history
- nuremberg
- retrieval
- hybrid-search
- international-law
pretty_name: Nuremberg Trials RAG Corpus
size_categories:
- 10K<n<100K
task_categories:
- question-answering
- text-retrieval
configs:
- config_name: default
  data_files: []
---

# Nuremberg Trials RAG Corpus

A preprocessed retrieval corpus built from the International Military Tribunal (IMT) proceedings at Nuremberg (1945–1946), structured for hybrid dense-sparse RAG pipelines. 46,325 chunks covering daily trial transcripts, prosecution documents, the final judgment, and supporting briefs.

This corpus underpins **[Nuremberg Scholar](https://github.com/DTufail/nuremberg-scholar)**, a RAG system using BGE-M3 hybrid retrieval, bge-reranker-v2-m3 cross-encoder reranking, and Llama-3.1-8B-Instruct generation.

## Corpus Statistics

| Field | Value |
|---|---|
| Total chunks | 46,325 |
| Session transcript chunks | ~39,600 |
| Document/brief chunks | ~6,700 |
| Date range | November 1945 – October 1946 |
| Chunk size | 512 tokens max, paragraph-boundary splits |
| Overlap | 50 tokens |
| Embedding model | BAAI/bge-m3 (1024-d dense + sparse lexical) |

## Collections

| Collection | Records | Description |
|---|---|---|
| `sessions` | ~39,600 | Daily trial transcripts with speaker attribution and page numbers |
| `judgment` | — | Final tribunal judgment (September–October 1946) |
| `key_docs` | — | Key prosecution documents and exhibits |
| `secondary` | ~6,700 | Nazi Conspiracy and Aggression (NCA) briefs and supporting documents |
| `vol1` | — | Volume 1 of the official trial record |

## File Structure

```
index/
  dense.faiss         # FAISS IndexFlatIP — 46,325 vectors, 1024-d float32
  metadata.jsonl      # Per-chunk metadata (one JSON object per line, row-aligned with FAISS)
  sparse.jsonl        # BGE-M3 sparse lexical weights (19,349 unique tokens, 7.1M non-zero entries)
  chunks.jsonl        # Full chunk text + metadata
```

## Chunk Schema

Each record in `chunks.jsonl`:

```json
{
  "chunk_id":    "sessions::01-02-46::0001",
  "collection":  "sessions",
  "slug":        "01-02-46",
  "source_url":  "https://avalon.law.yale.edu/imt/01-02-46.asp",
  "date_iso":    "1946-01-02",
  "speaker":     "COL. STOREY",
  "page_number": 255,
  "chunk_index": 1,
  "total_chunks": 79,
  "token_count": 548,
  "body":        "COL. STOREY: If the Tribunal please...",
  "text":        "[Date: 1946-01-02 | Source: 01-02-46 | Speaker: COL. STOREY | ...] COL. STOREY: ..."
}
```

`text` is the retrieval-formatted string passed to BGE-M3 at embed time. `body` is the raw chunk text.

## Usage

```python
from huggingface_hub import snapshot_download
from pathlib import Path
import faiss, json

path = snapshot_download(
    repo_id="dtufail/nuremberg-trials-corpus",
    repo_type="dataset",
    allow_patterns=["index/*"]
)
index_dir = Path(path) / "index"

# Load FAISS index
index = faiss.read_index(str(index_dir / "dense.faiss"))

# Load metadata
metadata = []
with open(index_dir / "metadata.jsonl") as f:
    for line in f:
        metadata.append(json.loads(line))

print(f"Loaded {index.ntotal} vectors, {len(metadata)} metadata records")
```

## Retrieval Pipeline

Built for a five-stage hybrid retrieval pipeline:

1. **Dense retrieval** — BGE-M3 query → FAISS inner product search (top-100)
2. **Sparse retrieval** — BGE-M3 sparse weights → inverted index lookup (top-100)
3. **RRF fusion** — Reciprocal Rank Fusion merges dense + sparse lists (top-25)
4. **Reranking** — bge-reranker-v2-m3 cross-encoder scores 25 candidates
5. **Generation** — Top-5 chunks → Llama-3.1-8B-Instruct (4-bit NF4 quantization)

## Sources

Trial transcripts sourced from:

- **[Yale Avalon Project](https://avalon.law.yale.edu/subject_menus/imt.asp)** — Primary source for IMT transcripts
- **[Harvard Nuremberg Trials Project](http://nuremberg.law.harvard.edu/)** — Used to fill gaps where Yale transcripts were incomplete or missing

Original transcripts are public domain. This derived corpus (chunking, embeddings, metadata extraction) is released under **CC BY 4.0**.

## Citation

```bibtex
@dataset{tufail2025nuremberg,
  author    = {Tufail, Daniyal},
  title     = {Nuremberg Trials RAG Corpus},
  year      = {2025},
  publisher = {HuggingFace},
  url       = {https://huggingface.co/datasets/dtufail/nuremberg-trials-corpus}
}
```
