# Nuremberg Scholar

RAG system over the International Military Tribunal proceedings at Nuremberg (1945-1946). Ask a question about the trials, get an answer grounded in the actual transcripts with every claim cited to a specific source passage.

**[Live Demo](https://huggingface.co/spaces/dtufail/nuremberg-scholar)** · **[Dataset](https://huggingface.co/datasets/dtufail/nuremberg-trials-corpus)**

---

## What it does

You ask: *"What did Goering say about the Luftwaffe?"*

The system searches 46,325 indexed passages from the trial record, retrieves the most relevant testimony, generates a cited answer, and then verifies every citation — stripping any the model hallucinated.

The corpus covers 221 trial session transcripts, the full Tribunal judgment (October 1946), key prosecution documents, and secondary source materials, all scraped from the Yale Avalon Project.

## Architecture

```
Query
  │
  ├── BGE-M3 encode (dense 1024-d + sparse lexical)
  │
  ├── FAISS search ──── top 100 dense candidates ──┐
  │                                                 ├── RRF fusion → top 25
  ├── CSR sparse search ─ top 100 sparse candidates ┘
  │
  ├── bge-reranker-v2-m3 cross-encoder → top 5
  │
  ├── Llama-3.1-8B via Groq API → cited answer
  │
  └── Citation verifier → strip hallucinated refs
```


## Optimisations

Seven rounds of measured optimisation, each with before/after numbers from terminal output:

1. **Groq API swap** — replaced local Llama-3.1-8B (12-18s/query on A10G) with Groq API (0.61s). Freed 10GB VRAM.
2. **Rerank input 100→25** — retrieval dropped from 1.23s to 0.65s with no quality loss.
3. **Max tokens 512→350** — eliminated mid-citation truncation.
4. **Metadata patch** — fixed secondary collection dates. 46,325 chunks now have 100% date_iso coverage.
5. **Semantic cache** — cosine-similarity LRU over query vectors (threshold 0.97). Repeat queries return in <1ms.
6. **Reranker bypass** — attempted and rejected. Max FAISS cosine sim on this corpus is 0.63. No safe threshold exists.
7. **Scipy CSR sparse index** — replaced Python dict-of-lists (608MB, 15ms/query) with CSR matrix (76MB, 1.7ms/query).

## Setup

```bash
git clone https://github.com/DTufail/nuremberg-project.git
cd nuremberg-project
pip install -r requirements.txt
```

You need a Groq API key:
```bash
export GROQ_API_KEY="your-key-here"
```

Run locally:
```bash
python app.py
```

The index files (~230MB) download automatically from HuggingFace on first run.

## Scraping the corpus yourself

If you want to rebuild from scratch rather than using the pre-built index:

```bash
python scraper.py          # Scrape Yale Avalon → output/sessions/, judgment/, key_docs/, secondary/
python chunker.py          # Chunk → output/chunks.jsonl
python embedder.py         # Embed → output/index/
```

The scraper handles all four Yale page number formats, redirect stubs, and container fallbacks. Harvard Law patches fill two sessions missing from Yale.

## Stack

| Component | Technology |
|---|---|
| Embedder | BAAI/bge-m3 (dense 1024-d + sparse lexical) |
| Vector store | FAISS IndexFlatIP |
| Sparse index | Scipy CSR matrix (19,349 tokens × 46,325 chunks) |
| Reranker | bge-reranker-v2-m3 |
| Generator | Llama-3.1-8B-Instruct via Groq |
| Cache | Semantic LRU (cosine sim threshold 0.97) |
| UI | Gradio |
| Deployment | HuggingFace Spaces |

## Source

Trial transcripts from the [Yale Avalon Project](https://avalon.law.yale.edu/subject_menus/imt.asp). Original transcripts are public domain. Two sessions patched from the [Harvard Law Nuremberg Trials Project](https://nuremberg.law.harvard.edu/).

## License

CC BY 4.0
