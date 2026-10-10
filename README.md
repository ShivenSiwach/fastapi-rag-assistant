# FastAPI Docs RAG Assistant

**A hybrid-retrieval RAG system over FastAPI's official documentation — built with FastAPI, to answer questions about FastAPI.**

Combines lexical search (BM25) and semantic search (Gemini embeddings + FAISS), fused with Reciprocal Rank Fusion, grounded generation with forced citations, and two independent evaluation suites that measure retrieval accuracy and answer faithfulness with real numbers — not screenshots.

---

## Table of Contents

- [Why This Project Exists](#why-this-project-exists)
- [Architecture](#architecture)
- [Key Engineering Decisions](#key-engineering-decisions)
- [Evaluation Results](#evaluation-results)
- [Setup](#setup)
- [Running the System](#running-the-system)
- [Docker](#docker)
- [MCP Server](#mcp-server)
- [API Reference](#api-reference)
- [Project Structure](#project-structure)
- [Engineering Challenges Solved](#engineering-challenges-solved)
- [What I'd Add Next](#what-id-add-next)
- [Tech Stack](#tech-stack)


---

## Why This Project Exists

Most portfolio RAG demos stop at "it returns an answer." That's not the hard part. The hard part — and the part this project actually measures — is:

1. **Does retrieval find the right document?** Not "does it look plausible," but a labeled eval with a measurable recall rate.
2. **Is the generated answer grounded in what was retrieved, or hallucinated?** Checked with an LLM judge scoring faithfulness against the exact retrieved context, not just relevancy.
3. **Does the system know what it doesn't know?** Verified against questions the documentation genuinely doesn't cover, checking that it declines instead of confidently inventing an answer.

This project answers all three with numbers.

---

## Architecture

```
data/raw/{tutorial,advanced}/*.md         (85 official FastAPI doc files)
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│  INGESTION                                                    │
│  src/chunk_naive.py  → fixed-size word chunking                │
│                         503 chunks, ~150 words each             │
└─────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│  EMBEDDING                                                     │
│  src/embed.py  → Gemini (gemini-embedding-001)                 │
│                   768-dim vectors, L2-normalized, batched +     │
│                   rate-limit-paced, resumable on failure        │
└─────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│  RETRIEVAL — three interchangeable strategies                  │
│                                                                 │
│  src/search.py         BM25Okapi (lexical, stopword-filtered)  │
│  src/search_bm25.py    FAISS IndexFlatIP (semantic)             │
│  src/search_hybrid.py  Reciprocal Rank Fusion of both            │
└─────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│  GENERATION                                                    │
│  src/generate.py  → Gemini, forced source citations,            │
│                      explicit permission to say "I don't know"  │
└─────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│  SERVING                                                        │
│  src/api.py  → FastAPI: POST /query · GET /retrieve · GET /health│
│                 Retriever loaded once at startup, not per-request│
└─────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│  EVALUATION                                                     │
│  src/eval_retrieval.py   recall@5 / MRR, BM25 vs vector vs hybrid│
│  src/eval_generation.py  LLM-judged faithfulness + relevancy +   │
│                           refusal-correctness on unanswerables   │
└─────────────────────────────────────────────────────────────┘
```

---

## Key Engineering Decisions

**Reciprocal Rank Fusion, not a weighted score blend.**
BM25 scores (~7.0) and cosine similarity scores (~0.7) live on incompatible scales — averaging them directly is mathematically invalid. RRF fuses by *rank position* instead of raw score, which sidesteps the scale mismatch entirely and is the standard approach for combining heterogeneous retrievers.

**Stopword filtering in BM25 was a measured fix, not a hunch.**
The naive tokenizer initially mis-ranked unrelated documentation above `request-files.md` for the query *"How do I upload a file in FastAPI?"* — high-frequency function words and the corpus-wide term "fastapi" were diluting the actual signal. Removing a small stopword list fixed it, verified with a direct before/after score comparison rather than assumed.

**LLM-judged generation eval, not retrieval recall alone.**
Recall@k only proves the retriever found the right document — it says nothing about whether the *generated answer* stays faithful to it. The generation eval scores faithfulness and relevancy independently, and separately verifies refusal behavior on questions the docs don't cover, because a system that fabricates confidently is worse than one that admits a gap.

**The retriever loads once at server startup, not per request.**
`src/api.py` uses FastAPI's `lifespan` context manager to build the FAISS index and BM25 corpus a single time when the service starts, rather than rebuilding a search index on every incoming request — the difference between a toy script and a real service.

---

## Evaluation Results

### Retrieval accuracy — BM25 vs. Vector vs. Hybrid
*32 labeled questions across natural-language, exact-keyword, and hard/distractor-heavy query types, over an 85-file, 503-chunk corpus.*

| Strategy | Recall@5 | MRR |
|----------|:---:|:---:|
| BM25 (lexical) | 0.938 | 0.854 |
| Vector (semantic) | 0.969 | 0.922 |
| **Hybrid (RRF)** | **1.000** | 0.907 |

**By query type:**

| Query type | BM25 recall@5 | Vector recall@5 | Hybrid recall@5 |
|---|:---:|:---:|:---:|
| Natural language (n=14) | 0.857 | 0.929 | **1.000** |
| Exact keyword (n=7) | 1.000 | 1.000 | 1.000 |
| Hard / distractor-heavy (n=11) | 1.000 | 1.000 | 1.000 |

Hybrid was the **only strategy with zero misses**. It recovered a question — *"How do I read a custom header from a request?"* — that **neither BM25 nor vector search found alone**; the correct chunk ranked outside the top 5 in both individual retrievers, but RRF's combined rank score lifted it in. That's fusion doing its actual job, not just averaging two similar answers.

Hybrid trades a marginally lower MRR than vector-only (0.907 vs 0.922) for perfect recall. For RAG specifically, **recall matters more than rank order** — the generator reads all 5 retrieved chunks, not only the first, so finding the right chunk anywhere in the top 5 matters more than finding it first.

> **Honest caveat:** this is a 32-question eval on one corpus. The recall gap between hybrid and vector-only is a single question (32/32 vs 31/32) — real and reproducible, not a large-sample statistical guarantee.

### Generation quality — LLM-judged faithfulness & relevancy

| Metric | Result |
|---|:---:|
| Avg faithfulness (1–5 scale) | **5.00 / 5** |
| Avg relevancy (1–5 scale) | **5.00 / 5** |
| Answers scoring faithfulness ≥ 4 | 12 / 12 |
| Unanswerable questions correctly declined (no fabrication) | 5 / 5 |

Five deliberately out-of-scope questions (Kubernetes deployment, Redis caching, Stripe integration, SMTP email, Elasticsearch) were included specifically to test refusal behavior — the system correctly stated the documentation didn't cover them in **every single case**, rather than inventing plausible-sounding but fictional instructions.

> **Honest caveats:**
> - Due to free-tier daily quota limits, 9 of 17 generation-eval questions were generated and judged with `gemini-2.5-flash`; the remaining 8 used `gemini-3.5-flash-lite` after the former's daily quota was exhausted mid-run. Scores are comparable in aggregate but not a strict single-model comparison.
> - An LLM judging its own model family can be lenient. Manual review of individual answers confirmed the faithfulness scores were deserved — no fabricated claims were found — but also surfaced a subtler pattern the automated score doesn't capture: some answers cite only 1 of the 5 retrieved chunks instead of synthesizing across all of them. Technically faithful, but under-using available context — a prompt-engineering opportunity, not a faithfulness failure.

---

## Setup

```bash
git clone <this-repo>
cd fastapi-rag-tutorial

python -m venv venv
# Windows: venv\Scripts\Activate.ps1
# Mac/Linux: source venv/bin/activate

pip install -r requirements.txt
cp .env.example .env   # add your GEMINI_API_KEY
```

Get a free Gemini API key at [aistudio.google.com/apikey](https://aistudio.google.com/apikey) — no credit card required, though free-tier rate limits apply (see [Engineering Challenges Solved](#engineering-challenges-solved)).

## Running the System

```bash
# 1. Chunk the documentation corpus (already committed to data/processed/,
#    but reproducible from data/raw/ if you want to rebuild it)
python src/chunk_naive.py

# 2. Generate embeddings — the only step that costs API calls
python src/embed.py

# 3. Serve the API
uvicorn api:app --reload --app-dir src
# → open http://127.0.0.1:8000/docs for interactive API documentation

# 4. Run the evaluation suites
python src/eval_retrieval.py       # ~15 seconds, no generation calls
python src/eval_generation.py      # ~5-10 minutes, resumable if quota-limited
```

## Docker

```bash
docker compose up --build
# → http://127.0.0.1:8000/docs
```

The image builds from pre-computed `data/processed/` (chunks + embeddings), not raw markdown — the source documentation isn't needed at runtime. `GEMINI_API_KEY` is injected at container start via environment variable; it is never baked into the image.

## MCP Server

The assistant is also available as an MCP server for Claude Desktop, Cursor and other MCP clients, with measured retrieval parity, a tool-selection eval and a Docker image. See [mcp_server/README.md](mcp_server/README.md).

## API Reference

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | Liveness check — confirms the retriever finished loading |
| `/retrieve` | GET | Retrieval only, no generation call (for debugging / cheap testing) |
| `/query` | POST | Full pipeline: hybrid retrieval → grounded generation → cited answer |

**Example request:**
```bash
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "How do I upload a file in FastAPI?"}'
```

**Example response:**
```json
{
  "answer": "To upload a file in FastAPI, you can define file parameters using `File` [source: request-files.md]...",
  "sources": [
    {"source_file": "request-files.md", "score": 0.032},
    {"source_file": "request-files.md", "score": 0.032}
  ]
}
```

## Project Structure

```
fastapi-rag-tutorial/
├── data/
│   ├── raw/                    # Source markdown (tutorial/ + advanced/ docs)
│   ├── processed/               # chunks.jsonl, embeddings.npy
│   └── eval_set.json            # 32 labeled retrieval questions
├── src/
│   ├── chunk_naive.py           # Step 1: chunking
│   ├── embed.py                 # Step 2: embedding generation
│   ├── search.py                # Vector search (FAISS)
│   ├── search_bm25.py           # Keyword search (BM25)
│   ├── search_hybrid.py         # RRF fusion
│   ├── generate.py              # Grounded generation
│   ├── api.py                   # FastAPI service
│   ├── eval_retrieval.py        # Retrieval accuracy eval
│   └── eval_generation.py       # LLM-judged generation eval
├── eval_results.json            # Retrieval eval output
├── eval_generation_results.json # Generation eval output
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## Engineering Challenges Solved

Real constraints hit and resolved during development — documented here because working through infrastructure limits is as much a part of shipping a system as writing the retrieval logic:

- **API key format migration.** Google transitioned Gemini API keys from the legacy `AIzaSy...` format to a new `AQ....` format mid-2026. Diagnosed by comparing the actual error against Google's own key-management page rather than assuming the newer format was wrong.
- **Free-tier daily quota (20 requests/day on `gemini-2.5-flash`).** Discovered mid-eval-run. Resolved by adding resumable checkpointing (progress saved after every question, so a quota wall doesn't lose completed work) and falling back to `gemini-3.5-flash-lite` for the remainder of the run — a documented, honest trade-off rather than a silent one.
- **Free-tier embedding rate limit (100 requests/minute).** Solved with batched requests, paced with a fixed delay between batches, and per-batch progress saving so a failure partway through doesn't require re-embedding from scratch.
- **BM25 retrieval quality regression from unfiltered stopwords.** Caught by direct before/after comparison on a real query, not assumed — see [Key Engineering Decisions](#key-engineering-decisions).
- **Windows `Copy-Item -Recurse` failures on nested directories.** Resolved by switching to `robocopy`, which handles nested directory trees far more reliably on Windows than PowerShell's native cmdlets.
- **FAISS's compiled backend requiring `libgomp1` inside the Docker image** — present implicitly on Windows but missing from the `python:3.11-slim` base image, causing a container-only import failure that never appeared locally.

## What I'd Add Next

- **MMR (Maximal Marginal Relevance) re-ranking** so retrieved chunks span multiple source files on broad questions, instead of occasionally all 5 coming from a single file.
- **A stricter generation prompt** to reduce the "cites 1 of 5 retrieved chunks" under-synthesis pattern surfaced during manual review.
- **A larger, more adversarial eval set.** The current "hard" question set turned out not to be hard enough to separate the three retrieval strategies — all three scored a perfect recall@5 on it, which is itself a useful finding about eval design.
- **Single-model generation eval**, re-run end-to-end on one consistent model once paid-tier or multi-key quota headroom is available.

## Tech Stack

**Language & Runtime:** Python 3.11
**LLM Provider:** Google Gemini API (embeddings, generation, and eval judging)
**Retrieval:** FAISS (vector search) · rank_bm25 (lexical search)
**Serving:** FastAPI · Uvicorn
**Containerization:** Docker · Docker Compose
**Evaluation:** Custom labeled eval sets · LLM-as-judge methodology

---

*Built as a hands-on deep dive into production RAG system design — from naive fixed-size chunking through hybrid retrieval, grounded generation, and measured evaluation, with every design decision backed by a comparison, not a guess.*
