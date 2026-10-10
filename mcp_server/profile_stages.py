"""Time each stage of hybrid search, using the project's own building blocks."""
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "src"))

t0 = time.perf_counter()
from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

import numpy as np
from google import genai
from search import build_index, embed_query, load_chunks
from search_bm25 import build_bm25, tokenize

print(f"imports: {time.perf_counter() - t0:.2f}s")

QUERY = "How do I read a custom header?"


def timed(label, fn, results):
    start = time.perf_counter()
    out = fn()
    results[label] = time.perf_counter() - start
    return out


for run in (1, 2, 3):
    r = {}
    chunks = timed("load_chunks", load_chunks, r)
    bm25 = timed("build_bm25", lambda: build_bm25(chunks), r)
    timed("bm25_score", lambda: bm25.get_scores(tokenize(QUERY)), r)
    emb = timed("load_embeddings", lambda: np.load("data/processed/embeddings.npy"), r)
    index = timed("build_faiss_index", lambda: build_index(emb), r)
    client = timed(
        "make_client",
        lambda: get_client(),
        r,
    )
    qvec = timed(
        "embed_query (Gemini)",
        lambda: embed_query(client, QUERY).reshape(1, -1),
        r,
    )
    timed("faiss_search", lambda: index.search(qvec, 20), r)

    total = sum(r.values())
    print(f"\nRun {run}: total {total:.2f}s")
    for label, secs in r.items():
        print(f"  {label:<22}{secs * 1000:>8.0f} ms  {secs / total * 100:>4.0f}%")