import os
from dotenv import load_dotenv
from google import genai

from search import load_chunks, embed_query, build_index  # step 5 (vector)
from search_bm25 import build_bm25, tokenize               # step 6 (BM25)

load_dotenv()

RRF_K = 60  # standard constant from the original RRF paper


def hybrid_search(query: str, k: int = 5, pool_size: int = 20):
    chunks = load_chunks()

    # --- Get BM25's full ranking ---
    bm25 = build_bm25(chunks)
    bm25_scores = bm25.get_scores(tokenize(query))
    bm25_ranking = bm25_scores.argsort()[::-1][:pool_size]  # indices, best first

    # --- Get vector search's full ranking ---
    import numpy as np
    embeddings = np.load("data/processed/embeddings.npy")
    index = build_index(embeddings)
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    query_vec = embed_query(client, query).reshape(1, -1)
    _, vector_ranking = index.search(query_vec, pool_size)
    vector_ranking = vector_ranking[0]

    # --- Fuse the two rankings with RRF ---
    rrf_scores = {}
    for rank, idx in enumerate(bm25_ranking):
        rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (RRF_K + rank + 1)
    for rank, idx in enumerate(vector_ranking):
        rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (RRF_K + rank + 1)

    top_indices = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:k]

    results = []
    for idx, score in top_indices:
        c = chunks[int(idx)]
        results.append({"source_file": c["source_file"], "score": round(score, 5), "text": c["text"]})
    return results


if __name__ == "__main__":
    for query in ["python-multipart", "How do I upload a file in FastAPI?"]:
        print(f"Query: {query}\n")
        for r in hybrid_search(query, k=5):
            print(f"[{r['score']}] {r['source_file']}")
            print(" ", r["text"][:150].replace("\n", " "), "...\n")
        print("=" * 60, "\n")