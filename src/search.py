import json
import os
import numpy as np
import faiss
from pathlib import Path
from dotenv import load_dotenv
from google import genai

load_dotenv()

CHUNKS_PATH = Path(__file__).parent.parent / "data" / "processed" / "chunks.jsonl"
EMBED_PATH = Path(__file__).parent.parent / "data" / "processed" / "embeddings.npy"
MODEL = "gemini-embedding-001"
EMBED_DIM = 768


def load_chunks():
    with open(CHUNKS_PATH) as f:
        return [json.loads(line) for line in f]


def embed_query(client, query: str) -> np.ndarray:
    resp = client.models.embed_content(
        model=MODEL,
        contents=[query],
        # RETRIEVAL_QUERY (not RETRIEVAL_DOCUMENT) -- the query side of a
        # search should be embedded differently than the documents being
        # searched, and Gemini's model is trained to expect this distinction.
        config={"output_dimensionality": EMBED_DIM, "task_type": "RETRIEVAL_QUERY"},
    )
    vec = np.array(resp.embeddings[0].values, dtype=np.float32)
    return vec / np.linalg.norm(vec)


def build_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    # IndexFlatIP = "Flat" (brute-force, exact, no approximation) index using
    # Inner Product. Since our vectors are normalized (length 1), inner
    # product == cosine similarity. "Flat"/exact is fine at our scale (325
    # vectors); large-scale systems trade exactness for speed with an
    # approximate index instead.
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index


def search(query: str, k: int = 5):
    api_key = os.environ.get("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)

    chunks = load_chunks()
    embeddings = np.load(EMBED_PATH)
    index = build_index(embeddings)

    query_vec = embed_query(client, query).reshape(1, -1)
    similarities, indices = index.search(query_vec, k)

    results = []
    for score, idx in zip(similarities[0], indices[0]):
        c = chunks[idx]
        results.append({"source_file": c["source_file"], "score": float(score), "text": c["text"]})
    return results


if __name__ == "__main__":
    query = "python-multipart"
    print(f"Query: {query}\n")
    for r in search(query, k=5):
        print(f"[{r['score']:.4f}] {r['source_file']}")
        print(" ", r["text"][:150].replace("\n", " "), "...\n")