import json
import re
from pathlib import Path
from rank_bm25 import BM25Okapi

CHUNKS_PATH = Path(__file__).parent.parent / "data" / "processed" / "chunks.jsonl"

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Common words that appear in nearly every chunk regardless of topic --
# without removing these, they dilute the score contribution of the words
# that actually distinguish one chunk from another.
_STOPWORDS = {
    "a", "an", "the", "how", "do", "does", "did", "i", "you", "is", "are",
    "in", "on", "of", "to", "for", "and", "or", "it", "this", "that",
    "with", "can", "what", "when", "where", "why", "be", "as", "if",
    "fastapi",
}


def tokenize(text: str):
    tokens = _TOKEN_RE.findall(text.lower())
    return [t for t in tokens if t not in _STOPWORDS]


def load_chunks():
    with open(CHUNKS_PATH) as f:
        return [json.loads(line) for line in f]


def build_bm25(chunks):
    tokenized_corpus = [tokenize(c["text"]) for c in chunks]
    return BM25Okapi(tokenized_corpus)


def search(query: str, k: int = 5):
    chunks = load_chunks()
    bm25 = build_bm25(chunks)

    scores = bm25.get_scores(tokenize(query))
    top_indices = scores.argsort()[::-1][:k]  # sort descending, take top k

    results = []
    for idx in top_indices:
        c = chunks[idx]
        results.append({"source_file": c["source_file"], "score": float(scores[idx]), "text": c["text"]})
    return results


if __name__ == "__main__":
    query = "python-multipart"
    print(f"Query: {query}\n")
    for r in search(query, k=5):
        print(f"[{r['score']:.4f}] {r['source_file']}")
        print(" ", r["text"][:150].replace("\n", " "), "...\n")