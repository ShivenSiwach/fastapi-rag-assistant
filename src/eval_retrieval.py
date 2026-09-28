import json
import os
import time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from google import genai

from search import load_chunks, embed_query, build_index, EMBED_PATH
from search_bm25 import build_bm25, tokenize

load_dotenv()

EVAL_PATH = Path(__file__).parent.parent / "data" / "eval_set.json"
RESULTS_PATH = Path(__file__).parent.parent / "eval_results.json"
K = 5          # how many top chunks we judge
POOL = 20      # how many candidates each retriever contributes before fusing
RRF_K = 60
STRATEGIES = ["bm25", "vector", "hybrid"]


def rrf_fuse(rankings, rrf_k=RRF_K):
    """Reciprocal Rank Fusion over several ranked lists of chunk indices."""
    scores = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking):
            scores[int(idx)] = scores.get(int(idx), 0) + 1 / (rrf_k + rank + 1)
    return [idx for idx, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]


def score_ranking(ranked_indices, chunks, expected, k=K):
    """Return (hit, reciprocal_rank) for one ranked list of chunk indices."""
    for pos, idx in enumerate(ranked_indices[:k], start=1):
        if chunks[int(idx)]["source_file"] in expected:
            return 1, 1 / pos
    return 0, 0.0


def summarize(rows):
    out = {}
    for s in STRATEGIES:
        out[s] = {
            f"recall@{K}": round(sum(r[s]["hit"] for r in rows) / len(rows), 3),
            "mrr": round(sum(r[s]["rr"] for r in rows) / len(rows), 3),
        }
    return out


def print_table(title, summary, n):
    print(f"\n=== {title} (n={n}) ===")
    print(f"{'strategy':<10}{'recall@' + str(K):<12}{'MRR':<8}")
    for s in STRATEGIES:
        print(f"{s:<10}{summary[s][f'recall@{K}']:<12}{summary[s]['mrr']:<8}")


def main():
    chunks = load_chunks()
    corpus_files = {c["source_file"] for c in chunks}
    eval_set = json.loads(EVAL_PATH.read_text(encoding="utf-8"))

    # Guard: bad ground truth would silently produce meaningless numbers.
    missing = sorted({f for q in eval_set for f in q["expected_sources"] if f not in corpus_files})
    if missing:
        print("These expected files are not in your corpus. Fix data/eval_set.json:")
        for f in missing:
            print("  -", f)
        raise SystemExit("Aborting: ground truth doesn't match the corpus.")

    bm25 = build_bm25(chunks)
    index = build_index(np.load(EMBED_PATH))
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

    rows = []
    for item in eval_set:
        q, expected = item["question"], set(item["expected_sources"])

        bm25_rank = bm25.get_scores(tokenize(q)).argsort()[::-1][:POOL]

        qvec = embed_query(client, q).reshape(1, -1)  # embedded once, shared by vector + hybrid
        _, vec_idx = index.search(qvec, POOL)
        vec_rank = vec_idx[0]

        rankings = {
            "bm25": bm25_rank,
            "vector": vec_rank,
            "hybrid": rrf_fuse([bm25_rank, vec_rank]),
        }

        row = {"id": item["id"], "type": item["type"], "question": q}
        for name, ranking in rankings.items():
            hit, rr = score_ranking(ranking, chunks, expected)
            row[name] = {"hit": hit, "rr": rr}
        rows.append(row)
        print(f"  evaluated {item['id']}")
        time.sleep(0.7)  # stay well under the free-tier rate limit

    print_table("Overall", summarize(rows), len(rows))
    for qtype in ["natural", "keyword", "hard"]:
        subset = [r for r in rows if r["type"] == qtype]
        print_table(f"{qtype} queries", summarize(subset), len(subset))

    print("\n=== Misses ===")
    for s in STRATEGIES:
        missed = [r["question"] for r in rows if r[s]["hit"] == 0]
        print(f"\n{s}: {len(missed)} missed")
        for m in missed:
            print("  -", m)

    RESULTS_PATH.write_text(json.dumps({"per_question": rows}, indent=2), encoding="utf-8")
    print(f"\nSaved detailed results -> {RESULTS_PATH}")


if __name__ == "__main__":
    main()