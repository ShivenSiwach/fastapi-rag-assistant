import time

from rag_backend import LocalBackend

t0 = time.perf_counter()
backend = LocalBackend()
print(f"init: {time.perf_counter() - t0:.2f}s")

queries = [
    "python-multipart",
    "python-multipart",  # repeated on purpose, to compare call 1 vs call 2
    "How do I upload a file in FastAPI?",
]

for q in queries:
    t = time.perf_counter()
    hits = backend.retrieve(q, k=5)
    elapsed = time.perf_counter() - t
    print(f"\nQuery: {q}  ({elapsed:.2f}s)")
    for h in hits:
        print(f"  [{h.score}] {h.source_file}")