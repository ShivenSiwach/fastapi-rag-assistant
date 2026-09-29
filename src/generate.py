import os
from dotenv import load_dotenv
from google import genai

from search_hybrid import hybrid_search

load_dotenv()

GEN_MODEL = "gemini-3.5-flash-lite"

SYSTEM_PROMPT = """You are a documentation assistant for FastAPI. Answer the \
user's question using ONLY the provided context chunks below. \
Cite the source file for every claim like [source: request-files.md]. \
If the context does not contain the answer, say so plainly instead of guessing."""


def build_context(hits):
    blocks = []
    for h in hits:
        blocks.append(f"[source: {h['source_file']}]\n{h['text']}")
    return "\n\n---\n\n".join(blocks)


def answer(query: str, k: int = 5):
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

    hits = hybrid_search(query, k=k)
    context = build_context(hits)
    prompt = f"{SYSTEM_PROMPT}\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:"

    resp = client.models.generate_content(model=GEN_MODEL, contents=prompt)
    return resp.text, hits


if __name__ == "__main__":
    query = "How do I deploy FastAPI to Kubernetes?"
    print(f"Query: {query}\n")
    response_text, hits = answer(query)
    print("=== Answer ===")
    print(response_text)
    print("\n=== Sources used ===")
    for h in hits:
        print(" -", h["source_file"])