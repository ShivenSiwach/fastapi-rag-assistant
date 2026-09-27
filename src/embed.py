import json
import os
import time
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from google import genai
 
load_dotenv()  # reads GEMINI_API_KEY from your .env file into the environment
 
CHUNKS_PATH = Path(__file__).parent.parent / "data" / "processed" / "chunks.jsonl"
EMBED_PATH = Path(__file__).parent.parent / "data" / "processed" / "embeddings.npy"
MODEL = "gemini-embedding-001"
EMBED_DIM = 768
BATCH_SIZE = 20
SECONDS_BETWEEN_BATCHES = 13  # keeps us under the free tier's 100 requests/minute cap
 
 
def load_chunks():
    with open(CHUNKS_PATH) as f:
        return [json.loads(line) for line in f]
 
 
def embed_batch(client, texts, retries=3):
    for attempt in range(retries):
        try:
            resp = client.models.embed_content(
                model=MODEL,
                contents=texts,
                config={"output_dimensionality": EMBED_DIM, "task_type": "RETRIEVAL_DOCUMENT"},
            )
            return [e.values for e in resp.embeddings]
        except Exception as e:
            if attempt == retries - 1:
                raise
            print(f"  retry {attempt + 1}: {e}")
            time.sleep(2 ** attempt)
 
 
def main():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("GEMINI_API_KEY not found. Check your .env file.")
 
    client = genai.Client(api_key=api_key)
    chunks = load_chunks()
    texts = [c["text"] for c in chunks]
    total_batches = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE
    est_minutes = round(total_batches * SECONDS_BETWEEN_BATCHES / 60, 1)
    print(f"Embedding {len(texts)} chunks with {MODEL}...")
    print(f"{total_batches} batches, ~{est_minutes} min (paced to respect free-tier rate limits)")
 
    all_vectors = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        vectors = embed_batch(client, batch)
        all_vectors.extend(vectors)
        print(f"  {min(i + BATCH_SIZE, len(texts))}/{len(texts)} done")
 
        # Save progress after every batch -- if a later batch fails, you
        # don't lose the API calls you already spent on earlier ones.
        partial = np.array(all_vectors, dtype=np.float32)
        np.save(EMBED_PATH, partial)
 
        time.sleep(SECONDS_BETWEEN_BATCHES)
 
    # Normalize each vector to length 1 (once, at the end, on the full
    # matrix) so a simple dot product at search time equals cosine
    # similarity -- simpler and faster for FAISS than raw cosine math.
    matrix = np.load(EMBED_PATH)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix = matrix / norms
    np.save(EMBED_PATH, matrix)
    print(f"Saved {matrix.shape} matrix -> {EMBED_PATH}")
 
 
if __name__ == "__main__":
    main()
 

