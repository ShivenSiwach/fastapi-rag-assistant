from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from search_hybrid import hybrid_search
from generate import answer as generate_answer

# A plain dict used as simple shared state across requests -- loaded once
# at startup, read by every request afterward.
_state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Runs once when the server starts. Warms up the retriever so the
    # FAISS index and BM25 corpus are already in memory before the first
    # request arrives, instead of rebuilding them on every call.
    print("Loading retriever...")
    from search import load_chunks, build_index, EMBED_PATH
    import numpy as np
    _state["chunks"] = load_chunks()
    _state["ready"] = True
    print(f"Ready. {len(_state['chunks'])} chunks loaded.")
    yield
    _state.clear()


app = FastAPI(
    title="FastAPI Docs RAG Assistant",
    description="Hybrid (BM25 + vector) retrieval-augmented Q&A over FastAPI's documentation.",
    version="1.0.0",
    lifespan=lifespan,
)


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, examples=["How do I upload a file in FastAPI?"])
    k: int = Field(5, ge=1, le=10)


class Source(BaseModel):
    source_file: str
    score: float


class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]


@app.get("/health")
def health():
    return {"status": "ok", "ready": _state.get("ready", False)}


@app.get("/retrieve")
def retrieve(question: str, k: int = 5):
    hits = hybrid_search(question, k=k)
    return {"hits": hits}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    text, hits = generate_answer(req.question, k=req.k)
    if not hits:
        raise HTTPException(status_code=404, detail="No relevant context found.")
    return QueryResponse(
        answer=text,
        sources=[Source(source_file=h["source_file"], score=h["score"]) for h in hits],
    )