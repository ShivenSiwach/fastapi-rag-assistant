"""MCP server exposing the FastAPI docs RAG system."""
from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
import logging
import threading

from rag_backend import LocalBackend, RagBackend, RagBackendError

mcp = MCPServer("fastapi-docs-rag")

for _name in ("httpx", "httpx2", "httpcore", "faiss", "google_genai"):
    logging.getLogger(_name).setLevel(logging.WARNING)
logging.getLogger("google_genai.models").setLevel(logging.ERROR)

_backend: RagBackend | None = None
_backend_lock = threading.Lock()


def get_backend() -> RagBackend:
    """Create the backend once (thread-safe), then reuse it."""
    global _backend
    if _backend is None:
        with _backend_lock:
            if _backend is None:
                _backend = LocalBackend()
    return _backend


def _warm_up() -> None:
    """Pay the one-time startup costs before the first real request."""
    try:
        get_backend().retrieve("warm up", k=1)
    except Exception:
        logging.getLogger(__name__).warning("warm-up failed", exc_info=True)


def _require_text(value: str, name: str) -> str:
    value = value.strip()
    if not value:
        raise ToolError(
            f"{name} must not be empty. Provide a question or search terms about FastAPI."
        )
    return value


@mcp.tool()
def search_fastapi_docs(query: str, top_k: int = 5) -> str:
    """Search the official FastAPI documentation (Tutorial and Advanced User Guide).

    Use this whenever the user asks how something works in FastAPI, or needs
    exact details such as parameter names, package names, or configuration.
    It returns numbered excerpts with their source file names. It does NOT
    write an answer: read the excerpts, answer the question yourself from
    them, and mention the source file names you relied on.

    Works with natural-language questions ("How do I upload a file?") and with
    exact terms ("python-multipart", "root_path"). If the excerpts do not
    cover the question, say so rather than guessing.

    Args:
        query: The question or search terms.
        top_k: How many excerpts to return (1 to 10, default 5).
    """
    query = _require_text(query, "query")
    top_k = max(1, min(top_k, 10))

    try:
        hits = get_backend().retrieve(query, k=top_k)
    except RagBackendError as exc:
        raise ToolError(str(exc)) from exc

    parts = [f'Found {len(hits)} excerpts from the FastAPI docs for: "{query}"']
    for rank, hit in enumerate(hits, start=1):
        parts.append(f"[{rank}] source: {hit.source_file}\n{hit.text.strip()}")
    return "\n\n---\n\n".join(parts)


@mcp.tool()
def ask_fastapi_docs(question: str, top_k: int = 5) -> str:
    """Get a finished, cited answer from the FastAPI documentation assistant.

    Use this only when the user explicitly wants the documentation assistant's
    own answer. For most questions prefer search_fastapi_docs: it is faster and
    does not use the answer-generation quota, which is small on the free tier.

    The answer cites source files like [source: request-files.md]. If the
    documentation does not cover the question, the answer says so instead of
    guessing: report that honestly rather than filling the gap yourself.

    Args:
        question: The question to answer.
        top_k: How many excerpts the assistant may read (1 to 10, default 5).
    """
    question = _require_text(question, "question")
    top_k = max(1, min(top_k, 10))

    try:
        result = get_backend().answer(question, k=top_k)
    except RagBackendError as exc:
        raise ToolError(
            f"{exc} If you only need documentation excerpts, "
            "search_fastapi_docs uses less quota and may still work."
        ) from exc

    files = list(dict.fromkeys(s.source_file for s in result.sources))
    return f"{result.text.strip()}\n\n---\nFiles retrieved: {', '.join(files)}"


if __name__ == "__main__":
    threading.Thread(target=_warm_up, daemon=True).start()
    mcp.run()
