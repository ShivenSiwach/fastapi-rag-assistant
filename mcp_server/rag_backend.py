"""Backend adapter: the only place the MCP server touches the RAG code."""
from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Protocol

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"


class RagBackendError(Exception):
    """A failure whose message is safe and useful to show to the AI model."""


@dataclass
class Chunk:
    """One retrieved piece of documentation."""
    source_file: str
    score: float
    text: str


@dataclass
class Answer:
    """A generated answer plus the chunks it was based on."""
    text: str
    sources: list[Chunk]


class RagBackend(Protocol):
    """What any backend must offer. MCP tools depend only on this."""

    def retrieve(self, query: str, k: int = 5) -> list[Chunk]: ...

    def answer(self, query: str, k: int = 5) -> Answer: ...


def _to_chunk(hit: dict) -> Chunk:
    return Chunk(
        source_file=hit["source_file"],
        score=float(hit["score"]),
        text=hit["text"],
    )


@contextmanager
def _translate_gemini_errors() -> Iterator[None]:
    """Turn Gemini API failures into RagBackendError with a clear message."""
    try:
        yield
    except Exception as exc:
        from google.genai import errors

        if not isinstance(exc, errors.APIError):
            raise
        code = getattr(exc, "code", None)
        if code == 429:
            msg = "The Gemini API quota is exhausted (free-tier limit). Try again later."
        else:
            msg = (
                f"The Gemini API returned an error (code {code}). "
                "This is usually temporary; try again in a moment."
            )
        raise RagBackendError(msg) from exc


class LocalBackend:
    """Calls the existing RAG modules directly, in this same process."""

    def __init__(self) -> None:
        os.chdir(PROJECT_ROOT)
        if str(SRC_DIR) not in sys.path:
            sys.path.insert(0, str(SRC_DIR))

        from dotenv import load_dotenv

        load_dotenv(PROJECT_ROOT / ".env")
        if not os.environ.get("GEMINI_API_KEY"):
            raise RagBackendError(
                "GEMINI_API_KEY is not set. Add it to the project's .env file "
                "or to the MCP client's env settings."
            )

        import generate
        import search_hybrid

        self._hybrid_search = search_hybrid.hybrid_search
        self._generate_answer = generate.answer

    def retrieve(self, query: str, k: int = 5) -> list[Chunk]:
        with _translate_gemini_errors():
            hits = self._hybrid_search(query, k=k)
        return [_to_chunk(h) for h in hits]

    def answer(self, query: str, k: int = 5) -> Answer:
        with _translate_gemini_errors():
            text, hits = self._generate_answer(query, k=k)
        return Answer(text=text, sources=[_to_chunk(h) for h in hits])