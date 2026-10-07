"""HTTP backend: calls the running FastAPI RAG service instead of importing its code."""
from __future__ import annotations

import httpx

from rag_backend import Answer, Chunk, RagBackendError

DEFAULT_URL = "http://127.0.0.1:8000"
# Fail fast if the service is down, but leave time for Gemini calls.
TIMEOUT = httpx.Timeout(60.0, connect=3.0)


def _status_message(code: int) -> str:
    if code == 429:
        return "The Gemini API quota is exhausted (free-tier limit). Try again later."
    if code == 422:
        return ("The documentation service rejected the request as invalid "
                "(for example, a question shorter than 3 characters).")
    if code == 404:
        return "The documentation service found no relevant context for this question."
    if code >= 500:
        return (f"The documentation service reported an internal error (HTTP {code}). "
                "This may be a Gemini quota limit or a temporary fault; try again later.")
    return f"The documentation service returned an unexpected status (HTTP {code})."


class HttpBackend:
    """Talks to the FastAPI service's /retrieve and /query endpoints."""

    def __init__(self, base_url: str = DEFAULT_URL,
                 transport: httpx.BaseTransport | None = None) -> None:
        self._base_url = base_url.rstrip("/")
        verify = self._base_url.startswith("https://")
        self._http = httpx.Client(
            base_url=self._base_url, timeout=TIMEOUT, transport=transport, verify=verify)

    def _request(self, method: str, path: str, **kwargs):
        try:
            resp = self._http.request(method, path, **kwargs)
        except httpx.TimeoutException as exc:
            raise RagBackendError(
                "The documentation service took too long to respond. Try again later."
            ) from exc
        except httpx.ConnectError as exc:
            raise RagBackendError(
                f"The documentation service at {self._base_url} is not reachable. "
                "It may not be running."
            ) from exc
        except httpx.HTTPError as exc:
            raise RagBackendError(
                "A network error occurred while contacting the documentation service."
            ) from exc

        if resp.status_code != 200:
            raise RagBackendError(_status_message(resp.status_code))
        try:
            return resp.json()
        except ValueError as exc:
            raise RagBackendError(
                "The documentation service returned a response that could not be read."
            ) from exc

    def retrieve(self, query: str, k: int = 5) -> list[Chunk]:
        data = self._request("GET", "/retrieve", params={"question": query, "k": k})
        try:
            return [Chunk(h["source_file"], float(h["score"]), h["text"])
                    for h in data["hits"]]
        except (KeyError, TypeError, ValueError) as exc:
            raise RagBackendError(
                "The documentation service returned an unexpected response format."
            ) from exc

    def answer(self, query: str, k: int = 5) -> Answer:
        data = self._request("POST", "/query", json={"question": query, "k": k})
        try:
            # /query returns file names and scores but no excerpt text.
            sources = [Chunk(s["source_file"], float(s["score"]), "")
                       for s in data["sources"]]
            return Answer(text=data["answer"], sources=sources)
        except (KeyError, TypeError, ValueError) as exc:
            raise RagBackendError(
                "The documentation service returned an unexpected response format."
            ) from exc

    def close(self) -> None:
        self._http.close()