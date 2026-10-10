import asyncio

import httpx
import pytest
from mcp import Client

import server
from http_backend import HttpBackend
from rag_backend import RagBackendError


def backend_with(handler):
    """An HttpBackend whose 'service' is the given function. No network involved."""
    return HttpBackend(transport=httpx.MockTransport(handler))


def reply(status=200, **body):
    return httpx.Response(status, json=body)


def test_retrieve_sends_query_params_and_parses_hits():
    seen = {}

    def handler(request):
        seen["path"] = request.url.path
        seen["params"] = dict(request.url.params)
        return reply(hits=[{"source_file": "a.md", "score": 0.03, "text": "alpha"}])

    chunks = backend_with(handler).retrieve("headers", k=3)
    assert seen["path"] == "/retrieve"
    assert seen["params"] == {"question": "headers", "k": "3"}
    assert (chunks[0].source_file, chunks[0].text) == ("a.md", "alpha")


def test_special_characters_survive_the_url():
    seen = {}

    def handler(request):
        seen["question"] = request.url.params["question"]
        return reply(hits=[])

    backend_with(handler).retrieve("a&b=c d?", k=1)
    assert seen["question"] == "a&b=c d?"


def test_answer_posts_json_and_has_no_excerpt_text():
    seen = {}

    def handler(request):
        seen["method"] = request.method
        seen["body"] = request.content
        return reply(answer="Use UploadFile [source: a.md].",
                     sources=[{"source_file": "a.md", "score": 0.03}])

    result = backend_with(handler).answer("upload a file", k=2)
    assert seen["method"] == "POST"
    assert b'"question"' in seen["body"] and b'"k"' in seen["body"]
    assert result.text.startswith("Use UploadFile")
    assert result.sources[0].source_file == "a.md"
    assert result.sources[0].text == ""  # /query returns file names, not excerpts


def _raising(exc_type, message):
    def handler(request):
        raise exc_type(message, request=request)
    return handler


def test_service_down_is_a_readable_error():
    backend = backend_with(_raising(httpx.ConnectError, "refused"))
    with pytest.raises(RagBackendError, match="not reachable"):
        backend.retrieve("x")


def test_timeout_is_a_readable_error():
    backend = backend_with(_raising(httpx.ReadTimeout, "slow"))
    with pytest.raises(RagBackendError, match="too long"):
        backend.retrieve("x")


@pytest.mark.parametrize("status, expected", [
    (429, "quota is exhausted"),
    (422, "rejected the request"),
    (404, "no relevant context"),
    (500, "internal error"),
    (418, "unexpected status"),
])
def test_error_statuses_become_readable_messages(status, expected):
    backend = backend_with(lambda request: httpx.Response(status, text="raw body"))
    with pytest.raises(RagBackendError, match=expected) as info:
        backend.answer("x")
    assert "raw body" not in str(info.value)  # service output never reaches the model


def test_unreadable_json_is_a_readable_error():
    backend = backend_with(lambda request: httpx.Response(200, text="<html>oops</html>"))
    with pytest.raises(RagBackendError, match="could not be read"):
        backend.retrieve("x")


def test_unexpected_shape_is_a_readable_error():
    backend = backend_with(lambda request: reply(something_else=[]))
    with pytest.raises(RagBackendError, match="unexpected response format"):
        backend.retrieve("x")


# --- the RAG_BACKEND switch and the full tool path ---

def call(tool, args):
    async def _run():
        async with Client(server.mcp) as client:
            return await client.call_tool(tool, args)
    return asyncio.run(_run())


def test_switch_picks_http(monkeypatch):
    monkeypatch.setattr(server, "_backend", None)
    monkeypatch.setenv("RAG_BACKEND", "http")
    assert isinstance(server.get_backend(), HttpBackend)


def test_switch_defaults_to_local(monkeypatch):
    monkeypatch.setattr(server, "_backend", None)
    monkeypatch.setattr(server, "LocalBackend", lambda: "local-stub")
    monkeypatch.delenv("RAG_BACKEND", raising=False)
    assert server.get_backend() == "local-stub"


def test_switch_rejects_unknown_value(monkeypatch):
    monkeypatch.setattr(server, "_backend", None)
    monkeypatch.setenv("RAG_BACKEND", "bogus")
    with pytest.raises(RagBackendError, match="Unknown RAG_BACKEND"):
        server.get_backend()


def test_http_quota_error_reaches_the_model(monkeypatch):
    backend = backend_with(lambda request: httpx.Response(429, text="boom"))
    monkeypatch.setattr(server, "_backend", backend)
    result = call("search_fastapi_docs", {"query": "anything"})
    assert result.is_error
    assert "quota is exhausted" in result.content[0].text