import asyncio
import os
import sys
from pathlib import Path

import pytest
from mcp import Client

import server
from rag_backend import Answer, Chunk, RagBackendError

MCP_DIR = Path(__file__).resolve().parent.parent / "mcp_server"


class FakeBackend:
    """Returns fixed results and remembers what it was asked."""

    def __init__(self):
        self.last_k = None

    def retrieve(self, query, k=5):
        self.last_k = k
        return [Chunk("a.md", 0.03, "alpha text"), Chunk("b.md", 0.02, "beta text")]

    def answer(self, query, k=5):
        self.last_k = k
        return Answer(
            "The answer [source: a.md].",
            [Chunk("a.md", 0.03, "x"), Chunk("a.md", 0.02, "y"), Chunk("b.md", 0.01, "z")],
        )


class BrokenBackend:
    def retrieve(self, query, k=5):
        raise RagBackendError("The Gemini API quota is exhausted (free-tier limit). Try again later.")

    def answer(self, query, k=5):
        raise RagBackendError("The Gemini API quota is exhausted (free-tier limit). Try again later.")


def call(tool, args):
    async def _run():
        async with Client(server.mcp) as client:
            return await client.call_tool(tool, args)

    return asyncio.run(_run())


def list_tools():
    async def _run():
        async with Client(server.mcp) as client:
            return (await client.list_tools()).tools

    return asyncio.run(_run())


@pytest.fixture
def fake(monkeypatch):
    backend = FakeBackend()
    monkeypatch.setattr(server, "_backend", backend)
    return backend


def test_lists_both_tools():
    names = {t.name for t in list_tools()}
    assert names == {"search_fastapi_docs", "ask_fastapi_docs"}


def _flat(text):
    """Collapse line breaks and indentation so wording checks ignore layout."""
    return " ".join(text.split())


def test_descriptions_steer_the_model():
    desc = {t.name: _flat(t.description) for t in list_tools()}
    assert "does NOT write an answer" in desc["search_fastapi_docs"]
    assert "prefer search_fastapi_docs" in desc["ask_fastapi_docs"]


def test_search_formats_numbered_excerpts(fake):
    result = call("search_fastapi_docs", {"query": "headers"})
    text = result.content[0].text
    assert not result.is_error
    assert "[1] source: a.md" in text
    assert "[2] source: b.md" in text
    assert "0.03" not in text  # raw RRF scores are not shown to the model


def test_top_k_is_clamped(fake):
    call("search_fastapi_docs", {"query": "x", "top_k": 500})
    assert fake.last_k == 10
    call("search_fastapi_docs", {"query": "x", "top_k": 0})
    assert fake.last_k == 1


def test_empty_query_is_a_readable_error(fake):
    result = call("search_fastapi_docs", {"query": "   "})
    assert result.is_error
    assert "must not be empty" in result.content[0].text


def test_backend_error_reaches_the_model(monkeypatch):
    monkeypatch.setattr(server, "_backend", BrokenBackend())
    result = call("search_fastapi_docs", {"query": "anything"})
    assert result.is_error
    assert "quota is exhausted" in result.content[0].text


def test_ask_error_points_to_search(monkeypatch):
    monkeypatch.setattr(server, "_backend", BrokenBackend())
    result = call("ask_fastapi_docs", {"question": "anything"})
    assert result.is_error
    assert "search_fastapi_docs" in result.content[0].text


def test_ask_lists_each_file_once(fake):
    result = call("ask_fastapi_docs", {"question": "anything"})
    assert "Files retrieved: a.md, b.md" in result.content[0].text


integration = pytest.mark.skipif(
    os.environ.get("RUN_INTEGRATION") != "1",
    reason="set RUN_INTEGRATION=1 to run (makes real Gemini calls)",
)


@integration
def test_stdio_end_to_end_from_another_folder(tmp_path):
    from mcp.client.stdio import StdioServerParameters, stdio_client

    params = StdioServerParameters(
        command=sys.executable,
        args=[str(MCP_DIR / "server.py")],
        cwd=str(tmp_path),
    )

    async def _run():
        async with Client(stdio_client(params)) as client:
            tools = await client.list_tools()
            result = await client.call_tool(
                "search_fastapi_docs", {"query": "python-multipart", "top_k": 3}
            )
            return [t.name for t in tools.tools], result

    names, result = asyncio.run(_run())
    assert "search_fastapi_docs" in names
    assert not result.is_error
    assert "request-files.md" in result.content[0].text