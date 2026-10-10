import asyncio

import pytest
from mcp import Client

import server
from rag_backend import Answer, Chunk, RagBackendError


class CountingBackend:
    def __init__(self):
        self.answer_calls = 0

    def retrieve(self, query, k=5):
        return [Chunk("a.md", 0.03, "alpha")]

    def answer(self, query, k=5):
        self.answer_calls += 1
        return Answer("ok [source: a.md]", [Chunk("a.md", 0.03, "x")])


class FailingBackend:
    def retrieve(self, query, k=5):
        raise RagBackendError("The Gemini API quota is exhausted (free-tier limit). Try again later.")

    def answer(self, query, k=5):
        raise RagBackendError("The Gemini API quota is exhausted (free-tier limit). Try again later.")


def call(tool, args):
    async def _run():
        async with Client(server.mcp) as client:
            return await client.call_tool(tool, args)
    return asyncio.run(_run())


@pytest.fixture
def backend(monkeypatch):
    b = CountingBackend()
    monkeypatch.setattr(server, "_backend", b)
    monkeypatch.setattr(server, "_ask_calls", 0)  # restored after each test
    return b


def test_ask_is_allowed_up_to_the_budget(backend, monkeypatch):
    monkeypatch.setenv("ASK_BUDGET", "2")
    assert not call("ask_fastapi_docs", {"question": "one"}).is_error
    assert not call("ask_fastapi_docs", {"question": "two"}).is_error
    third = call("ask_fastapi_docs", {"question": "three"})
    assert third.is_error
    assert "budget" in third.content[0].text
    assert "search_fastapi_docs" in third.content[0].text
    assert backend.answer_calls == 2  # the third call never reached the backend


def test_search_is_never_limited(backend, monkeypatch):
    monkeypatch.setenv("ASK_BUDGET", "0")
    for _ in range(3):
        assert not call("search_fastapi_docs", {"query": "headers"}).is_error


def test_zero_budget_disables_ask(backend, monkeypatch):
    monkeypatch.setenv("ASK_BUDGET", "0")
    assert call("ask_fastapi_docs", {"question": "anything"}).is_error
    assert backend.answer_calls == 0


def test_empty_question_does_not_use_the_budget(backend, monkeypatch):
    monkeypatch.setenv("ASK_BUDGET", "1")
    assert call("ask_fastapi_docs", {"question": "   "}).is_error  # rejected as empty
    assert not call("ask_fastapi_docs", {"question": "real question"}).is_error


def test_failed_attempts_still_count(monkeypatch):
    monkeypatch.setattr(server, "_backend", FailingBackend())
    monkeypatch.setattr(server, "_ask_calls", 0)
    monkeypatch.setenv("ASK_BUDGET", "1")
    first = call("ask_fastapi_docs", {"question": "one"})
    assert "quota is exhausted" in first.content[0].text  # the backend error
    second = call("ask_fastapi_docs", {"question": "two"})
    assert "budget" in second.content[0].text  # now the budget message


def test_default_budget_is_five(monkeypatch):
    monkeypatch.delenv("ASK_BUDGET", raising=False)
    assert server._ask_budget() == 5


def test_invalid_budget_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("ASK_BUDGET", "many")
    assert server._ask_budget() == 5