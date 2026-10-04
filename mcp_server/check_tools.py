import asyncio

from mcp import Client

import server
from rag_backend import RagBackendError


def show(title, result, limit=500):
    print(f"\n===== {title} =====")
    print("is_error:", result.is_error)
    print(result.content[0].text[:limit])


class QuotaBackend:
    """A fake backend that always fails the way an exhausted quota does."""

    def retrieve(self, query, k=5):
        raise RagBackendError("The Gemini API quota is exhausted (free-tier limit). Try again later.")

    def answer(self, query, k=5):
        raise RagBackendError("The Gemini API quota is exhausted (free-tier limit). Try again later.")


async def main():
    async with Client(server.mcp) as client:
        listed = await client.list_tools()
        print("Tools:", [t.name for t in listed.tools])

        show("EMPTY QUERY", await client.call_tool("search_fastapi_docs", {"query": "   "}))
        show(
            "ASK: answerable",
            await client.call_tool("ask_fastapi_docs", {"question": "How do I upload a file in FastAPI?"}),
            1500,
        )
        show(
            "ASK: not in the docs",
            await client.call_tool("ask_fastapi_docs", {"question": "How do I deploy FastAPI to Kubernetes?"}),
            1500,
        )

        # Swap in the fake backend: no real API calls, no quota used.
        server._backend = QuotaBackend()
        show("SIMULATED QUOTA ERROR (search)", await client.call_tool("search_fastapi_docs", {"query": "anything"}))
        show("SIMULATED QUOTA ERROR (ask)", await client.call_tool("ask_fastapi_docs", {"question": "anything"}))


asyncio.run(main())