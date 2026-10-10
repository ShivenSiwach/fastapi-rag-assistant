import asyncio

from mcp import Client
from server import mcp


async def main():
    async with Client(mcp) as client:
        listed = await client.list_tools()
        for tool in listed.tools:
            print("TOOL:", tool.name)
            print("DESCRIPTION:\n", tool.description)
            print("SCHEMA:", tool.input_schema)

        print("\n===== NORMAL CALL =====")
        result = await client.call_tool(
            "search_fastapi_docs", {"query": "python-multipart", "top_k": 3}
        )
        print("is_error:", result.is_error)
        print(result.content[0].text)

        print("\n===== EMPTY QUERY =====")
        result = await client.call_tool("search_fastapi_docs", {"query": "   "})
        print("is_error:", result.is_error)
        print(result.content[0].text)


asyncio.run(main())