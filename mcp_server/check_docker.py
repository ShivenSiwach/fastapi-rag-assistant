import asyncio
import os
import sys

from mcp import Client
from mcp.client.stdio import StdioServerParameters, stdio_client

args = sys.argv[1:] or ["run", "-i", "--rm", "fastapi-docs-mcp"]
# The client's default environment is too small for `docker compose` to find
# its plugin, so pass the full environment through.
params = StdioServerParameters(command="docker", args=args, env=dict(os.environ))


async def main():
    async with Client(stdio_client(params)) as client:
        tools = await client.list_tools()
        print([t.name for t in tools.tools])
        result = await client.call_tool(
            "search_fastapi_docs", {"query": "python-multipart", "top_k": 3}
        )
        print("is_error:", result.is_error)
        print(result.content[0].text[:300])


asyncio.run(main())