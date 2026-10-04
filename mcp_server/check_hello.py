import asyncio

from mcp import Client
from hello_server import mcp


async def main():
    async with Client(mcp) as client:
        listed = await client.list_tools()
        for tool in listed.tools:
            print("---- TOOL ----")
            print(tool.model_dump_json(indent=2))

        print("---- CALL add(2, 3) ----")
        result = await client.call_tool("add", {"a": 2, "b": 3})
        print(result)


asyncio.run(main())