# test_multi_server.py
"""
Verify both MCP servers are up and tool routing works.

Run servers first:
  python mcp_server_http.py        # terminal 1 — port 8000
  python mcp_server_websearch.py   # terminal 2 — port 8001

Then:
  python test_multi_server.py      # terminal 3
"""

import asyncio
import json
from mcp import ClientSession
from mcp.client.sse import sse_client

RESUME_URL    = "http://localhost:8000/sse"
SEARCH_URL = "http://localhost:8001/sse"


async def test_server(name: str, url: str, tool_name: str, arguments: dict):
    print(f"\n── {name} ({url}) ───────────────────────────")
    async with sse_client(url) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # List tools
            tools = await session.list_tools()
            print(f"Tools: {[t.name for t in tools.tools]}")

            # Call a tool
            print(f"Calling {tool_name}({arguments}) ...")
            result = await session.call_tool(tool_name, arguments)
            data   = json.loads(result.content[0].text)
            print(f"Result preview: {json.dumps(data)[:300]} ...")


async def main():
    # Test RAG server
    await test_server(
        name      = "RESUME server",
        url       = RESUME_URL,
        tool_name = "resume_stats",
        arguments = {},
    )

    # Test web search server
    await test_server(
        name      = "Web search server",
        url       = SEARCH_URL,
        tool_name = "web_search",
        arguments = {"query": "testing web search", "max_results": 2},
    )

    print("\n✓ Both servers reachable. Tool routing ready.")


asyncio.run(main())