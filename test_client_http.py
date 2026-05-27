# test_client.py
"""
MCP test client using HTTP/SSE transport.

Instead of launching a subprocess (stdio), this client connects to
a running MCP server over HTTP. Run the server first:
  python mcp_server.py

Then in another terminal:
  python test_client.py
"""

import asyncio
import json
from mcp import ClientSession
from mcp.client.sse import sse_client

SERVER_URL = "http://localhost:8000/sse"


async def main():
    print(f"Connecting to MCP server at {SERVER_URL} ...\n")

    # sse_client opens a persistent GET /sse connection and returns
    # (read_stream, write_stream) — the same interface as stdio_client.
    # Everything below this line is identical to the stdio client.
    async with sse_client(SERVER_URL) as (read, write):
        async with ClientSession(read, write) as session:

            await session.initialize()
            print("Connected.\n")

            # ── List available tools ────────────────────
            tools_response = await session.list_tools()
            print("Available tools:")
            for tool in tools_response.tools:
                print(f"  • {tool.name}: {tool.description}")
            print()

            # ── rag_stats ───────────────────────────────
            print("── rag_stats ──────────────────────────────")
            result = await session.call_tool("rag_stats", {})
            stats = json.loads(result.content[0].text)
            print(json.dumps(stats, indent=2))
            print()

            # ── rag_search (retrieval only, no LLM) ─────
            print("── rag_search ─────────────────────────────")
            result = await session.call_tool("rag_search", {
                "query": "developer experience",
                "top_k": 2,
            })
            chunks = json.loads(result.content[0].text)
            for i, chunk in enumerate(chunks, 1):
                print(f"  [{i}] score={chunk['score']}  source={chunk['source']}")
                print(f"       {chunk['text'][:120]} ...")
            print()

            # ── rag_query (full RAG: retrieve + LLM) ────
            print("── rag_query ──────────────────────────────")
            result = await session.call_tool("rag_query", {
                "question": "has he worked in on apache kafka? if so in which company?",
            })
            data = json.loads(result.content[0].text)
            print("Answer:", data["answer"])
            print()
            print("Sources used:")
            for src in data["sources"]:
                print(f"  • {src['source']} (score={src['score']})")


asyncio.run(main())