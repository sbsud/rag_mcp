# mcp_server_websearch.py
"""
Web search MCP server — runs independently on port 8001.
No dependency on the RAG pipeline. Completely standalone.

Tools:
  web_search   — search the web via DuckDuckGo (no API key needed)
  web_fetch    — fetch full text of a URL

Run:
  python mcp_server_websearch.py
"""

import json
import logging
import time
import requests
import uvicorn
from starlette.applications import Starlette
from starlette.routing import Mount, Route
from starlette.responses import Response
from mcp.server.sse import SseServerTransport
from mcp.server import Server
from mcp import types
from ddgs import DDGS
from starlette.requests import Request
from starlette.responses import JSONResponse

import config
from logging_config import setup_logging
 
setup_logging()
logger = logging.getLogger(__name__)

server = Server("websearch-server")


@server.list_tools()
async def list_tools() -> list[types.Tool]:
    logger.debug("list_tools called")
    return [
        types.Tool(
            name="web_search",
            description=(
                "Search the web for current, real-time, or public information. "
                "Use this for job listings, recent news, public profiles, live data, "
                "or anything not likely to be in private ingested documents. "
                "Do NOT use for questions about personal documents, resumes, or "
                "private knowledge base content."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Number of results to return (default 5, max 10)",
                    },
                },
                "required": ["query"],
            },
        ),
        types.Tool(
            name="web_fetch",
            description=(
                "Fetch and return the plain text content of a specific public URL. "
                "Use this after web_search when you need the full content of a page, "
                "not just the search snippet. Do NOT use for private or internal URLs."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The full URL to fetch"
                    },
                },
                "required": ["url"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    logger.info("Tool called: %s  args=%s", name, json.dumps(arguments)[:120])
    start = time.perf_counter()

    if name == "web_search":
        query       = arguments["query"]
        max_results = min(arguments.get("max_results", 5), 10)

        with DDGS() as ddgs:
            raw = list(ddgs.text(query, max_results=max_results))

        # Shape results clearly so the LLM can reason over them
        results = [
            {
                "title":   r.get("title", ""),
                "url":     r.get("href", ""),
                "snippet": r.get("body", ""),
            }
            for r in raw
        ]

        return [types.TextContent(
            type="text",
            text=json.dumps({"query": query, "results": results}, indent=2)
        )]

    elif name == "web_fetch":
        url = arguments["url"]
        try:
            resp = requests.get(url, timeout=10, headers={
                "User-Agent": "Mozilla/5.0 (compatible; research-bot/1.0)"
            })
            resp.raise_for_status()

            # Strip HTML tags with a simple approach — no extra dependencies
            import re
            text = re.sub(r"<[^>]+>", " ", resp.text)
            text = re.sub(r"\s+", " ", text).strip()

            return [types.TextContent(
                type="text",
                text=json.dumps({
                    "url":     url,
                    "content": text[:5000]    # cap to avoid flooding context
                }, indent=2)
            )]
        except Exception as e:
            return [types.TextContent(
                type="text",
                text=json.dumps({"url": url, "error": str(e)})
            )]

    else:
        raise ValueError(f"Unknown tool: {name}")

async def health(request: Request) -> JSONResponse:
    from ddgs import DDGS
    query = request.query_params.get("q", "python")
    try:
        raw = list(DDGS().text(query, max_results=2))
        return JSONResponse({
            "status":       "ok" if raw else "empty",
            "result_count": len(raw),
            "first_title":  raw[0]["title"] if raw else None,
        })
    except Exception as e:
        return JSONResponse({"status": "error", "detail": str(e)}, status_code=500)

def make_app() -> Starlette:
    sse_transport = SseServerTransport("/messages")

    async def handle_sse(request):
        client = request.client.host if request.client else "unknown"
        logger.info("SSE connection opened from %s", client)
        try:
            async with sse_transport.connect_sse(
                request.scope, request.receive, request._send
            ) as streams:
                await server.run(
                    streams[0], streams[1],
                    server.create_initialization_options(),
                )
        finally:
            logger.info("SSE connection closed from %s", client) 
        return Response()           
    return Starlette(routes=[
        Route("/sse", endpoint=handle_sse),
        Route("/health",   endpoint=health),  
        Mount("/messages", app=sse_transport.handle_post_message),
    ])


if __name__ == "__main__":
    logger.info("Starting web search MCP server on port %d",
                config.MCP_WEBSEARCH_PORT)
    app = make_app()
    uvicorn.run(app, host="0.0.0.0", port=config.MCP_WEBSEARCH_PORT)