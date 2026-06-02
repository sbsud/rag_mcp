# mcp_server.py
"""
MCP Server over HTTP using SSE (Server-Sent Events) transport.

MCP's HTTP transport works like this:
  - Client sends tool calls as POST to /messages
  - Server streams responses back as SSE on GET /sse
  - Starlette is the HTTP framework; uvicorn is the ASGI runner

Run:
  python mcp_server.py
  # listens on http://localhost:8000
"""

import logging
import json
import uvicorn
from starlette.applications import Starlette
from starlette.routing import Mount, Route
from starlette.responses import Response
from mcp.server.sse import SseServerTransport
from mcp.server import Server
from mcp import types

import rag_pipeline
import retriever
import vectorstore
import embedder
import ingest
import config
from logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


# ── Create the MCP server (same as before) ─────────────
server = Server("resume-local-server")


# ── Tool declarations (identical to stdio version) ──────
@server.list_tools()
async def list_tools() -> list[types.Tool]:
    logger.debug("list_tools called")
    return [
        types.Tool(
            name="resume_query",
            description=(
                "Answer questions about Sudarshan's resume: work experience, job titles, "
                "companies, employment dates, competencies, skills, education, certifications, and projects. "
                "Use this for any question about his professional background and career history. "
                "Do NOT use for current job listings, salary data, or anything not in the resume."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "top_k":    {"type": "integer"},
                },
                "required": ["question"],
            },
        ),
        types.Tool(
            name="resume_search",
            description="Search the knowledge base and return raw chunks without generating an answer.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "top_k": {"type": "integer"},
                },
                "required": ["query"],
            },
        ),
        # types.Tool(
        #     # name="rag_ingest_text",
        #     description="Add raw text to the knowledge base.",
        #     inputSchema={
        #         "type": "object",
        #         "properties": {
        #             "text":   {"type": "string"},
        #             "source": {"type": "string"},
        #         },
        #         "required": ["text", "source"],
        #     },
        # ),
        types.Tool(
            name="resume_list_docs",
            description="List all source documents currently in the knowledge base.",
            inputSchema={"type": "object", "properties": {}},
        ),
        types.Tool(
            name="resume_stats",
            description="Return statistics about the knowledge base.",
            inputSchema={"type": "object", "properties": {}},
        ),
    ]


# ── Tool handlers (identical to stdio version) ──────────
@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    logger.info("Tool called: %s  args=%s", name, json.dumps(arguments)[:120])

    import time
    start = time.perf_counter()

    if name == "resume_query":
        result = rag_pipeline.query(
            question=arguments["question"],
            top_k=arguments.get("top_k"),
        )
        output = {
            "answer":  result["answer"],
            "sources": [
                {"source": c["source"], "score": c["score"], "excerpt": c["text"][:200]}
                for c in result["sources"]
            ],
        }
        return [types.TextContent(type="text", text=json.dumps(output, indent=2))]

    elif name == "resume_search":
        chunks = retriever.retrieve(
            query=arguments["query"],
            top_k=arguments.get("top_k"),
        )
        return [types.TextContent(type="text", text=json.dumps(chunks, indent=2))]

    # elif name == "rag_ingest_text":
    #     text   = arguments["text"]
    #     source = arguments.get("source", "inline")
    #     chunks = ingest.chunk_text(text, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
    #     embeddings = embedder.embed(chunks)
    #     import uuid
    #     ids   = [str(uuid.uuid4()) for _ in chunks]
    #     metas = [{"source": source, "chunk_id": str(i)} for i in range(len(chunks))]
    #     vectorstore.get_store().add(ids=ids, embeddings=embeddings, documents=chunks, metadatas=metas)
    #     return [types.TextContent(
    #         type="text",
    #         text=json.dumps({"status": "ok", "chunks_added": len(chunks), "source": source}),
    #     )]

    elif name == "resume_list_docs":
        docs = vectorstore.get_store().list_docs()
        return [types.TextContent(type="text", text=json.dumps(docs, indent=2))]

    elif name == "resume_stats":
        stats = {
            "total_chunks":   vectorstore.get_store().count(),
            "embed_provider": config.EMBED_PROVIDER,
            "embed_model":    config.EMBED_MODEL,
            "llm_provider":   config.LLM_PROVIDER,
            "llm_model":      config.LLM_MODEL,
            "vectorstore":    config.VECTORSTORE_PROVIDER,
            "chunk_size":     config.CHUNK_SIZE,
            "chunk_overlap":  config.CHUNK_OVERLAP,
            "retrieval_top_k": config.RETRIEVAL_TOP_K,
        }
        return [types.TextContent(type="text", text=json.dumps(stats, indent=2))]

    else:
        raise ValueError(f"Unknown tool: {name}")


# ── Wire up HTTP/SSE transport ──────────────────────────
#
# SseServerTransport manages two HTTP endpoints:
#   GET  /sse       — client connects here to receive streamed responses
#   POST /messages  — client sends tool calls here
#
# The transport handles the MCP protocol framing over these two channels.
# Your tool code above doesn't change at all — only the transport layer does.

def make_app() -> Starlette:
    sse_transport = SseServerTransport("/messages")

    async def handle_sse(request):
        """
        Client opens a persistent GET /sse connection.
        The server streams MCP messages back over this channel.
        """
        client = request.client.host if request.client else "unknown"
        logger.info("SSE connection opened from %s", client)

        try:
            async with sse_transport.connect_sse(
                request.scope, request.receive, request._send
            ) as streams:
                await server.run(
                    streams[0],
                    streams[1],
                    server.create_initialization_options(),
                )
        finally:
            logger.info("SSE connection closed from %s", client)        
    
        return Response()

    return Starlette(
        routes=[
            Route("/sse", endpoint=handle_sse),          # SSE stream
            Mount("/messages", app=sse_transport.handle_post_message),  # tool calls
        ]
    )


if __name__ == "__main__":
    app = make_app()

    logger.info("Starting Local Resume MCP server on port %d", config.MCP_LOCAL_RESUME_HTTP_PORT)
    logger.info("Models: LLM=%s  embed=%s", config.LLM_MODEL, config.EMBED_MODEL)
    logger.info("Vector store: %s chunks in '%s'",
                vectorstore.get_store().count(), config.VECTORSTORE_COLLECTION)

    uvicorn.run(app, host="0.0.0.0", port=config.MCP_LOCAL_RESUME_HTTP_PORT)