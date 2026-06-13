# mcp_server.py
"""
MCP Server — exposes the RAG pipeline as callable tools.

Tools exposed:
  • rag_query       – Full RAG: retrieve + generate answer
  • rag_search      – Retrieval only (no LLM call)
  • rag_ingest_text – Add raw text directly (no file needed)
  • rag_list_docs   – Show what's in the knowledge base
  • rag_stats       – Show vector store stats

Run:
  python mcp_server.py

The server communicates over stdio (stdin/stdout).
"""

import json
import sys
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types

import rag_pipeline
import retriever
import vectorstore
import embedder
import ingest
import config


# ── Create the MCP server instance ────────────────────
server = Server(config.MCP_SERVER_NAME)


# ── Declare available tools ────────────────────────────
@server.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="rag_query",
            description=(
                "Ask a question. Retrieves relevant chunks from the knowledge base "
                "and uses the local LLM to generate an answer grounded in those chunks."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "The question to answer"},
                    "top_k":    {"type": "integer", "description": "Number of chunks to retrieve (default 5)"},
                },
                "required": ["question"],
            },
        ),
        types.Tool(
            name="rag_search",
            description=(
                "Search the knowledge base and return raw chunks without generating an answer. "
                "Useful when you want to see source passages directly."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "top_k": {"type": "integer", "description": "Number of results (default 5)"},
                },
                "required": ["query"],
            },
        ),
        types.Tool(
            name="rag_ingest_text",
            description="Add raw text to the knowledge base. Useful for ingesting snippets without a file.",
            inputSchema={
                "type": "object",
                "properties": {
                    "text":   {"type": "string",  "description": "Text content to ingest"},
                    "source": {"type": "string",  "description": "Label for this text (e.g. 'meeting_notes')"},
                },
                "required": ["text", "source"],
            },
        ),
        types.Tool(
            name="rag_list_docs",
            description="List all source documents currently in the knowledge base.",
            inputSchema={"type": "object", "properties": {}},
        ),
        types.Tool(
            name="rag_stats",
            description="Return statistics about the knowledge base (chunk count, model config, etc.).",
            inputSchema={"type": "object", "properties": {}},
        ),
    ]


# ── Handle tool calls ──────────────────────────────────
@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:

    if name == "rag_query":
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

    elif name == "rag_search":
        chunks = retriever.retrieve(
            query=arguments["query"],
            top_k=arguments.get("top_k"),
        )
        return [types.TextContent(type="text", text=json.dumps(chunks, indent=2))]

    elif name == "rag_ingest_text":
        text   = arguments["text"]
        source = arguments.get("source", "inline")
        chunks = ingest.chunk_text(text, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
        embeddings = embedder.embed(chunks)
        import uuid
        ids = [str(uuid.uuid4()) for _ in chunks]
        metas = [{"source": source, "chunk_id": str(i)} for i in range(len(chunks))]
        vectorstore.get_store().add(ids=ids, embeddings=embeddings, documents=chunks, metadatas=metas)
        return [types.TextContent(
            type="text",
            text=json.dumps({"status": "ok", "chunks_added": len(chunks), "source": source}),
        )]

    elif name == "rag_list_docs":
        docs = vectorstore.get_store().list_docs()
        return [types.TextContent(type="text", text=json.dumps(docs, indent=2))]

    elif name == "rag_stats":
        stats = {
            "total_chunks":  vectorstore.get_store().count(),
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


# ── Start the server ───────────────────────────────────
async def main():
    async with stdio_server() as streams:
        await server.run(
            streams[0],   # stdin
            streams[1],   # stdout
            server.create_initialization_options(),
        )


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())