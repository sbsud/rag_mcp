import asyncio, json, os
from mcp.server import Server
from mcp.server.sse import SseServerTransport
from mcp import types
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.requests import Request
from starlette.responses import JSONResponse
import uvicorn
import logging
from logging_config import setup_logging
from email_client import send_email, check_credentials

setup_logging()
logger = logging.getLogger(__name__)

# ── In-memory draft store (keyed by draft_id) ─────────────
_drafts: dict[str, dict] = {}
_draft_counter = 0

server = Server("email_server")

@server.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="draft_email",
            description=(
                "Create a draft email to a recipient. Returns a draft_id for review. "
                "Use this when the user wants to compose an email — job application, "
                "outreach, or follow-up. Do NOT send automatically; always draft first "
                "and let the user confirm before calling send_email."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "to":      {"type": "string", "description": "Recipient email address"},
                    "subject": {"type": "string", "description": "Email subject line"},
                    "body":    {"type": "string", "description": "Full email body text"},
                },
                "required": ["to", "subject", "body"],
            },
        ),
        types.Tool(
            name="send_email",
            description=(
                "Send a previously drafted email. Requires a draft_id from draft_email. "
                "ONLY call this after the user has explicitly confirmed they want to send. "
                "Do NOT call this without a draft_id or without user confirmation."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "draft_id": {"type": "string", "description": "ID returned by draft_email"},
                },
                "required": ["draft_id"],
            },
        ),
        types.Tool(
            name="list_sent",
            description=(
                "List recently sent emails (subject, recipient, timestamp). "
                "Use to check whether an email to a specific recipient was already sent."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "description": "Max number of entries to return", "default": 5},
                },
            },
        ),
    ]

# Sent log — in-memory for now; Part 4 can persist this to ChromaDB or SQLite
_sent_log: list[dict] = []

@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    global _draft_counter

    if name == "draft_email":
        _draft_counter += 1
        draft_id = f"draft_{_draft_counter}"
        _drafts[draft_id] = {
            "to":      arguments["to"],
            "subject": arguments["subject"],
            "body":    arguments["body"],
        }
        preview = (
            f"DRAFT {draft_id}\n"
            f"To: {arguments['to']}\n"
            f"Subject: {arguments['subject']}\n"
            f"---\n{arguments['body']}"
        )
        logger.info("Draft created: %s → %s", draft_id, arguments["to"])
        return [types.TextContent(type="text", text=preview)]

    elif name == "send_email":
        draft_id = arguments["draft_id"]
        if draft_id not in _drafts:
            return [types.TextContent(type="text", text=f"Error: draft_id '{draft_id}' not found.")]
        draft = _drafts.pop(draft_id)
        result = send_email(draft["to"], draft["subject"], draft["body"])
        import datetime
        _sent_log.append({**result, "timestamp": datetime.datetime.now().isoformat()})
        return [types.TextContent(type="text", text=json.dumps(result))]

    elif name == "list_sent":
        limit = arguments.get("limit", 5)
        return [types.TextContent(type="text", text=json.dumps(_sent_log[-limit:]))]

    return [types.TextContent(type="text", text=f"Unknown tool: {name}")]


# ── Starlette app (mirrors mcp_server_websearch.py) ───────
sse_transport = SseServerTransport("/messages")

async def handle_sse(request: Request):
    async with sse_transport.connect_sse(request.scope, request.receive, request._send) as streams:
        await server.run(streams[0], streams[1], server.create_initialization_options())

async def handle_messages(request: Request):
    await sse_transport.handle_post_message(request.scope, request.receive, request._send)

async def health(request: Request):
    ok = check_credentials()
    return JSONResponse({"status": "ok" if ok else "smtp_error", "smtp_auth": ok})

app = Starlette(routes=[
    Route("/sse",      handle_sse),
    Route("/messages", handle_messages, methods=["POST"]),
    Route("/health",   health),
])

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)