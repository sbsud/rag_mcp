from mcp.server.sse import SseServerTransport
from mcp.server import Server
import logging
from starlette.applications import Starlette
from starlette.routing import Mount, Route
from starlette.responses import Response

import uvicorn

from logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

def make_app(server: Server) -> Starlette:
    # server = Server(server_name)
    sse_transport = SseServerTransport("/messages")
    async def handle_sse(request):

        client = request.client.host if request.client else "unknown"

        try:
            async with sse_transport.connect_sse(
                request.scope, request.receive, request._send
            ) as streams:
                await server.run(
                    streams[0],
                    streams[1],
                    server.create_initialization_options(),
                )
            return Response()
        finally:
            logger.info("SSE connection closed from %s", client)

    return Starlette(
        routes=[
            Route("/sse", endpoint=handle_sse),
            Mount("/messages", app=sse_transport.handle_post_message)
        ]
    )

def deploy(server: Server, port: int):
    app = make_app(server)
    uvicorn.run(app, host="0.0.0.0", port=port)