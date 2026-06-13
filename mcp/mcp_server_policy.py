import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import logging
import uvicorn
from mcp.server import Server
from mcp import types
from mcp.server.sse import SseServerTransport

from starlette.applications import Starlette
from starlette.routing import Mount, Route
import rag.rag_pipeline as rag_pipeline

import config
import app_utils
from logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

TARGET = "ecommerce"
TOOL_NAME = "policy"
server = Server(TOOL_NAME +"-server")

# Tool declarations
@server.list_tools()
async def list_tools() -> list[types.Tool]:
    logger.debug("list_tools called")
    return [
        types.Tool(
            name="policy_query",
            description="",
            inputSchema={
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                },
                "required": ["question"]
            }
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    logger.info("Tool called: %s args:%s", name, json.dumps(arguments)[:120])

    if name == "policy_query":
        result = rag_pipeline.query(
            corpus="policy",
            question=arguments["question"]
        )

        output = {
            "answer": result["answer"],
            "sources": [
                {"source": source["source"], "score": source["score"], "excerpt": source["text"][:200]}    
                for source in result["sources"]
            ]
        }
        logger.debug("output ********** type %s \nvalue %s", type(output), output)
        return [types.TextContent(type="text", text=json.dumps(output, indent=2))]
    

if __name__ == "__main__":
    port = config.TARGET[TARGET]["mcp_port"][TOOL_NAME]
    app_utils.deploy(server, port)


# def make_app() -> Starlette:
#     sse_transport = SseServerTransport("/messages")
#     async def handle_sse(request):

#         client = request.client.host if request.client else "unknown"
# #       log cient

#         try:
#             async with sse_transport.connect_sse(
#                 request.scope, request.receive, request._send
#             ) as streams:
#                 await server.run(
#                     streams[0],
#                     streams[1],
#                     server.create_initialization_options(),
#                 )
#         finally:
#             logger.info("SSE connection closed from %s", client)

#     return Starlette(
#         routes=[
#             Route("/sse", endpoint=handle_sse),
#             Mount("/messages", app=sse_transport.handle_post_message)
#         ]
#     )

    



# if __name__ == "__main__":
#     app = make_app()
#     uvicorn.run(app, host="0.0.0.0", port=8003)
