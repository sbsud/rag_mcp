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
TOOL_NAME = "complaints-mcp"
server = Server(TOOL_NAME +"-server")

# Tool declarations
@server.list_tools()
async def list_tools() -> list[types.Tool]:
    logger.debug("list_tools called")
    return [
        types.Tool(
            name="complaints_query",
            description=(
                "Search the Amazon Appliances customer complaints corpus to find complaint "
                "patterns, identify brands with high complaint volume, or retrieve sample "
                "review text for a specific rating tier. "
                "Do NOT use this tool for resolution policies, refund rules, or SLA "
                "information — use policy_query for those."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "rating": {
                        "type": ["number", "null"],
                        "description": "Filter to a specific star rating, e.g. 1.0 for 1-star reviews. Use null or omit entirely if no rating filter is needed — do not pass null."
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of complaint samples to retrieve. Use 10-20 for broad pattern analysis across many complaints; use 3-5 for specific brand or narrow questions."
                    }                    
                },
                "required": ["question"]
            }
        )
    ]

MAX_TOP_K = 8
@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    try:
        logger.info("Tool called: %s args:%s", name, json.dumps(arguments)[:120])

        if name == "complaints_query":
            requested_top_k = arguments.get("top_k") or config.RETRIEVAL_TOP_K
            safe_top_k = min(requested_top_k, MAX_TOP_K)
            rating = arguments.get("rating")
            where = {"rating": float(rating)} if rating is not None else None
            result = rag_pipeline.query(
                corpus="complaints",
                question=arguments["question"],
                top_k=safe_top_k,
                where=where,
            )

            output = {
                "answer": result["answer"],
                "sources": [
                    {"source": source["source"], "score": source["score"], "excerpt": source["text"][:150]}    
                    for source in result["sources"]
                ]
            }
            logger.debug("output ********** type %s \nvalue %s", type(output), output)
            return [types.TextContent(type="text", text=json.dumps(output, indent=2))]
    except Exception as e:
        import traceback
        logger.error("call_tool exception: %s\n%s", e, traceback.format_exc())
        return [types.TextContent(type="text", text=str(e))]        
    

if __name__ == "__main__":
    port = int(os.environ["MCP_PORT"])
    app_utils.deploy(server, port, service_name=TOOL_NAME)

