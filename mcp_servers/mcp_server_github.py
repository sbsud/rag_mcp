import sys
import os


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
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
from consul_utils import get_kv
# from config import GITHUB_REPO
# from config import GITHUB_REPO_OWNER



setup_logging()
logger = logging.getLogger(__name__)

TARGET = "ecommerce"
TOOL_NAME = "github-mcp"
server = Server(TOOL_NAME +"-server")
# Tool declarations
@server.list_tools()
async def list_tools() -> list[types.Tool]:
    logger.debug("list_tools called")
    return [
        types.Tool(
            name="create_github_issue_action",
            description=(
                "File a GitHub issue summarising a customer complaint pattern. "
                "Call this AT MOST ONCE per investigation — after it succeeds, the task "
                "is complete; do not call it again. "
                "Only call this after complaints_query and policy_query have both been "
                "called in this session. Use the ACTUAL complaint pattern and policy "
                "clause text from those tool results."
                ),
            inputSchema={
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "A specific issue title naming the brand(s) and complaint type, e.g. 'GE Appliances — defective product complaints'."},
                    "complaint_pattern": {"type": "string", "description": "The specific complaint pattern found by complaints_query, including brand names and complaint themes — copy the actual finding, not a generic label."},
                    "source_category": {"type": "string", "description": "The product category these complaints came from, e.g. 'Appliances'."},
                    "policy_clause": {"type": "string", 
                                      "description": (
                                          "The exact policy clause text retrieved by policy_query that applies "
                                            "to this complaint pattern. Copy it verbatim or closely paraphrase it — "
                                            "do not invent a plausible-sounding clause. If policy_query found no "
                                            "applicable clause, set this field to 'No matching policy clause found' "
                                            "rather than fabricating one."
                                            )},
                },
                "required": ["title", "complaint_pattern", "source_category","policy_clause"]
            }
        )
    ]

NO_POLICY_MARKERS = ["no matching policy", "not clearly stated", "not found", "no applicable"]
PLACEHOLDER_MARKERS = ["brand x", "top complaint pattern found", "to be determined", "tbd"]



@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    logger.info("Tool called: %s args:%s", name, json.dumps(arguments)[:120])

    if name == "create_github_issue_action":
        GITHUB_REPO = get_kv(key="rag_mcp/ecommerce/github/repo")
        GITHUB_REPO_OWNER = get_kv(key="rag_mcp/ecommerce/github/owner")
        pattern = arguments.get("complaint_pattern", "").lower()
        clause = arguments.get("policy_clause", "").lower()
        if any(m in pattern for m in PLACEHOLDER_MARKERS) or any(m in clause for m in PLACEHOLDER_MARKERS):
            logger.warning("Rejected create_github_issue — placeholder content detected: %s", arguments)
            return [types.TextContent(type="text", text=json.dumps(
                {"status": "rejected", "reason": "complaint_pattern or policy_clause looks like a placeholder, not real retrieved content. Re-check complaints_query and policy_query results before retrying."}
            ))]

        if any(m in clause for m in NO_POLICY_MARKERS):
            logger.info("No applicable policy clause — issue not created")
            return [types.TextContent(type="text", text=json.dumps(
                {"status": "skipped", "reason": "No applicable policy clause found; issue not filed."}
            ))]
        
        body = (
            f"## Complaint pattern\n{arguments['complaint_pattern']}\n\n"
            f"## Source category\n{arguments['source_category']}\n\n"
            f"## Policy clause referenced\n{arguments['policy_clause']}\n"
        )
        logger.info("issue body == "+ body)
        token = os.getenv("GITHUB_TOKEN")
        logger.info("github token =="+ token)
        if not token:
            logger.error("GITHUB_TOKEN not set — cannot create issue")
            return [types.TextContent(type="text", text=json.dumps(
                {"status": "error", "message": "GITHUB_TOKEN not configured"}
            ))]
        try:
            resp = requests.post(
                f"https://api.github.com/repos/{GITHUB_REPO_OWNER}/{GITHUB_REPO}/issues",
                headers={
                    "Authorization": f"Bearer {token}",
                    "Accept": "application/vnd.github+json",
                },
                json={"title": arguments["title"], "body": body},
            )
            logger.info("GitHub issue create — status=%d", resp.status_code)
            logger.info("GitHub response body: %s", resp.text[:500])
            resp.raise_for_status()
        except requests.exceptions.RequestException as e:
            logger.error("GitHub issue creation failed: %s", e, exc_info=True)
            return [types.TextContent(type="text", text=json.dumps(
                {"status": "error", "message": str(e)}
            ))]
        # logger.info()
        # logger.info(f"HTTP Status Code: {resp.status_code}")
        # logger.info(f"Response Body Text: {resp.text}")
        issue = resp.json()
        logger.info("Created issue #%s: %s", issue["number"], issue["html_url"])

        output = {"status": "created", "issue_number": issue["number"], "url": issue["html_url"]}
        return [types.TextContent(type="text", text=json.dumps(output, indent=2))]

    

if __name__ == "__main__":
    port = int(os.environ["MCP_PORT"])
    app_utils.deploy(server, port, service_name=TOOL_NAME)

