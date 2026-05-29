# multi_agent.py
"""
ReAct agent connected to multiple MCP servers simultaneously.

The agent fetches tools from all servers at startup and builds
a routing table: tool_name → session. When the LLM calls a tool,
the router dispatches to the correct server automatically.

Servers:
  RAG server:    http://localhost:8000/sse  (rag_query, rag_search, ...)
  Search server: http://localhost:8001/sse  (web_search, web_fetch)

Run:
  python mcp_server_http.py        # terminal 1
  python mcp_server_websearch.py   # terminal 2
  python multi_agent.py            # terminal 3
"""

import logging
import time
import asyncio
import json
import requests
import config
from mcp import ClientSession
from mcp.client.sse import sse_client
from logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


# ── Server registry ────────────────────────────────────────────────────────
# Add or remove servers here. The agent code below never changes.
MCP_SERVERS = [
    {"name": "resume_local",    "url": "http://localhost:8000/sse"},
    {"name": "web_search", "url": "http://localhost:8001/sse"},
]

MAX_STEPS = 10


# ── Connect to all servers and build tool registry ─────────────────────────

async def build_tool_registry(sessions: dict) -> tuple[list[dict], dict]:
    """
    Connects to every server, collects their tools, and returns:
      - ollama_tools:   list of tools in Ollama's format (sent to LLM)
      - tool_router:    dict mapping tool_name → ClientSession
                        so we know which server to call for each tool
    """
    ollama_tools = []
    tool_router  = {}

    for server_name, session in sessions.items():
        response = await session.list_tools()
        logger.debug("Server '%s' exposes %d tool(s)", server_name,
                     len(response.tools))

        for t in response.tools:
            # Register which session handles this tool
            tool_router[t.name] = session

            # Convert to Ollama format
            ollama_tools.append({
                "type": "function",
                "function": {
                    "name":        t.name,
                    "description": t.description,
                    "parameters":  t.inputSchema,
                }
            })
            logger.info("Registered tool: %-24s ← server='%s'",
                        t.name, server_name)
    logger.info("Tool registry built: %d tool(s) across %d server(s)",
                len(ollama_tools), len(sessions))

    return ollama_tools, tool_router


# ── Call a tool via the correct server ────────────────────────────────────

async def call_tool(tool_router: dict, tool_name: str, arguments: dict) -> str:
    """
    Looks up which server owns this tool and calls it.
    The LLM and the rest of the agent don't need to know
    which server a tool lives on.
    """
    session = tool_router.get(tool_name)
    if not session:
        logger.error("No server registered for tool: %s", tool_name)        
        return json.dumps({"error": f"No server found for tool: {tool_name}"})

    logger.debug("Dispatching '%s' with args=%s", tool_name,
                 json.dumps(arguments)[:120])
    start  = time.perf_counter()
    result = await session.call_tool(tool_name, arguments)
    elapsed = time.perf_counter() - start
 
    text = result.content[0].text
    logger.debug("Tool '%s' returned %d chars in %.2fs", tool_name,
                 len(text), elapsed)
    return text


# ── The agent loop ─────────────────────────────────────────────────────────

async def run_agent(goal: str, verbose: bool = True) -> str:
    """
    Opens SSE connections to all servers, builds the tool registry,
    then runs the ReAct loop until the LLM produces a final answer.
    """
    # Open all SSE connections concurrently using AsyncExitStack
    from contextlib import AsyncExitStack
    logger.info("═" * 60)
    logger.info("Agent starting — goal: '%s'", goal[:100])

    async with AsyncExitStack() as stack:

        # Connect to every server and initialise sessions
        sessions = {}
        for srv in MCP_SERVERS:
            logger.info("Connecting to MCP server '%s' at %s",
                        srv["name"], srv["url"])
            try:
                read, write = await stack.enter_async_context(sse_client(srv["url"]))
                session     = await stack.enter_async_context(ClientSession(read, write))
                await session.initialize()
                sessions[srv["name"]] = session
                logger.info("Connected to '%s'", srv["name"])
            except Exception as e:
                logger.error("Failed to connect to server '%s': %s",
                             srv["name"], e, exc_info=True)
                raise

        # if verbose:
        #     print(f"\nGoal: {goal}")
        #     print(f"Connected to {len(sessions)} server(s). Available tools:")

        ollama_tools, tool_router = await build_tool_registry(sessions)
        # history = [{"role": "user", "content": goal}]
        history = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant with access to tools. "
                    "When asked to search for jobs matching someone's skills: "
                    "first retrieve the skills using the resume tool, "
                    "then build a search query from the skill names themselves — "
                    "never use the person's name as a search term for job listings. "
                    "A good job search query looks like: "
                    "'Kafka Flink Kubernetes Staff Engineer jobs Bangalore site:linkedin.com OR site:naukri.com'"
                )
            },
            {"role": "user", "content": goal}
        ]

        if verbose:
            print()

        # ── ReAct loop ─────────────────────────────────────────────────────
        for step in range(MAX_STEPS):
            logger.info("── Step %d/%d ──────────────────────────────",
                        step, MAX_STEPS)
            logger.debug("Sending %d message(s) in history to LLM", len(history))
            llm_start = time.perf_counter()

            # Ask Ollama what to do next
            response = requests.post(
                f"{config.LLM_BASE_URL}/api/chat",
                json={
                    "model":    config.LLM_MODEL,
                    "stream":   False,
                    "tools":    ollama_tools,
                    "messages": history,
                    "options":  {"temperature": 0.1},
                }
            )
            response.raise_for_status()
            message = response.json()["message"]

            llm_elapsed = time.perf_counter() - llm_start
            logger.info("LLM responded in %.2fs", llm_elapsed)

            # Tool call?
            if message.get("tool_calls"):
                for tool_call in message["tool_calls"]:
                    fn        = tool_call["function"]
                    tool_name = fn["name"]
                    arguments = fn["arguments"]

                    logger.info("LLM decided: call '%s'  args=%s",
                                tool_name, json.dumps(arguments)[:120])


                    # Route to the correct server
                    tool_result = await call_tool(tool_router, tool_name, arguments)

                    if verbose:
                        preview = tool_result[:300].replace('\n', ' ')
                        print(f"         → {preview}...\n")

                    # Append to history — LLM sees this on next iteration
                    history.append({
                        "role":       "assistant",
                        "content":    "",
                        "tool_calls": [tool_call],
                    })
                    history.append({
                        "role":    "tool",
                        "content": tool_result[:2000],  # cap to keep context lean
                        "name":    tool_name,
                    })

            # No tool call = LLM is done
            else:
                final_answer = message["content"].strip()
                logger.info("Agent finished in %d step(s)", step)
                logger.info("Final answer (%d chars): %s...",
                            len(final_answer), final_answer[:120])
                logger.info("═" * 60)

                return final_answer
    
    logger.warning("Agent reached MAX_STEPS=%d without finishing", MAX_STEPS)
    return "Reached maximum steps without a final answer."


# ── Queries ────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # # Query 1: pure RAG — should go only to the RAG server
    # print("=" * 60)
    # answer = asyncio.run(run_agent(
    #     "What are Sudarshan's key technical skills?"
    # ))
    # print("Answer:", answer)

    # print("###################################")

    # # Query 2: pure web search — should go only to the search server
    # print("\n" + "=" * 60)
    # answer = asyncio.run(run_agent(
    #     "What are the top Python developer jobs listed in Bangalore right now?"
    # ))
    # print("Answer:", answer)

    # print("=======================================================")

    # Query 3: chained — RAG first to get skills, then web search for jobs
    print("\n" + "=" * 60)
    answer = asyncio.run(run_agent(
        "Step 1: Use resume_query to find Sudarshan's technical skills. "
        "Step 2: From the skills returned, construct a job search query using "
        "the actual skill names (like 'Kafka Kubernetes Staff Engineer jobs Bangalore') "
        "— do NOT use the person's name in the search query. "
        "Step 3: Use web_search with that skills-based query to find matching "
        "job listings on LinkedIn, Naukri, or similar job boards. "
        "Step 4: Return the top matching roles with their links."
    ))
    print("Answer:", answer)