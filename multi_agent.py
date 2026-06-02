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
import os
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

        ollama_tools, tool_router = await build_tool_registry(sessions)
        history = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant with access to tools. "
                    "When asked to search for jobs matching someone's competencies, skills: "
                    "first retrieve the top competencies, skills using the resume tool, "
                    "then build a search query from the top competency names themselves — "
                    "never use the person's name as a search term for job listings. "
                    # "A good job search query looks like: "
                    # "'Kafka Flink Kubernetes Staff Engineer jobs Bangalore site:linkedin.com OR site:naukri.com'"
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

            logger.debug("HISTORY %s", history)
            message   = _call_llm(history, ollama_tools)

            llm_elapsed = time.perf_counter() - llm_start
            logger.info("LLM responded in %.2fs  provider=%s  model=%s",
                        llm_elapsed, config.LLM_PROVIDER, config.LLM_MODEL)
            # Tool call?
            if message.get("tool_calls"):
                # Re-serialise arguments to JSON string for history storage.
                # Groq sends arguments as a string, we parse them to dict for tool dispatch,
                # but Groq requires them back as a string when the message goes into history.
                history_tool_calls = []
                for tc in message["tool_calls"]:
                    history_tc = {
                        "id":       tc.get("id", ""),
                        "type":     tc.get("type", "function"),
                        "function": {
                            "name":      tc["function"]["name"],
                            "arguments": (
                                json.dumps(tc["function"]["arguments"])
                                if isinstance(tc["function"]["arguments"], dict)
                                else tc["function"]["arguments"]   # already a string (Ollama path)
                            ),
                        }
                    }
                    history_tool_calls.append(history_tc)

                history.append({
                    "role":       "assistant",
                    "content":    message.get("content") or "",
                    "tool_calls": history_tool_calls,
                })

                for tc, history_tc in zip(message["tool_calls"], history_tool_calls):
                    fn           = tc["function"]
                    tool_name    = fn["name"]
                    arguments    = fn["arguments"]   # dict — already parsed, used for dispatch
                    tool_call_id = tc.get("id", tool_name)

                    logger.info("LLM decided: call '%s'  args=%s",
                                tool_name, json.dumps(arguments))

                    tool_result = await call_tool(tool_router, tool_name, arguments)

                    if len(tool_result) > 2000:
                        logger.warning("Tool result truncated: %d → 2000 chars", len(tool_result))
                        tool_result = tool_result[:2000]

                    history.append({
                        "role":         "tool",
                        "content":      tool_result,
                        "tool_call_id": tool_call_id,
                        "name":         tool_name,
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


def _call_llm(messages: list, tools: list) -> dict:
    """
    Calls the LLM with tool support.
    Handles both Ollama and OpenAI-compatible (Groq, Together AI) providers.
    Always returns a normalised message dict:
      {"content": str, "tool_calls": [...] or None}
    with tool_calls arguments always as a parsed dict (not a JSON string).
    """
    if config.LLM_PROVIDER == "ollama":
        return _call_ollama(messages, tools)
    elif config.LLM_PROVIDER == "openai_compatible":
        return _call_openai_compatible(messages, tools)
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {config.LLM_PROVIDER}")


def _call_ollama(messages: list, tools: list) -> dict:
    url = f"{config.LLM_BASE_URL}/api/chat"
    logger.debug("LLM call: ollama  url=%s  model=%s", url, config.LLM_MODEL)

    response = requests.post(url, json={
        "model":    config.LLM_MODEL,
        "stream":   False,
        "tools":    tools,
        "messages": messages,
        "options":  {"temperature": 0.1},
    })
    response.raise_for_status()
    message = response.json()["message"]

    # Ollama returns tool_call arguments as a dict already — no parsing needed
    return message


def _call_openai_compatible(messages: list, tools: list) -> dict:
    url     = f"{config.LLM_BASE_URL}/v1/chat/completions"
    headers = {"Authorization": f"Bearer {os.getenv('LLM_API_KEY', 'none')}"}
    logger.debug("REMOVE headers %s", headers)
    logger.debug("LLM call: openai_compatible  url=%s  model=%s", url, config.LLM_MODEL)

    response = requests.post(url, headers=headers, json={
        "model":       config.LLM_MODEL,
        "tools":       tools,
        "messages":    messages,
        "temperature": 0.1,
    })

    if not response.ok:
        print(f"=== GROQ ERROR {response.status_code} ===")
        print(response.text)           # Groq always puts the reason here
        print("=== END GROQ ERROR ===")

    response.raise_for_status()
    logger.debug("LLM response \n%s", json.dumps(response.json(),indent=2))
    # OpenAI format wraps the message in choices[0]
    message = response.json()["choices"][0]["message"]

    # OpenAI/Groq returns tool_call arguments as a JSON *string* — parse it to dict
    # so the rest of the agent sees a consistent format regardless of provider
    if message.get("tool_calls"):
        for tc in message["tool_calls"]:
            args = tc["function"]["arguments"]
            if isinstance(args, str):
                tc["function"]["arguments"] = json.loads(args)

    return message


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
        "Step 1: Use resume_query to find Sudarshan's top 3 technical job competencies. "
        "Step 2: From the competencies returned, construct a job search query using "
        # "the actual skill names (like 'Kafka Kubernetes Staff Engineer jobs Bangalore') "
        "Sudarshan's actual top 3 technical job competency names"
        "— do NOT use the person's name in the search query. "
        "Step 3: Use web_search with that competencies-based query to find matching "
        "job listings from companies on LinkedIn, Naukri, or similar job boards. "
        "- do NOT return person profiles or advice or blog posts from LinkedIn or similar with matching skills or links to lists of jobs. " \
        "Step 4: Return the top matching roles with their links."
    ))
    print("Answer:", answer)