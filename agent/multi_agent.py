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
import os
import argparse
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
# _target="ecommerce"
# _port_config = config.TARGET[_target]["mcp_port"]
# MCP_SERVERS = [
#     {"name": "complaints",    "url": f"http://localhost:{_port_config["complaints"]}/sse"},
#     {"name": "policy", "url": f"http://localhost:{_port_config["policy"]}/sse"},
#     {"name": "github", "url": f"http://localhost:{_port_config["github"]}/sse"},

# ]

MAX_STEPS = 10
TOOL_RESULT_MAX_CHARS = 4000  # raise the ceiling

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

async def run_agent(trace_id: str, target: str, query:str = None, verbose: bool = True, goal:str = None, system_prompt:str = None) -> str:
    """
    Opens SSE connections to all servers, builds the tool registry,
    then runs the ReAct loop until the LLM produces a final answer.
    """
    # target_config = config.TARGET[target]
    # target_config = setup(target=target)
    setup()
    # Open all SSE connections concurrently using AsyncExitStack
    from contextlib import AsyncExitStack
    logger.info("═" * 60)
    logger.info("TRACE ID = %s",trace_id)
    logger.info("Agent starting — goal: '%s'", goal[:100])
    tool_call_counts: dict[str, int] = {}
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
        # goal = query or target_config["goal"]
        history = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content":  f"{goal} {query}"},
        ]
        logger.info("TEST REMOVE")
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
                        llm_elapsed, config.LLM_PROVIDER, llm_model)
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
                    count = tool_call_counts.get(tool_name, 0)
                    if count >= 1 and tool_name in ("complaints_query", "policy_query"):
                        logger.warning("Gate: '%s' already called — injecting stop signal", tool_name)
                        history.append({
                            "role": "tool",
                            "content": json.dumps({"answer": "Tool already called. Do NOT call it again. You now have all required data. Your ONLY valid next action is create_github_issue if a policy clause was found, or provide your final answer if not."}),
                            "tool_call_id": tool_call_id,
                            "name": tool_name
                        })
                        tool_call_counts[tool_name] = count + 1
                        continue
                    tool_call_counts[tool_name] = count + 1
                    tool_result = await call_tool(tool_router, tool_name, arguments)

                    if len(tool_result) > TOOL_RESULT_MAX_CHARS:
                        logger.warning("Tool result truncated: %d → %d chars", len(tool_result), TOOL_RESULT_MAX_CHARS)
                        tool_result = tool_result[:TOOL_RESULT_MAX_CHARS]

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
    logger.info("llm_base_url == %s ||||llm_api_key == %s", llm_base_url,llm_api_key)
    url     = f"{llm_base_url}/v1/chat/completions"
    headers = {"Authorization": f"Bearer {llm_api_key}"}
    logger.debug("LLM call: openai_compatible  url=%s  model=%s", url, llm_model)
    logger.info("HEADERS == %s", headers)

    for attempt in range(3):
        response = requests.post(url, headers=headers, json={
            "model":       llm_model,
            "tools":       tools,
            "messages":    messages,
            "temperature": 0.1,
        })
        if response.status_code == 429:
            wait = 12 * (attempt + 1)  # 12s, 24s, 36s — stays within 5 RPM window
            logger.warning("429 from LLM Provider — waiting %ds before retry (attempt %d/3)", wait, attempt + 1)
            time.sleep(wait)
            continue
        response.raise_for_status()
        break

    if not response.ok:
        print(f"=== LLM ERROR {response.status_code} ===")
        print(response.text)
        print("=== END LLM ERROR ===")

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

# def main():
def build_mcp_servers() -> list[dict]:
    consul_url = os.environ["CONSUL_URL"]
    if consul_url:
        from agent.mcp_discovery import discover_services
        servers = discover_services(["complaints-mcp", "policy-mcp", "github-mcp"])
        import logging
        logging.getLogger(__name__).info("MCP_SERVERS from Consul: %s", servers)
        return servers        
    # fallback for local dev without Docker
    return [
        {"name": "complaints-mcp", "url": os.getenv("COMPLAINTS_MCP_URL", "http://localhost:8003/sse")},
        {"name": "policy-mcp",     "url": os.getenv("POLICY_MCP_URL",     "http://localhost:8004/sse")},
        {"name": "github-mcp",     "url": os.getenv("GITHUB_MCP_URL",     "http://localhost:8005/sse")},
    ]

def setup() -> dict:
    global llm_base_url, llm_api_key, llm_model, MCP_SERVERS
    llm_base_url = os.environ['LLM_BASE_URL']
    llm_api_key = os.environ['LLM_API_KEY']
    llm_model = os.environ['LLM_MODEL']
    # _port_config = config.TARGET[target]["mcp_port"]

    MCP_SERVERS = build_mcp_servers()
    return
#     MCP_SERVERS = [
#         {"name": "complaints",    "url": f"{os.getenv('COMPLAINTS_MCP_URL')}"},
#         {"name": "policy", "url": f"{os.getenv('POLICY_MCP_URL')}"},
#         {"name": "github", "url": f"{os.getenv('GITHUB_MCP_URL')}"},

# ]
    # target_config = config.TARGET[target]
    # return target_config


# if __name__ == "__main__":

#     parser = argparse.ArgumentParser()
#     parser.add_argument("--target", required=True, choices=config.TARGET.keys())
#     args = parser.parse_args()
#     global _target 
#     _target = args.target;

#     target_config = setup(target=_target)
#     print("\n" + "=" * 60)
    
#     answer = asyncio.run(run_agent(target_config, target=_target))
#     print("Answer:", answer)