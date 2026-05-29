# agent.py
"""
ReAct agent using the MCP SDK for tool communication
and Ollama's HTTP API for LLM reasoning.

Run:
  python mcp_server_http.py   # terminal 1
  python agent.py             # terminal 2
"""

import asyncio
import json
import requests
import config
from mcp import ClientSession
from mcp.client.sse import sse_client

MCP_SSE_URL = "http://localhost:8000/sse"
MAX_STEPS   = 10


# ── Fetch tools via MCP SDK ────────────────────────────────────────────────
async def fetch_mcp_tools(session: ClientSession) -> list[dict]:
    """
    Fetches tools from the MCP server via a proper SSE session
    and converts them to Ollama's tool format.
    """
    tools_response = await session.list_tools()

    ollama_tools = []
    for t in tools_response.tools:
        ollama_tools.append({
            "type": "function",
            "function": {
                "name":        t.name,
                "description": t.description,
                "parameters":  t.inputSchema,
            }
        })
    return ollama_tools


# ── Call a tool via MCP SDK ────────────────────────────────────────────────
async def call_mcp_tool(session: ClientSession, tool_name: str, arguments: dict) -> str:
    """
    Calls a tool via the established MCP SSE session.
    """
    result = await session.call_tool(tool_name, arguments)
    return result.content[0].text


# ── The agent loop ─────────────────────────────────────────────────────────
async def run_agent(goal: str, verbose: bool = True) -> str:
    """
    Opens one MCP SSE session and runs the full agent loop within it.
    """
    async with sse_client(MCP_SSE_URL) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools   = await fetch_mcp_tools(session)
            history = [{"role": "user", "content": goal}]

            if verbose:
                print(f"\nGoal: {goal}")
                print(f"Tools: {[t['function']['name'] for t in tools]}\n")

            for step in range(MAX_STEPS):

                # ── Ask Ollama what to do next ─────────────────────────────
                response = requests.post(
                    f"{config.LLM_BASE_URL}/api/chat",
                    json={
                        "model":    config.LLM_MODEL,
                        "stream":   False,
                        "tools":    tools,
                        "messages": history,
                        "options":  {"temperature": 0.1},
                    },
                    stream=True
                )
                response.raise_for_status()
                # message = response.json()["message"]
                full_message = {"role": "assistant", "content": "", "tool_calls": []}
                print(f"Step {step+1} thinking: ", end="", flush=True)
                for line in response.iter_lines():
                    if not line:
                        continue
                    chunk = json.loads(line)
                    msg   = chunk.get("message", {})
                    if msg.get("content"):
                        print(msg["content"], end="", flush=True)
                        full_message["content"] += msg["content"]
                    if msg.get("tool_calls"):
                        full_message["tool_calls"] = msg["tool_calls"]
                    if chunk.get("done"):
                        print()
                        break

                message = full_message

                # ── Tool call? ─────────────────────────────────────────────
                if message.get("tool_calls"):
                    for tool_call in message["tool_calls"]:
                        fn        = tool_call["function"]
                        tool_name = fn["name"]
                        arguments = fn["arguments"]

                        if verbose:
                            print(f"Step {step+1}: calling {tool_name}({json.dumps(arguments)})")

                        tool_result = await call_mcp_tool(session, tool_name, arguments)

                        if verbose:
                            preview = tool_result[:200].replace('\n', ' ')
                            print(f"         → {preview}...\n")

                        # Append call + result to history so LLM sees them next iteration
                        history.append({
                            "role":       "assistant",
                            "content":    "",
                            "tool_calls": [tool_call],
                        })
                        history.append({
                            "role":    "tool",
                            "content": tool_result,
                            "name":    tool_name,
                        })

                # ── No tool call = LLM is done ─────────────────────────────
                else:
                    final_answer = message["content"].strip()
                    if verbose:
                        print(f"Done in {step+1} step(s).\n")
                    return final_answer

    return "Reached maximum steps without a final answer."


# ── Entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    answer = asyncio.run(run_agent(
        "Has Sudarshan worked on in AI or ML and not just done course work? Give examples of work done."
    ))
    print("Answer:", answer)