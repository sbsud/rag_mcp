from typing import Annotated
from fastapi import FastAPI, Header
from pydantic import BaseModel
import uuid
import asyncio
import os
from agent.multi_agent import run_agent
from agent.kv_config import get_kv



class AgentRequest(BaseModel):
    query: str
    max_steps: int = 10

class AgentResponse(BaseModel):
    trace_id: str
    answer: str
    steps: int
    elapsed_time: int

app = FastAPI()

@app.post("/query")
async def accept_query(agent_query: AgentRequest, trace_id: Annotated[str | None, Header()] = None):

    query_trace_id = trace_id or str(uuid.uuid4())
    goal = get_kv(
                "rag_mcp/ecommerce/goal",
                fallback=os.getenv("AGENT_GOAL", "")
            )
    system_prompt = get_kv(
                "rag_mcp/ecommerce/system_prompt",
                fallback=os.getenv("AGENT_SYSTEM_PROMPT", "")
            )
    answer = await run_agent(trace_id=query_trace_id, target="ecommerce", query=agent_query.query, goal=goal, system_prompt=system_prompt)
    return AgentResponse(trace_id=query_trace_id, answer=answer, steps=3, elapsed_time=55)

