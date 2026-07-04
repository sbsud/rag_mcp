# test_scripts/test_policy_rag.py
import asyncio
from mcp.client.sse import sse_client
from mcp import ClientSession

async def test():
    async with sse_client("http://localhost:8004/sse") as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(
                "policy_query",
                {"question": "what is the policy for defective products"}
            )
            print(result.content[0].text)

asyncio.run(test())