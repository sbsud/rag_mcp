# test_client.py — a minimal MCP client for testing
import asyncio, json
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

SERVER_CMD  = ".venv/bin/python"
SERVER_ARGS = ["mcp_server.py"]

async def main():
    server_params = StdioServerParameters(command=SERVER_CMD, args=SERVER_ARGS)
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # List tools
            tools = await session.list_tools()
            print("Available tools:", [t.name for t in tools.tools])

            # Call rag_query
            result = await session.call_tool(
                "rag_query",
                {"question": "which company did Sudarshan work at last? consider work experience is provided in reverse chronological order."}
            )
            print(result)

            data = json.loads(result.content[0].text)
            print("\nAnswer:", data["answer"])

asyncio.run(main())