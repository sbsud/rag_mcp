# test_tool_calling.py
import requests

response = requests.post("http://localhost:11434/api/chat", json={
    "model":  "qwen2.5:7b",
    "stream": False,
    "tools": [{
        "type": "function",
        "function": {
            "name":        "web_search",
            "description": "Search the web for information",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"}
                },
                "required": ["query"]
            }
        }
    }],
    "messages": [{"role": "user", "content": "Search for Python jobs in Bangalore"}]
})

message = response.json()["message"]
print("tool_calls present:", bool(message.get("tool_calls")))
print("content:", message.get("content", "")[:100])

if message.get("tool_calls"):
    for tc in message["tool_calls"]:
        print("tool name:", tc["function"]["name"])
        print("arguments:", tc["function"]["arguments"])