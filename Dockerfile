# Dockerfile for MCP servers (complaints, policy, github)
FROM python:3.11-slim
WORKDIR /app
COPY requirements-base.txt .
RUN pip install --no-cache-dir -r requirements-base.txt
COPY config.py logging_config.py ./
COPY store/ store/
COPY rag/ rag/
COPY llm/ llm/
COPY mcp/ mcp/

