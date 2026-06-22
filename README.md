# rag_mcp

A fully local RAG pipeline exposed as MCP servers, connected to a ReAct agent.
Built and documented as a public learning series — each branch is a working state that corresponds to a blog post.

---

## Blog series

| Branch | Post | What it introduces |
|--------|------|--------------------|
| `main` | [Part 1 — Building a fully local RAG pipeline with an MCP server](https://medium.com/@_sudarshans/building-a-fully-local-rag-pipeline-with-an-mcp-server-what-i-learned-the-hard-way-421ccb7e0645) | RAG pipeline, ChromaDB, single MCP server over HTTP/SSE |
| `multi-mcp-and-agent` | Part 2 — Grounded answers, smarter routing | Multiple MCP servers, ReAct agent, tool description as router |
| `action_mcp` ← you are here | Part 3 — Grounded language, untethered action | Action-capable agent, real side effects, server-side grounding gate |

---

## What this branch demonstrates

The agent connects to three MCP servers and runs a multi-step investigation over an e-commerce complaint dataset:

1. **`complaints_query`** — retrieve the top complaint pattern from a ChromaDB collection of Amazon Appliances reviews
2. **`policy_query`** — find the applicable resolution policy clause from a separate corpus
3. **`create_github_issue_action`** — file a real GitHub issue if, and only if, a genuine policy clause was found

The central finding from building this: **prompt-level grounding instructions do not transfer across separate LLM calls**. In the same agent session, the LLM correctly retrieved "no matching policy clause found" in one generation, then fabricated a policy clause and filed a real GitHub issue in the next. Two explicit prompt rewrites failed to fix this. The fix that worked was a **server-side keyword gate in `mcp_server_github.py`** — the MCP server inspects the `policy_clause` argument value before allowing execution, and rejects calls that contain fabricated or placeholder content.

This is not a claim that the problem is solved. The gate is a heuristic matched to observed failure phrases. The transferable principle is where in the architecture the check has to sit: at the action boundary, enforced by the system, not requested of the model.

---

## Architecture

```
multi_agent.py              ← ReAct loop (reason → act → observe → repeat)
    │
    ├── mcp/mcp_server_complaints.py   port 8003   complaints_query
    ├── mcp/mcp_server_policy.py       port 8004   policy_query
    └── mcp/mcp_server_github.py       port 8005   create_github_issue_action
                                                    └── keyword gate (runs before execution)

Each MCP server calls:
    rag/rag_pipeline.py
        ├── store/retriever.py   → embed query → search ChromaDB
        └── llm/llm.py           → generate answer from retrieved chunks

Two ChromaDB collections (populated separately before the agent runs):
    corpus_complaints   ← Amazon Appliances review JSONL
    corpus_policy       ← plain-text resolution policy document
```

The agent does not know which server owns which tool. `multi_agent.py` builds a `tool_name → session` routing table at startup from the tool descriptions advertised by each server. Adding or removing a server requires no changes to the agent loop.

---

## Repository structure

```
config.py                     all settings: models, ports, chunk sizes, targets
multi_agent.py                ReAct agent loop

mcp/
  mcp_server_complaints.py    complaints retrieval MCP server (port 8003)
  mcp_server_policy.py        policy retrieval MCP server (port 8004)
  mcp_server_github.py        GitHub action MCP server with grounding gate (port 8005)
  app_utils.py                shared SSE/Starlette wiring used by all three servers

rag/
  rag_pipeline.py             retriever + prompt template + LLM orchestration

store/
  embedder.py                 text → vector  (sentence-transformers or Ollama)
  vectorstore.py              ChromaDB add/query wrapper; manages named collections
  retriever.py                embed query, vector search, return ranked chunks

llm/
  llm.py                      LLM call abstraction (Ollama and OpenAI-compatible)

ingest/
  ingest.py                   CLI: load, chunk, embed, store  (.jsonl / .txt / .pdf)
  embed_corpus.py             pre-compute embeddings for large JSONL corpora (.npy)

logging_config.py             coloured console + rotating file handler, configured once
test_scripts/                 standalone debugging scripts for individual components
```

---

## Quick start

### Prerequisites

- Python 3.11+
- [Groq API key](https://console.groq.com) — free tier is sufficient; alternatively, a local Ollama install (see [Switching to Ollama](#switching-to-ollama) below)
- GitHub personal access token with `repo` scope, pointed at a test repository you own
- Amazon Appliances review dataset in JSONL format — available from [McAuley-Lab/Amazon-Reviews-2023](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023) on HuggingFace
- A plain-text resolution policy document to use as the policy corpus (create your own, or use any e-commerce returns/refunds policy text)

### 1. Clone and install

```bash
git clone https://github.com/sbsud/rag_mcp.git
cd rag_mcp
git checkout action_mcp

pip install -r requirements.txt
```

First run will download `BAAI/bge-m3` (~2 GB) from HuggingFace. Subsequent runs load it from cache.

### 2. Set environment variables

```bash
export GROQ_API_KEY=your_groq_key_here
export GITHUB_TOKEN=your_github_token_here
```

### 3. Configure your GitHub target repository

In `mcp/mcp_server_github.py`, set:

```python
GITHUB_REPO_OWNER = "your-github-username"
GITHUB_REPO       = "your-test-repo-name"
```

### 4. Pre-compute embeddings (recommended for large JSONL files)

The complaints corpus can be tens of thousands of records. Pre-computing embeddings once avoids re-embedding on every ingest run:

```bash
python ingest/embed_corpus.py --path data/amazon_Appliances.jsonl
```

This writes a `.npy` file alongside the JSONL. `ingest.py` detects and loads it automatically.

### 5. Ingest both corpora

```bash
# Complaints (JSONL format)
python ingest/ingest.py \
  --target ecommerce \
  --collection complaints \
  --path data/amazon_Appliances.jsonl

# Policy (plain text)
python ingest/ingest.py \
  --target ecommerce \
  --collection policy \
  --path data/ecommerce_policy.txt
```

Valid `--collection` values for `--target ecommerce` are `complaints` and `policy`.
These map to ChromaDB collections `corpus_complaints` and `corpus_policy` as defined in `config.py`.

### 6. Start the three MCP servers

Each needs its own terminal:

```bash
# Terminal 1
python mcp/mcp_server_complaints.py   # listens on port 8003

# Terminal 2
python mcp/mcp_server_policy.py       # listens on port 8004

# Terminal 3
python mcp/mcp_server_github.py       # listens on port 8005
```

### 7. Run the agent

```bash
# Terminal 4
python multi_agent.py
```

The agent runs the full investigation and prints the final answer.
Structured logs are written to `logs/rag_mcp.log`. Console output is coloured by log level.

---

## Switching to Ollama

All provider switching is in `config.py` — nothing else changes.

**LLM:**

```python
# Groq (default in this branch)
LLM_PROVIDER = "openai_compatible"
LLM_BASE_URL = "https://api.groq.com/openai"
LLM_MODEL    = "llama-3.3-70b-versatile"

# Local Ollama
# LLM_PROVIDER = "ollama"
# LLM_BASE_URL = "http://localhost:11434"
# LLM_MODEL    = "qwen2.5:7b"
```

**Embeddings:**

```python
# sentence-transformers (default — no Ollama needed)
EMBED_PROVIDER = "sentence_transformers"
EMBED_MODEL    = "BAAI/bge-m3"

# Ollama
# EMBED_PROVIDER = "ollama"
# EMBED_BASE_URL = "http://localhost:11434"
# EMBED_MODEL    = "mxbai-embed-large"
```

If you change the embedding model after ingesting, you must re-ingest — ChromaDB will contain vectors from the old model and similarity scores will be meaningless.

---

## Stack

| Component | Choice in this branch |
|-----------|----------------------|
| Embeddings | `sentence-transformers` — `BAAI/bge-m3` |
| Vector store | ChromaDB, local persistent |
| LLM (agent + RAG) | Groq — `llama-3.3-70b-versatile` |
| MCP transport | MCP Python SDK — HTTP/SSE via Starlette + uvicorn |
| Agent pattern | ReAct loop in `multi_agent.py` |
