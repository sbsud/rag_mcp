# config.py
import logging
from pathlib import Path

# ── LLM ────────────────────────────────────────────────
# LLM_PROVIDER = "ollama"          # "ollama" | "openai_compatible"
LLM_PROVIDER = "openai_compatible"
# LLM_BASE_URL  = "http://localhost:11434"
LLM_BASE_URL = "https://api.groq.com/openai"
# LLM_MODEL     = "llama3.2:1b"  
# LLM_MODEL = "qwen2.5:7b"     # any model in `ollama list`
# LLM_MODEL = "qwen2.5:3b" 
# LLM_MODEL    = "qwen-qwq-32b"   # or "llama-3.3-70b-versatile", "mixtral-8x7b-32768"
# LLM_MODEL = "llama3-groq-70b-8192-tool-use-preview"
LLM_MODEL = "llama-3.3-70b-versatile"
LLM_TEMPERATURE = 0.1
LLM_MAX_TOKENS  = 1024

# ── EMBEDDINGS ─────────────────────────────────────────
EMBED_PROVIDER = "ollama"        # "ollama" | "sentence_transformers"
EMBED_BASE_URL = "http://localhost:11434"
# EMBED_MODEL    = "nomic-embed-text"   # must match `ollama pull` name
EMBED_MODEL = "mxbai-embed-large"
# If using sentence_transformers instead:
# EMBED_PROVIDER = "sentence_transformers"
# EMBED_MODEL    = "all-MiniLM-L6-v2"

# ── VECTOR STORE ───────────────────────────────────────
VECTORSTORE_PROVIDER  = "chromadb"   # "chromadb" | "faiss"
VECTORSTORE_PATH      = str(Path(__file__).parent / "data" / ".chroma")
VECTORSTORE_COLLECTION = "rag_docs"

# ── RETRIEVAL ──────────────────────────────────────────
RETRIEVAL_TOP_K       = 5        # how many chunks to fetch
RETRIEVAL_STRATEGY    = "cosine" # "cosine" (only option for now; extend here)

# ── CHUNKING ───────────────────────────────────────────
CHUNK_SIZE    = 400    # characters per chunk
CHUNK_OVERLAP = 80     # overlap between consecutive chunks

# ── MCP SERVER ─────────────────────────────────────────
MCP_SERVER_NAME = "rag-server"

LOG_LEVEL_NAME = "DEBUG"   # change to "DEBUG" for verbose console
LOG_DIR        = "logs/"

MCP_LOCAL_RESUME_HTTP_PORT = 8000
MCP_WEBSEARCH_PORT = 8001