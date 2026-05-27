# config.py
from pathlib import Path

# ── LLM ────────────────────────────────────────────────
LLM_PROVIDER = "ollama"          # "ollama" | "openai_compatible"
LLM_BASE_URL  = "http://localhost:11434"
LLM_MODEL     = "llama3.2:1b"       # any model in `ollama list`
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
CHUNK_OVERLAP = 250     # overlap between consecutive chunks

# ── MCP SERVER ─────────────────────────────────────────
MCP_SERVER_NAME = "rag-server"