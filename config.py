# config.py
from dotenv import load_dotenv
load_dotenv()
import os
import logging
from pathlib import Path

# ── LLM ────────────────────────────────────────────────
LLM_PROVIDER    = os.getenv("LLM_PROVIDER",    "openai_compatible")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))
LLM_MAX_TOKENS  = int(os.getenv("LLM_MAX_TOKENS",    "1024"))

# ── EMBEDDINGS ─────────────────────────────────────────
EMBED_PROVIDER  = os.getenv("EMBED_PROVIDER",  "sentence_transformers")
EMBED_MODEL     = os.getenv("EMBED_MODEL",      "BAAI/bge-m3")


# ── VECTOR STORE ───────────────────────────────────────
VECTORSTORE_PROVIDER  = "chromadb"   # "chromadb" | "faiss"
# CHROMA_HOST     = os.getenv("CHROMA_HOST",     "localhost")
# CHROMA_PORT     = int(os.getenv("CHROMA_PORT", "8000"))



# ── RETRIEVAL ──────────────────────────────────────────
RETRIEVAL_TOP_K       = 5        # how many chunks to fetch
RETRIEVAL_STRATEGY    = "cosine" # "cosine" (only option for now; extend here)

# ── CHUNKING ───────────────────────────────────────────
CHUNK_SIZE    = 400    # characters per chunk
CHUNK_OVERLAP = 80     # overlap between consecutive chunks

LOG_LEVEL_NAME = "DEBUG"   # change to "DEBUG" for verbose console
LOG_DIR        = "logs/"

# GITHUB_REPO = "scratch_repo_for_tests"
# GITHUB_REPO_OWNER = "sbsud"