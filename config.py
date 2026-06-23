# config.py
from dotenv import load_dotenv
load_dotenv()
import logging
from pathlib import Path

# ── LLM ────────────────────────────────────────────────
LLM_PROVIDER = "openai_compatible"          # "ollama" | "openai_compatible"
# LLM_BASE_URL  = "http://localhost:11434"
# LLM_BASE_URL = "https://api.groq.com/openai"
# LLM_MODEL = "llama-3.3-70b-versatile"
# config.py
LLM_BASE_URL = "https://api.cerebras.ai"
LLM_MODEL = "gpt-oss-120b"
LLM_TEMPERATURE = 0.1
LLM_MAX_TOKENS  = 1024

# ── EMBEDDINGS ─────────────────────────────────────────
EMBED_PROVIDER = "sentence_transformers"
EMBED_MODEL = "BAAI/bge-m3"  # same model, loaded directly


# ── VECTOR STORE ───────────────────────────────────────
VECTORSTORE_PROVIDER  = "chromadb"   # "chromadb" | "faiss"
VECTORSTORE_PATH      = str(Path(__file__).parent / "chromadb" / ".chroma")
VECTORSTORE_COLLECTION = "corpus"
VECTORSTORE_CORPUS_COLLECTION = "corpus"

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

MCP_WEBSEARCH_PORT = 8001

TARGET = {

    "ecommerce": {
        "name": "ecommerce",
        "collections": {
            "complaints": "corpus_complaints",
            "policy": "corpus_policy"
        },
        "description": "ecomnerce configurations",
        "system_prompt": ("You are a customer complaint intelligence agent. "
                        "You have tools: complaints_query, policy_query,"
                         " create_github_issue. "
                        "Never call create_github_issue unless complaints_query and policy_query has already been called in this session. "
                        "GitHub issue body must always include the actual complaint pattern found, source category, policy clause referenced."),
        "goal": (
                    "Step 1: Use complaints_query to find the top complaint pattern, including specific brand names and complaint themes. "
                    "Step 2: Use policy_query to find the relevant resolution policy for that exact pattern. "
                    "If no policy clause genuinely applies to the complaint pattern found in Step 1, do not proceed to Step 3 — "
                    "report that no applicable policy was found instead. "
                    "Step 3: Only if a genuinely applicable policy clause was found, use create_github_issue ONCE, "
                    "passing the specific findings from steps 1 and 2 verbatim. "
                    "After create_github_issue returns successfully, you are DONE — report the issue URL as your final answer."
                ),
        "mcp_port": {
            "complaints": 8003,
            "policy": 8004,
            "github": 8005,
        },
    },

}
GITHUB_REPO = "scratch_repo_for_tests"
GITHUB_REPO_OWNER = "sbsud"