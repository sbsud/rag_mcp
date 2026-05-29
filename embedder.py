# embedder.py
"""
Embedder module.
Swap EMBED_PROVIDER in config.py to change the backend.
Contract: embed(texts: list[str]) -> list[list[float]]
"""
import logging
import time
import requests
import config

logger = logging.getLogger(__name__)

def embed(texts: list[str]) -> list[list[float]]:
    """
    Embed a list of texts. Returns a list of float vectors.
    One vector per input text.
    """
    logger.debug("Embedding %d text(s) with [%s:%s]",
                 len(texts), config.EMBED_PROVIDER, config.EMBED_MODEL)

    if config.EMBED_PROVIDER == "ollama":
        return _embed_ollama(texts)
    elif config.EMBED_PROVIDER == "sentence_transformers":
        return _embed_sentence_transformers(texts)
    else:
        raise ValueError(f"Unknown EMBED_PROVIDER: {config.EMBED_PROVIDER}")


# ── Backend A: Ollama ──────────────────────────────────
def _embed_ollama(texts: list[str]) -> list[list[float]]:
    url = f"{config.EMBED_BASE_URL}/api/embed"
    logger.debug("POST %s  model=%s  inputs=%d", url, config.EMBED_MODEL, len(texts))
    try:
        response = requests.post(url, json={
        "model": config.EMBED_MODEL,
        "input": texts,
        })
        response.raise_for_status()
    except:
        logger.error("Ollama embed request failed: %s", e)
        raise

    # return response.json()["embeddings"]   # list[list[float]]
    embeddings = response.json()["embeddings"]
    logger.debug("Received %d embedding(s) from Ollama", len(embeddings))
    return embeddings



# ── Backend B: sentence-transformers (CPU, no Ollama needed) ──
def _embed_sentence_transformers(texts: list[str]) -> list[list[float]]:
    # pip install sentence-transformers
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(config.EMBED_MODEL)
    return model.encode(texts, convert_to_numpy=True).tolist()