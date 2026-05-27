# embedder.py
"""
Embedder module.
Swap EMBED_PROVIDER in config.py to change the backend.
Contract: embed(texts: list[str]) -> list[list[float]]
"""

import requests
import config


def embed(texts: list[str]) -> list[list[float]]:
    """
    Embed a list of texts. Returns a list of float vectors.
    One vector per input text.
    """
    if config.EMBED_PROVIDER == "ollama":
        return _embed_ollama(texts)
    elif config.EMBED_PROVIDER == "sentence_transformers":
        return _embed_sentence_transformers(texts)
    else:
        raise ValueError(f"Unknown EMBED_PROVIDER: {config.EMBED_PROVIDER}")


# ── Backend A: Ollama ──────────────────────────────────
def _embed_ollama(texts: list[str]) -> list[list[float]]:
    url = f"{config.EMBED_BASE_URL}/api/embed"
    response = requests.post(url, json={
        "model": config.EMBED_MODEL,
        "input": texts,
    })
    response.raise_for_status()
    return response.json()["embeddings"]   # list[list[float]]


# ── Backend B: sentence-transformers (CPU, no Ollama needed) ──
def _embed_sentence_transformers(texts: list[str]) -> list[list[float]]:
    # pip install sentence-transformers
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(config.EMBED_MODEL)
    return model.encode(texts, convert_to_numpy=True).tolist()