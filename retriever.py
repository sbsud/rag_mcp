# retriever.py
"""
Retriever module.
Takes a query string, embeds it, and fetches the top-k similar chunks
from the vector store.
"""

import config
import embedder
import vectorstore


def retrieve(query: str, top_k: int = None) -> list[dict]:
    """
    Returns a list of dicts:
      [{"text": ..., "source": ..., "score": ..., "chunk_id": ...}, ...]
    Ordered by relevance (best first).
    """
    top_k = top_k or config.RETRIEVAL_TOP_K

    # 1. Embed the query (same model used at ingest time — critical!)
    [query_vec] = embedder.embed([query])

    # 2. Search the vector store
    store = vectorstore.get_store()
    results = store.query(query_vec, top_k=top_k)

    # 3. Package results
    chunks = []
    for text, meta, dist in zip(
        results["documents"],
        results["metadatas"],
        results["distances"],
    ):
        chunks.append({
            "text":     text,
            "source":   meta.get("source", "unknown"),
            "chunk_id": meta.get("chunk_id", "?"),
            "score":    round(1 - dist, 4),  # cosine distance → similarity
        })

    return chunks