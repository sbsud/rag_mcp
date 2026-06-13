# retriever.py
"""
Retriever module.
Takes a query string, embeds it, and fetches the top-k similar chunks
from the vector store.
"""
import logging
import time
import config
from store import embedder
from store import vectorstore

logger = logging.getLogger(__name__)

def retrieve(corpus: str, query: str, top_k: int = None) -> list[dict]:
    """
    Returns a list of dicts:
      [{"text": ..., "source": ..., "score": ..., "chunk_id": ...}, ...]
    Ordered by relevance (best first).
    """
    top_k = top_k or config.RETRIEVAL_TOP_K
    logger.info("Retrieve: query='%s'  top_k=%d", query[:80], top_k)

    start = time.perf_counter()

    # 1. Embed the query (same model used at ingest time — critical!)
    logger.debug("Embedding query...")
    [query_vec] = embedder.embed([query])

    # 2. Search the vector store
    logger.debug("Searching vector store...")
    store = vectorstore.get_store()
    results = store.query(corpus, query_vec, top_k=top_k)

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
        
    elapsed = time.perf_counter() - start
    scores  = [c["score"] for c in chunks]
    sources = list({c["source"] for c in chunks})
    logger.info("Retrieve complete: %d chunk(s) in %.2fs — scores=%s  sources=%s",
                len(chunks), elapsed, scores, sources)
 
    if chunks and chunks[0]["score"] < 0.4:
        logger.warning("Top result score %.3f is low — retrieval may be unreliable "
                       "for query: '%s'", chunks[0]["score"], query[:80])

    return chunks