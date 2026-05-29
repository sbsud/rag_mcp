# vectorstore.py
"""
VectorStore module.
Swap VECTORSTORE_PROVIDER in config.py to change the backend.
Contract:
  add(ids, embeddings, documents, metadatas)
  query(query_embedding, top_k) -> {"ids": [...], "documents": [...], "metadatas": [...], "distances": [...]}
  count() -> int
  list_docs() -> list[dict]
"""
import logging
import os
import config


logger = logging.getLogger(__name__)

def _get_store():
    if config.VECTORSTORE_PROVIDER == "chromadb":
        return ChromaStore()
    else:
        raise ValueError(f"Unknown VECTORSTORE_PROVIDER: {config.VECTORSTORE_PROVIDER}")


# ── Singleton so the DB is opened once ────────────────
_store = None
def get_store():
    global _store
    if _store is None:
        # _store = _get_store()
        logger.debug("Initialising vector store: provider=%s  path=%s  collection=%s",
                config.VECTORSTORE_PROVIDER, config.VECTORSTORE_PATH,
                config.VECTORSTORE_COLLECTION)
        _store = _get_store()
        logger.info("Vector store ready — %d chunks in collection '%s'",
                    _store.count(), config.VECTORSTORE_COLLECTION)
    return _store



# ── ChromaDB implementation ────────────────────────────
class ChromaStore:
    """
    ChromaDB stores vectors in a local directory (VECTORSTORE_PATH).
    No separate server process needed.
    """
    def __init__(self):
        import chromadb
        os.makedirs(config.VECTORSTORE_PATH, exist_ok=True)
        self.client = chromadb.PersistentClient(path=config.VECTORSTORE_PATH)
        self.collection = self.client.get_or_create_collection(
            name=config.VECTORSTORE_COLLECTION,
            metadata={"hnsw:space": "cosine"},  # use cosine similarity
        )

    def add(self, ids, embeddings, documents, metadatas):
        logger.debug("Adding %d chunks to collection '%s'",
                     len(ids), config.VECTORSTORE_COLLECTION)
        self.collection.add(
            ids=ids, embeddings=embeddings,
            documents=documents, metadatas=metadatas,
        )
        logger.info("Stored %d chunks — collection total: %d",
                    len(ids), self.collection.count())


    def query(self, query_embedding: list[float], top_k: int) -> dict:
        logger.debug("Vector search: top_k=%d  collection='%s'",
                top_k, config.VECTORSTORE_COLLECTION)

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        distances = results["distances"][0]
        scores    = [round(1 - d, 4) for d in distances]
        sources   = [m.get("source", "?") for m in results["metadatas"][0]]
        logger.debug("Search returned %d result(s) — scores: %s  sources: %s",
                     len(scores), scores, sources)

        return {
            "ids":       results["ids"][0],
            "documents": results["documents"][0],
            "metadatas": results["metadatas"][0],
            "distances": results["distances"][0],
        }

    def count(self) -> int:
        return self.collection.count()

    def list_docs(self) -> list[dict]:
        """Return unique source documents (by metadata['source'])."""
        all_items = self.collection.get(include=["metadatas"])
        seen = set()
        docs = []
        for meta in all_items["metadatas"]:
            src = meta.get("source", "unknown")
            if src not in seen:
                seen.add(src)
                docs.append(meta)
        logger.debug("list_docs: %d unique source(s)", len(docs))
        return docs