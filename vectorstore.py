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

import config


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
        _store = _get_store()
    return _store


# ── ChromaDB implementation ────────────────────────────
class ChromaStore:
    """
    ChromaDB stores vectors in a local directory (VECTORSTORE_PATH).
    No separate server process needed.
    """
    def __init__(self):
        import chromadb
        import os
        os.makedirs(config.VECTORSTORE_PATH, exist_ok=True)
        self.client = chromadb.PersistentClient(path=config.VECTORSTORE_PATH)
        self.collection = self.client.get_or_create_collection(
            name=config.VECTORSTORE_COLLECTION,
            metadata={"hnsw:space": "cosine"},  # use cosine similarity
        )

    def add(self, ids, embeddings, documents, metadatas):
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

    def query(self, query_embedding: list[float], top_k: int) -> dict:
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
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
        return docs