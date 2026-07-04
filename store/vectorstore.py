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
_target = "ecommerce"

def _load_collections_from_consul(target: str) -> dict:
    """
    Reads collection name mappings from Consul KV.
    Falls back to env vars if Consul is unavailable.
    Returns e.g. {"complaints": "corpus_complaints", "policy": "corpus_policy"}
    """
    consul_url = os.getenv("CONSUL_URL")
    prefix = f"rag_mcp/{target}/collections"

    if consul_url:
        try:
            import requests
            r = requests.get(f"{consul_url}/v1/kv/{prefix}?keys", timeout=3)
            r.raise_for_status()
            keys = r.json()  # list of full key paths
            collections = {}
            for key in keys:
                name = key.split("/")[-1]  # e.g. "complaints"
                val_r = requests.get(f"{consul_url}/v1/kv/{key}?raw", timeout=3)
                val_r.raise_for_status()
                collections[name] = val_r.text
                logger.info("Loaded collection from Consul: %s → %s", name, val_r.text)
            return collections
        except Exception as e:
            logger.warning("Consul collection lookup failed: %s — using env var fallback", e)

    # fallback for local dev without Consul
    return {
        "complaints": os.getenv("COMPLAINTS_COLLECTION", "corpus_complaints"),
        "policy":     os.getenv("POLICY_COLLECTION",     "corpus_policy"),
    }


def _get_store():
    if config.VECTORSTORE_PROVIDER == "chromadb":
        return ChromaStore(_target)
    else:
        raise ValueError(f"Unknown VECTORSTORE_PROVIDER: {config.VECTORSTORE_PROVIDER}")


# ── Singleton so the DB is opened once ────────────────
_store = None
def get_store():
    global _store
    if _store is None:
        # _store = _get_store()
        logger.debug("Initialising vector store: provider=%s ",
                config.VECTORSTORE_PROVIDER)
        _store = _get_store()
        # logger.info("Vector store ready — %d chunks in collection '%s'",
        #             _store.count(), config.VECTORSTORE_CORPUS_COLLECTION)
    return _store



# ── ChromaDB implementation ────────────────────────────
class ChromaStore:
    """
    ChromaDB stores vectors in a local directory (VECTORSTORE_PATH).
    No separate server process needed.
    """

    def __init__(self, target:str):
        import chromadb
        # os.makedirs(config.VECTORSTORE_PATH, exist_ok=True)
        # self.client = chromadb.PersistentClient(path=config.VECTORSTORE_PATH)
        import chromadb
        import os
        chroma_host = os.getenv("CHROMA_HOST", "localhost")
        chroma_port = int(os.getenv("CHROMA_PORT", "8000"))
        self.client = chromadb.HttpClient(host=chroma_host, port=chroma_port)
        self.target = target
        self._collection_names = _load_collections_from_consul(target)
        self.collections = {}
        for logical_name, chroma_name in self._collection_names.items():
            logger.info("Creating collection %s → %s", logical_name, chroma_name)
            self.collections[logical_name] = self.client.get_or_create_collection(
                name=chroma_name,
                metadata={"hnsw:space": "cosine"},
            )        
        # collections = config.TARGET[target]["collections"]
        # self.collections = {}
        # for collection in collections:
        #     coll_name = collections[collection]
        #     logger.info("Creating collection %s", coll_name)
        #     self.collections[collection] = self.client.get_or_create_collection(
        #         name=coll_name,
        #         metadata={"hnsw:space": "cosine"},  # use cosine similarity
        #     )


    def add(self, collection, ids, embeddings, documents, metadatas):
        coll = self.collections[collection]
        max_batch = coll._client.get_max_batch_size()  # chromadb exposes this
        total = len(ids)
        logger.debug("Adding %d chunks to collection '%s' in batches of %d",
                    total, collection, max_batch)

        for i in range(0, total, max_batch):
            end = i + max_batch
            coll.add(
                ids=ids[i:end],
                embeddings=embeddings[i:end],
                documents=documents[i:end],
                metadatas=metadatas[i:end],
            )
            logger.info("Added batch %d-%d of %d", i, min(end, total), total)

        logger.info("Stored %d chunks — collection total: %d", total, coll.count())


    def query(self, collection:str, query_embedding: list[float], top_k: int, where: dict = None) -> dict:
        # collection_name = config.TARGET[self.target]["collections"][collection]
        collection_name = self._collection_names[collection]
        logger.debug("Vector search: top_k=%d  collection='%s'",
                top_k, collection_name)
        
        logger.info("Collection name for query is %s", collection_name)
        coll = self.collections[collection]

        kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        results = coll.query(**kwargs)
        distances = results["distances"][0]
        scores    = [round(1 - d, 4) for d in distances]
        sources   = [m.get("source", "?") for m in results["metadatas"][0]]
        logger.info("Search returned %d result(s) — scores: %s  sources: %s",
                     len(scores), scores, sources)

        return {
            "ids":       results["ids"][0],
            "documents": results["documents"][0],
            "metadatas": results["metadatas"][0],
            "distances": results["distances"][0],
        }

    def count(self, collection:str) -> int:
        return self.collections[collection].count()
    
    def list_docs(self, collection: str) -> list[dict]:
        """Return unique source documents (by metadata['source'])."""
        all_items = self.collections[collection].get(include=["metadatas"])
        seen = set()
        docs = []
        for meta in all_items["metadatas"]:
            src = meta.get("source", "unknown")
            if src not in seen:
                seen.add(src)
                docs.append(meta)
        logger.debug("list_docs: %d unique source(s)", len(docs))
        return docs    