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
        logger.debug("Initialising vector store: provider=%s  path=%s  target=%s",
                config.VECTORSTORE_PROVIDER, config.VECTORSTORE_PATH,
                _target)
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
    # def __init__(self):
    #     import chromadb
    #     os.makedirs(config.VECTORSTORE_PATH, exist_ok=True)
    #     self.client = chromadb.PersistentClient(path=config.VECTORSTORE_PATH)
    #     self.collection = self.client.get_or_create_collection(
    #         name=config.VECTORSTORE_CORPUS_COLLECTION,
    #         metadata={"hnsw:space": "cosine"},  # use cosine similarity
    #     )

    def __init__(self, target:str):
        import chromadb
        os.makedirs(config.VECTORSTORE_PATH, exist_ok=True)
        self.client = chromadb.PersistentClient(path=config.VECTORSTORE_PATH)
        self.target = target
        collections = config.TARGET[target]["collections"]
        self.collections = {}
        for collection in collections:
            coll_name = collections[collection]
            logger.info("Creating collection %s", coll_name)
            self.collections[collection] = self.client.get_or_create_collection(
                name=coll_name,
                metadata={"hnsw:space": "cosine"},  # use cosine similarity
            )
        print("── self.collections ─────────────────────────────")
        print(self.collections)
        print("── self.collections ─────────────────────────────")

    # def add(self, ids, embeddings, documents, metadatas):
    #     logger.debug("Adding %d chunks to collection '%s'",
    #                  len(ids), config.VECTORSTORE_CORPUS_COLLECTION)
    #     self.collection.add(
    #         ids=ids, embeddings=embeddings,
    #         documents=documents, metadatas=metadatas,
    #     )
    #     logger.info("Stored %d chunks — collection total: %d",
    #                 len(ids), self.collection.count())

    def add(self, collection, ids, embeddings, documents, metadatas):
        logger.debug("Adding %d chunks to collection '%s'",
                     len(ids), collection)
        self.collections[collection].add(
            ids=ids, embeddings=embeddings,
            documents=documents, metadatas=metadatas,
        )
        logger.info("Stored %d chunks — collection total: %d",
                    len(ids), self.collections[collection].count())


    # def query(self, query_embedding: list[float], top_k: int) -> dict:
    #     logger.debug("Vector search: top_k=%d  collection='%s'",
    #             top_k, config.VECTORSTORE_CORPUS_COLLECTION)

    #     results = self.collection.query(
    #         query_embeddings=[query_embedding],
    #         n_results=top_k,
    #         include=["documents", "metadatas", "distances"],
    #     )
    #     distances = results["distances"][0]
    #     scores    = [round(1 - d, 4) for d in distances]
    #     sources   = [m.get("source", "?") for m in results["metadatas"][0]]
    #     logger.debug("Search returned %d result(s) — scores: %s  sources: %s",
    #                  len(scores), scores, sources)

    #     return {
    #         "ids":       results["ids"][0],
    #         "documents": results["documents"][0],
    #         "metadatas": results["metadatas"][0],
    #         "distances": results["distances"][0],
    #     }

    def query(self, collection:str, query_embedding: list[float], top_k: int) -> dict:
        collection_name = config.TARGET[self.target]["collections"][collection]
        logger.debug("Vector search: top_k=%d  collection='%s'",
                top_k, collection_name)
        
        logger.info("Collection name for query is %s", collection_name)
        logger.info("#############################################")
        coll = self.collections[collection]
        logger.info("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        print("── self.collections ─────────────────────────────")
        print(coll)
        print("── self.collections ─────────────────────────────")
        results = coll.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        # print(results)
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


    # def count(self) -> int:
    #     return self.collection.count()

    def count(self, collection:str) -> int:
        return self.collections[collection].count()

    # def list_docs(self) -> list[dict]:
    #     """Return unique source documents (by metadata['source'])."""
    #     all_items = self.collection.get(include=["metadatas"])
    #     seen = set()
    #     docs = []
    #     for meta in all_items["metadatas"]:
    #         src = meta.get("source", "unknown")
    #         if src not in seen:
    #             seen.add(src)
    #             docs.append(meta)
    #     logger.debug("list_docs: %d unique source(s)", len(docs))
    #     return docs
    
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