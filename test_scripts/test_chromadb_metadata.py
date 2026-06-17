import chromadb

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config

client = chromadb.PersistentClient(path=config.VECTORSTORE_PATH)
coll = client.get_collection("corpus_complaints")
result = coll.get(limit=3, include=["metadatas", "documents"])
for doc, meta in zip(result["documents"], result["metadatas"]):
    print(meta)
    print(doc[:100])
    print("---")