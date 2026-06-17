import chromadb
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from  rag import rag_pipeline

# client = chromadb.PersistentClient(path=config.VECTORSTORE_PATH)
# coll = client.get_collection("corpus_complaints")
# print(f"Total chunks: {coll.count()}")

# print("########################################")

# result = coll.get(limit=5, include=["metadatas", "documents"])
# for doc, meta in zip(result["documents"], result["metadatas"]):
#     print(meta)
#     print(doc[:150])
#     print("---")


# print("#######################################")

# from store import embedder, vectorstore

# query = "washing machine stopped working complaints"
# [query_vec] = embedder.embed([query])

# store = vectorstore.get_store()
# results = store.query("complaints", query_vec, top_k=5)

# for doc, meta, dist in zip(results["documents"], results["metadatas"], results["distances"]):
#     print(f"score={round(1-dist, 4)}  brand={meta.get('brand')}  rating={meta.get('rating')}")
#     print(doc[:150])
#     print("---")

# print("#######################################")

# sample = coll.get(limit=5, include=["metadatas"])
# for m in sample["metadatas"]:
#     print(m.get("rating"), type(float(m.get("rating"))))

# sample = coll.get(limit=20, include=["metadatas"])
# ratings = [float(m.get("rating")) for m in sample["metadatas"]]
# print(set(ratings))

# filtered = coll.query(
#     query_embeddings=[query_vec],
#     n_results=5,
#     where={"rating": {"$lte": 4.0}},
# )
# # print("filtered returned rows %d", len(filtered))
# # print(filtered["metadatas"])
# for meta in filtered["metadatas"][0]:
#     print(meta.get("brand"), meta.get("rating"))

# from collections import Counter

# ratings = []
# batch_size = 5000
# offset = 0
# total = coll.count()

# while offset < total:
#     batch = coll.get(limit=batch_size, offset=offset, include=["metadatas"])
#     ratings.extend(m.get("rating") for m in batch["metadatas"])
#     offset += batch_size

# # print(Counter(ratings))

# print(rag_pipeline.query("complaints", question="what are 1-star complaints about? List the top 3 brands.")["answer"])
# print("-----------------------------------")
# # print(rag_pipeline.query("complaints", question="What do reviews complain about most? List the top 3 brands.",top_k=20, where={"rating": 2.0})["answer"])


# import chromadb, config
# client = chromadb.PersistentClient(path=config.VECTORSTORE_PATH)
# coll = client.get_collection("corpus_complaints")
# # result = coll.get(where={"brand": "GE APPLIANCES", "rating": 1.0}, include=["metadatas"])
# # for m in result["metadatas"]:
# #     print(m.get("total_complaints"))

# result = coll.get(
#     where={"$and": [{"brand": "GE APPLIANCES"}, {"rating": 1.0}]},
#     include=["metadatas"]
# )
# for m in result["metadatas"]:
#     print(m.get("total_complaints"))


# print("______________________________")
# result = coll.get(
#     where={"$and": [{"brand": "GE APPLIANCES"}, {"rating": 1.0}]},
#     include=["metadatas", "documents"]
# )
# for m, d in zip(result["metadatas"], result["documents"]):
#     print(m.get("category"), "-", m.get("total_complaints"))
#     print(d[:100])
#     print("---")

print("_____________________________")
result = rag_pipeline.query("policy", "defective appliances resolution")
for s in result["sources"]:
    print(s["text"][:150])