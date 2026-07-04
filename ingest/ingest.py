# ingest.py
"""
Ingestion CLI.
Usage:
  python ingest.py path/to/file.txt
  python ingest.py path/to/dir/       # ingests all .txt and .pdf files
  python ingest.py --clear            # wipe the vector store
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import json
import logging
import time
import sys
import uuid
import numpy as np
from pathlib import Path
from store import embedder
from store import vectorstore
import config
from logging_config import setup_logging

logger = None



def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    """Split text into overlapping windows."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start += size - overlap
    logger.debug("chunk_text: input=%d chars → %d chunks (size=%d overlap=%d)",
                 len(text), len(chunks), size, overlap)

    return chunks


def chunk_record(record: dict, size: int, overlap:int) -> list[dict]:
    
    text = record["text"]
    metadata = record["metadata"]

    if len(text) <= size:
        return [{"text": text, "metadata": metadata}]
    
    sub_texts = chunk_text(text=text, size=size, overlap=overlap)
    logger.debug("Record (%d chars, category=%s) split into %d sub-chunks",
                  len(text), metadata.get("category", "?"), len(sub_texts))

    return [{"text": t, "metadata": metadata} for t in sub_texts]


def load_jsonl_records(path: Path) -> list[dict]:
    records = []

    with open(path, encoding="utf8") as file:
        for line_num, line in enumerate(file, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except Exception as e:
                logger.error("Bad JSON on line %d of %s: %s", line_num, path.name, e)
    
    logger.debug("load_jsonl_records: %s -> %d record(s)", path.name, len(records))
    return records



def load_file(path: Path) -> str:
    logger.debug("load_file: %s  suffix=%s", path.name, path.suffix)
    if path.suffix.lower() == ".pdf":
        return _load_pdf(path)

    text = path.read_text(encoding="utf-8", errors="replace")
    logger.debug("Loaded plain text: %d chars from %s", len(text), path.name)
    return text


def _load_pdf(path: Path) -> str:
    # Option 1: pdfminer — handles layout much better than PyPDF2
    # pip install pdfminer.six
    logger.info("Extracting PDF text: %s", path.name)
    try:

        from pdfminer.high_level import extract_text
        text = extract_text(str(path))
        logger.info("PDF extraction complete: %s → %d chars", path.name, len(text))
        return text

    except Exception as e:
        logger.error("PDF extraction failed for %s: %s", path.name, e)
        raise


def embed_in_batches(documents: list[str], batch_size: int = 256) -> list[list[float]]:
    all_embeddings = []
    for i in range(0, len(documents), batch_size):
        batch = documents[i : i + batch_size]
        logger.info("Embedding batch %d/%d", i // batch_size + 1, 
                    (len(documents) + batch_size - 1) // batch_size)
        all_embeddings.extend(embedder.embed(batch))
    return all_embeddings

def ingest_jsonl_file(collection: str, path: Path) -> int:
    logger.info("── Ingesting JSONL: %s", path.name)
    start = time.perf_counter()

    records = load_jsonl_records(path)

    documents = []
    metadatas = []
    if not records:
        logger.warning("No records found in %s — skipping", path.name)
        return 0

    for record_idx, record in enumerate(records):
        pieces = chunk_record(record, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
        for sub_idx, peice in enumerate(pieces):
            documents.append(peice["text"])
            meta = dict(peice["metadata"])    
            meta["source"] = path.name
            meta["path"] = str(path)
            meta["chunk_id"] = f"{record_idx}_{sub_idx}"
            metadatas.append(meta)

    logger.info("'%s': %d record(s) -> %d chunk(s)", path.name, len(records), len(documents))

    npy_path = path.with_suffix(".npy")
    print(npy_path)
    if npy_path.exists():
        logger.info("Loading pre-computed embeddings from %s", npy_path)
        embeddings = np.load(npy_path).tolist()
        if len(embeddings) != len(documents):
            logger.error(
                "Embedding count (%d) != document count (%d) — corpus changed since "
                "embed_corpus.py was run. Re-run embed_corpus.py, or delete the .npy "
                "to fall back to live embedding.",
                len(embeddings), len(documents)
            )
            raise ValueError("Stale .npy file — embedding/document count mismatch")
    else:
        logger.info("No .npy found — embedding live (slow path)")
        embeddings = embedder.embed(documents)        



    # embeddings = embedder.embed(documents)
    # embeddings = embed_in_batches(documents=documents)
    ids = [str(uuid.uuid4()) for _ in documents]

    store = vectorstore.get_store()
    store.add(collection=collection, ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)

    elapsed = time.perf_counter() - start
    logger.info("Ingested '%s': %d chunk(s) in %.2fs", path.name, len(documents), elapsed)

    return len(documents)

def ingest_file(collection: str, path:Path) -> int:
    if path.suffix.lower() == ".jsonl":
        return ingest_jsonl_file(collection, path)
    else:
        return ingest_text_file(collection, path)    


def ingest_text_file(collection: str, path: Path) -> int:
    """Ingest a single file. Returns number of chunks added."""
    logger.info("── Ingesting: %s", path.name)
    start = time.perf_counter()

    text = load_file(path)
    text = text.strip()
    if not text:
        logger.warning("Empty content after loading %s — skipping", path.name)
        return 0

    chunks = chunk_text(text, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
    logger.info("Chunked '%s' into %d chunk(s)", path.name, len(chunks))

    # Embed all chunks in one batch (faster than one-by-one)
    logger.info("Embedding %d chunk(s) with [%s:%s]...",
                len(chunks), config.EMBED_PROVIDER, config.EMBED_MODEL)
    embeddings = embedder.embed(chunks)

    # Build metadata for each chunk
    ids       = [str(uuid.uuid4()) for _ in chunks]
    metadatas = [
        {"source": path.name, "chunk_id": str(i), "path": str(path)}
        for i in range(len(chunks))
    ]

    store = vectorstore.get_store()
    store.add(collection=collection, ids=ids, embeddings=embeddings, documents=chunks, metadatas=metadatas)

    elapsed = time.perf_counter() - start
    logger.info("Ingested '%s': %d chunks in %.2fs", path.name, len(chunks), elapsed)

    return len(chunks)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    parser.add_argument("--collection", required=True)
    parser.add_argument("--path", required=True)
    args = parser.parse_args()

    # if args.collection not in config.TARGET[args.target]["collections"]:
    #     parser.error(f"argument --collection: invalid choice: '{args.collection}' (choose from [{", ".join(config.TARGET[args.target]["collections"])}] when --target is '{args.target}')")

    setup_logging()
    global logger 
    logger = logging.getLogger(__name__)

    path = Path(args.path)
    files = list(path.rglob("*.txt")) + list(path.rglob("*.pdf")) + list(path.rglob("*.jsonl")) \
        if path.is_dir() else [path]
    logger.info("Found %d file(s) to ingest in '%s'", len(files), path)

    total = 0
    for f in files:
        try:
            total += ingest_file(args.collection, f)
        except Exception as e:
            logger.error("Failed to ingest %s: %s", f.name, e, exc_info=True)


    logger.info("Ingestion complete — %d total chunks in collection %s.", total, args.collection)

    # target_config = config.TARGET[args.target]


if __name__ == "__main__":
    main()