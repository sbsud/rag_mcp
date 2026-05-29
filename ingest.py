# ingest.py
"""
Ingestion CLI.
Usage:
  python ingest.py path/to/file.txt
  python ingest.py path/to/dir/       # ingests all .txt and .pdf files
  python ingest.py --clear            # wipe the vector store
"""
import logging
import time
import sys
import uuid
from pathlib import Path
import embedder
import vectorstore
import config
from logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


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


# def load_file(path: Path) -> str:
#     if path.suffix.lower() == ".pdf":
#         from PyPDF2 import PdfReader
#         reader = PdfReader(str(path))
#         return "\n".join(page.extract_text() or "" for page in reader.pages)
#     else:
#         return path.read_text(encoding="utf-8", errors="replace")

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


def ingest_file(path: Path) -> int:
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
    store.add(ids=ids, embeddings=embeddings, documents=chunks, metadatas=metadatas)

    elapsed = time.perf_counter() - start
    logger.info("Ingested '%s': %d chunks in %.2fs", path.name, len(chunks), elapsed)

    return len(chunks)


def main():
    args = sys.argv[1:]
    if not args:
        logger.error("Usage: python ingest.py <file_or_dir> [--clear]")
        sys.exit(1)

    if "--clear" in args:
        logger.info("Clearing vector store collection '%s'",
                    config.VECTORSTORE_COLLECTION)

        import chromadb, os
        client = chromadb.PersistentClient(path=config.VECTORSTORE_PATH)
        client.delete_collection(config.VECTORSTORE_COLLECTION)
        logger.info("Vector store cleared")
        return

    path = Path(args[0])
    files = list(path.rglob("*.txt")) + list(path.rglob("*.pdf")) \
        if path.is_dir() else [path]
    
    logger.info("Found %d file(s) to ingest in '%s'", len(files), path)
    
    total = 0
    for f in files:
        try:
            total += ingest_file(f)
        except Exception as e:
            logger.error("Failed to ingest %s: %s", f.name, e, exc_info=True)

    # store = vectorstore.get_store()
    logger.info("Ingestion complete — %d total chunks in store", total)

if __name__ == "__main__":
    main()