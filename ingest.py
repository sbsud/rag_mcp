# ingest.py
"""
Ingestion CLI.
Usage:
  python ingest.py path/to/file.txt
  python ingest.py path/to/dir/       # ingests all .txt and .pdf files
  python ingest.py --clear            # wipe the vector store
"""

import sys
import uuid
from pathlib import Path
import embedder
import vectorstore
import config


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    """Split text into overlapping windows."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start += size - overlap
    return chunks


# def load_file(path: Path) -> str:
#     if path.suffix.lower() == ".pdf":
#         from PyPDF2 import PdfReader
#         reader = PdfReader(str(path))
#         return "\n".join(page.extract_text() or "" for page in reader.pages)
#     else:
#         return path.read_text(encoding="utf-8", errors="replace")

def load_file(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        return _load_pdf(path)
    else:
        return path.read_text(encoding="utf-8", errors="replace")


def _load_pdf(path: Path) -> str:
    # Option 1: pdfminer — handles layout much better than PyPDF2
    # pip install pdfminer.six
    from pdfminer.high_level import extract_text
    return extract_text(str(path))

def ingest_file(path: Path) -> int:
    """Ingest a single file. Returns number of chunks added."""
    print(f"  Loading: {path.name}")
    text = load_file(path)
    text = text.strip()
    if not text:
        print(f"  ⚠ Empty file, skipping.")
        return 0

    chunks = chunk_text(text, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
    print(f"  Chunking: {len(chunks)} chunks (size={config.CHUNK_SIZE}, overlap={config.CHUNK_OVERLAP})")

    # Embed all chunks in one batch (faster than one-by-one)
    print(f"  Embedding with [{config.EMBED_PROVIDER}:{config.EMBED_MODEL}] ...")
    embeddings = embedder.embed(chunks)

    # Build metadata for each chunk
    ids       = [str(uuid.uuid4()) for _ in chunks]
    metadatas = [
        {"source": path.name, "chunk_id": str(i), "path": str(path)}
        for i in range(len(chunks))
    ]

    store = vectorstore.get_store()
    store.add(ids=ids, embeddings=embeddings, documents=chunks, metadatas=metadatas)
    print(f"  ✓ Added {len(chunks)} chunks from {path.name}")
    return len(chunks)


def main():
    args = sys.argv[1:]
    if not args:
        print("Usage: python ingest.py <file_or_dir> [--clear]")
        sys.exit(1)

    if "--clear" in args:
        import chromadb, os
        client = chromadb.PersistentClient(path=config.VECTORSTORE_PATH)
        client.delete_collection(config.VECTORSTORE_COLLECTION)
        print("✓ Vector store cleared.")
        return

    path = Path(args[0])
    files = list(path.rglob("*.txt")) + list(path.rglob("*.pdf")) \
        if path.is_dir() else [path]

    total = 0
    for f in files:
        total += ingest_file(f)

    store = vectorstore.get_store()
    print(f"\n✓ Done. Total chunks in store: {store.count()}")


if __name__ == "__main__":
    main()