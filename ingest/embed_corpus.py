# ingest/embed_corpus.py
"""
One-time embedding precomputation.
Reads a JSONL corpus, embeds all texts, saves vectors as .npy.
Run this ONCE per corpus version. Re-run only if the corpus text changes.

Usage:
  python ingest/embed_corpus.py --path data_dump/amazon_Appliances_complaints.jsonl
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import logging
import time
from pathlib import Path
import numpy as np

from store import embedder
import config
from logging_config import setup_logging

logger = None


def load_texts(path: Path) -> list[str]:
    texts = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            texts.append(record["text"])
    return texts


def embed_in_batches(texts: list[str], batch_size: int = 256) -> list[list[float]]:
    all_embeddings = []
    total_batches = (len(texts) + batch_size - 1) // batch_size
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        batch_num = i // batch_size + 1
        logger.info("Embedding batch %d/%d", batch_num, total_batches)
        all_embeddings.extend(embedder.embed(batch))
    return all_embeddings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True)
    args = parser.parse_args()

    setup_logging()
    global logger
    logger = logging.getLogger(__name__)

    path = Path(args.path)
    out_path = path.with_suffix(".npy")

    logger.info("Loading texts from %s", path)
    texts = load_texts(path)
    logger.info("Loaded %d texts", len(texts))

    start = time.perf_counter()
    embeddings = embed_in_batches(texts, batch_size=256)
    elapsed = time.perf_counter() - start
    logger.info("Embedded %d texts in %.1fs", len(embeddings), elapsed)

    arr = np.array(embeddings, dtype=np.float32)
    np.save(out_path, arr)
    logger.info("Saved embeddings to %s  shape=%s", out_path, arr.shape)


if __name__ == "__main__":
    main()