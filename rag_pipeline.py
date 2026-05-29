# rag_pipeline.py
"""
Full RAG pipeline.
retrieve() → build prompt → llm.generate()
"""

import logging
import time
import retriever
import llm


logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = """\
You are a helpful assistant. Answer the question using ONLY the context provided below.
If the context does not contain enough information, say "I don't have enough information to answer that."

## Context
{context}

## Question
{question}

## Answer
"""


def query(question: str, top_k: int = None) -> dict:
    """
    Run the full RAG pipeline.
    Returns:
      {
        "answer":   str,
        "sources":  list[dict],   # the retrieved chunks
        "question": str,
      }
    """
    logger.info("RAG query: '%s'  top_k=%s", question[:80], top_k or "default")
    start = time.perf_counter()

    # Step 1: Retrieve relevant chunks
    chunks = retriever.retrieve(question, top_k=top_k)

    if not chunks:
        return {
            "answer":   "No relevant documents found in the knowledge base.",
            "sources":  [],
            "question": question,
        }

    # Step 2: Build context string
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        context_parts.append(
            f"[{i}] Source: {chunk['source']} (relevance: {chunk['score']})\n{chunk['text']}"
        )
    context = "\n\n---\n\n".join(context_parts)
    logger.debug("Prompt context built from %d chunk(s), total %d chars",
                 len(chunks), len(context))

    # Step 3: Fill prompt template
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)
    logger.debug("Full prompt length: %d chars", len(prompt))

    # Step 4: Generate answer
    answer = llm.generate(prompt)

    elapsed = time.perf_counter() - start
    logger.info("RAG query complete in %.2fs — answer_len=%d chars",
                elapsed, len(answer))


    return {
        "answer":   answer,
        "sources":  chunks,
        "question": question,
    }