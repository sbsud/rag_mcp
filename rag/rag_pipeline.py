# rag_pipeline.py
"""
Full RAG pipeline.
retrieve() → build prompt → llm.generate()
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import logging
import time
from store import retriever
from llm import llm


logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = """\
You are a helpful assistant. Answer the question using ONLY the context provided below.
If a retrieved clause is reasonably applicable to the question — even if it does not name the exact product or complaint type — cite it and explain how it applies.
Only say "I don't have enough information" if no retrieved clause has any reasonable relevance to the question.
Do not invent clauses not present in the context.


## Context
{context}

## Question
{question}

## Answer
"""




def query(corpus: str, question: str, top_k: int = None, where: dict = None) -> dict:
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
    chunks = retriever.retrieve(corpus, question, top_k=top_k, where=where)

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
    logger.debug(answer)
    elapsed = time.perf_counter() - start
    logger.info("RAG query complete in %.2fs — answer_len=%d chars",
                elapsed, len(answer))


    return {
        "answer":   answer,
        "sources":  chunks,
        "question": question,
    }