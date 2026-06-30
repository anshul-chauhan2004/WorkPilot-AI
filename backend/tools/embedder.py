"""
tools/embedder.py — Gemini Text Embedding Wrapper

Wraps gemini-embedding-001 for two use cases:
    - RETRIEVAL_DOCUMENT: embed chunks when ingesting a PDF
    - RETRIEVAL_QUERY:    embed a user's question for similarity search

gemini-embedding-001 specs:
    Dimensions: 3072
    Max input tokens: ~2048
    Task types: RETRIEVAL_DOCUMENT, RETRIEVAL_QUERY, SEMANTIC_SIMILARITY

Performance:
    embed_chunks() uses a ThreadPoolExecutor with up to MAX_WORKERS concurrent
    API calls, reducing large document ingestion from O(n) sequential seconds
    to O(n/workers) — e.g. 100 chunks goes from ~70s to ~15s with 5 workers.
"""

import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Literal

import google.generativeai as genai

from config import settings

logger = logging.getLogger(__name__)

# Embedding model — gemini-embedding-001 is available on this API key (3072-dim)
EMBEDDING_MODEL = "models/gemini-embedding-001"

# Max concurrent embedding calls. Free tier allows ~1500 req/min, so 5 workers
# gives ~300 req/min peak — well within limits.
MAX_WORKERS = 5

TaskType = Literal["RETRIEVAL_DOCUMENT", "RETRIEVAL_QUERY", "SEMANTIC_SIMILARITY"]


def _configure() -> None:
    genai.configure(api_key=settings.GEMINI_API_KEY)


def embed_text(text: str, task_type: TaskType = "RETRIEVAL_QUERY") -> list[float]:
    """
    Embed a single string using gemini-embedding-001.

    Args:
        text:      Text to embed (max ~2048 tokens)
        task_type: "RETRIEVAL_QUERY" for user questions,
                   "RETRIEVAL_DOCUMENT" for storing chunks

    Returns:
        3072-dimensional float vector
    """
    _configure()
    result = genai.embed_content(
        model=EMBEDDING_MODEL,
        content=text,
        task_type=task_type,
    )
    return result["embedding"]


def _embed_one(args: tuple) -> tuple[int, list[float]]:
    """
    Worker function for ThreadPoolExecutor.
    Returns (original_index, embedding) so order is preserved.
    """
    idx, text, task_type = args
    _configure()
    result = genai.embed_content(
        model=EMBEDDING_MODEL,
        content=text,
        task_type=task_type,
    )
    return idx, result["embedding"]


def embed_chunks(
    texts: list[str],
    task_type: TaskType = "RETRIEVAL_DOCUMENT",
    delay_between_requests: float = 0.0,  # kept for API compat, no longer used
) -> list[list[float]]:
    """
    Embed a list of text chunks concurrently, returning one vector per chunk.

    Uses a ThreadPoolExecutor with MAX_WORKERS (5) concurrent API calls,
    which is ~5x faster than sequential for large documents.

    Args:
        texts:                    List of chunk strings to embed
        task_type:                "RETRIEVAL_DOCUMENT" for chunks being stored
        delay_between_requests:   Deprecated; kept for API compatibility

    Returns:
        List of 3072-dim vectors in the SAME ORDER as input texts
    """
    _configure()

    n = len(texts)
    if n == 0:
        return []

    # Small documents: just embed sequentially (no thread overhead)
    if n <= 3:
        return [embed_text(t, task_type) for t in texts]

    logger.info(f"Embedding {n} chunks with {MAX_WORKERS} concurrent workers…")
    start = time.time()

    results: list[list[float] | None] = [None] * n
    args_list = [(i, text, task_type) for i, text in enumerate(texts)]

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(_embed_one, args): args[0] for args in args_list}
        completed = 0
        for future in as_completed(futures):
            try:
                idx, embedding = future.result()
                results[idx] = embedding
                completed += 1
                if completed % 10 == 0:
                    logger.info(f"  Embedded {completed}/{n} chunks…")
            except Exception as e:
                original_idx = futures[future]
                logger.error(f"Embedding failed for chunk {original_idx}: {e}")
                raise

    elapsed = time.time() - start
    logger.info(f"✅ Embedded {n} chunks in {elapsed:.1f}s ({elapsed/n:.2f}s/chunk avg)")

    return results  # type: ignore[return-value]


def embed_query(query: str) -> list[float]:
    """Convenience wrapper: embed a user query for similarity search."""
    return embed_text(query, task_type="RETRIEVAL_QUERY")
