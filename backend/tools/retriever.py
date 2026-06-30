"""
tools/retriever.py — ChromaDB Similarity Search

High-level retrieval interface used by agents.
Combines embedder + chroma_client into a single call.

Usage in agents:
    from tools.retriever import retrieve_context, retrieve_for_document

    # General knowledge base search (all docs)
    chunks = retrieve_context("What is the vacation policy?", n_results=5)

    # Search within a specific uploaded document
    chunks = retrieve_for_document("Summarize the key points", doc_id="uuid-here")
"""

from typing import Optional

from db.chroma_client import collection_count, query_chunks
from tools.embedder import embed_query


def retrieve_context(
    query: str,
    n_results: int = 5,
    min_relevance: float = 1.5,
) -> list[dict]:
    """
    Retrieve the most relevant chunks from the knowledge base for a query.

    Filters out chunks with cosine distance > min_relevance (very dissimilar).
    Cosine distance range: 0.0 (identical) to 2.0 (opposite).
    A threshold of 1.5 keeps anything at least vaguely relevant.

    Args:
        query:         User's question or search text
        n_results:     Max chunks to return
        min_relevance: Distance threshold — chunks further than this are dropped

    Returns:
        List of chunk dicts ordered by relevance (closest first).
        Empty list if no documents are indexed or no relevant chunks found.
    """
    if collection_count() == 0:
        return []  # no documents indexed yet — agents fall back to general knowledge

    query_vector = embed_query(query)
    chunks = query_chunks(query_vector, n_results=n_results)

    # Filter by relevance threshold
    relevant = [c for c in chunks if c["distance"] <= min_relevance]
    return relevant


def retrieve_for_document(
    query: str,
    doc_id: str,
    n_results: int = 8,
) -> list[dict]:
    """
    Retrieve chunks from a specific document only.

    Used by the Document Agent when the user asks about a specific uploaded file.

    Args:
        query:     Question or instruction about the document
        doc_id:    UUID of the target document
        n_results: Max chunks to return

    Returns:
        List of chunk dicts from that document ordered by relevance.
    """
    if collection_count() == 0:
        return []

    query_vector = embed_query(query)
    return query_chunks(query_vector, n_results=n_results, doc_id=doc_id)


def format_context_block(chunks: list[dict]) -> str:
    """
    Format retrieved chunks into a clean [CONTEXT] block for agent prompts.

    Example output:
        [SOURCE: company_handbook.pdf | Page 3]
        Employees are entitled to 20 days of paid vacation per year...

        [SOURCE: company_handbook.pdf | Page 7]
        Remote work requests must be submitted 48 hours in advance...

    Args:
        chunks: List of chunk dicts from retrieve_context()

    Returns:
        Formatted string to inject into an agent's prompt, or empty string if no chunks.
    """
    if not chunks:
        return ""

    lines = []
    for chunk in chunks:
        source_label = f"[SOURCE: {chunk['filename']} | Page {chunk['page_num']}]"
        lines.append(f"{source_label}\n{chunk['text']}")

    return "\n\n".join(lines)


def has_indexed_documents() -> bool:
    """True if at least one document chunk is stored in ChromaDB."""
    return collection_count() > 0
