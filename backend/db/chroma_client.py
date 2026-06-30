"""
db/chroma_client.py — ChromaDB Persistent Client for WorkPilot AI

Manages the single ChromaDB collection that stores all document embeddings.

Collection design:
    name:       workpilot_docs
    similarity: cosine
    dimension:  3072 (gemini-embedding-001)

Each stored chunk has:
    id:         "{doc_id}_{chunk_idx}"
    document:   raw chunk text
    embedding:  3072-dim float list from Gemini
    metadata:   doc_id, filename, page_num, chunk_idx, uploaded_at
"""

from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings

from config import settings

# ── Client & Collection ───────────────────────────────────────────────────────

_client: Optional[chromadb.PersistentClient] = None
_collection = None

COLLECTION_NAME = "workpilot_docs"


def get_client() -> chromadb.PersistentClient:
    """Return (or create) the singleton ChromaDB client."""
    global _client
    if _client is None:
        persist_path = Path(settings.CHROMA_PERSIST_PATH)
        persist_path.mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(
            path=str(persist_path),
            settings=Settings(anonymized_telemetry=False, allow_reset=True),
        )
    return _client


def get_collection():
    """Return (or create) the workpilot_docs collection."""
    global _collection
    if _collection is None:
        client = get_client()
        _collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},  # cosine similarity
        )
    return _collection


# ── Write Operations ──────────────────────────────────────────────────────────

def add_chunks(
    chunk_ids: list[str],
    texts: list[str],
    embeddings: list[list[float]],
    metadatas: list[dict],
) -> None:
    """
    Store a batch of text chunks with their embeddings in ChromaDB.

    Args:
        chunk_ids:  Unique IDs, e.g. ["docid_0", "docid_1", ...]
        texts:      Raw chunk text strings
        embeddings: Pre-computed 3072-dim float vectors (from gemini-embedding-001)
        metadatas:  Dicts with doc_id, filename, page_num, chunk_idx, etc.
    """
    collection = get_collection()
    collection.add(
        ids=chunk_ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
    )


def delete_document(doc_id: str) -> int:
    """Delete all chunks belonging to a document. Returns number deleted."""
    collection = get_collection()
    results = collection.get(where={"doc_id": doc_id})
    if results["ids"]:
        collection.delete(ids=results["ids"])
        return len(results["ids"])
    return 0


# ── Read Operations ───────────────────────────────────────────────────────────

def query_chunks(
    query_embedding: list[float],
    n_results: int = 5,
    doc_id: Optional[str] = None,
) -> list[dict]:
    """
    Find the most similar chunks to a query embedding.

    Args:
        query_embedding: 3072-dim vector from gemini-embedding-001
        n_results:       How many chunks to return
        doc_id:          If set, restrict search to this document only

    Returns:
        List of dicts: {text, doc_id, filename, page_num, chunk_idx, distance}
        Ordered by similarity (most similar first).
    """
    collection = get_collection()

    # Need at least n_results chunks in the collection
    total = collection.count()
    if total == 0:
        return []

    effective_n = min(n_results, total)

    kwargs: dict = {
        "query_embeddings": [query_embedding],
        "n_results": effective_n,
        "include": ["documents", "metadatas", "distances"],
    }
    if doc_id:
        kwargs["where"] = {"doc_id": doc_id}

    results = collection.query(**kwargs)

    chunks = []
    for text, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append(
            {
                "text": text,
                "doc_id": meta.get("doc_id", ""),
                "filename": meta.get("filename", ""),
                "page_num": meta.get("page_num", 0),
                "chunk_idx": meta.get("chunk_idx", 0),
                "distance": round(dist, 4),
            }
        )
    return chunks


def collection_count() -> int:
    """Total number of chunks stored across all documents."""
    return get_collection().count()
