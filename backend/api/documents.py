"""
api/documents.py — Document Ingestion & Listing API

Endpoints:
    POST /documents/ingest   Upload a PDF → parse → embed → store in ChromaDB
    GET  /documents          List all indexed documents

Ingestion pipeline (per request):
    1. Receive multipart PDF upload
    2. Save raw file to uploads/ with UUID filename
    3. Parse PDF into text chunks (PyMuPDF)
    4. Embed all chunks (Gemini text-embedding-004) — sequential with rate-limit delay
    5. Store chunks + embeddings in ChromaDB
    6. Save document metadata to SQLite
    7. Return IngestResponse with doc_id and chunk count
"""

import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.responses import JSONResponse

from config import settings
from db.sqlite_client import save_document, list_documents, get_document, delete_document_record
from db.chroma_client import add_chunks, delete_document
from tools.pdf_parser import parse_pdf, get_page_count
from tools.embedder import embed_chunks
from schemas.document import IngestResponse, DocumentInfo, DocumentListResponse

router = APIRouter()


# ── POST /documents/ingest ────────────────────────────────────────────────────

@router.post(
    "/documents/ingest",
    response_model=IngestResponse,
    summary="Upload and index a PDF document",
    description=(
        "Uploads a PDF, parses it into chunks, embeds each chunk with Gemini "
        "text-embedding-004, and stores everything in ChromaDB. "
        "After ingestion, the Knowledge Agent will use this document for RAG."
    ),
)
async def ingest_document(file: UploadFile = File(...)) -> IngestResponse:
    # ── Validate file type ────────────────────────────────────────────────────
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    doc_id = str(uuid.uuid4())
    original_filename = file.filename

    # ── Save raw file to uploads/ ─────────────────────────────────────────────
    uploads_dir = Path(settings.UPLOADS_PATH)
    uploads_dir.mkdir(parents=True, exist_ok=True)
    saved_filename = f"{doc_id}.pdf"
    file_path = uploads_dir / saved_filename

    try:
        content = await file.read()
        file_size_bytes = len(content)
        with open(file_path, "wb") as f:
            f.write(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")

    # ── Parse PDF into chunks ─────────────────────────────────────────────────
    try:
        chunks = parse_pdf(
            file_path=str(file_path),
            doc_id=doc_id,
            filename=original_filename,
        )
        page_count = get_page_count(str(file_path))
    except ValueError as e:
        file_path.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        file_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"PDF parsing failed: {e}")

    chunk_count = len(chunks)

    # ── Embed all chunks (Gemini text-embedding-004) ──────────────────────────
    try:
        texts = [c["text"] for c in chunks]
        embeddings = embed_chunks(texts, task_type="RETRIEVAL_DOCUMENT")
    except Exception as e:
        file_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Embedding failed: {e}")

    # ── Store in ChromaDB ─────────────────────────────────────────────────────
    try:
        add_chunks(
            chunk_ids=[c["chunk_id"] for c in chunks],
            texts=texts,
            embeddings=embeddings,
            metadatas=[
                {
                    "doc_id": c["doc_id"],
                    "filename": c["filename"],
                    "page_num": c["page_num"],
                    "chunk_idx": c["chunk_idx"],
                }
                for c in chunks
            ],
        )
    except Exception as e:
        file_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Vector store write failed: {e}")

    # ── Save metadata to SQLite ───────────────────────────────────────────────
    try:
        save_document(
            doc_id=doc_id,
            filename=original_filename,
            file_path=str(file_path),
            file_size_bytes=file_size_bytes,
            page_count=page_count,
            chunk_count=chunk_count,
            status="indexed",
        )
    except Exception as e:
        # Non-fatal: ChromaDB already has the data; log but continue
        print(f"Warning: failed to save document metadata to SQLite: {e}")

    return IngestResponse(
        doc_id=doc_id,
        filename=original_filename,
        page_count=page_count,
        chunk_count=chunk_count,
        message=f"Successfully indexed '{original_filename}' — {chunk_count} chunks from {page_count} pages ready for RAG.",
    )


# ── GET /documents ────────────────────────────────────────────────────────────

@router.get(
    "/documents",
    response_model=DocumentListResponse,
    summary="List all indexed documents",
)
async def get_documents() -> DocumentListResponse:
    docs = list_documents()
    return DocumentListResponse(
        documents=[DocumentInfo(**d) for d in docs],
        total=len(docs),
    )


# ── GET /documents/{doc_id} ───────────────────────────────────────────────────

@router.get(
    "/documents/{doc_id}",
    response_model=DocumentInfo,
    summary="Get metadata for a specific document",
)
async def get_document_detail(doc_id: str) -> DocumentInfo:
    doc = get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found.")
    return DocumentInfo(**doc)


# ── DELETE /documents/{doc_id} ────────────────────────────────────────────────

@router.delete(
    "/documents/{doc_id}",
    summary="Delete a document and remove it from the vector store",
    description=(
        "Removes all ChromaDB vectors for this document, deletes the physical PDF "
        "from uploads/, and removes the SQLite metadata record."
    ),
)
async def delete_document_endpoint(doc_id: str):
    # Verify it exists
    doc = get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found.")

    errors = []

    # 1. Remove vectors from ChromaDB
    try:
        deleted_chunks = delete_document(doc_id)
    except Exception as e:
        errors.append(f"ChromaDB: {e}")
        deleted_chunks = 0

    # 2. Delete the physical PDF file
    try:
        file_path = Path(doc.get("file_path", ""))
        if file_path.exists():
            file_path.unlink()
    except Exception as e:
        errors.append(f"File: {e}")

    # 3. Delete SQLite record
    try:
        delete_document_record(doc_id)
    except Exception as e:
        errors.append(f"SQLite: {e}")

    return {
        "deleted": True,
        "doc_id": doc_id,
        "filename": doc["filename"],
        "chunks_removed": deleted_chunks,
        "warnings": errors,
    }
