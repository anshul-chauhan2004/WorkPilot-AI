"""
schemas/document.py — Pydantic models for the /documents API
"""

from typing import Optional
from pydantic import BaseModel


class IngestResponse(BaseModel):
    doc_id: str
    filename: str
    page_count: int
    chunk_count: int
    message: str


class DocumentInfo(BaseModel):
    doc_id: str
    filename: str
    file_size_bytes: int
    page_count: int
    chunk_count: int
    status: str          # "indexed" | "failed"
    uploaded_at: str


class DocumentListResponse(BaseModel):
    documents: list[DocumentInfo]
    total: int
