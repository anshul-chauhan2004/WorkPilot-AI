"""
tools/pdf_parser.py — PDF to Chunks Pipeline

Converts a PDF file into a list of text chunks suitable for embedding.

Chunking strategy:
    1. Extract text per-page using PyMuPDF (fitz)
    2. Pages with > MAX_CHUNK_CHARS are split into overlapping sub-chunks
    3. Each chunk carries full provenance metadata (doc_id, filename, page, idx)

Chunk sizes are tuned for Gemini text-embedding-004's optimal input range (~500-800 chars).
"""

import re
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF

MAX_CHUNK_CHARS = 800    # target chunk size
OVERLAP_CHARS = 100      # overlap between consecutive chunks on the same page


def _clean_text(text: str) -> str:
    """Normalize whitespace and remove junk characters from extracted PDF text."""
    # Collapse multiple whitespace/newlines into single space
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def _split_long_text(text: str, max_chars: int = MAX_CHUNK_CHARS, overlap: int = OVERLAP_CHARS) -> list[str]:
    """
    Split a long string into overlapping chunks of at most max_chars characters.
    Tries to split at sentence boundaries ('. ') first, then falls back to hard split.
    """
    if len(text) <= max_chars:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + max_chars

        if end >= len(text):
            chunks.append(text[start:].strip())
            break

        # Try to find a sentence boundary near the end of the window
        split_at = text.rfind(". ", start, end)
        
        # If no period is found, or it's so close to the start that subtracting 
        # the overlap would cause an infinite loop, fall back to a hard split.
        if split_at == -1 or (split_at + 1 - overlap) <= start:
            split_at = end  # hard split fallback

        chunk = text[start : split_at + 1].strip()
        if chunk:
            chunks.append(chunk)

        # Step back by overlap, guaranteed to advance forward now
        start = split_at + 1 - overlap

    return [c for c in chunks if len(c) > 30]  # discard tiny fragments


def parse_pdf(
    file_path: str,
    doc_id: str,
    filename: str,
) -> list[dict]:
    """
    Parse a PDF file into a list of chunk dicts ready for embedding.

    Args:
        file_path: Absolute path to the PDF on disk
        doc_id:    UUID for this document (used as ChromaDB metadata)
        filename:  Original upload filename shown to users

    Returns:
        List of chunk dicts:
        {
            "chunk_id":  str   — unique ID: "{doc_id}_{chunk_idx}"
            "text":      str   — chunk text content
            "doc_id":    str
            "filename":  str
            "page_num":  int   — 1-indexed page number
            "chunk_idx": int   — 0-indexed position across all chunks
        }

    Raises:
        ValueError: if the file is not a valid PDF or produces no text
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {file_path}")

    chunks: list[dict] = []
    global_chunk_idx = 0

    try:
        doc = fitz.open(str(path))
    except Exception as e:
        raise ValueError(f"Could not open PDF: {e}")

    for page_num in range(len(doc)):
        page = doc[page_num]
        raw_text = page.get_text("text")
        cleaned = _clean_text(raw_text)

        if not cleaned or len(cleaned) < 30:
            continue  # skip blank/near-blank pages

        sub_chunks = _split_long_text(cleaned)

        for sub_text in sub_chunks:
            chunks.append(
                {
                    "chunk_id": f"{doc_id}_{global_chunk_idx}",
                    "text": sub_text,
                    "doc_id": doc_id,
                    "filename": filename,
                    "page_num": page_num + 1,  # 1-indexed
                    "chunk_idx": global_chunk_idx,
                }
            )
            global_chunk_idx += 1

    doc.close()

    if not chunks:
        raise ValueError("PDF produced no extractable text. It may be a scanned image PDF.")

    return chunks


def get_page_count(file_path: str) -> int:
    """Return the number of pages in a PDF without full parsing."""
    doc = fitz.open(file_path)
    count = len(doc)
    doc.close()
    return count
