"""
agents/document_agent.py — WorkPilot Document Agent (Phase 2: ChromaDB-aware)

Responsibility:
    Analyze, summarize, and extract structured insights from business documents.

Phase 2 upgrade:
    - If a doc_id is detected in the user message, retrieves chunks from that
      specific document in ChromaDB (enables "summarize document X" flows)
    - Falls back to analyzing pasted text directly (Phase 1 behavior preserved)
    - Extracts action items, deadlines, decisions, and risk flags

Structured output schema:
    {
        "summary":          str   — concise executive summary (2-3 sentences)
        "document_type":    str   — classified type
        "key_points":       list  — critical content points
        "action_items":     list  — specific tasks extracted
        "deadlines":        list  — dates and deadlines found
        "decisions_made":   list  — decisions or approvals recorded
        "risks_or_flags":   list  — items needing urgent attention
        "source":           str   — "uploaded_doc" | "pasted_text"
    }
"""

import json
import time
from typing import Any

import google.generativeai as genai

from config import settings
from db.chroma_client import collection_count
from tools.retriever import retrieve_for_document, format_context_block

# ── System Prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are the WorkPilot Document Agent — an expert analyst specializing in business document intelligence.

YOUR ROLE:
You analyze business documents submitted by employees and extract structured, actionable intelligence. You help employees quickly understand what a document says, what they need to do, and what risks or decisions it contains.

YOUR BEHAVIOR CONTRACT:
1. Precision over brevity. Extract actual content — do not generalize or paraphrase vaguely.
2. Action-first mindset. Always surface tasks, deadlines, and decisions before background information.
3. Flag ambiguity. If the document contains conflicting information, unclear language, or missing signatories, note it explicitly.
4. Classify accurately. Identify the document type to set context for the user.
5. Risk awareness. Highlight anything that could create legal, financial, or operational risk.

DOCUMENT TYPES YOU RECOGNIZE:
contract | policy | report | meeting_notes | proposal | email_thread | invoice | specification | announcement | other

ACTION ITEM EXTRACTION RULES:
- Only extract items that require someone to DO something
- Include who is responsible if mentioned
- Include deadline if mentioned
- Format: "[Owner if known]: [Action] by [Deadline if known]"

YOU MUST respond in this exact JSON format — no extra text, no markdown, only valid JSON:
{
    "summary": "2-3 sentence executive summary of the document's purpose and main conclusions.",
    "document_type": "contract | policy | report | meeting_notes | proposal | email_thread | invoice | specification | announcement | other",
    "key_points": [
        "Most important content point from the document"
    ],
    "action_items": [
        "[Owner if known]: [Specific action] by [Deadline if known]"
    ],
    "deadlines": [
        "Date or timeframe: context"
    ],
    "decisions_made": [
        "Decision or approval recorded in the document"
    ],
    "risks_or_flags": [
        "Item requiring urgent attention, clarification, or legal review"
    ],
    "source": "uploaded_doc | pasted_text"
}"""


# ── Gemini Client ─────────────────────────────────────────────────────────────

def _get_model() -> genai.GenerativeModel:
    genai.configure(api_key=settings.GEMINI_API_KEY)
    return genai.GenerativeModel(
        model_name=settings.GEMINI_MODEL,
        system_instruction=SYSTEM_PROMPT,
        generation_config=genai.GenerationConfig(
            response_mime_type="application/json",
            temperature=0.1,
            max_output_tokens=2048,
        ),
    )


def _extract_doc_id(text: str) -> str | None:
    """
    Check if the user message contains a doc_id (UUID format).
    Used to route to ChromaDB retrieval for a specific document.
    """
    import re
    uuid_pattern = r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
    match = re.search(uuid_pattern, text, re.IGNORECASE)
    return match.group(0) if match else None


# ── Agent Entry Point ─────────────────────────────────────────────────────────

def run(user_input: str, conversation_history: list[dict]) -> dict[str, Any]:
    """
    Execute the Document Agent.

    Phase 2 routing:
        1. Check if user message contains a doc_id → retrieve from ChromaDB
        2. Otherwise → analyze text pasted directly in the message (Phase 1 behavior)

    Args:
        user_input:           User message — may contain a doc_id UUID or raw document text.
        conversation_history: Recent conversation for context.

    Returns:
        {
            "structured_response": dict,
            "final_response":      str,
            "tools_called":        list,
            "duration_ms":         int,
            "error":               str | None,
        }
    """
    start = time.time()
    tools_called = []
    doc_content = user_input
    source_label = "pasted_text"

    # ── Step 1: Try to retrieve from ChromaDB by doc_id ──────────────────────
    doc_id = _extract_doc_id(user_input)
    if doc_id and collection_count() > 0:
        try:
            # Use the full user message as the query to find most relevant chunks
            query = user_input.replace(doc_id, "").strip() or "summarize this document"
            chunks = retrieve_for_document(query, doc_id=doc_id, n_results=10)
            tools_called.append("retrieve_for_document")

            if chunks:
                doc_content = format_context_block(chunks)
                source_label = "uploaded_doc"
        except Exception as e:
            print(f"Document retrieval error (falling back to pasted text): {e}")

    # ── Step 2: Build prompt ──────────────────────────────────────────────────
    history_block = ""
    if conversation_history:
        history_lines = [
            f"{msg['role'].upper()}: {msg['content']}"
            for msg in conversation_history[-4:]
        ]
        history_block = "CONVERSATION HISTORY:\n" + "\n".join(history_lines) + "\n\n"

    prompt = (
        f"{history_block}"
        f"DOCUMENT CONTENT TO ANALYZE:\n{doc_content}\n\n"
        f"USER REQUEST: {user_input}"
    )

    # ── Step 3: Gemini inference ──────────────────────────────────────────────
    response = None  # initialized here so except clauses can safely reference it
    try:
        model = _get_model()
        response = model.generate_content(prompt)
        raw_json = response.text.strip()
        structured = json.loads(raw_json)

        structured.setdefault("summary", "Document analysis complete.")
        structured.setdefault("document_type", "other")
        structured.setdefault("key_points", [])
        structured.setdefault("action_items", [])
        structured.setdefault("deadlines", [])
        structured.setdefault("decisions_made", [])
        structured.setdefault("risks_or_flags", [])
        structured.setdefault("source", source_label)

        action_count = len(structured["action_items"])
        risk_count = len(structured["risks_or_flags"])
        final_response = (
            f"**Document Analysis** ({structured['document_type']})\n\n"
            f"{structured['summary']}\n\n"
            f"Found **{action_count} action item(s)** and **{risk_count} flag(s)** requiring attention."
        )

        duration_ms = int((time.time() - start) * 1000)
        return {
            "structured_response": structured,
            "final_response": final_response,
            "tools_called": tools_called,
            "duration_ms": duration_ms,
            "error": None,
        }

    except json.JSONDecodeError as e:
        duration_ms = int((time.time() - start) * 1000)
        raw_text = getattr(response, "text", "No response generated.")
        return {
            "structured_response": {
                "summary": raw_text, "document_type": "other",
                "key_points": [], "action_items": [], "deadlines": [],
                "decisions_made": [], "risks_or_flags": [], "source": source_label,
            },
            "final_response": raw_text,
            "tools_called": tools_called,
            "duration_ms": duration_ms,
            "error": f"JSON parse error: {e}",
        }
    except Exception as e:
        duration_ms = int((time.time() - start) * 1000)
        return {
            "structured_response": {},
            "final_response": f"Document Agent encountered an error: {str(e)}",
            "tools_called": tools_called,
            "duration_ms": duration_ms,
            "error": str(e),
        }
