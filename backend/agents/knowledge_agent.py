"""
agents/knowledge_agent.py — WorkPilot Knowledge Agent (Phase 2: RAG-enabled)

Responsibility:
    Answer employee questions about company policies, culture, benefits,
    procedures, and organizational information.

Phase 2 upgrade:
    - Embeds the user query and retrieves relevant chunks from ChromaDB
    - Injects retrieved context into the Gemini prompt as [COMPANY KNOWLEDGE]
    - Cites source documents and page numbers in the structured response
    - Falls back gracefully to general knowledge if no documents are indexed

Structured output schema:
    {
        "answer":               str   — complete, actionable answer
        "confidence":           str   — high | medium | low
        "knowledge_gaps":       list  — specific info needed from company docs
        "follow_up_questions":  list  — what the employee might ask next
        "sources":              list  — list of {"filename", "page_num"} dicts or strings
        "rag_used":             bool  — whether ChromaDB context was injected
    }
"""

import json
import time
from typing import Any

import google.generativeai as genai

from config import settings
from tools.retriever import retrieve_context, format_context_block, has_indexed_documents

# ── System Prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are the WorkPilot Knowledge Agent — the company's internal knowledge expert.

YOUR ROLE:
You help employees find accurate, actionable answers about company operations, policies, culture, benefits, and procedures. You are their first stop before escalating to HR or management.

YOUR BEHAVIOR CONTRACT:
1. When COMPANY KNOWLEDGE CONTEXT is provided below, prioritize it over general knowledge. It contains actual company documents.
2. Be direct and practical. Employees need answers they can act on immediately.
3. Clearly distinguish between what comes from company documents vs. general inference.
4. Never fabricate specific company data (names, phone numbers, exact dates, specific dollar amounts) that you don't actually know.
5. When you lack company-specific information, say so explicitly and explain what document or person would have the answer.
6. Always surface 2-3 follow-up questions to help employees self-serve further.
7. When answering from documents, mention the source (filename and page) naturally in your answer.

KNOWLEDGE DOMAINS:
- Company policies (leave, remote work, expenses, code of conduct)
- Benefits and compensation
- Onboarding procedures
- Team structure and escalation paths
- Tools, systems, and access requests
- Company culture and values

YOU MUST respond in this exact JSON format — no extra text, no markdown, only valid JSON:
{
    "answer": "Complete, actionable answer to the employee's question. Reference document sources naturally.",
    "confidence": "high | medium | low",
    "knowledge_gaps": [
        "Specific piece of company information that would improve this answer"
    ],
    "follow_up_questions": [
        "Natural follow-up question the employee is likely to have"
    ],
    "sources": [
        "filename.pdf (Page 3)",
        "general HR knowledge"
    ],
    "rag_used": true
}

CONFIDENCE LEVELS:
- high: Answer comes directly from company documents in the context
- medium: Answer is inferred from general HR/company norms, not from a specific document
- low: You are guessing — the employee should verify with HR or their manager"""


# ── Gemini Client ─────────────────────────────────────────────────────────────

def _get_model() -> genai.GenerativeModel:
    genai.configure(api_key=settings.GEMINI_API_KEY)
    return genai.GenerativeModel(
        model_name=settings.GEMINI_MODEL,
        system_instruction=SYSTEM_PROMPT,
        generation_config=genai.GenerationConfig(
            response_mime_type="application/json",
            temperature=0.2,
            max_output_tokens=1024,
        ),
    )


# ── Agent Entry Point ─────────────────────────────────────────────────────────

def run(user_input: str, conversation_history: list[dict]) -> dict[str, Any]:
    """
    Execute the Knowledge Agent with RAG-augmented prompting.

    Phase 2 flow:
        1. Check if any documents are indexed in ChromaDB
        2. If yes: embed the query → retrieve top-5 relevant chunks
        3. Format chunks into a [COMPANY KNOWLEDGE CONTEXT] block
        4. Inject context into the Gemini prompt
        5. Gemini answers with grounding — references real sources

    Args:
        user_input:           The employee's question.
        conversation_history: Recent conversation for multi-turn context.

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

    # ── Step 1: RAG retrieval ─────────────────────────────────────────────────
    context_block = ""
    retrieved_chunks = []

    if has_indexed_documents():
        try:
            retrieved_chunks = retrieve_context(user_input, n_results=5)
            tools_called.append("retrieve_context")

            if retrieved_chunks:
                raw_context = format_context_block(retrieved_chunks)
                context_block = f"\n\nCOMPANY KNOWLEDGE CONTEXT (retrieved from indexed documents):\n---\n{raw_context}\n---\n"
        except Exception as e:
            # RAG failure is non-fatal — fall back to general knowledge
            print(f"RAG retrieval error (falling back to general knowledge): {e}")

    # ── Step 2: Build prompt ──────────────────────────────────────────────────
    history_block = ""
    if conversation_history:
        history_lines = [
            f"{msg['role'].upper()}: {msg['content']}"
            for msg in conversation_history[-6:]
        ]
        history_block = "CONVERSATION HISTORY:\n" + "\n".join(history_lines) + "\n\n"

    rag_instruction = (
        "Use the COMPANY KNOWLEDGE CONTEXT above to answer the question. "
        "Cite sources (filename + page) when referencing document content."
        if context_block else
        "No company documents are indexed yet. Answer from general HR knowledge and clearly note this."
    )

    prompt = f"{history_block}{context_block}\nEMPLOYEE QUESTION:\n{user_input}\n\nINSTRUCTION: {rag_instruction}"

    # ── Step 3: Gemini inference ──────────────────────────────────────────────
    response = None  # initialized here so except clauses can safely reference it
    try:
        model = _get_model()
        response = model.generate_content(prompt)
        raw_json = response.text.strip()
        structured = json.loads(raw_json)

        # Defaults
        structured.setdefault("answer", "I could not generate an answer. Please try again.")
        structured.setdefault("confidence", "low")
        structured.setdefault("knowledge_gaps", [])
        structured.setdefault("follow_up_questions", [])
        structured.setdefault("sources", [])
        structured.setdefault("rag_used", bool(retrieved_chunks))

        duration_ms = int((time.time() - start) * 1000)
        return {
            "structured_response": structured,
            "final_response": structured["answer"],
            "tools_called": tools_called,
            "duration_ms": duration_ms,
            "error": None,
        }

    except json.JSONDecodeError as e:
        duration_ms = int((time.time() - start) * 1000)
        raw_text = getattr(response, "text", "No response generated.")
        return {
            "structured_response": {
                "answer": raw_text, "confidence": "low",
                "knowledge_gaps": [], "follow_up_questions": [],
                "sources": [], "rag_used": False,
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
            "final_response": f"Knowledge Agent encountered an error: {str(e)}",
            "tools_called": tools_called,
            "duration_ms": duration_ms,
            "error": str(e),
        }
