"""
agents/knowledge_agent.py — WorkPilot Knowledge Agent

Responsibility:
    Answer employee questions about company policies, culture, benefits,
    procedures, and organizational information.

Unique traits:
    - Distinguishes between verified knowledge vs. inference
    - Explicitly flags knowledge gaps (what it would need from real docs)
    - Always suggests follow-up questions to encourage self-service

Structured output schema:
    {
        "answer":               str   — complete, actionable answer
        "confidence":           str   — high | medium | low
        "knowledge_gaps":       list  — specific info needed from company docs
        "follow_up_questions":  list  — what the employee might ask next
        "sources":              list  — knowledge sources referenced
    }
"""

import json
import time
from datetime import datetime, timezone
from typing import Any

import google.generativeai as genai

from config import settings

# ── System Prompt ─────────────────────────────────────────────────────────────
# This prompt is unique to the Knowledge Agent. It establishes a distinct
# persona, behavior contract, and exact JSON output format.

SYSTEM_PROMPT = """You are the WorkPilot Knowledge Agent — the company's internal knowledge expert.

YOUR ROLE:
You help employees find accurate, actionable answers about company operations, policies, culture, benefits, and procedures. You are their first stop before escalating to HR or management.

YOUR BEHAVIOR CONTRACT:
1. Be direct and practical. Employees need answers they can act on immediately.
2. Clearly distinguish between what you know confidently vs. what you are inferring.
3. Never fabricate specific company data (names, phone numbers, exact dates, specific dollar amounts) that you don't actually know.
4. When you lack company-specific information, say so explicitly and explain what document or person would have the answer.
5. Always surface 2-3 follow-up questions to help employees self-serve further.

KNOWLEDGE DOMAINS:
- Company policies (leave, remote work, expenses, code of conduct)
- Benefits and compensation
- Onboarding procedures
- Team structure and escalation paths
- Tools, systems, and access requests
- Company culture and values

YOU MUST respond in this exact JSON format — no extra text, no markdown, only valid JSON:
{
    "answer": "Complete, actionable answer to the employee's question. Be thorough but concise.",
    "confidence": "high | medium | low",
    "knowledge_gaps": [
        "Specific piece of company information that would improve this answer"
    ],
    "follow_up_questions": [
        "Natural follow-up question the employee is likely to have"
    ],
    "sources": [
        "general HR knowledge",
        "company policy inference"
    ]
}

CONFIDENCE LEVELS:
- high: You are certain this is standard practice at most companies
- medium: This is likely correct but may vary by company policy
- low: You are inferring — the employee should verify with HR or their manager"""


# ── Gemini Client ─────────────────────────────────────────────────────────────
def _get_model() -> genai.GenerativeModel:
    """Lazily configure and return the Gemini model for this agent."""
    genai.configure(api_key=settings.GEMINI_API_KEY)
    return genai.GenerativeModel(
        model_name=settings.GEMINI_MODEL,
        system_instruction=SYSTEM_PROMPT,
        generation_config=genai.GenerationConfig(
            response_mime_type="application/json",
            temperature=0.2,        # low temperature for factual, consistent answers
            max_output_tokens=1024,
        ),
    )


# ── Agent Entry Point ─────────────────────────────────────────────────────────
def run(user_input: str, conversation_history: list[dict]) -> dict[str, Any]:
    """
    Execute the Knowledge Agent.

    Args:
        user_input: The employee's question.
        conversation_history: List of {"role": str, "content": str} dicts for context.

    Returns:
        {
            "structured_response": dict,   # parsed JSON from Gemini
            "final_response":      str,    # human-readable extracted answer
            "tools_called":        list,   # empty in Phase 1; populated in Phase 3
            "duration_ms":         int,
            "error":               str | None,
        }
    """
    start = time.time()

    # Build the prompt — include recent conversation context if available
    context_block = ""
    if conversation_history:
        context_lines = [
            f"{msg['role'].upper()}: {msg['content']}"
            for msg in conversation_history[-6:]  # last 3 turns
        ]
        context_block = "CONVERSATION HISTORY:\n" + "\n".join(context_lines) + "\n\n"

    prompt = f"{context_block}EMPLOYEE QUESTION:\n{user_input}"

    try:
        model = _get_model()
        response = model.generate_content(prompt)
        raw_json = response.text.strip()

        structured = json.loads(raw_json)

        # Validate required fields; fill defaults if missing
        structured.setdefault("answer", "I could not generate an answer. Please try again.")
        structured.setdefault("confidence", "low")
        structured.setdefault("knowledge_gaps", [])
        structured.setdefault("follow_up_questions", [])
        structured.setdefault("sources", ["general knowledge"])

        duration_ms = int((time.time() - start) * 1000)

        return {
            "structured_response": structured,
            "final_response": structured["answer"],
            "tools_called": [],     # Phase 1: no tools
            "duration_ms": duration_ms,
            "error": None,
        }

    except json.JSONDecodeError as e:
        duration_ms = int((time.time() - start) * 1000)
        # Fallback: return the raw text as the answer
        raw_text = getattr(response, "text", "No response generated.")
        return {
            "structured_response": {"answer": raw_text, "confidence": "low", "knowledge_gaps": [], "follow_up_questions": [], "sources": []},
            "final_response": raw_text,
            "tools_called": [],
            "duration_ms": duration_ms,
            "error": f"JSON parse error: {e}",
        }
    except Exception as e:
        duration_ms = int((time.time() - start) * 1000)
        return {
            "structured_response": {},
            "final_response": f"Knowledge Agent encountered an error: {str(e)}",
            "tools_called": [],
            "duration_ms": duration_ms,
            "error": str(e),
        }
