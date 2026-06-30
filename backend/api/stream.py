"""
api/stream.py — GET /api/v1/chat/stream

Streaming endpoint that:
  1. Runs the full orchestrator pipeline (intent → agent → RAG) synchronously
     in a thread pool, exactly like /chat, to get the full metadata
  2. Then streams the final answer token-by-token via Server-Sent Events (SSE)
     so the frontend can render characters as they arrive (ChatGPT-style)

SSE event format:
  data: {"type": "meta",  "agent": "...", "intent": "...", "conversation_id": "..."}
  data: {"type": "token", "text": "word "}
  data: {"type": "done",  "duration_ms": 1234, "rag_used": false, "sources": [...]}
  data: [DONE]
"""

import asyncio
import json
import time
from typing import AsyncGenerator

import google.generativeai as genai
from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from agents.orchestrator import run as run_orchestrator
from config import settings
from schemas.chat import ChatRequest

router = APIRouter()


# ── Streaming generator ──────────────────────────────────────────────────────

async def _stream_events(message: str, conversation_id: str) -> AsyncGenerator[str, None]:
    """
    Async generator that yields SSE-formatted strings.
    Step 1: run the full orchestrator pipeline in a thread (blocking).
    Step 2: stream the final response token-by-token with Gemini streaming.
    """

    # ── Step 1: run orchestrator in thread pool ──────────────────────────────
    loop = asyncio.get_event_loop()
    try:
        state = await loop.run_in_executor(
            None, run_orchestrator, message, conversation_id
        )
    except Exception as e:
        yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
        return

    # Extract metadata from orchestrator result
    agent_used      = state.get("selected_agent") or "unknown"
    intent          = state.get("intent") or "unknown"
    agent_response  = state.get("agent_response") or {}
    rag_used        = bool(agent_response.get("rag_used", False))
    sources         = agent_response.get("sources", []) or []
    full_answer     = state.get("final_response") or "No response generated."

    # ── Step 2: send metadata frame ──────────────────────────────────────────
    yield f"data: {json.dumps({'type': 'meta', 'agent': agent_used, 'intent': intent, 'conversation_id': conversation_id})}\n\n"

    # ── Step 3: stream the final answer token-by-token ───────────────────────
    stream_start = time.time()
    try:
        genai.configure(api_key=settings.GEMINI_API_KEY)
        model = genai.GenerativeModel(
            model_name=settings.GEMINI_MODEL,
            generation_config=genai.GenerationConfig(
                temperature=0.2,
                max_output_tokens=2048,
            ),
        )

        STREAM_PROMPT = (
            "You are WorkPilot AI, an enterprise assistant. "
            "Present the following answer clearly using markdown formatting "
            "(use **bold**, bullet points, numbered lists, and ## headings where appropriate). "
            "Do not add any introduction or preamble — output only the formatted answer.\n\n"
            f"Answer to present:\n{full_answer}"
        )

        # Use the synchronous streaming API, yielding chunks via run_in_executor
        # to avoid blocking the event loop
        response = await loop.run_in_executor(
            None,
            lambda: model.generate_content(STREAM_PROMPT, stream=True),
        )

        # Iterate over chunks — each chunk has .text
        for chunk in response:
            token = chunk.text if hasattr(chunk, "text") and chunk.text else ""
            if token:
                yield f"data: {json.dumps({'type': 'token', 'text': token})}\n\n"
                # Small sleep to let the event loop breathe and flush to client
                await asyncio.sleep(0)

    except Exception:
        # Fallback: stream the pre-computed answer in word chunks
        words = full_answer.split(" ")
        for i, word in enumerate(words):
            chunk = word + (" " if i < len(words) - 1 else "")
            yield f"data: {json.dumps({'type': 'token', 'text': chunk})}\n\n"
            await asyncio.sleep(0.015)

    # ── Step 4: done frame ───────────────────────────────────────────────────
    duration_ms = int((time.time() - stream_start) * 1000)
    yield f"data: {json.dumps({'type': 'done', 'duration_ms': duration_ms, 'rag_used': rag_used, 'sources': sources})}\n\n"
    yield "data: [DONE]\n\n"


# ── Route ────────────────────────────────────────────────────────────────────

@router.post(
    "/chat/stream",
    summary="Stream a WorkPilot AI response via SSE",
    description=(
        "Runs the full multi-agent pipeline then streams the final answer "
        "token-by-token using Server-Sent Events. "
        "Events: meta → token (repeated) → done → [DONE]."
    ),
)
async def chat_stream(request: ChatRequest) -> StreamingResponse:
    conversation_id = request.get_or_create_conversation_id()

    return StreamingResponse(
        _stream_events(request.message, conversation_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control":     "no-cache",
            "X-Accel-Buffering": "no",   # disable nginx buffering
            "Connection":        "keep-alive",
        },
    )
