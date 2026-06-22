"""
api/chat.py — POST /chat endpoint

Request flow:
    1. Validate ChatRequest (message + optional conversation_id)
    2. Run orchestrator in thread executor (Gemini SDK is sync)
    3. Return ChatResponse with full trace

The endpoint is intentionally synchronous in logic but wrapped in
asyncio.run_in_executor so FastAPI's event loop is not blocked.
"""

import asyncio
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from agents.orchestrator import run as run_orchestrator
from schemas.chat import ChatRequest, ChatResponse, TraceStep, ErrorResponse

router = APIRouter()


def _execute_pipeline(message: str, conversation_id: str) -> dict:
    """
    Blocking function that runs the full LangGraph pipeline.
    Runs in a thread pool via run_in_executor — keeps event loop free.
    """
    return run_orchestrator(
        user_input=message,
        conversation_id=conversation_id,
    )


@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Send a message to WorkPilot AI",
    description=(
        "Routes the message through the LangGraph orchestrator to the appropriate "
        "specialized agent (Knowledge, Document, or Task). Returns the agent's "
        "structured response plus a full reasoning trace."
    ),
)
async def chat(request: ChatRequest) -> ChatResponse:
    conversation_id = request.get_or_create_conversation_id()

    try:
        # Run the blocking orchestrator in a thread pool
        loop = asyncio.get_event_loop()
        state = await loop.run_in_executor(
            None,
            _execute_pipeline,
            request.message,
            conversation_id,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Orchestrator error: {str(e)}")

    # Map state trace dicts → TraceStep Pydantic models
    trace_steps = [
        TraceStep(
            step=t["step"],
            timestamp=t["timestamp"],
            agent=t.get("agent"),
            input_summary=t["input_summary"],
            output_summary=t["output_summary"],
            data=t["data"],
            duration_ms=t.get("duration_ms"),
        )
        for t in state.get("trace", [])
    ]

    # Calculate total duration from trace
    total_duration = sum(t.duration_ms or 0 for t in trace_steps)

    final_response = state.get("final_response") or "No response generated."

    return ChatResponse(
        conversation_id=conversation_id,
        message=final_response,
        agent_used=state.get("selected_agent") or "unknown",
        intent=state.get("intent") or "unknown",
        intent_confidence=state.get("intent_confidence") or "low",
        structured_response=state.get("agent_response") or {},
        trace=trace_steps,
        duration_ms=total_duration,
    )
