"""
schemas/chat.py — Pydantic models for the /chat API

These define the contract between frontend and backend.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
import uuid


# ── Request ───────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=8000, description="User's message")
    conversation_id: Optional[str] = Field(
        default=None,
        description="Existing conversation ID for multi-turn. Omit to start a new conversation.",
    )

    def get_or_create_conversation_id(self) -> str:
        return self.conversation_id or str(uuid.uuid4())


# ── Trace Step ────────────────────────────────────────────────────────────────

class TraceStep(BaseModel):
    step: str                   # e.g. "intent_classification", "agent_execution", "persistence"
    timestamp: str              # ISO 8601
    agent: Optional[str]        # which agent ran this step
    input_summary: str          # brief description of input
    output_summary: str         # brief description of output
    data: Dict[str, Any]        # full structured data for this step
    duration_ms: Optional[int]  # how long this step took


# ── Response ──────────────────────────────────────────────────────────────────

class ChatResponse(BaseModel):
    conversation_id: str
    message: str                        # human-readable final response
    agent_used: str                     # which agent handled the request
    intent: str                         # classified intent
    intent_confidence: str              # high | medium | low
    structured_response: Dict[str, Any] # raw structured output from the agent
    trace: List[TraceStep]              # full reasoning trace (for Activity Viewer)
    duration_ms: int                    # total request processing time


# ── Error ─────────────────────────────────────────────────────────────────────

class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
