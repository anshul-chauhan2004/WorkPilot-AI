"""
api/history.py — Conversation History & Agent Activity API

Endpoints:
    GET /history                    List recent conversations (paginated)
    GET /history/{conversation_id}  Full thread: messages + agent reasoning trace
"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from db.sqlite_client import list_conversations, get_conversation_thread

router = APIRouter()


# ── Schemas ───────────────────────────────────────────────────────────────────

class ConversationSummary(BaseModel):
    id: str
    created_at: str
    updated_at: str
    preview: str
    message_count: int
    last_agent: str | None


class HistoryListResponse(BaseModel):
    conversations: list[ConversationSummary]
    total: int


class MessageOut(BaseModel):
    id: str
    role: str
    content: str
    created_at: str


class AgentLogOut(BaseModel):
    id: str
    user_input: str
    intent: str | None
    intent_confidence: str | None
    intent_reasoning: str | None
    selected_agent: str | None
    tools_called: list
    final_response: str | None
    duration_ms: int | None
    created_at: str


class ConversationThread(BaseModel):
    id: str
    created_at: str
    updated_at: str
    messages: list[MessageOut]
    agent_logs: list[AgentLogOut]


# ── Routes ────────────────────────────────────────────────────────────────────

@router.get(
    "/history",
    response_model=HistoryListResponse,
    summary="List conversation history",
    description=(
        "Returns recent conversations ordered by last activity. "
        "Each entry includes the first message as a preview and which agent last responded."
    ),
)
def get_history(limit: int = Query(50, ge=1, le=200)):
    conversations = list_conversations(limit=limit)
    return HistoryListResponse(
        conversations=conversations,
        total=len(conversations),
    )


@router.get(
    "/history/{conversation_id}",
    response_model=ConversationThread,
    summary="Get full conversation thread",
    description=(
        "Returns the complete message history and full agent reasoning trace "
        "for a single conversation. Useful for the Agent Activity Viewer."
    ),
)
def get_thread(conversation_id: str):
    thread = get_conversation_thread(conversation_id)
    if not thread:
        raise HTTPException(
            status_code=404,
            detail=f"Conversation '{conversation_id}' not found",
        )
    return thread
