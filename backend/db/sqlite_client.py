"""
db/sqlite_client.py — SQLite persistence layer for WorkPilot AI

Tables:
  conversations   — chat session registry
  messages        — individual user/assistant messages
  agent_logs      — full reasoning trace per request (powers Agent Activity Viewer)
"""

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import create_engine, text

from config import settings

# ── Engine ────────────────────────────────────────────────────────────────────
_db_path = Path(settings.SQLITE_DB_PATH)
_db_path.parent.mkdir(parents=True, exist_ok=True)

engine = create_engine(
    f"sqlite:///{_db_path}",
    connect_args={"check_same_thread": False},  # needed for FastAPI threading
    echo=False,
)


# ── Schema Initialisation ─────────────────────────────────────────────────────
def init_db() -> None:
    """Create all tables if they don't already exist. Safe to call on every startup."""
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS conversations (
                id          TEXT PRIMARY KEY,
                created_at  TEXT NOT NULL,
                updated_at  TEXT NOT NULL
            )
        """))

        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS messages (
                id               TEXT PRIMARY KEY,
                conversation_id  TEXT NOT NULL REFERENCES conversations(id),
                role             TEXT NOT NULL,   -- 'user' | 'assistant'
                content          TEXT NOT NULL,
                created_at       TEXT NOT NULL
            )
        """))

        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS agent_logs (
                id               TEXT PRIMARY KEY,
                conversation_id  TEXT NOT NULL REFERENCES conversations(id),
                user_input       TEXT NOT NULL,
                intent           TEXT,            -- knowledge | document | task
                intent_confidence TEXT,           -- high | medium | low
                intent_reasoning  TEXT,
                selected_agent   TEXT,
                tools_called     TEXT,            -- JSON array of tool names
                agent_response   TEXT,            -- JSON structured agent output
                final_response   TEXT,            -- human-readable response
                trace            TEXT,            -- JSON full reasoning trace
                duration_ms      INTEGER,         -- total processing time
                created_at       TEXT NOT NULL
            )
        """))


# ── Helper Functions ──────────────────────────────────────────────────────────
def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_id() -> str:
    return str(uuid.uuid4())


def ensure_conversation(conversation_id: str) -> str:
    """Create conversation row if it doesn't exist. Returns the conversation_id."""
    with engine.begin() as conn:
        existing = conn.execute(
            text("SELECT id FROM conversations WHERE id = :id"),
            {"id": conversation_id},
        ).fetchone()

        if not existing:
            now = now_iso()
            conn.execute(
                text("INSERT INTO conversations (id, created_at, updated_at) VALUES (:id, :ca, :ua)"),
                {"id": conversation_id, "ca": now, "ua": now},
            )
    return conversation_id


def save_message(conversation_id: str, role: str, content: str) -> str:
    """Persist a single message. Returns the new message id."""
    msg_id = new_id()
    with engine.begin() as conn:
        conn.execute(
            text("""
                INSERT INTO messages (id, conversation_id, role, content, created_at)
                VALUES (:id, :cid, :role, :content, :ca)
            """),
            {"id": msg_id, "cid": conversation_id, "role": role, "content": content, "ca": now_iso()},
        )
        # bump conversation updated_at
        conn.execute(
            text("UPDATE conversations SET updated_at = :ua WHERE id = :id"),
            {"ua": now_iso(), "id": conversation_id},
        )
    return msg_id


def save_agent_log(
    conversation_id: str,
    user_input: str,
    intent: str,
    intent_confidence: str,
    intent_reasoning: str,
    selected_agent: str,
    tools_called: list,
    agent_response: dict,
    final_response: str,
    trace: list,
    duration_ms: int,
) -> str:
    """Save the full reasoning trace for one request. Returns log id."""
    log_id = new_id()
    with engine.begin() as conn:
        conn.execute(
            text("""
                INSERT INTO agent_logs (
                    id, conversation_id, user_input, intent, intent_confidence,
                    intent_reasoning, selected_agent, tools_called, agent_response,
                    final_response, trace, duration_ms, created_at
                ) VALUES (
                    :id, :cid, :user_input, :intent, :intent_confidence,
                    :intent_reasoning, :selected_agent, :tools_called, :agent_response,
                    :final_response, :trace, :duration_ms, :ca
                )
            """),
            {
                "id": log_id,
                "cid": conversation_id,
                "user_input": user_input,
                "intent": intent,
                "intent_confidence": intent_confidence,
                "intent_reasoning": intent_reasoning,
                "selected_agent": selected_agent,
                "tools_called": json.dumps(tools_called),
                "agent_response": json.dumps(agent_response),
                "final_response": final_response,
                "trace": json.dumps(trace),
                "duration_ms": duration_ms,
                "ca": now_iso(),
            },
        )
    return log_id


def get_recent_messages(conversation_id: str, limit: int = 10) -> list[dict]:
    """Fetch the last N messages for conversation context (multi-turn memory)."""
    with engine.connect() as conn:
        rows = conn.execute(
            text("""
                SELECT role, content FROM messages
                WHERE conversation_id = :cid
                ORDER BY created_at DESC
                LIMIT :limit
            """),
            {"cid": conversation_id, "limit": limit},
        ).fetchall()
    # Return in chronological order
    return [{"role": r.role, "content": r.content} for r in reversed(rows)]
