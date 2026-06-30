"""
db/sqlite_client.py — SQLite persistence layer for WorkPilot AI

Tables:
  conversations   — chat session registry
  messages        — individual user/assistant messages
  agent_logs      — full reasoning trace per request (powers Agent Activity Viewer)
  documents       — metadata for uploaded/ingested PDFs (Phase 2)
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

        # Phase 2: document metadata table
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS documents (
                id                TEXT PRIMARY KEY,   -- doc_id (UUID)
                filename          TEXT NOT NULL,       -- original upload filename
                file_path         TEXT NOT NULL,       -- path in uploads/
                file_size_bytes   INTEGER NOT NULL,
                page_count        INTEGER NOT NULL,
                chunk_count       INTEGER NOT NULL,
                status            TEXT NOT NULL,       -- 'indexed' | 'failed'
                uploaded_at       TEXT NOT NULL
            )
        """))

        # Phase 3: tasks table — AI-generated tasks persisted by Task Agent
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS tasks (
                id                  TEXT PRIMARY KEY,
                plan_id             TEXT NOT NULL,      -- groups tasks from same plan
                plan_title          TEXT NOT NULL,
                title               TEXT NOT NULL,
                description         TEXT NOT NULL,
                priority            TEXT NOT NULL,      -- high | medium | low
                status              TEXT NOT NULL DEFAULT 'todo',  -- todo | in_progress | done
                owner               TEXT,
                estimated_hours     REAL,
                acceptance_criteria TEXT,
                dependencies        TEXT,               -- JSON array of sibling task IDs
                conversation_id     TEXT,               -- which chat created this plan
                created_at          TEXT NOT NULL
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


# ── History Helpers (Phase 4) ─────────────────────────────────────────────────

def list_conversations(limit: int = 50) -> list[dict]:
    """
    Return recent conversations ordered by last activity (newest first).
    Each entry includes the first user message as a preview.
    """
    with engine.connect() as conn:
        convs = conn.execute(
            text("""
                SELECT c.id, c.created_at, c.updated_at,
                       (SELECT content FROM messages
                        WHERE conversation_id = c.id AND role = 'user'
                        ORDER BY created_at ASC LIMIT 1) AS first_message,
                       (SELECT COUNT(*) FROM messages WHERE conversation_id = c.id) AS message_count,
                       (SELECT selected_agent FROM agent_logs
                        WHERE conversation_id = c.id
                        ORDER BY created_at DESC LIMIT 1) AS last_agent
                FROM conversations c
                ORDER BY c.updated_at DESC
                LIMIT :limit
            """),
            {"limit": limit},
        ).fetchall()
    return [
        {
            "id": r.id,
            "created_at": r.created_at,
            "updated_at": r.updated_at,
            "preview": (r.first_message or "")[:120],
            "message_count": r.message_count or 0,
            "last_agent": r.last_agent,
        }
        for r in convs
    ]


def get_conversation_thread(conversation_id: str) -> dict | None:
    """
    Return a full conversation thread: messages + agent_logs ordered by time.
    Returns None if the conversation doesn't exist.
    """
    with engine.connect() as conn:
        conv = conn.execute(
            text("SELECT id, created_at, updated_at FROM conversations WHERE id = :id"),
            {"id": conversation_id},
        ).fetchone()

        if not conv:
            return None

        messages = conn.execute(
            text("""
                SELECT id, role, content, created_at
                FROM messages
                WHERE conversation_id = :cid
                ORDER BY created_at ASC
            """),
            {"cid": conversation_id},
        ).fetchall()

        logs = conn.execute(
            text("""
                SELECT id, user_input, intent, intent_confidence, intent_reasoning,
                       selected_agent, tools_called, final_response, duration_ms, created_at
                FROM agent_logs
                WHERE conversation_id = :cid
                ORDER BY created_at ASC
            """),
            {"cid": conversation_id},
        ).fetchall()

    return {
        "id": conv.id,
        "created_at": conv.created_at,
        "updated_at": conv.updated_at,
        "messages": [
            {
                "id": m.id,
                "role": m.role,
                "content": m.content,
                "created_at": m.created_at,
            }
            for m in messages
        ],
        "agent_logs": [
            {
                "id": lg.id,
                "user_input": lg.user_input,
                "intent": lg.intent,
                "intent_confidence": lg.intent_confidence,
                "intent_reasoning": lg.intent_reasoning,
                "selected_agent": lg.selected_agent,
                "tools_called": json.loads(lg.tools_called or "[]"),
                "final_response": lg.final_response,
                "duration_ms": lg.duration_ms,
                "created_at": lg.created_at,
            }
            for lg in logs
        ],
    }


# ── Document Helpers (Phase 2) ────────────────────────────────────────────────

def save_document(
    doc_id: str,
    filename: str,
    file_path: str,
    file_size_bytes: int,
    page_count: int,
    chunk_count: int,
    status: str = "indexed",
) -> str:
    """Persist document metadata after successful ingestion. Returns doc_id."""
    with engine.begin() as conn:
        conn.execute(
            text("""
                INSERT INTO documents
                    (id, filename, file_path, file_size_bytes, page_count, chunk_count, status, uploaded_at)
                VALUES
                    (:id, :fn, :fp, :fs, :pc, :cc, :st, :ua)
            """),
            {
                "id": doc_id,
                "fn": filename,
                "fp": file_path,
                "fs": file_size_bytes,
                "pc": page_count,
                "cc": chunk_count,
                "st": status,
                "ua": now_iso(),
            },
        )
    return doc_id


def list_documents() -> list[dict]:
    """Return all indexed documents ordered by upload time (newest first)."""
    with engine.connect() as conn:
        rows = conn.execute(
            text("""
                SELECT id, filename, file_size_bytes, page_count, chunk_count, status, uploaded_at
                FROM documents
                ORDER BY uploaded_at DESC
            """)
        ).fetchall()
    return [
        {
            "doc_id": r.id,
            "filename": r.filename,
            "file_size_bytes": r.file_size_bytes,
            "page_count": r.page_count,
            "chunk_count": r.chunk_count,
            "status": r.status,
            "uploaded_at": r.uploaded_at,
        }
        for r in rows
    ]


def get_document(doc_id: str) -> dict | None:
    """Fetch a single document's metadata by ID. Returns None if not found."""
    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT * FROM documents WHERE id = :id"),
            {"id": doc_id},
        ).fetchone()
    if not row:
        return None
    return {
        "doc_id": row.id,
        "filename": row.filename,
        "file_path": row.file_path,
        "file_size_bytes": row.file_size_bytes,
        "page_count": row.page_count,
        "chunk_count": row.chunk_count,
        "status": row.status,
        "uploaded_at": row.uploaded_at,
    }


def delete_document_record(doc_id: str) -> bool:
    """Delete a document row from SQLite. Returns True if a row was deleted."""
    with engine.begin() as conn:
        result = conn.execute(
            text("DELETE FROM documents WHERE id = :id"),
            {"id": doc_id},
        )
    return result.rowcount > 0


# ── Task Helpers (Phase 3) ────────────────────────────────────────────────────

def create_task(
    plan_id: str,
    plan_title: str,
    title: str,
    description: str,
    priority: str,
    owner: str | None = None,
    estimated_hours: float | None = None,
    acceptance_criteria: str | None = None,
    dependencies: list | None = None,
    conversation_id: str | None = None,
) -> str:
    """Persist a single task row. Returns the new task id."""
    task_id = new_id()
    with engine.begin() as conn:
        conn.execute(
            text("""
                INSERT INTO tasks (
                    id, plan_id, plan_title, title, description, priority,
                    status, owner, estimated_hours, acceptance_criteria,
                    dependencies, conversation_id, created_at
                ) VALUES (
                    :id, :plan_id, :plan_title, :title, :description, :priority,
                    'todo', :owner, :est_hours, :ac, :deps, :conv_id, :ca
                )
            """),
            {
                "id": task_id,
                "plan_id": plan_id,
                "plan_title": plan_title,
                "title": title,
                "description": description,
                "priority": priority,
                "owner": owner,
                "est_hours": estimated_hours,
                "ac": acceptance_criteria,
                "deps": json.dumps(dependencies or []),
                "conv_id": conversation_id,
                "ca": now_iso(),
            },
        )
    return task_id


def list_tasks(
    status: str | None = None,
    plan_id: str | None = None,
    limit: int = 200,
) -> list[dict]:
    """Return tasks ordered by plan creation, then priority. Filters are optional."""
    query = "SELECT * FROM tasks"
    conditions: list[str] = []
    params: dict = {"limit": limit}

    if status:
        conditions.append("status = :status")
        params["status"] = status
    if plan_id:
        conditions.append("plan_id = :plan_id")
        params["plan_id"] = plan_id

    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    query += " ORDER BY created_at ASC LIMIT :limit"

    with engine.connect() as conn:
        rows = conn.execute(text(query), params).fetchall()

    return [
        {
            "id": r.id,
            "plan_id": r.plan_id,
            "plan_title": r.plan_title,
            "title": r.title,
            "description": r.description,
            "priority": r.priority,
            "status": r.status,
            "owner": r.owner,
            "estimated_hours": r.estimated_hours,
            "acceptance_criteria": r.acceptance_criteria,
            "dependencies": json.loads(r.dependencies or "[]"),
            "conversation_id": r.conversation_id,
            "created_at": r.created_at,
        }
        for r in rows
    ]


def get_task(task_id: str) -> dict | None:
    """Fetch a single task by ID. Returns None if not found."""
    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT * FROM tasks WHERE id = :id"),
            {"id": task_id},
        ).fetchone()
    if not row:
        return None
    return {
        "id": row.id,
        "plan_id": row.plan_id,
        "plan_title": row.plan_title,
        "title": row.title,
        "description": row.description,
        "priority": row.priority,
        "status": row.status,
        "owner": row.owner,
        "estimated_hours": row.estimated_hours,
        "acceptance_criteria": row.acceptance_criteria,
        "dependencies": json.loads(row.dependencies or "[]"),
        "conversation_id": row.conversation_id,
        "created_at": row.created_at,
    }


def update_task_status(task_id: str, new_status: str) -> bool:
    """
    Update a task's status. Valid values: 'todo', 'in_progress', 'done'.
    Returns True if the task was found and updated, False otherwise.
    """
    valid = {"todo", "in_progress", "done"}
    if new_status not in valid:
        raise ValueError(f"Invalid status '{new_status}'. Must be one of {valid}")

    with engine.begin() as conn:
        result = conn.execute(
            text("UPDATE tasks SET status = :status WHERE id = :id"),
            {"status": new_status, "id": task_id},
        )
    return result.rowcount > 0


def delete_task(task_id: str) -> bool:
    """Delete a task by ID. Returns True if deleted, False if not found."""
    with engine.begin() as conn:
        result = conn.execute(
            text("DELETE FROM tasks WHERE id = :id"),
            {"id": task_id},
        )
    return result.rowcount > 0


def get_task_stats() -> dict:
    """Return aggregate counts by status for dashboard display."""
    with engine.connect() as conn:
        rows = conn.execute(
            text("SELECT status, COUNT(*) as count FROM tasks GROUP BY status")
        ).fetchall()
    stats = {"todo": 0, "in_progress": 0, "done": 0, "total": 0}
    for r in rows:
        stats[r.status] = r.count
        stats["total"] += r.count
    return stats
