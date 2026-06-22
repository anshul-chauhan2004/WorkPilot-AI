"""
main.py — WorkPilot AI FastAPI Application Entry Point

Startup sequence:
    1. Load settings (validates GEMINI_API_KEY exists)
    2. Initialize SQLite schema (creates tables if needed)
    3. Mount API routers
    4. Serve via uvicorn

Run locally:
    uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""

import warnings
# Suppress LangGraph/LangChain pending-deprecation noise at startup
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*allowed_objects.*")
warnings.filterwarnings("ignore", message=".*LangChain.*")

import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import settings
from db.sqlite_client import init_db
from api.chat import router as chat_router

# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize DB schema on startup. Replaces deprecated @app.on_event."""
    init_db()
    print(f"✅ WorkPilot AI started")
    print(f"   Model   : {settings.GEMINI_MODEL}")
    print(f"   Database: {settings.SQLITE_DB_PATH}")
    print(f"   Docs    : http://localhost:{settings.API_PORT}/docs")
    yield
    # (shutdown logic goes here in future)


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="WorkPilot AI",
    description=(
        "Enterprise agentic AI platform. "
        "Multi-agent system powered by LangGraph + Gemini. "
        "Agents: Knowledge, Document, Task."
    ),
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ── CORS ──────────────────────────────────────────────────────────────────────
# Permissive for local development. Tighten before production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Startup (removed) ─────────────────────────────────────────────────────────
# Startup logic moved to lifespan() context manager above.


# ── Routers ───────────────────────────────────────────────────────────────────

app.include_router(chat_router, prefix="/api/v1", tags=["Chat"])


# ── Health Check ──────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
async def health():
    """Quick liveness check."""
    return {
        "status": "ok",
        "model": settings.GEMINI_MODEL,
        "version": "0.1.0",
    }


# ── Dev Runner ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import pathlib
    _backend = pathlib.Path(__file__).parent

    uvicorn.run(
        "main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=True,
        reload_dirs=[str(_backend)],
        reload_excludes=["*/venv/*", "*/data/*", "*/chroma_store/*", "*/uploads/*", "*/__pycache__/*"],
    )
