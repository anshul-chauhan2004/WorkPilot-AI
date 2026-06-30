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
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from config import settings
from db.sqlite_client import init_db
from api.chat import router as chat_router
from api.documents import router as documents_router
from api.tasks import router as tasks_router
from api.history import router as history_router
from api.stream import router as stream_router

# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize DB schema on startup. Replaces deprecated @app.on_event."""
    init_db()
    print(f"✅ WorkPilot AI started")
    print(f"   Model   : {settings.GEMINI_MODEL}")
    print(f"   Database: {settings.SQLITE_DB_PATH}")
    print(f"   Docs    : http://localhost:{settings.API_PORT}/docs")
    print(f"   Demo UI : http://localhost:{settings.API_PORT}/demo")
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
    allow_origins=[
        "http://localhost:3000", "http://127.0.0.1:3000",
        "http://localhost:8000", "http://127.0.0.1:8000",  # demo UI
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Startup (removed) ─────────────────────────────────────────────────────────
# Startup logic moved to lifespan() context manager above.


# ── Routers ───────────────────────────────────────────────────────────────────

app.include_router(chat_router,      prefix="/api/v1", tags=["Chat"])
app.include_router(stream_router,    prefix="/api/v1", tags=["Stream"])
app.include_router(documents_router, prefix="/api/v1", tags=["Documents"])
app.include_router(tasks_router,     prefix="/api/v1", tags=["Tasks"])
app.include_router(history_router,   prefix="/api/v1", tags=["History"])

# ── Demo UI ───────────────────────────────────────────────────────────────────
_static_dir = Path(__file__).parent / "static"
_static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")

@app.get("/demo", include_in_schema=False)
async def demo_ui():
    """Serve the single-page demo UI."""
    return FileResponse(str(_static_dir / "demo.html"))


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
        reload_excludes=[
            "*/venv/*", "*/data/*", "*/chroma_store/*",
            "*/uploads/*", "*/static/*", "*/__pycache__/*",
        ],
    )
