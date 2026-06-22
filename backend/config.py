"""
config.py — WorkPilot AI Settings

Loads all configuration from environment variables.
GEMINI_MODEL is intentionally configurable so no model names
are hardcoded anywhere in the application.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from the backend directory
load_dotenv(Path(__file__).parent / ".env")


class Settings:
    # ── Gemini ────────────────────────────────────────────────────
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

    # ── Storage ───────────────────────────────────────────────────
    SQLITE_DB_PATH: str = os.getenv("SQLITE_DB_PATH", "data/workpilot.db")
    CHROMA_PERSIST_PATH: str = os.getenv("CHROMA_PERSIST_PATH", "chroma_store")
    UPLOADS_PATH: str = os.getenv("UPLOADS_PATH", "uploads")

    # ── Server ────────────────────────────────────────────────────
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))

    def validate(self) -> None:
        """Fail fast if required config is missing."""
        if not self.GEMINI_API_KEY:
            raise ValueError(
                "GEMINI_API_KEY is not set.\n"
                "Copy backend/.env.example → backend/.env and add your key."
            )


settings = Settings()
settings.validate()

# Ensure required directories exist at startup
for path_str in [settings.SQLITE_DB_PATH, settings.CHROMA_PERSIST_PATH, settings.UPLOADS_PATH]:
    path = Path(path_str)
    # For files (e.g. .db), create the parent dir; for dirs create directly
    target = path.parent if path.suffix else path
    target.mkdir(parents=True, exist_ok=True)
