# api package — use relative imports to avoid Pylance circular-import warnings
from .chat import router as chat_router
from .documents import router as documents_router

__all__ = ["chat_router", "documents_router"]
