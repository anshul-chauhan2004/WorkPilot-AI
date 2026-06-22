# WorkPilot AI

**Enterprise agentic AI platform** — AI agents that help employees with company knowledge, document understanding, task planning, and onboarding.

Built with [LangGraph](https://langchain-ai.github.io/langgraph/) multi-agent orchestration and [Gemini](https://ai.google.dev/).

---

## What It Does

WorkPilot AI routes every employee request to a specialized AI agent:

| Agent | What it handles |
|---|---|
| **Knowledge Agent** | Company policies, benefits, procedures, culture Q&A |
| **Document Agent** | Analyze PDFs, summarize documents, extract action items |
| **Task Agent** | Create task lists, onboarding plans, work breakdowns |
| **Orchestrator** | Classifies intent and routes to the right agent automatically |

Every request produces a **full reasoning trace** — what the orchestrator decided, which agent ran, what it returned — powering a future Agent Activity Viewer.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend framework | [FastAPI](https://fastapi.tiangolo.com/) |
| Agent framework | [LangGraph](https://langchain-ai.github.io/langgraph/) |
| LLM | [Gemini API](https://ai.google.dev/) (model configurable via `.env`) |
| Embeddings | Gemini `text-embedding-004` (Phase 2+) |
| Vector DB | [ChromaDB](https://www.trychroma.com/) persistent storage (Phase 2+) |
| Relational DB | SQLite via [SQLAlchemy](https://www.sqlalchemy.org/) |
| Frontend | Next.js + Tailwind (Phase 5) |

---

## Current Features (Phase 1)

- ✅ LangGraph multi-agent orchestrator with conditional routing
- ✅ Three specialized agents — each with a unique system prompt and structured JSON output
- ✅ Full reasoning trace logged per request (intent, agent selected, tools called, response)
- ✅ SQLite persistence: conversations, messages, agent execution logs
- ✅ `POST /api/v1/chat` endpoint — accepts message + optional conversation ID
- ✅ `GET /health` endpoint
- ✅ Auto-generated interactive API docs at `/docs`
- ✅ GEMINI_MODEL configurable from `.env` — no hardcoded model names

---

## Project Structure

```
WorkPilot-AI/
├── backend/
│   ├── main.py                   # FastAPI entry point
│   ├── config.py                 # Settings from .env
│   ├── requirements.txt
│   ├── .env.example              # Template — copy to .env
│   │
│   ├── agents/
│   │   ├── orchestrator.py       # LangGraph StateGraph — routes & traces
│   │   ├── knowledge_agent.py    # Company Q&A specialist
│   │   ├── document_agent.py     # Document analysis specialist
│   │   └── task_agent.py         # Work planning specialist
│   │
│   ├── api/
│   │   └── chat.py               # POST /api/v1/chat
│   │
│   ├── db/
│   │   └── sqlite_client.py      # SQLite schema + helpers
│   │
│   └── schemas/
│       └── chat.py               # Pydantic request/response models
│
├── frontend/                     # Next.js app (Phase 5)
├── .gitignore
├── pyrightconfig.json
└── README.md
```

---

## Setup & Run

### Prerequisites
- Python 3.11+
- A [Gemini API key](https://aistudio.google.com/apikey)

### 1. Clone and enter the project
```bash
git clone https://github.com/your-username/WorkPilot-AI.git
cd WorkPilot-AI
```

### 2. Create and activate a virtual environment
```bash
cd backend
python3 -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate         # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure environment
```bash
cp .env.example .env
# Open .env and set your GEMINI_API_KEY
```

### 5. Start the server
```bash
python main.py
```

Server starts at `http://localhost:8000`

---

## API Reference

### `GET /health`
Quick liveness check.
```json
{ "status": "ok", "model": "gemini-2.0-flash", "version": "0.1.0" }
```

### `POST /api/v1/chat`
Send a message to the agent system.

**Request:**
```json
{
  "message": "What is the company remote work policy?",
  "conversation_id": "optional-uuid-for-multi-turn"
}
```

**Response:**
```json
{
  "conversation_id": "uuid",
  "message": "Human-readable answer from the agent",
  "agent_used": "knowledge_agent",
  "intent": "knowledge",
  "intent_confidence": "high",
  "structured_response": { ... },
  "trace": [ ... ],
  "duration_ms": 1240
}
```

### Interactive Docs
Visit `http://localhost:8000/docs` for the full Swagger UI.

---

## Example Requests

```bash
# Knowledge Agent — company Q&A
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is the company vacation policy?"}'

# Task Agent — work planning
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Create an onboarding plan for a new backend engineer"}'

# Document Agent — document analysis
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Summarize this: All employees must complete security training by Q3. IT will send calendar invites. Non-compliance results in system access suspension."}'
```

---

## Changing the Gemini Model

Edit `backend/.env` — no code changes required:
```bash
GEMINI_MODEL=gemini-2.0-flash       # default
GEMINI_MODEL=gemini-2.0-flash-lite  # cheaper
GEMINI_MODEL=gemini-1.5-pro         # more capable
```

---

## Development Roadmap

| Phase | Status | Description |
|---|---|---|
| **Phase 1** | ✅ Complete | LangGraph multi-agent backend, /chat endpoint, SQLite persistence |
| **Phase 2** | Planned | RAG pipeline — PDF ingestion, ChromaDB embeddings, Knowledge Agent grounding |
| **Phase 3** | Planned | Agent tools — `create_task()`, `summarize_document()`, `create_onboarding_plan()` |
| **Phase 4** | Planned | Full persistence — multi-turn memory, agent execution history API |
| **Phase 5** | Planned | Next.js frontend — chat workspace, document upload, task board, agent activity viewer |

---

## Contributing

1. Fork the repo
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes
4. Open a pull request

---

## License

MIT
