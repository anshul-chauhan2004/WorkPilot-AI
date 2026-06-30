"""
agents/task_agent.py — WorkPilot Task Agent

Responsibility:
    Translate high-level employee goals into structured, sequenced, executable
    task plans. Specializes in work breakdown, onboarding plan generation,
    and project planning.

Unique traits:
    - Never produces vague tasks ("do the thing") — always specific and verifiable
    - Sequences tasks by dependency, not just importance
    - Includes acceptance criteria for every task
    - Tailors onboarding plans to specific roles and contexts
    - Estimates effort in hours, not story points or vague effort labels

Structured output schema:
    {
        "plan_title":            str   — descriptive name for this plan
        "plan_type":             str   — task_list | onboarding_plan | project_plan
        "tasks":                 list  — list of structured task objects
        "total_estimated_hours": int
        "recommended_sequence":  list  — task titles in suggested execution order
        "dependencies_map":      dict  — task → list of tasks it depends on
        "notes":                 str   — additional context or caveats
    }
"""

import json
import time
from typing import Any

import google.generativeai as genai

from config import settings
from tools.task_tools import save_plan_to_db, get_tasks_summary

# ── System Prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are the WorkPilot Task Agent — an expert work planner and onboarding specialist.

YOUR ROLE:
You help employees and managers translate goals, projects, and onboarding requirements into clear, executable task plans. Every plan you create can be handed directly to an employee and immediately acted upon.

YOUR BEHAVIOR CONTRACT:
1. Specificity is non-negotiable. Every task must have a clear, unambiguous description. "Set up development environment" is not a task. "Install Node.js 20 LTS, clone the main repository, and run npm install to verify zero errors" is a task.
2. Sequence by dependency. Identify which tasks block others and surface this clearly.
3. Realistic estimates. Provide honest hour estimates — do not underestimate to seem optimistic.
4. Acceptance criteria. Every task must include a verifiable "done" condition.
5. Role awareness. For onboarding plans, tailor tasks to the specific role, seniority level, and team context provided.

PLAN TYPES:
- task_list: Ad-hoc list of tasks for a specific goal
- onboarding_plan: Structured plan for a new employee joining a team
- project_plan: Multi-phase plan for a larger initiative

TASK OBJECT FORMAT:
{
    "id": "t1",
    "title": "Short task title",
    "description": "Specific, detailed description of exactly what to do",
    "priority": "high | medium | low",
    "estimated_hours": 2,
    "dependencies": ["t1", "t2"],
    "owner": "suggested owner or 'assignee'",
    "acceptance_criteria": "Specific, observable condition that proves this task is complete"
}

YOU MUST respond in this exact JSON format — no extra text, no markdown, only valid JSON:
{
    "plan_title": "Descriptive title for this plan",
    "plan_type": "task_list | onboarding_plan | project_plan",
    "tasks": [
        {
            "id": "t1",
            "title": "Task title",
            "description": "Specific description of what to do",
            "priority": "high | medium | low",
            "estimated_hours": 2,
            "dependencies": [],
            "owner": "assignee",
            "acceptance_criteria": "Verifiable completion condition"
        }
    ],
    "total_estimated_hours": 10,
    "recommended_sequence": ["t1", "t2", "t3"],
    "dependencies_map": {
        "t2": ["t1"],
        "t3": ["t1", "t2"]
    },
    "notes": "Any important caveats, assumptions, or recommendations"
}

For onboarding plans, include tasks covering: tool access, key introductions, documentation reading, first small contribution, and 30/60/90 day milestones."""


# ── Gemini Client ─────────────────────────────────────────────────────────────
def _get_model() -> genai.GenerativeModel:
    genai.configure(api_key=settings.GEMINI_API_KEY)
    return genai.GenerativeModel(
        model_name=settings.GEMINI_MODEL,
        system_instruction=SYSTEM_PROMPT,
        generation_config=genai.GenerationConfig(
            response_mime_type="application/json",
            temperature=0.4,        # moderate — creative in decomposition, not in facts
            max_output_tokens=3000, # onboarding plans can be large
        ),
    )


# ── Agent Entry Point ─────────────────────────────────────────────────────────
def run(
    user_input: str,
    conversation_history: list[dict],
    conversation_id: str | None = None,
) -> dict[str, Any]:
    """
    Execute the Task Agent.

    Args:
        user_input:           Employee's goal or planning request.
        conversation_history: Recent conversation for role/context hints.
        conversation_id:      Chat session ID (used to link saved tasks to the chat).

    Returns:
        {
            "structured_response": dict,
            "final_response":      str,
            "tools_called":        list,
            "duration_ms":         int,
            "error":               str | None,
        }
    """
    start = time.time()

    context_block = ""
    if conversation_history:
        context_lines = [
            f"{msg['role'].upper()}: {msg['content']}"
            for msg in conversation_history[-4:]
        ]
        context_block = "CONVERSATION HISTORY:\n" + "\n".join(context_lines) + "\n\n"

    prompt = f"{context_block}PLANNING REQUEST:\n{user_input}"
    response = None  # initialized before try so except clauses can safely reference it
    try:
        model = _get_model()
        response = model.generate_content(prompt)
        raw_json = response.text.strip()

        structured = json.loads(raw_json)

        structured.setdefault("plan_title", "Work Plan")
        structured.setdefault("plan_type", "task_list")
        structured.setdefault("tasks", [])
        structured.setdefault("total_estimated_hours", 0)
        structured.setdefault("recommended_sequence", [])
        structured.setdefault("dependencies_map", {})
        structured.setdefault("notes", "")

        # ── Persist tasks to SQLite (Phase 3) ────────────────────────────────
        saved_ids: list[str] = []
        tools_called: list[str] = []
        try:
            saved_ids = save_plan_to_db(structured, conversation_id=conversation_id)
            tools_called = ["save_plan_to_db"]
        except Exception as save_err:
            # Non-fatal — agent still returns plan even if DB save fails
            tools_called = [f"save_plan_to_db:ERROR:{save_err}"]

        task_count = len(structured["tasks"])
        saved_count = len(saved_ids)
        total_hours = structured["total_estimated_hours"]
        high_priority = sum(1 for t in structured["tasks"] if t.get("priority") == "high")

        # Build human-readable response
        save_line = (
            f"✅ Saved {saved_count} tasks to your task board."
            if saved_count > 0
            else "⚠️ Could not save tasks to DB — plan generated but not persisted."
        )
        board_summary = get_tasks_summary()

        notes_line = f"\n\n{structured['notes']}" if structured["notes"] else ""

        final_response = (
            f"**{structured['plan_title']}** ({structured['plan_type'].replace('_', ' ').title()})\n\n"
            f"Created **{task_count} tasks** totalling ~{total_hours} hours. "
            f"**{high_priority} high-priority** task(s) to start immediately.\n\n"
            f"{save_line}\n"
            f"{board_summary}"
            f"{notes_line}"
        ).strip()

        duration_ms = int((time.time() - start) * 1000)
        return {
            "structured_response": structured,
            "final_response": final_response,
            "tools_called": tools_called,
            "duration_ms": duration_ms,
            "error": None,
        }

    except json.JSONDecodeError as e:
        duration_ms = int((time.time() - start) * 1000)
        raw_text = getattr(response, "text", "No response generated.")
        return {
            "structured_response": {"plan_title": "Plan", "plan_type": "task_list", "tasks": [], "total_estimated_hours": 0, "recommended_sequence": [], "dependencies_map": {}, "notes": raw_text},
            "final_response": raw_text,
            "tools_called": [],
            "duration_ms": duration_ms,
            "error": f"JSON parse error: {e}",
        }
    except Exception as e:
        duration_ms = int((time.time() - start) * 1000)
        return {
            "structured_response": {},
            "final_response": f"Task Agent encountered an error: {str(e)}",
            "tools_called": [],
            "duration_ms": duration_ms,
            "error": str(e),
        }
