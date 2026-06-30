"""
tools/task_tools.py — Task Persistence Tools for the Task Agent

Called by task_agent.py after Gemini generates a structured plan.
Saves every task in the plan to the SQLite tasks table.

Tools:
    save_plan_to_db  — persist all tasks from an AI plan, returns task IDs
    get_tasks_summary — human-readable stats string for the agent response
"""

import logging
import uuid

from db.sqlite_client import (
    create_task,
    get_task_stats,
)

logger = logging.getLogger(__name__)


def save_plan_to_db(plan: dict, conversation_id: str | None = None) -> list[str]:
    """
    Persist every task in an AI-generated plan to SQLite.

    Args:
        plan:            The structured plan dict from task_agent (JSON parsed).
        conversation_id: ID of the chat that triggered this plan (for traceability).

    Returns:
        List of newly created task IDs (one per task in the plan).
    """
    plan_id = str(uuid.uuid4())
    plan_title = plan.get("plan_title", "Untitled Plan")
    tasks_data = plan.get("tasks", [])

    if not tasks_data:
        logger.warning("save_plan_to_db called with zero tasks — nothing saved.")
        return []

    saved_ids: list[str] = []

    for task in tasks_data:
        try:
            task_db_id = create_task(
                plan_id=plan_id,
                plan_title=plan_title,
                title=task.get("title", "Untitled Task"),
                description=task.get("description", ""),
                priority=task.get("priority", "medium"),
                owner=task.get("owner"),
                estimated_hours=task.get("estimated_hours"),
                acceptance_criteria=task.get("acceptance_criteria"),
                dependencies=task.get("dependencies", []),
                conversation_id=conversation_id,
            )
            saved_ids.append(task_db_id)
        except Exception as e:
            logger.error(f"Failed to save task '{task.get('title')}': {e}")

    logger.info(
        f"✅ Saved {len(saved_ids)}/{len(tasks_data)} tasks "
        f"for plan '{plan_title}' (plan_id={plan_id})"
    )
    return saved_ids


def get_tasks_summary() -> str:
    """
    Return a one-line summary of all tasks in the DB.
    Used by the task agent to append stats to its response.

    Example: "📋 Task board: 12 todo, 3 in_progress, 5 done (20 total)"
    """
    try:
        stats = get_task_stats()
        return (
            f"📋 Task board: {stats['todo']} todo, "
            f"{stats['in_progress']} in progress, "
            f"{stats['done']} done "
            f"({stats['total']} total)"
        )
    except Exception as e:
        logger.error(f"get_tasks_summary failed: {e}")
        return ""
