"""
api/tasks.py — Tasks REST API

Endpoints:
    GET  /tasks              List all tasks (optional ?status=todo&plan_id=xxx)
    GET  /tasks/{id}         Get a single task
    PATCH /tasks/{id}/status Update task status (todo | in_progress | done)
    DELETE /tasks/{id}       Delete a task
    GET  /tasks/stats        Aggregate counts by status
"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from db.sqlite_client import (
    list_tasks,
    get_task,
    update_task_status,
    delete_task,
    get_task_stats,
)

router = APIRouter()


# ── Schemas ───────────────────────────────────────────────────────────────────

class TaskOut(BaseModel):
    id: str
    plan_id: str
    plan_title: str
    title: str
    description: str
    priority: str
    status: str
    owner: str | None
    estimated_hours: float | None
    acceptance_criteria: str | None
    dependencies: list[str]
    conversation_id: str | None
    created_at: str


class TaskListResponse(BaseModel):
    tasks: list[TaskOut]
    total: int


class StatusUpdate(BaseModel):
    status: str  # todo | in_progress | done


class TaskStatsResponse(BaseModel):
    todo: int
    in_progress: int
    done: int
    total: int


# ── Routes ────────────────────────────────────────────────────────────────────

@router.get(
    "/tasks",
    response_model=TaskListResponse,
    summary="List all tasks",
    description="Returns tasks from all AI-generated plans. Filter by status or plan_id.",
)
def get_tasks(
    status: str | None = Query(None, description="Filter: todo | in_progress | done"),
    plan_id: str | None = Query(None, description="Filter by plan ID"),
    limit: int = Query(200, ge=1, le=1000),
):
    valid_statuses = {"todo", "in_progress", "done"}
    if status and status not in valid_statuses:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid status '{status}'. Must be one of {sorted(valid_statuses)}",
        )

    tasks = list_tasks(status=status, plan_id=plan_id, limit=limit)
    return TaskListResponse(tasks=tasks, total=len(tasks))


@router.get(
    "/tasks/stats",
    response_model=TaskStatsResponse,
    summary="Get task statistics",
    description="Returns aggregate task counts grouped by status.",
)
def get_stats():
    return get_task_stats()


@router.get(
    "/tasks/{task_id}",
    response_model=TaskOut,
    summary="Get a single task",
)
def get_single_task(task_id: str):
    task = get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")
    return task


@router.patch(
    "/tasks/{task_id}/status",
    response_model=TaskOut,
    summary="Update task status",
    description="Cycle a task through: todo → in_progress → done",
)
def patch_task_status(task_id: str, body: StatusUpdate):
    valid = {"todo", "in_progress", "done"}
    if body.status not in valid:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid status '{body.status}'. Must be one of {sorted(valid)}",
        )

    updated = update_task_status(task_id, body.status)
    if not updated:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")

    task = get_task(task_id)
    return task


@router.delete(
    "/tasks/{task_id}",
    summary="Delete a task",
    description="Permanently delete a task by ID.",
)
def remove_task(task_id: str):
    deleted = delete_task(task_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")
    return {"deleted": True, "id": task_id}
