"""
agents/orchestrator.py — WorkPilot LangGraph Orchestrator

Architecture:
    This module defines the LangGraph StateGraph that powers the multi-agent
    routing system. The graph has 5 nodes:

    1. classify_intent   — Gemini classifies the user's intent into one of 3 routes
    2. knowledge_agent   — handles company Q&A requests
    3. document_agent    — handles document analysis requests
    4. task_agent        — handles task planning requests
    5. persist_trace     — saves full reasoning trace to SQLite

    Every request produces a full trace logged to agent_logs, which powers
    the Agent Activity Viewer in Phase 5.

LangGraph State (WorkPilotState):
    Typed dict passed through every node. Each node reads from and writes
    to this shared state.
"""

import json
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, TypedDict

import google.generativeai as genai
from langgraph.graph import END, StateGraph

from config import settings
from db.sqlite_client import (
    ensure_conversation,
    get_recent_messages,
    save_agent_log,
    save_message,
)
from agents import knowledge_agent as knowledge_agent_module
from agents import document_agent as document_agent_module
from agents import task_agent as task_agent_module


# ── State Definition ──────────────────────────────────────────────────────────

class WorkPilotState(TypedDict):
    # Input
    user_input: str
    conversation_id: str

    # Intent classification
    intent: Optional[str]               # knowledge | document | task
    intent_confidence: Optional[str]    # high | medium | low
    intent_reasoning: Optional[str]
    intent_entities: Optional[List[str]]

    # Agent execution
    selected_agent: Optional[str]
    agent_response: Optional[Dict[str, Any]]  # structured output from agent
    final_response: Optional[str]
    tools_called: Optional[List[str]]

    # Reasoning trace (appended at each step)
    trace: List[Dict[str, Any]]

    # Meta
    start_time: float
    error: Optional[str]


# ── Intent Classifier Prompt ──────────────────────────────────────────────────
# This is the orchestrator's unique system prompt — distinct from all agent prompts.

INTENT_CLASSIFIER_PROMPT = """You are the WorkPilot Orchestrator — the routing intelligence of an enterprise AI platform.

YOUR ROLE:
Analyze employee messages and route them to the correct specialized agent. Your routing decision determines the quality of the employee's experience, so be precise.

AVAILABLE AGENTS:
1. knowledge
   Route here for: questions about company policies, culture, benefits, procedures, team structure, tools, org chart, "how do I", "what is the policy on", "who should I contact", general company FAQs
   
2. document
   Route here for: requests to summarize a document, analyze text that has been pasted, extract action items from a document, review a contract, understand a report, anything where the user has provided or refers to a specific piece of text/document

3. task
   Route here for: requests to create a plan, break down a project, generate an onboarding plan, organize work, "create tasks for", "help me plan", "what are the steps to", project management requests

ROUTING DECISION RULES:
- If the message contains pasted document text → document
- If the message asks "what is" or "how does" about company operations → knowledge
- If the message asks to "create", "plan", "organize", or "break down" work → task
- When ambiguous between knowledge and task → knowledge
- When ambiguous between document and knowledge (no pasted text) → knowledge

YOU MUST respond in this exact JSON format — no extra text, no markdown, only valid JSON:
{
    "intent": "knowledge | document | task",
    "confidence": "high | medium | low",
    "reasoning": "One sentence explaining why you chose this agent and what the key signal was.",
    "key_entities": ["Important nouns and concepts extracted from the message"]
}"""


# ── Node: Classify Intent ─────────────────────────────────────────────────────

def classify_intent_node(state: WorkPilotState) -> WorkPilotState:
    """
    Node 1: Use Gemini to classify user intent and decide routing.
    Adds a 'intent_classification' step to the trace.
    """
    step_start = time.time()

    genai.configure(api_key=settings.GEMINI_API_KEY)
    model = genai.GenerativeModel(
        model_name=settings.GEMINI_MODEL,
        system_instruction=INTENT_CLASSIFIER_PROMPT,
        generation_config=genai.GenerationConfig(
            response_mime_type="application/json",
            temperature=0.0,  # deterministic routing — no creativity needed
            max_output_tokens=256,
        ),
    )

    try:
        response = model.generate_content(state["user_input"])
        classification = json.loads(response.text.strip())

        intent = classification.get("intent", "knowledge")
        confidence = classification.get("confidence", "low")
        reasoning = classification.get("reasoning", "Default routing to knowledge agent.")
        entities = classification.get("key_entities", [])

    except Exception as e:
        # Safe fallback
        intent, confidence, reasoning, entities = "knowledge", "low", f"Fallback due to error: {e}", []

    step_duration = int((time.time() - step_start) * 1000)

    trace_step = {
        "step": "intent_classification",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "agent": "orchestrator",
        "input_summary": f"User message ({len(state['user_input'])} chars)",
        "output_summary": f"Routed to '{intent}' agent with {confidence} confidence",
        "data": {
            "intent": intent,
            "confidence": confidence,
            "reasoning": reasoning,
            "key_entities": entities,
            "model_used": settings.GEMINI_MODEL,
        },
        "duration_ms": step_duration,
    }

    return {
        **state,
        "intent": intent,
        "intent_confidence": confidence,
        "intent_reasoning": reasoning,
        "intent_entities": entities,
        "trace": state["trace"] + [trace_step],
    }


# ── Routing Function ──────────────────────────────────────────────────────────

def route_to_agent(state: WorkPilotState) -> str:
    """
    Conditional edge function: maps intent → next node name.
    LangGraph calls this after classify_intent_node to determine the next step.
    """
    intent_map = {
        "knowledge": "knowledge_agent",
        "document": "document_agent",
        "task": "task_agent",
    }
    intent = state.get("intent") or "knowledge"  # narrow str | None → str
    return intent_map.get(intent, "knowledge_agent")


# ── Node: Knowledge Agent ─────────────────────────────────────────────────────

def knowledge_agent_node(state: WorkPilotState) -> WorkPilotState:
    """Node 2a: Run the Knowledge Agent and record its trace step."""
    step_start = time.time()
    history = get_recent_messages(state["conversation_id"], limit=10)

    result = knowledge_agent_module.run(state["user_input"], history)

    step_duration = result["duration_ms"]
    trace_step = {
        "step": "agent_execution",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "agent": "knowledge_agent",
        "input_summary": f"Q: {state['user_input'][:100]}...",
        "output_summary": f"Answer with {result['structured_response'].get('confidence', '?')} confidence",
        "data": {
            "agent": "knowledge_agent",
            "structured_response": result["structured_response"],
            "tools_called": result["tools_called"],
            "error": result["error"],
        },
        "duration_ms": step_duration,
    }

    return {
        **state,
        "selected_agent": "knowledge_agent",
        "agent_response": result["structured_response"],
        "final_response": result["final_response"],
        "tools_called": result["tools_called"],
        "trace": state["trace"] + [trace_step],
        "error": result["error"],
    }


# ── Node: Document Agent ──────────────────────────────────────────────────────

def document_agent_node(state: WorkPilotState) -> WorkPilotState:
    """Node 2b: Run the Document Agent and record its trace step."""
    history = get_recent_messages(state["conversation_id"], limit=6)

    result = document_agent_module.run(state["user_input"], history)

    trace_step = {
        "step": "agent_execution",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "agent": "document_agent",
        "input_summary": f"Document/text: {state['user_input'][:100]}...",
        "output_summary": f"Type: {result['structured_response'].get('document_type', '?')}, {len(result['structured_response'].get('action_items', []))} action items",
        "data": {
            "agent": "document_agent",
            "structured_response": result["structured_response"],
            "tools_called": result["tools_called"],
            "error": result["error"],
        },
        "duration_ms": result["duration_ms"],
    }

    return {
        **state,
        "selected_agent": "document_agent",
        "agent_response": result["structured_response"],
        "final_response": result["final_response"],
        "tools_called": result["tools_called"],
        "trace": state["trace"] + [trace_step],
        "error": result["error"],
    }


# ── Node: Task Agent ──────────────────────────────────────────────────────────

def task_agent_node(state: WorkPilotState) -> WorkPilotState:
    """Node 2c: Run the Task Agent and record its trace step."""
    history = get_recent_messages(state["conversation_id"], limit=6)

    result = task_agent_module.run(
        state["user_input"],
        history,
        conversation_id=state["conversation_id"],
    )

    task_count = len(result["structured_response"].get("tasks", []))
    trace_step = {
        "step": "agent_execution",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "agent": "task_agent",
        "input_summary": f"Planning request: {state['user_input'][:100]}...",
        "output_summary": f"Created {task_count} tasks, {result['structured_response'].get('total_estimated_hours', 0)}h total",
        "data": {
            "agent": "task_agent",
            "structured_response": result["structured_response"],
            "tools_called": result["tools_called"],
            "error": result["error"],
        },
        "duration_ms": result["duration_ms"],
    }

    return {
        **state,
        "selected_agent": "task_agent",
        "agent_response": result["structured_response"],
        "final_response": result["final_response"],
        "tools_called": result["tools_called"],
        "trace": state["trace"] + [trace_step],
        "error": result["error"],
    }


# ── Node: Persist Trace ───────────────────────────────────────────────────────

def persist_trace_node(state: WorkPilotState) -> WorkPilotState:
    """
    Node 3 (final): Save the full execution to SQLite.
    - Saves user message
    - Saves assistant response
    - Saves full agent_log (reasoning trace)
    """
    step_start = time.time()
    total_duration = int((time.time() - state["start_time"]) * 1000)

    try:
        ensure_conversation(state["conversation_id"])

        # Persist user message
        save_message(state["conversation_id"], "user", state["user_input"])

        # Persist assistant response
        final_resp = state.get("final_response") or "No response generated."
        save_message(state["conversation_id"], "assistant", final_resp)

        # Persist full reasoning trace
        save_agent_log(
            conversation_id=state["conversation_id"],
            user_input=state["user_input"],
            intent=state.get("intent") or "unknown",
            intent_confidence=state.get("intent_confidence") or "low",
            intent_reasoning=state.get("intent_reasoning") or "",
            selected_agent=state.get("selected_agent") or "unknown",
            tools_called=state.get("tools_called") or [],
            agent_response=state.get("agent_response") or {},
            final_response=final_resp,
            trace=state["trace"],
            duration_ms=total_duration,
        )
        persist_status = "success"
    except Exception as e:
        persist_status = f"error: {e}"

    step_duration = int((time.time() - step_start) * 1000)

    trace_step = {
        "step": "persistence",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "agent": "orchestrator",
        "input_summary": "Final response + trace",
        "output_summary": f"Saved to SQLite ({persist_status}). Total: {total_duration}ms",
        "data": {
            "conversation_id": state["conversation_id"],
            "total_duration_ms": total_duration,
            "persist_status": persist_status,
        },
        "duration_ms": step_duration,
    }

    return {
        **state,
        "trace": state["trace"] + [trace_step],
    }


# ── Graph Assembly ────────────────────────────────────────────────────────────

def build_graph():
    """
    Assemble and compile the LangGraph StateGraph.

    Graph edges:
        START → classify_intent
        classify_intent → [knowledge_agent | document_agent | task_agent]  (conditional)
        knowledge_agent → persist_trace
        document_agent  → persist_trace
        task_agent      → persist_trace
        persist_trace   → END
    """
    graph = StateGraph(WorkPilotState)

    # Register nodes
    graph.add_node("classify_intent", classify_intent_node)
    graph.add_node("knowledge_agent", knowledge_agent_node)
    graph.add_node("document_agent", document_agent_node)
    graph.add_node("task_agent", task_agent_node)
    graph.add_node("persist_trace", persist_trace_node)

    # Entry point
    graph.set_entry_point("classify_intent")

    # Conditional routing after classification
    graph.add_conditional_edges(
        "classify_intent",
        route_to_agent,
        {
            "knowledge_agent": "knowledge_agent",
            "document_agent": "document_agent",
            "task_agent": "task_agent",
        },
    )

    # All agents → persist → END
    graph.add_edge("knowledge_agent", "persist_trace")
    graph.add_edge("document_agent", "persist_trace")
    graph.add_edge("task_agent", "persist_trace")
    graph.add_edge("persist_trace", END)

    return graph.compile()


# ── Public API ────────────────────────────────────────────────────────────────

# Compiled graph (singleton — built once at import time)
_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def run(user_input: str, conversation_id: str) -> Dict[str, Any]:
    """
    Execute the full multi-agent pipeline for one user message.

    Args:
        user_input:       The employee's message.
        conversation_id:  UUID string for the conversation session.

    Returns:
        WorkPilotState dict with all fields populated, including full trace.
    """
    initial_state: WorkPilotState = {
        "user_input": user_input,
        "conversation_id": conversation_id,
        "intent": None,
        "intent_confidence": None,
        "intent_reasoning": None,
        "intent_entities": None,
        "selected_agent": None,
        "agent_response": None,
        "final_response": None,
        "tools_called": None,
        "trace": [],
        "start_time": time.time(),
        "error": None,
    }

    graph = get_graph()
    final_state = graph.invoke(initial_state)
    return final_state
