"""
Wires Planner -> Content -> Assessment -> Monitor into a LangGraph
StateGraph, with the conditional edge back to Planner when Monitor
sets replan_flag. This is the executable version of Fig. 2 in the
synopsis.

advance_topic() is exported for use by app.py so the topic-advance
logic lives in exactly one place.
"""
from langgraph.graph import StateGraph, END

from src.state import LearnerState
from src.agents.planner import planner_agent
from src.agents.content import content_agent
from src.agents.assessment import assessment_agent
from src.agents.monitor import monitor_agent


def advance_topic(state: LearnerState) -> LearnerState:
    """
    Pop the just-taught topic off the queue and point to the next one.
    Used both by the LangGraph pipeline (advance_topic node) and by
    app.py (_advance alias) — single definition, no duplication.
    """
    state = dict(state)
    seq = list(state["topic_sequence"])
    if state["topic_pointer"] in seq:
        seq.remove(state["topic_pointer"])
    state["topic_sequence"] = seq
    state["topic_pointer"]  = seq[0] if seq else None
    return state


def _route_after_monitor(state: LearnerState) -> str:
    if state.get("replan_flag"):
        return "advance_and_replan"
    if not state.get("topic_sequence"):
        return END
    return "content"


def _advance_and_replan(state: LearnerState) -> LearnerState:
    """
    On a replan triggered by low mastery/struggle: pop the current topic
    off the queue first so we don't loop forever on the same topic, then
    let the Planner re-sequence whatever remains.
    """
    state = dict(state)
    seq = list(state["topic_sequence"])
    current = state.get("topic_pointer")
    if current and current in seq:
        seq.remove(current)
    state["topic_sequence"] = seq
    state["topic_pointer"]  = seq[0] if seq else None
    state["replan_flag"]    = True   # keep flag so planner knows it was a replan
    return state


def _route_after_advance_and_replan(state: LearnerState) -> str:
    if not state.get("topic_sequence"):
        return END
    return "planner"


def build_graph():
    graph = StateGraph(LearnerState)

    graph.add_node("planner", planner_agent)
    graph.add_node("content", content_agent)
    graph.add_node("assessment", assessment_agent)
    graph.add_node("monitor", monitor_agent)
    graph.add_node("advance_topic", advance_topic)
    graph.add_node("advance_and_replan", _advance_and_replan)

    graph.set_entry_point("planner")
    graph.add_edge("planner", "content")
    graph.add_edge("content", "assessment")
    graph.add_edge("assessment", "monitor")

    graph.add_conditional_edges(
        "monitor",
        _route_after_monitor,
        {"advance_and_replan": "advance_and_replan", "content": "advance_topic", END: END},
    )
    graph.add_edge("advance_topic", "content")

    graph.add_conditional_edges(
        "advance_and_replan",
        _route_after_advance_and_replan,
        {"planner": "planner", END: END},
    )

    return graph.compile()
