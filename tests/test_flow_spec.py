"""
Deterministic tests for the Traditional vs AI experimental flow (no API key needed).

Run:
    pytest tests/test_flow_spec.py -v

Covers:
  - 25-question pre-test / post-test / fixed practice bank structure
  - practice bank questions are distinct from pre/post tests
  - fixed traditional topic order
  - per-topic corpus extraction for the control arm
  - AI-arm remediation cap (spec §12) via src.remediation.topic_end_decision
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.state import new_state
from src.remediation import topic_end_decision, MAX_REMEDIATION_ATTEMPTS
from src.notes import topic_order, topic_notes
from src.agents.planner import _load_graph


def _load(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _question_ids(data: dict) -> list:
    return [q["id"] for s in data["sections"] for q in s["questions"]]


# ── test banks ──────────────────────────────────────────────────────────────

def test_pretest_is_25_questions_sectioned():
    pre = _load("data/pretest_python.json")
    assert pre["form"] == "A"
    assert len(pre["sections"]) == 5
    assert sum(len(s["questions"]) for s in pre["sections"]) == 25


def test_posttest_is_25_questions_sectioned():
    post = _load("data/posttest_python.json")
    assert len(post["sections"]) == 5
    assert sum(len(s["questions"]) for s in post["sections"]) == 25


def test_practice_bank_has_25_questions_5_per_topic():
    bank = _load("data/practice_python.json")
    assert bank["domain"] == "python_programming"
    assert len(bank["sections"]) == 5
    for sec in bank["sections"]:
        assert len(sec["questions"]) == 5, sec["topic"]


def test_practice_bank_distinct_from_pre_and_post_tests():
    pre   = set(_question_ids(_load("data/pretest_python.json")))
    post  = set(_question_ids(_load("data/posttest_python.json")))
    prac  = set(_question_ids(_load("data/practice_python.json")))
    assert len(prac) == 25
    assert not (prac & pre), "practice ids collide with pre-test"
    assert not (prac & post), "practice ids collide with post-test"


def test_practice_bank_topics_match_graph():
    graph_topics  = set(_load_graph("python_programming").keys())
    bank_topics   = {s["topic"] for s in _load("data/practice_python.json")["sections"]}
    assert bank_topics == graph_topics


def test_practice_bank_questions_same_across_students():
    # The bank is a static file — it cannot vary per student (spec §4).
    assert os.path.exists("data/practice_python.json")


# ── traditional arm order ───────────────────────────────────────────────────

def test_traditional_topic_order_is_fixed():
    assert topic_order("python_programming") == [
        "variables_datatypes", "control_flow", "loops", "functions", "oop_basics",
    ]


def test_topic_notes_extracts_each_section():
    for topic in topic_order("python_programming"):
        notes = topic_notes("python_programming", topic)
        assert notes and notes.startswith("## "), topic
        assert len(notes) > 40, topic


def test_topic_notes_unknown_topic_returns_empty():
    assert topic_notes("python_programming", "unknown_topic") == ""


# ── AI-arm remediation cap (spec §12) ───────────────────────────────────────

def test_max_remediation_attempts_config_sane():
    assert 1 <= MAX_REMEDIATION_ATTEMPTS <= 3


def test_no_replan_just_advances():
    state = new_state("S1", "python_programming")
    state["topic_pointer"] = "functions"
    decision, state = topic_end_decision(state, replan=False)
    assert decision == "advance"
    assert state["engagement"]["total_replans"] == 0
    assert state["needs_review"] == {}
    assert state["replan_flag"] is False


def test_replan_remediates_up_to_cap_then_marks_needs_review():
    state = new_state("S2", "python_programming")
    state["topic_pointer"] = "loops"

    outcomes = []
    for _ in range(MAX_REMEDIATION_ATTEMPTS):
        decision, state = topic_end_decision(state, replan=True)
        outcomes.append(decision)

    assert outcomes == ["remediate"] * MAX_REMEDIATION_ATTEMPTS
    assert state["engagement"]["remediation_counts"]["loops"] == MAX_REMEDIATION_ATTEMPTS
    assert state["engagement"]["total_remediations"] == MAX_REMEDIATION_ATTEMPTS

    decision, state = topic_end_decision(state, replan=True)
    assert decision == "advance_needs_review"
    assert state["needs_review"]["loops"] is True
    assert state["engagement"]["total_replans"] == MAX_REMEDIATION_ATTEMPTS + 1
    assert state["engagement"]["total_remediations"] == MAX_REMEDIATION_ATTEMPTS


def test_needs_review_topic_is_scheduled_later_by_planner():
    from src.agents.planner import planner_node
    state = new_state("S3", "python_programming")
    state["ablation_mode"] = "no_adaptive_planning"   # keep the test offline (no LLM)
    state["needs_review"]["loops"] = True             # loops is 3rd in the natural order
    state["pretest_per_topic"] = {"loops": {"correct": 0, "total": 5, "pct": 0.0}}
    state["mastery"] = {"loops": 0.0}
    state = planner_node(state)
    # needs_review topic must appear last in the plan (scheduled for later)
    assert state["topic_sequence"][-1] == "loops"
    events = [e.get("event") for e in state["session_history"]]
    assert "needs_review_scheduled_later" in events


def test_new_state_has_research_counters():
    state = new_state("S4", "python_programming")
    assert state["group"] == "experimental"
    assert state["traditional_index"] == 0
    assert state["topics_attempted"] == []
    assert state["needs_review"] == {}
    assert state["engagement"]["total_replans"] == 0
    assert state["engagement"]["total_remediations"] == 0
    assert "learning_time_sec" in state["engagement"]
