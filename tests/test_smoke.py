"""
Tests that run WITHOUT any API key - covers everything deterministic
(Planner's ordering, Monitor's mastery math, SQLite persistence, and
the eval scripts). Run this after every change to confirm nothing
foundational broke, before spending API credits on a full session.

    pytest tests/test_smoke.py -v
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.planner import _load_graph, _topological_order
from src.agents.monitor import monitor_node
from src.state import new_state
from src.memory.store import load_state, save_state, delete_state


def test_planner_produces_valid_prerequisite_ordering():
    topics = _load_graph("python_programming")
    order = _topological_order(topics, mastery={})

    assert len(order) == len(topics), "every topic should appear exactly once"

    pos = {t: i for i, t in enumerate(order)}
    for topic, info in topics.items():
        for prereq in info["prerequisites"]:
            assert pos[prereq] < pos[topic], f"{prereq} must come before {topic}"


def test_planner_skips_already_mastered_topics():
    topics = _load_graph("python_programming")
    order = _topological_order(topics, mastery={"variables_datatypes": 0.95})
    assert "variables_datatypes" not in order


def test_monitor_updates_mastery_with_ema():
    state = new_state("test_student", "python_programming")
    state["topic_pointer"] = "variables_datatypes"
    state["last_grade"] = {"correct": True, "score": 1.0, "flagged": False}

    state = monitor_node(state)
    assert state["mastery"]["variables_datatypes"] == 0.4  # EMA_ALPHA(0.4)*1.0 + 0.6*0.0


def test_monitor_triggers_replan_on_low_mastery():
    state = new_state("test_student", "python_programming")
    state["topic_pointer"] = "variables_datatypes"
    state["last_grade"] = {"correct": False, "score": 0.0, "flagged": False}

    state = monitor_node(state)
    assert state["replan_flag"] is True


def test_sqlite_persistence_roundtrip(tmp_path):
    db_path = str(tmp_path / "test.db")
    state = load_state(db_path, "S_test", "python_programming")
    assert state["mastery"] == {}

    state["mastery"]["variables_datatypes"] = 0.6
    save_state(db_path, state)

    reloaded = load_state(db_path, "S_test", "python_programming")
    assert reloaded["mastery"]["variables_datatypes"] == 0.6


def test_eval_learning_gain_runs_on_sample_data():
    from eval.learning_gain import analyze
    result = analyze("results/sample_pretest_scores.csv", "results/sample_posttest_scores.csv")
    assert "mean_gain_experimental" in result
    assert "mean_gain_control" in result
    assert result["n_experimental"] == 6


def test_eval_grading_reliability_runs_on_sample_data():
    from eval.grading_reliability import analyze
    result = analyze("results/sample_instructor_grades.csv")
    assert "quadratic_weighted_kappa" in result
