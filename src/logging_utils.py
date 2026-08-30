"""
Every agent call gets logged here, in real time, as it happens.

The log is the raw data eval/coordination_efficiency.py reads.
Each record includes a session_id (student_id + domain + session_count)
so the coordination eval can group by SESSION rather than by student —
preventing multi-session data from inflating task_completion_rate.
"""
import json
import os
import time
import functools
from datetime import datetime, timezone

DEFAULT_LOG_PATH = "logs/agent_calls.jsonl"


def _append(log_path: str, record: dict) -> None:
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def log_node_call(agent_name: str, log_path: str = DEFAULT_LOG_PATH):
    """
    Decorator for a LangGraph/Streamlit node function: `def node(state) -> state`.

    Times the call, diffs state before/after to find changed keys, and appends
    one JSON line to the agent calls log.  A call is "redundant" if no state
    keys changed (it re-did work for nothing) — used by coordination_efficiency.

    session_id format: "<student_id>__<domain>__<session_count>"
    This lets eval scripts group by session rather than student so that
    multi-session pilots don't have inflated task_completion_rate.
    """
    def decorator(node_fn):
        @functools.wraps(node_fn)
        def wrapper(state, *args, **kwargs):
            start    = time.perf_counter()
            start_ts = datetime.now(timezone.utc).isoformat()

            before_snapshot = {k: state.get(k) for k in state.keys()}
            new_state       = node_fn(state, *args, **kwargs)
            after_snapshot  = {k: new_state.get(k) for k in new_state.keys()}

            changed_keys = [
                k for k in after_snapshot
                if before_snapshot.get(k) != after_snapshot.get(k)
            ]
            redundant = len(changed_keys) == 0

            # Build session_id so eval can group by session, not just student
            student_id    = state.get("student_id", "unknown")
            domain        = state.get("domain", "unknown")
            session_count = state.get("engagement", {}).get("session_count", 1)
            session_id    = f"{student_id}__{domain}__{session_count}"

            end  = time.perf_counter()
            path = os.getenv("LOG_PATH", log_path)
            _append(path, {
                "agent":            agent_name,
                "student_id":       student_id,
                "session_id":       session_id,
                "domain":           domain,
                "start_ts":         start_ts,
                "latency_seconds":  round(end - start, 4),
                "changed_keys":     changed_keys,
                "redundant":        redundant,
                "step_count":       new_state.get("step_count"),
            })
            return new_state
        return wrapper
    return decorator
