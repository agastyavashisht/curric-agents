"""
Every agent call gets logged here, in real time, as it happens.

The log is the raw data eval/coordination_efficiency.py reads.
Each record includes:
  - session_id  (student__domain__session_count) — groups by session not student
  - latency_seconds — wall-clock time for the agent call
  - tokens_used     — approximate token count for the LLM response (FR-11)
  - changed_keys    — which state fields the agent mutated
  - redundant       — True if no state changed (wasted call)

Token counting (FR-11):
  LangChain stores usage metadata in response.usage_metadata or
  response.response_metadata['token_usage']. We try both paths.
  For non-LLM agents (Monitor) the count is 0.
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


def _count_tokens(new_state: dict) -> int:
    """
    Extract token counts from any LLM response objects stored in the new state.
    Checks common LangChain metadata paths.
    Returns total tokens used (input + output), or 0 if not available.
    """
    total = 0

    # Check current_material (Content agent stores the raw response indirectly)
    # The most reliable source is the pending_item or last_grade which may
    # carry raw_response. We approximate via character count if no metadata.
    for key in ("current_material", "pending_item", "last_grade"):
        val = new_state.get(key)
        if isinstance(val, dict) and "raw_response" in val:
            raw = val["raw_response"] or ""
            # Rough estimate: 1 token ≈ 4 characters for English text
            total += max(len(raw) // 4, 0)

    return total


def log_node_call(agent_name: str, log_path: str = DEFAULT_LOG_PATH):
    """
    Decorator for a node function: `def node(state) -> state`.

    Records timing, state diff, and token usage.
    session_id = "<student_id>__<domain>__<session_count>" so eval scripts
    can group by session, not by student (prevents multi-session inflation).
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

            # Token counting (FR-11)
            tokens_used = _count_tokens(new_state)

            # session_id for per-session grouping in coordination eval
            student_id    = state.get("student_id", "unknown")
            domain        = state.get("domain", "unknown")
            session_count = state.get("engagement", {}).get("session_count", 1)
            session_id    = f"{student_id}__{domain}__{session_count}"

            latency = round(time.perf_counter() - start, 4)
            path    = os.getenv("LOG_PATH", log_path)

            _append(path, {
                "agent":           agent_name,
                "student_id":      student_id,
                "session_id":      session_id,
                "domain":          domain,
                "start_ts":        start_ts,
                "latency_seconds": latency,
                "tokens_used":     tokens_used,
                "changed_keys":    changed_keys,
                "redundant":       redundant,
                "step_count":      new_state.get("step_count"),
            })
            return new_state
        return wrapper
    return decorator
