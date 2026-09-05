"""
Persistent memory layer — local SQLite + Supabase dual-store.

Load priority:  Supabase (online, shared across machines) → SQLite (local fallback)
Save strategy:  Write to BOTH simultaneously (non-blocking for Supabase)

This means:
  - Students can resume from any device (state lives in Supabase)
  - Sessions survive network outages (state also in local SQLite)
  - Researcher sees all 20 students' data in one dashboard
"""
import sqlite3
import json
import os
from datetime import datetime, timezone

from src.state import LearnerState, new_state

_SCHEMA = """
CREATE TABLE IF NOT EXISTS learner_state (
    student_id TEXT NOT NULL,
    domain     TEXT NOT NULL,
    state_json TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (student_id, domain)
);
"""


def _connect(db_path: str) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute(_SCHEMA)
    return conn


def load_state(db_path: str, student_id: str, domain: str) -> LearnerState:
    """
    Returns the learner's saved state, or a fresh one if none exists.
    Prefers Supabase when configured (shared across all machines).
    """
    # ── 1. Try Supabase first ─────────────────────────────────────────────────
    try:
        from src.online_db import load_learner_state_online, is_configured
        if is_configured():
            online_state = load_learner_state_online(student_id, domain)
            if online_state:
                # Also cache locally so offline access works
                _save_local(db_path, online_state)
                return online_state
    except Exception as e:
        print(f"[SUPABASE] load_state fallback to SQLite: {e}")

    # ── 2. Fall back to local SQLite ──────────────────────────────────────────
    conn = _connect(db_path)
    try:
        row = conn.execute(
            "SELECT state_json FROM learner_state WHERE student_id=? AND domain=?",
            (student_id, domain),
        ).fetchone()
        if row is None:
            return new_state(student_id, domain)
        loaded = json.loads(row[0])
        # Back-fill any fields added after this state was first saved
        fresh = new_state(student_id, domain)
        for key, default in fresh.items():
            if key not in loaded:
                loaded[key] = default
        return loaded
    finally:
        conn.close()


def _save_local(db_path: str, state: LearnerState) -> None:
    """Write state to local SQLite only."""
    conn = _connect(db_path)
    try:
        conn.execute(
            """INSERT INTO learner_state (student_id, domain, state_json, updated_at)
               VALUES (?,?,?,?)
               ON CONFLICT(student_id, domain)
               DO UPDATE SET state_json=excluded.state_json, updated_at=excluded.updated_at""",
            (
                state["student_id"],
                state["domain"],
                json.dumps(state),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def save_state(db_path: str, state: LearnerState) -> None:
    """
    Persists the learner's state to BOTH local SQLite and Supabase.
    Called after every graded answer and at session end.
    """
    # ── 1. Local SQLite (always, synchronous) ─────────────────────────────────
    _save_local(db_path, state)

    # ── 2. Supabase (when configured, non-blocking) ───────────────────────────
    try:
        from src.online_db import save_learner_state_online, is_configured
        if is_configured():
            save_learner_state_online(state)
    except Exception as e:
        print(f"[SUPABASE] save_state sync failed (data safe in SQLite): {e}")


def delete_state(db_path: str, student_id: str, domain: str) -> None:
    """Reset a test student between dry-runs."""
    conn = _connect(db_path)
    try:
        conn.execute(
            "DELETE FROM learner_state WHERE student_id=? AND domain=?",
            (student_id, domain),
        )
        conn.commit()
    finally:
        conn.close()
