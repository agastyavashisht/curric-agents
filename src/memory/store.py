"""
Structured, persistent memory layer. This is the piece that makes a
learner's profile survive across sessions instead of resetting every
time they log back in - the whole point of "persistent memory" in the
synopsis. SQLite is enough for a prototype; swap for Postgres later
without changing the interface below.
"""
import sqlite3
import json
import os
from datetime import datetime, timezone
from typing import Optional

from src.state import LearnerState, new_state

_SCHEMA = """
CREATE TABLE IF NOT EXISTS learner_state (
    student_id TEXT NOT NULL,
    domain TEXT NOT NULL,
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
    """Returns the learner's saved state, or a fresh one if none exists yet."""
    conn = _connect(db_path)
    try:
        row = conn.execute(
            "SELECT state_json FROM learner_state WHERE student_id = ? AND domain = ?",
            (student_id, domain),
        ).fetchone()
        if row is None:
            return new_state(student_id, domain)
        return json.loads(row[0])
    finally:
        conn.close()


def save_state(db_path: str, state: LearnerState) -> None:
    """Persists the learner's state. Called after every graph run, not every node."""
    conn = _connect(db_path)
    try:
        conn.execute(
            """INSERT INTO learner_state (student_id, domain, state_json, updated_at)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(student_id, domain)
               DO UPDATE SET state_json = excluded.state_json, updated_at = excluded.updated_at""",
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


def delete_state(db_path: str, student_id: str, domain: str) -> None:
    """Useful for resetting a test student between pilot dry-runs."""
    conn = _connect(db_path)
    try:
        conn.execute(
            "DELETE FROM learner_state WHERE student_id = ? AND domain = ?",
            (student_id, domain),
        )
        conn.commit()
    finally:
        conn.close()
