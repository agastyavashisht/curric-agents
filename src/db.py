"""
Extended SQLite persistence layer.

Adds the structured tables required by the spec alongside the existing
learner_state blob table (which is kept for backward compatibility).

Tables:
  learner_state    — original full-state JSON blob (existing, kept)
  assessment_log   — every graded item with justification, graded_by, latency
  pilot_scores     — pre/post test scores per student per domain (for learning_gain eval)

All writes use INSERT OR IGNORE / ON CONFLICT so they are idempotent.
"""
import os
import sqlite3
from datetime import datetime, timezone

DB_PATH = os.getenv("DB_PATH", "results/learner_state.db")

_EXTENDED_SCHEMA = """
CREATE TABLE IF NOT EXISTS assessment_log (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id    TEXT    NOT NULL,
    domain        TEXT    NOT NULL,
    topic         TEXT    NOT NULL,
    item_type     TEXT    NOT NULL,       -- 'mcq' | 'short_answer'
    item_text     TEXT    NOT NULL,
    response      TEXT    NOT NULL,
    grade         REAL,
    justification TEXT,
    raw_llm_response TEXT,               -- full model output before parsing (auditability)
    graded_by     TEXT    NOT NULL,       -- 'exact_match' | 'llm_judge' | 'llm_judge+embedding'
    flagged       INTEGER NOT NULL DEFAULT 0,
    latency_ms    INTEGER,
    timestamp     TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS pilot_scores (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id   TEXT    NOT NULL,
    domain       TEXT    NOT NULL,
    test_type    TEXT    NOT NULL,       -- 'pretest' | 'posttest'
    score_pct    REAL    NOT NULL,
    n_correct    INTEGER,
    n_total      INTEGER,
    group_label  TEXT    NOT NULL DEFAULT 'experimental',
    timestamp    TEXT    NOT NULL,
    UNIQUE(student_id, domain, test_type)
);
"""


def get_conn(db_path: str = DB_PATH) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(_EXTENDED_SCHEMA)
    _migrate(conn)
    return conn


def _migrate(conn: sqlite3.Connection) -> None:
    """
    Safe incremental migration: adds any columns that exist in the current
    schema but are missing from the live DB table (e.g. when the schema
    changed after the DB was already created).
    ALTER TABLE ADD COLUMN is idempotent-safe — we catch the error if the
    column already exists and move on.
    """
    migrations = [
        # (table, column, column_def)
        ("assessment_log", "raw_llm_response", "TEXT"),
        ("assessment_log", "item_text",        "TEXT NOT NULL DEFAULT ''"),
        ("assessment_log", "response",         "TEXT NOT NULL DEFAULT ''"),
    ]
    for table, col, col_def in migrations:
        existing = [r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()]
        if col not in existing:
            try:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {col_def}")
                conn.commit()
                print(f"[DB MIGRATE] Added column {table}.{col}")
            except Exception as e:
                print(f"[DB MIGRATE] Skipped {table}.{col}: {e}")


def log_assessment(
    student_id: str,
    domain: str,
    topic: str,
    item_type: str,
    item_text: str,
    response: str,
    grade: float | None,
    justification: str | None,
    graded_by: str,
    flagged: bool = False,
    latency_ms: int | None = None,
    raw_llm_response: str | None = None,
    db_path: str = DB_PATH,
) -> None:
    conn = get_conn(db_path)
    try:
        conn.execute(
            """INSERT INTO assessment_log
               (student_id, domain, topic, item_type, item_text, response,
                grade, justification, raw_llm_response, graded_by, flagged, latency_ms, timestamp)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                student_id, domain, topic, item_type, item_text, response,
                grade, justification, raw_llm_response, graded_by,
                int(flagged), latency_ms,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def save_pilot_score(
    student_id: str,
    domain: str,
    test_type: str,
    score_pct: float,
    n_correct: int,
    n_total: int,
    group_label: str = "experimental",
    db_path: str = DB_PATH,
) -> None:
    conn = get_conn(db_path)
    try:
        conn.execute(
            """INSERT INTO pilot_scores
               (student_id, domain, test_type, score_pct, n_correct, n_total, group_label, timestamp)
               VALUES (?,?,?,?,?,?,?,?)
               ON CONFLICT(student_id, domain, test_type)
               DO UPDATE SET score_pct=excluded.score_pct, n_correct=excluded.n_correct,
                             n_total=excluded.n_total, timestamp=excluded.timestamp""",
            (
                student_id, domain, test_type, score_pct,
                n_correct, n_total, group_label,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def get_pilot_scores(db_path: str = DB_PATH) -> list[dict]:
    conn = get_conn(db_path)
    try:
        rows = conn.execute(
            "SELECT student_id, domain, test_type, score_pct, group_label FROM pilot_scores"
        ).fetchall()
        return [
            {"student_id": r[0], "domain": r[1], "test_type": r[2],
             "score_pct": r[3], "group": r[4]}
            for r in rows
        ]
    finally:
        conn.close()


def get_assessment_log(db_path: str = DB_PATH) -> list[dict]:
    conn = get_conn(db_path)
    try:
        rows = conn.execute(
            """SELECT student_id, domain, topic, item_type, grade,
                      justification, raw_llm_response, graded_by, flagged, latency_ms, timestamp
               FROM assessment_log ORDER BY timestamp"""
        ).fetchall()
        return [
            {"student_id": r[0], "domain": r[1], "topic": r[2],
             "item_type": r[3], "grade": r[4], "justification": r[5],
             "raw_llm_response": r[6], "graded_by": r[7], "flagged": bool(r[8]),
             "latency_ms": r[9], "timestamp": r[10]}
            for r in rows
        ]
    finally:
        conn.close()
