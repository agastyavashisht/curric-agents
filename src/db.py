"""
Extended SQLite persistence layer — with Supabase dual-write.

Strategy:
  • Every write goes to BOTH local SQLite and Supabase (when configured).
  • Reads come from Supabase when online (so the dashboard shows all students),
    falling back to local SQLite when offline.
  • On Streamlit Cloud, DB_PATH is automatically set to /tmp/ so SQLite
    works as a within-session cache even though the filesystem is ephemeral.

Tables:
  learner_state    — full-state JSON blob per student×domain
  assessment_log   — every graded item with justification, raw LLM response
  pilot_scores     — pre/post test scores per student (for learning_gain eval)
  survey_responses — post-study Likert questionnaire answers
"""
import os
import sqlite3
from datetime import datetime, timezone

# On Streamlit Cloud the cwd is ephemeral — use /tmp for SQLite so it at
# least persists within a single session (Supabase is the real store).
_IS_CLOUD = os.getenv("STREAMLIT_SHARING_MODE") or os.getenv("STREAMLIT_SERVER_HEADLESS")
_DEFAULT_DB = "/tmp/learner_state.db" if _IS_CLOUD else "results/learner_state.db"
DB_PATH = os.getenv("DB_PATH", _DEFAULT_DB)

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
    ablation_mode TEXT   NOT NULL DEFAULT 'full',
    timestamp    TEXT    NOT NULL,
    UNIQUE(student_id, domain, test_type)
);

CREATE TABLE IF NOT EXISTS survey_responses (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id   TEXT    NOT NULL,
    domain       TEXT    NOT NULL,
    group_label  TEXT    NOT NULL DEFAULT 'experimental',
    q1_ease      INTEGER,   -- 1-5 Likert: ease of use
    q2_helpful   INTEGER,   -- 1-5 Likert: helpfulness
    q3_adaptive  INTEGER,   -- 1-5 Likert: felt personalised
    q4_recommend INTEGER,   -- 1-5 Likert: would recommend
    q5_prefer    INTEGER,   -- 1-5 Likert: preferred over traditional
    comments     TEXT,
    timestamp    TEXT    NOT NULL
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
        ("pilot_scores",   "ablation_mode",    "TEXT NOT NULL DEFAULT 'full'"),
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
    # ── 1. Local SQLite (always) ──────────────────────────────────────────────
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

    # ── 2. Supabase (when configured, non-blocking) ───────────────────────────
    try:
        from src.online_db import log_assessment_online
        log_assessment_online(
            student_id, domain, topic, item_type, item_text, response,
            grade, justification, graded_by, flagged, latency_ms, raw_llm_response,
        )
    except Exception as e:
        print(f"[SUPABASE] log_assessment sync failed (data safe in SQLite): {e}")


def save_pilot_score(
    student_id: str,
    domain: str,
    test_type: str,
    score_pct: float,
    n_correct: int,
    n_total: int,
    group_label: str = "experimental",
    db_path: str = DB_PATH,
    ablation_mode: str = "full",
) -> None:
    # ── 1. Local SQLite (always) ──────────────────────────────────────────────
    conn = get_conn(db_path)
    try:
        conn.execute(
            """INSERT INTO pilot_scores
               (student_id, domain, test_type, score_pct, n_correct, n_total,
                group_label, ablation_mode, timestamp)
               VALUES (?,?,?,?,?,?,?,?,?)
               ON CONFLICT(student_id, domain, test_type)
               DO UPDATE SET score_pct=excluded.score_pct, n_correct=excluded.n_correct,
                             n_total=excluded.n_total, ablation_mode=excluded.ablation_mode,
                             timestamp=excluded.timestamp""",
            (
                student_id, domain, test_type, score_pct,
                n_correct, n_total, group_label, ablation_mode,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        conn.commit()
    finally:
        conn.close()

    # ── 2. Supabase (when configured) ────────────────────────────────────────
    try:
        from src.online_db import save_pilot_score_online
        save_pilot_score_online(
            student_id, domain, test_type, score_pct,
            n_correct, n_total, group_label,
        )
    except Exception as e:
        print(f"[SUPABASE] save_pilot_score sync failed (data safe in SQLite): {e}")


def save_survey_response(
    student_id: str,
    domain: str,
    group_label: str,
    q1_ease: int,
    q2_helpful: int,
    q3_adaptive: int,
    q4_recommend: int,
    q5_prefer: int,
    comments: str = "",
    db_path: str = DB_PATH,
) -> None:
    ts = datetime.now(timezone.utc).isoformat()
    # ── 1. Local SQLite ───────────────────────────────────────────────────────
    conn = get_conn(db_path)
    try:
        conn.execute(
            """INSERT INTO survey_responses
               (student_id, domain, group_label, q1_ease, q2_helpful,
                q3_adaptive, q4_recommend, q5_prefer, comments, timestamp)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (
                student_id, domain, group_label,
                q1_ease, q2_helpful, q3_adaptive, q4_recommend, q5_prefer,
                comments, ts,
            ),
        )
        conn.commit()
    finally:
        conn.close()

    # ── 2. Supabase (when configured) ────────────────────────────────────────
    try:
        from src.online_db import _post, is_configured
        if is_configured():
            _post("survey_responses", {
                "student_id":  student_id,
                "domain":      domain,
                "group_label": group_label,
                "q1_ease":     q1_ease,
                "q2_helpful":  q2_helpful,
                "q3_adaptive": q3_adaptive,
                "q4_recommend": q4_recommend,
                "q5_prefer":   q5_prefer,
                "comments":    comments,
                "timestamp":   ts,
            })
    except Exception as e:
        print(f"[SUPABASE] save_survey_response sync failed (data safe in SQLite): {e}")


def get_survey_responses(db_path: str = DB_PATH) -> list[dict]:
    # Prefer Supabase (shows all students across machines)
    try:
        from src.online_db import _get, is_configured
        if is_configured():
            rows = _get("survey_responses", {"order": "timestamp.asc"})
            if rows:
                return [
                    {"student_id": r["student_id"], "domain": r["domain"],
                     "group": r.get("group_label", "experimental"),
                     "q1_ease": r.get("q1_ease"), "q2_helpful": r.get("q2_helpful"),
                     "q3_adaptive": r.get("q3_adaptive"), "q4_recommend": r.get("q4_recommend"),
                     "q5_prefer": r.get("q5_prefer"), "comments": r.get("comments", ""),
                     "timestamp": r.get("timestamp", "")}
                    for r in rows
                ]
    except Exception as e:
        print(f"[SUPABASE] get_survey_responses fallback to SQLite: {e}")

    # Fall back to local SQLite
    conn = get_conn(db_path)
    try:
        rows = conn.execute(
            """SELECT student_id, domain, group_label,
                      q1_ease, q2_helpful, q3_adaptive, q4_recommend, q5_prefer,
                      comments, timestamp
               FROM survey_responses ORDER BY timestamp"""
        ).fetchall()
        return [
            {"student_id": r[0], "domain": r[1], "group": r[2],
             "q1_ease": r[3], "q2_helpful": r[4], "q3_adaptive": r[5],
             "q4_recommend": r[6], "q5_prefer": r[7],
             "comments": r[8], "timestamp": r[9]}
            for r in rows
        ]
    finally:
        conn.close()


def get_pilot_scores(db_path: str = DB_PATH) -> list[dict]:
    # Prefer Supabase (shows all students across machines)
    try:
        from src.online_db import get_pilot_scores_online, is_configured
        if is_configured():
            rows = get_pilot_scores_online()
            if rows:
                return rows
    except Exception as e:
        print(f"[SUPABASE] get_pilot_scores fallback to SQLite: {e}")

    # Fall back to local SQLite
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
    # Prefer Supabase (shows all students across machines)
    try:
        from src.online_db import get_assessment_log_online, is_configured
        if is_configured():
            rows = get_assessment_log_online()
            if rows:
                return rows
    except Exception as e:
        print(f"[SUPABASE] get_assessment_log fallback to SQLite: {e}")

    # Fall back to local SQLite
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
