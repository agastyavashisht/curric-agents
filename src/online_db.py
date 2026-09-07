"""
Supabase online database layer.

Uses Supabase's PostgREST HTTP API via `requests` — no psycopg2,
no C extensions, nothing that App Control can block.

Strategy: DUAL-WRITE
  Every write goes to BOTH local SQLite and Supabase.
  Reads come from Supabase when online, SQLite as fallback.
  This means:
    - Data is never lost if the network drops mid-session
    - The researcher dashboard always shows live online data
    - Students can use the app even without internet (local only)

Setup:
  1. Create a free project at https://supabase.com
  2. Run the SQL in src/supabase_schema.sql in Supabase SQL Editor
  3. Add to .env:
       SUPABASE_URL=https://<project>.supabase.co
       SUPABASE_ANON_KEY=<anon/public key from Settings → API>

The anon key is safe to commit — it only allows the operations
defined in your Row Level Security (RLS) policies.
"""

from __future__ import annotations

import os
import json
import time
from datetime import datetime, timezone
from typing import Any

import requests
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(), override=True)

def _get_secret(key: str, default: str = "") -> str:
    """
    Read a secret from Streamlit secrets (cloud) or .env (local).
    Called per-request, NOT at module import time, so st.secrets is available.
    """
    try:
        import streamlit as st
        val = st.secrets.get(key, "")
        if val:
            return str(val)
    except Exception:
        pass
    return os.getenv(key, default)


def _url() -> str:
    """Return Supabase URL — read fresh each call so Cloud secrets work."""
    return _get_secret("SUPABASE_URL", "").rstrip("/")


def _key() -> str:
    """Return Supabase anon key — read fresh each call so Cloud secrets work."""
    return _get_secret("SUPABASE_ANON_KEY", "")


_TIMEOUT = 8   # seconds per request


def _online() -> bool:
    """True if Supabase is configured — evaluated per-call, not at import time."""
    return bool(_url() and _key())


def _headers() -> dict:
    k = _key()
    return {
        "apikey":        k,
        "Authorization": f"Bearer {k}",
        "Content-Type":  "application/json",
        "Prefer":        "return=minimal",
    }


def _post(table: str, payload: dict) -> bool:
    """Insert one row into a Supabase table. Returns True on success."""
    url = _url()
    if not url or not _key():
        return False
    try:
        r = requests.post(
            f"{url}/rest/v1/{table}",
            headers={**_headers(), "Prefer": "return=minimal,resolution=merge-duplicates"},
            json=payload,
            timeout=_TIMEOUT,
        )
        if r.status_code in (200, 201):
            return True
        print(f"[SUPABASE] POST {table} → {r.status_code}: {r.text[:200]}")
        return False
    except Exception as e:
        print(f"[SUPABASE] POST {table} error: {e}")
        return False


def _get(table: str, params: dict | None = None) -> list[dict]:
    """Select rows from a Supabase table. Returns [] on failure."""
    url = _url()
    if not url or not _key():
        return []
    try:
        r = requests.get(
            f"{url}/rest/v1/{table}",
            headers={**_headers(), "Prefer": "return=representation"},
            params=params or {},
            timeout=_TIMEOUT,
        )
        if r.status_code == 200:
            return r.json()
        print(f"[SUPABASE] GET {table} → {r.status_code}: {r.text[:200]}")
        return []
    except Exception as e:
        print(f"[SUPABASE] GET {table} error: {e}")
        return []


# ── public API ────────────────────────────────────────────────────────────────

def is_configured() -> bool:
    """Returns True if SUPABASE_URL and SUPABASE_ANON_KEY are set."""
    return _online()


def test_connection() -> tuple[bool, str]:
    """Ping Supabase. Returns (success, message)."""
    url = _url()
    if not url or not _key():
        return False, "SUPABASE_URL or SUPABASE_ANON_KEY not set in .env"
    try:
        r = requests.get(
            f"{url}/rest/v1/pilot_scores",
            headers=_headers(),
            params={"limit": "1"},
            timeout=_TIMEOUT,
        )
        if r.status_code == 200:
            return True, "Connected to Supabase"
        if r.status_code == 401:
            return False, "Invalid SUPABASE_ANON_KEY"
        if r.status_code == 404:
            return False, "Table 'pilot_scores' not found — run the schema SQL first"
        return False, f"HTTP {r.status_code}: {r.text[:120]}"
    except requests.exceptions.ConnectionError:
        return False, "Cannot reach Supabase — check internet connection"
    except Exception as e:
        return False, str(e)


def save_pilot_score_online(
    student_id: str,
    domain: str,
    test_type: str,
    score_pct: float,
    n_correct: int,
    n_total: int,
    group_label: str = "experimental",
) -> bool:
    return _post("pilot_scores", {
        "student_id":  student_id,
        "domain":      domain,
        "test_type":   test_type,
        "score_pct":   score_pct,
        "n_correct":   n_correct,
        "n_total":     n_total,
        "group_label": group_label,
        "timestamp":   datetime.now(timezone.utc).isoformat(),
    })


def log_assessment_online(
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
) -> bool:
    return _post("assessment_log", {
        "student_id":       student_id,
        "domain":           domain,
        "topic":            topic,
        "item_type":        item_type,
        "item_text":        item_text,
        "response":         response,
        "grade":            grade,
        "justification":    justification,
        "raw_llm_response": raw_llm_response,
        "graded_by":        graded_by,
        "flagged":          flagged,
        "latency_ms":       latency_ms,
        "timestamp":        datetime.now(timezone.utc).isoformat(),
    })


def save_learner_state_online(state: dict) -> bool:
    """Upsert the full learner state blob to Supabase."""
    return _post("learner_state", {
        "student_id": state["student_id"],
        "domain":     state["domain"],
        "state_json": json.dumps(state),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    })


def load_learner_state_online(student_id: str, domain: str) -> dict | None:
    """Load learner state from Supabase. Returns None if not found."""
    rows = _get("learner_state", {
        "student_id": f"eq.{student_id}",
        "domain":     f"eq.{domain}",
        "select":     "state_json",
        "limit":      "1",
    })
    if rows and rows[0].get("state_json"):
        try:
            return json.loads(rows[0]["state_json"])
        except Exception:
            return None
    return None


def get_pilot_scores_online() -> list[dict]:
    rows = _get("pilot_scores", {"order": "timestamp.desc"})
    return [
        {
            "student_id": r["student_id"],
            "domain":     r["domain"],
            "test_type":  r["test_type"],
            "score_pct":  r["score_pct"],
            "group":      r.get("group_label", "experimental"),
        }
        for r in rows
    ]


def get_assessment_log_online() -> list[dict]:
    rows = _get("assessment_log", {"order": "timestamp.asc"})
    return [
        {
            "student_id":       r["student_id"],
            "domain":           r["domain"],
            "topic":            r["topic"],
            "item_type":        r["item_type"],
            "grade":            r.get("grade"),
            "justification":    r.get("justification"),
            "raw_llm_response": r.get("raw_llm_response"),
            "graded_by":        r.get("graded_by", ""),
            "flagged":          bool(r.get("flagged", False)),
            "latency_ms":       r.get("latency_ms"),
            "timestamp":        r.get("timestamp", ""),
        }
        for r in rows
    ]
