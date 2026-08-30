"""
Monitor Agent.

Deterministic (no LLM call): mastery updates, forgetting-curve decay,
and drift/struggle detection are numeric operations that must be
fully auditable and reproducible.

Forgetting-curve model (FR-9):
    Ebbinghaus-style exponential decay applied at the START of each
    monitor call for every topic the student hasn't reviewed recently.
    Decay formula:  m_decayed = m * exp(-lambda * days_since_review)
    DECAY_LAMBDA = 0.05 → a topic reviewed 14 days ago retains ~50% of
    its mastery estimate.  Topics reviewed today are unaffected.

Mastery update (FR-8):
    EMA: m_new = alpha * observed_score + (1-alpha) * m_old
    After decay is applied first, so a returning student who aced a
    topic 2 weeks ago still gets credit for reviewing it now.

Replan trigger (FR-2):
    Fires when updated (post-decay) mastery < DRIFT_THRESHOLD,
    or when the student has answered incorrectly N times in a row
    on the same topic (STRUGGLE_STREAK).
"""
import math
from datetime import datetime, timezone
from typing import Optional

from src.state import LearnerState
from src.logging_utils import log_node_call

EMA_ALPHA          = 0.4   # weight given to the most recent observed score
DRIFT_THRESHOLD    = 0.35  # mastery below this after update triggers replan
STRUGGLE_STREAK    = 2     # consecutive wrong/low answers on one topic also triggers replan
DECAY_LAMBDA       = 0.05  # Ebbinghaus decay rate (per day); 0.05 → ~50% retention at 14 days


def _days_since(iso_ts: Optional[str]) -> float:
    """Returns fractional days since an ISO-8601 timestamp, or 0 if None."""
    if not iso_ts:
        return 0.0
    try:
        reviewed = datetime.fromisoformat(iso_ts)
        if reviewed.tzinfo is None:
            reviewed = reviewed.replace(tzinfo=timezone.utc)
        delta = datetime.now(timezone.utc) - reviewed
        return max(delta.total_seconds() / 86_400, 0.0)
    except (ValueError, TypeError):
        return 0.0


def _apply_forgetting_decay(mastery: dict, last_reviewed: dict) -> dict:
    """
    Returns a NEW mastery dict with Ebbinghaus decay applied to every
    topic that hasn't been reviewed today.
    Topics never reviewed (no timestamp) are not decayed — they start at 0
    anyway, and decaying 0 would just produce 0 with extra noise.
    """
    decayed = {}
    for topic, m in mastery.items():
        days = _days_since(last_reviewed.get(topic))
        if days > 0 and m > 0:
            m = round(m * math.exp(-DECAY_LAMBDA * days), 4)
        decayed[topic] = max(m, 0.0)
    return decayed


def _update_mastery(old: float, correct: bool, score: Optional[float]) -> float:
    observed = score if score is not None else (1.0 if correct else 0.0)
    return round(EMA_ALPHA * observed + (1 - EMA_ALPHA) * old, 4)


def monitor_node(state: LearnerState) -> LearnerState:
    state = dict(state)
    grade = state.get("last_grade")
    topic = state.get("topic_pointer")

    # ── 1. Apply forgetting-curve decay to ALL topics first ───────────────────
    mastery      = dict(state.get("mastery", {}))
    last_reviewed = dict(state.get("last_reviewed", {}))
    mastery = _apply_forgetting_decay(mastery, last_reviewed)
    state["mastery"] = mastery

    if grade is None or topic is None:
        # decay-only pass (e.g. session opened after a gap, before any answer)
        return state

    # ── 2. EMA update for the topic just answered ──────────────────────────────
    old_mastery = mastery.get(topic, 0.0)
    new_mastery = _update_mastery(old_mastery, grade.get("correct", False), grade.get("score"))
    mastery = dict(mastery)
    mastery[topic] = new_mastery
    state["mastery"] = mastery

    # ── 3. Record that this topic was reviewed now ─────────────────────────────
    last_reviewed = dict(last_reviewed)
    last_reviewed[topic] = datetime.now(timezone.utc).isoformat()
    state["last_reviewed"] = last_reviewed

    # ── 4. Struggle-streak tracking ───────────────────────────────────────────
    engagement = dict(state["engagement"])
    streaks = dict(engagement.get("streaks", {}))
    if new_mastery < old_mastery or not grade.get("correct", False):
        streaks[topic] = streaks.get(topic, 0) + 1
    else:
        streaks[topic] = 0
    engagement["streaks"] = streaks
    state["engagement"] = engagement

    # ── 5. Replan check ───────────────────────────────────────────────────────
    drifted   = new_mastery < DRIFT_THRESHOLD
    struggling = streaks[topic] >= STRUGGLE_STREAK
    state["replan_flag"] = bool(drifted or struggling)

    # ── 6. Misconception tracking ─────────────────────────────────────────────
    if grade.get("flagged"):
        misconceptions = dict(state.get("misconceptions", {}))
        tags = list(misconceptions.get(topic, []))
        tag = grade.get("misconception_tag", "unspecified")
        if tag and tag not in tags:
            tags.append(tag)
        misconceptions[topic] = tags
        state["misconceptions"] = misconceptions

    # ── 7. Audit log ──────────────────────────────────────────────────────────
    state["session_history"] = state["session_history"] + [
        {
            "event":            "mastery_update",
            "topic":            topic,
            "mastery_before_decay": old_mastery,
            "mastery_after_update": new_mastery,
            "replan_triggered": state["replan_flag"],
            "decay_applied":    mastery.get(topic, new_mastery) != new_mastery,
        }
    ]
    state["step_count"] = state.get("step_count", 0) + 1
    return state


monitor_agent = log_node_call("monitor")(monitor_node)
