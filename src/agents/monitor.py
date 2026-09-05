"""
Monitor Agent.

Deterministic (no LLM call) — fully auditable and reproducible.

─────────────────────────────────────────────────────────────────────
Knowledge Tracing — Bayesian Knowledge Tracing (FR-8, paper Eq. 1)
─────────────────────────────────────────────────────────────────────
Mastery of concept c for student is modelled as a latent probability
P(L_c) updated after each observed response using the BKT recurrence:

  P(L_c^{t+1}) = P(L_c^t | obs_t) · (1 − P(F))
               + (1 − P(L_c^t | obs_t)) · P(T)

where:
  P(T)  = probability of learning (not-knowing → knowing after practice)
  P(F)  = probability of forgetting (knowing → not-knowing)
  P(G)  = probability of a correct guess when not knowing
  P(S)  = probability of a slip (wrong answer when knowing)

P(L_c^t | obs_t) is computed via Bayes' rule:
  If obs_t = correct:
    P(L | correct) = P(L) · (1 − P(S))
                   / [P(L)·(1−P(S)) + (1−P(L))·P(G)]
  If obs_t = incorrect:
    P(L | incorrect) = P(L) · P(S)
                     / [P(L)·P(S) + (1−P(L))·(1−P(G))]

Default parameters (calibrated for a short-interaction pilot):
  P(T) = 0.30  — moderate learning rate per episode
  P(F) = 0.05  — slow forgetting within a session
  P(G) = 0.25  — random-guess probability for a 4-option MCQ
  P(S) = 0.10  — realistic slip rate for a knowledgeable student

─────────────────────────────────────────────────────────────────────
Forgetting-curve decay (FR-9, Ebbinghaus between sessions)
─────────────────────────────────────────────────────────────────────
Applied BEFORE the BKT update for topics not reviewed recently.
  m_decayed = m · exp(−λ · days_since_review)   λ = 0.05
This captures between-session forgetting that BKT P(F) does not
model (BKT P(F) covers within-episode slipping).

─────────────────────────────────────────────────────────────────────
Replan trigger (FR-2)
─────────────────────────────────────────────────────────────────────
Fires when updated mastery < DRIFT_THRESHOLD, or when the student
has answered incorrectly STRUGGLE_STREAK times in a row on the same
topic (signals persistent misunderstanding, not just a slip).
"""
import math
from datetime import datetime, timezone
from typing import Optional

from src.state import LearnerState
from src.logging_utils import log_node_call

# ── BKT parameters (paper §V-C) ───────────────────────────────────────────────
P_T = 0.30   # P(Transit)  — probability of learning per episode
P_F = 0.05   # P(Forget)   — within-episode forgetting
P_G = 0.25   # P(Guess)    — correct guess when not knowing (1/4 for MCQ)
P_S = 0.10   # P(Slip)     — wrong answer when knowing

# ── control thresholds ────────────────────────────────────────────────────────
MASTERY_THRESHOLD = 0.70   # topic considered mastered and skipped
DRIFT_THRESHOLD   = 0.35   # replan if mastery drops below this
STRUGGLE_STREAK   = 2      # consecutive wrong answers also triggers replan

# ── between-session Ebbinghaus decay ─────────────────────────────────────────
DECAY_LAMBDA = 0.05        # exp(-lambda * days); 0.05 → ~50% retention at 14 days


# ── BKT update functions ──────────────────────────────────────────────────────

def _bkt_posterior(prior: float, correct: bool) -> float:
    """
    Compute P(L_c^t | obs_t) via Bayes' rule.
    Paper Eq. 1, Bayes step.
    """
    if correct:
        num = prior * (1.0 - P_S)
        den = num + (1.0 - prior) * P_G
    else:
        num = prior * P_S
        den = num + (1.0 - prior) * (1.0 - P_G)
    return num / den if den > 0 else prior


def _bkt_update(prior: float, correct: bool) -> float:
    """
    Full BKT recurrence: posterior → next prior.
    Paper Eq. 1:
      P(L^{t+1}) = P(L|obs) · (1 − P(F)) + (1 − P(L|obs)) · P(T)
    Returns new mastery estimate clamped to [0, 1].
    """
    posterior = _bkt_posterior(prior, correct)
    new_mastery = posterior * (1.0 - P_F) + (1.0 - posterior) * P_T
    return round(min(max(new_mastery, 0.0), 1.0), 4)


# ── between-session Ebbinghaus decay ─────────────────────────────────────────

def _days_since(iso_ts: Optional[str]) -> float:
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
    Ebbinghaus decay for topics not reviewed today (between-session forgetting).
    Topics never reviewed are not decayed — they start at P(L_0) ≈ 0.
    """
    decayed = {}
    for topic, m in mastery.items():
        days = _days_since(last_reviewed.get(topic))
        if days > 0 and m > 0:
            m = round(m * math.exp(-DECAY_LAMBDA * days), 4)
        decayed[topic] = max(m, 0.0)
    return decayed


# ── main node ─────────────────────────────────────────────────────────────────

def monitor_node(state: LearnerState) -> LearnerState:
    state = dict(state)
    grade = state.get("last_grade")
    topic = state.get("topic_pointer")

    # ── 1. Apply between-session forgetting decay to ALL topics ───────────────
    mastery       = dict(state.get("mastery", {}))
    last_reviewed = dict(state.get("last_reviewed", {}))
    mastery = _apply_forgetting_decay(mastery, last_reviewed)
    state["mastery"] = mastery

    if grade is None or topic is None:
        # decay-only pass (session opened after a gap, before any answer)
        return state

    # ── 2. BKT update for the topic just answered (paper Eq. 1) ──────────────
    prior       = mastery.get(topic, 0.0)
    correct     = bool(grade.get("correct", False))
    # For short-answer, use the normalised score to decide correct/incorrect
    # (score >= 0.6 = correct per grading rubric threshold)
    score = grade.get("score")
    if score is not None:
        correct = score >= 0.6
    new_mastery = _bkt_update(prior, correct)

    mastery = dict(mastery)
    mastery[topic] = new_mastery
    state["mastery"] = mastery

    # ── 3. Record that this topic was reviewed now ────────────────────────────
    last_reviewed        = dict(last_reviewed)
    last_reviewed[topic] = datetime.now(timezone.utc).isoformat()
    state["last_reviewed"] = last_reviewed

    # ── 4. Struggle-streak tracking ───────────────────────────────────────────
    engagement = dict(state["engagement"])
    streaks    = dict(engagement.get("streaks", {}))
    if not correct:
        streaks[topic] = streaks.get(topic, 0) + 1
    else:
        streaks[topic] = 0
    engagement["streaks"] = streaks
    state["engagement"]   = engagement

    # ── 5. Replan check ───────────────────────────────────────────────────────
    drifted    = new_mastery < DRIFT_THRESHOLD
    struggling = streaks[topic] >= STRUGGLE_STREAK
    state["replan_flag"] = bool(drifted or struggling)

    # ── 6. Misconception tracking ─────────────────────────────────────────────
    if grade.get("flagged"):
        misconceptions = dict(state.get("misconceptions", {}))
        tags = list(misconceptions.get(topic, []))
        tag  = grade.get("misconception_tag", "unspecified")
        if tag and tag not in tags:
            tags.append(tag)
        misconceptions[topic] = tags
        state["misconceptions"] = misconceptions

    # ── 7. Audit log — full BKT trace for reproducibility (NFR auditability) ──
    state["session_history"] = state["session_history"] + [
        {
            "event":            "mastery_update",
            "topic":            topic,
            "bkt_prior":        prior,
            "bkt_posterior":    round(_bkt_posterior(prior, correct), 4),
            "bkt_new_mastery":  new_mastery,
            "bkt_params":       {"P_T": P_T, "P_F": P_F, "P_G": P_G, "P_S": P_S},
            "correct":          correct,
            "replan_triggered": state["replan_flag"],
        }
    ]
    state["step_count"] = state.get("step_count", 0) + 1
    return state


monitor_agent = log_node_call("monitor")(monitor_node)
