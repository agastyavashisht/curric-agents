"""
Remediation logic for the AI / Experimental group (spec §12).

After a student completes all q_per_topic practice questions on a topic,
topic_end_decision() determines what happens next:

  "remediate"            — student is weak, under the remediation cap.
                           Planner stays on same topic, Content generates a
                           DIFFERENT lesson, Assessment generates a fresh
                           non-repeated question.

  "advance"              — student is ready to move on (mastery OK or
                           remediation cap hit → mark needs_review and move).

  "advance_needs_review" — cap hit; topic marked needs_review=True so the
                           Planner will append it at the END of the sequence
                           for a later review pass (spec §12 + §4).

Avoiding infinite loops (spec §12):
  MAX_REMEDIATION_ATTEMPTS = 2
  After 2 unsuccessful remediation rounds on the same topic, the student
  is allowed to proceed. The topic is flagged needs_review=True so it
  reappears at the end of the sequence in the same session.
"""

from __future__ import annotations

MAX_REMEDIATION_ATTEMPTS = 1   # topic appears max twice: original + 1 retry
                               # (spec §12 recommended 2-3; reduced to 1 for pilot UX)

# Threshold below which we attempt remediation (same as Monitor's DRIFT_THRESHOLD)
_REMEDIATION_MASTERY_THRESHOLD = 0.45


def topic_end_decision(
    state: dict,
    replan_triggered: bool,
) -> tuple[str, dict]:
    """
    Called at the END of a topic's practice questions (q_done == q_per_topic).

    Returns (decision, updated_state) where decision is one of:
      "remediate"            — re-teach this topic (different lesson + new questions)
      "advance"              — mastery sufficient, move to next topic
      "advance_needs_review" — mastery insufficient but remediation cap hit;
                               mark for later review and advance anyway

    The state's engagement["remediation_counts"][topic] is incremented on
    each remediation decision and read to enforce the cap.
    """
    state     = dict(state)
    topic     = state.get("topic_pointer")
    mastery   = state.get("mastery", {})
    eng       = dict(state.get("engagement", {}))
    rem_counts = dict(eng.get("remediation_counts", {}))

    if topic is None:
        return "advance", state

    current_mastery = mastery.get(topic, 0.0)
    needs_remediation = (
        replan_triggered
        or current_mastery < _REMEDIATION_MASTERY_THRESHOLD
    )

    if not needs_remediation:
        # Student is doing well — advance normally
        return "advance", state

    # Student needs remediation — check the cap
    attempts_so_far = rem_counts.get(topic, 0)

    if attempts_so_far >= MAX_REMEDIATION_ATTEMPTS:
        # Cap hit — mark for review, let student advance (spec §12)
        needs_review = dict(state.get("needs_review", {}))
        needs_review[topic] = True
        state["needs_review"] = needs_review

        # NOTE: do NOT add to total_remediations here — the cap hit is not
        # a new remediation attempt; the prior attempts were already counted.
        state["engagement"] = eng

        state["session_history"] = list(state.get("session_history", [])) + [
            {
                "event":   "remediation_cap_hit",
                "topic":   topic,
                "mastery": current_mastery,
                "cap":     MAX_REMEDIATION_ATTEMPTS,
            }
        ]
        return "advance_needs_review", state

    # Remediate: increment counter and return
    rem_counts[topic] = attempts_so_far + 1
    eng["remediation_counts"] = rem_counts
    eng["total_remediations"]  = eng.get("total_remediations", 0) + 1
    state["engagement"] = eng

    # Clear the replan flag so Monitor doesn't re-trigger it mid-topic
    state["replan_flag"] = False

    state["session_history"] = list(state.get("session_history", [])) + [
        {
            "event":        "remediation_started",
            "topic":        topic,
            "mastery":      current_mastery,
            "attempt":      rem_counts[topic],
            "cap":          MAX_REMEDIATION_ATTEMPTS,
        }
    ]
    return "remediate", state
