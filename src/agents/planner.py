"""
Planner Agent  — personalised curriculum sequencing.

Personalisation layers (applied in order):
─────────────────────────────────────────────
1. Pre-test bootstrap: if the student just completed a pre-test, set initial
   mastery estimates from their score so topics they answered correctly get
   credit immediately (they skip what they know).

2. Mastery-gated skip: topics whose mastery >= MASTERY_THRESHOLD are removed
   from the plan entirely — the student has already demonstrated competence.

3. Kahn topological sort: produces a prerequisite-valid baseline order over
   the remaining topics.

4. LLM re-rank: the LLM is given the student's full mastery vector AND known
   misconceptions and asked to promote topics with bigger gaps AND known
   misconceptions to earlier positions, without violating prerequisites.
   If the LLM output is invalid the deterministic baseline is kept (safe
   fallback, FR-2).

5. Bloom-level annotation: each topic in the returned plan carries the
   recommended Bloom level for this student based on their mastery — ensuring
   the Content and Assessment agents generate depth-appropriate material.
"""
import json
from typing import List

from src.state import LearnerState
from src.logging_utils import log_node_call
from src.config import get_llm, extract_content, _strip_fences

MASTERY_THRESHOLD = 0.70    # skip topics at or above this mastery


# ── graph loading ─────────────────────────────────────────────────────────────
def _load_graph(domain: str) -> dict:
    domain_to_file = {
        "python_programming": "data/prerequisite_graph.json",
        "ml_basics":          "data/ml_basics_graph.json",
        "signal_processing":  "data/signal_processing_graph.json",
        "physiology_basics":  "data/physiology_graph.json",
        "data_analysis":      "data/data_analysis_graph.json",
    }
    path = domain_to_file.get(domain)
    if path is None:
        raise ValueError(
            f"Unknown domain '{domain}'. "
            f"Supported: {list(domain_to_file.keys())}"
        )
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["topics"]


# ── personalisation helpers ───────────────────────────────────────────────────
def _pretest_bootstrap(mastery: dict, topics: dict, pretest_score_pct: float | None) -> dict:
    """
    Layer 1 — Pre-test bootstrap.

    Convert a raw pre-test % score into initial per-topic mastery estimates.
    Strategy: assume the student's knowledge is proportional to their total
    score and map it linearly across the topic difficulty ordering.

    Topics in the early part of the topological order get higher initial
    mastery; later (harder) topics get lower, scaled by the overall score.
    This gives a smooth, conservative starting point rather than treating
    every topic as zero.
    """
    if pretest_score_pct is None:
        return mastery

    mastery = dict(mastery)
    score   = max(0.0, min(pretest_score_pct / 100.0, 1.0))

    # Sort topics by their position in a plain topological order (no mastery)
    plain_topo = _topological_order(topics, {}, mastery_threshold=1.1)  # threshold > 1 → skip nothing
    n = len(plain_topo)

    for rank, topic in enumerate(plain_topo):
        if topic in mastery and mastery[topic] > 0:
            continue    # student already has real history — don't overwrite
        # Earlier (easier) topics benefit more from a high score.
        # Positional weight: 1.0 for first topic, 0.0 for last.
        positional = 1.0 - (rank / max(n - 1, 1))
        # Blend: for a 100% score, early topics start at 1.0; for 0%, all start at 0.
        # For a 60% score, early topic ≈ 0.60, late topic ≈ 0.
        initial = round(score * (0.4 + 0.6 * positional), 3)
        mastery[topic] = initial

    return mastery


def _topological_order(topics: dict, mastery: dict,
                        mastery_threshold: float = MASTERY_THRESHOLD) -> List[str]:
    """Kahn's algorithm; skips already-mastered topics."""
    mastered  = {t for t in topics if mastery.get(t, 0.0) >= mastery_threshold}
    remaining = {t: set(info["prerequisites"]) for t, info in topics.items()
                 if t not in mastered}
    order     = []
    satisfied = set(mastered)

    while remaining:
        ready = sorted(t for t, prereqs in remaining.items() if not (prereqs - satisfied))
        if not ready:
            # cycle or unsatisfiable — return what we have
            order.extend(sorted(remaining.keys()))
            break
        order.append(ready[0])
        satisfied.add(ready[0])
        del remaining[ready[0]]

    return order


def _is_valid_order(sequence: list, topics: dict, mastery: dict,
                    threshold: float = MASTERY_THRESHOLD) -> bool:
    mastered = {t for t in topics if mastery.get(t, 0.0) >= threshold}
    seen     = set(mastered)
    if set(sequence) != (set(topics) - mastered):
        return False
    for topic in sequence:
        if set(topics[topic].get("prerequisites", [])) - seen:
            return False
        seen.add(topic)
    return True


def _bloom_level(mastery_score: float) -> str:
    """Map current mastery to a recommended Bloom cognitive level."""
    if mastery_score < 0.25:
        return "remember"         # recall definitions and facts
    elif mastery_score < 0.45:
        return "understand"       # explain in own words
    elif mastery_score < 0.65:
        return "apply"            # use in a new example
    elif mastery_score < 0.80:
        return "analyze"          # break down and compare
    else:
        return "evaluate"         # critique and justify


# ── LLM re-rank ───────────────────────────────────────────────────────────────
_RERANK_PROMPT = """You are a curriculum planner personalising a learning path for one student.

A prerequisite-valid BASELINE order is provided. Re-rank it so that:
  1. Topics where the student has LARGER mastery gaps appear EARLIER.
  2. Topics with KNOWN MISCONCEPTIONS are prioritised over equally-gapped topics.
  3. NO prerequisite constraint is violated.

You may only move a topic earlier if ALL its prerequisites appear before it.

Domain: {domain}
Baseline order (prerequisite-valid): {order}
Student mastery per topic (0=none, 1=mastered; omitted = 0): {mastery}
Known misconceptions per topic (empty list = none identified): {misconceptions}
Prerequisites map (topic -> required prior topics): {prereqs}

Respond ONLY as JSON:
{{"topic_sequence": ["topic_id", ...], "rationale": "one sentence"}}
No markdown fences. topic_sequence must contain EXACTLY the same ids as the baseline.
"""


def _llm_rerank(domain: str, baseline: List[str], topics: dict,
                mastery: dict, misconceptions: dict) -> tuple[List[str], str]:
    """
    Returns (reranked_sequence, rationale).
    Falls back to baseline if LLM output is invalid.
    """
    if not baseline:
        return baseline, "empty plan"

    llm = get_llm("planner")
    prereqs = {t: topics[t].get("prerequisites", []) for t in baseline}
    # include only topics in the plan
    m_sub = {t: round(mastery.get(t, 0.0), 3) for t in baseline}
    mi_sub = {t: misconceptions.get(t, []) for t in baseline if misconceptions.get(t)}

    prompt = _RERANK_PROMPT.format(
        domain=domain,
        order=json.dumps(baseline),
        mastery=json.dumps(m_sub),
        misconceptions=json.dumps(mi_sub),
        prereqs=json.dumps(prereqs),
    )
    try:
        raw      = _strip_fences(extract_content(llm.invoke(prompt)))
        parsed   = json.loads(raw)
        proposed = parsed.get("topic_sequence", [])
        rationale = parsed.get("rationale", "LLM re-ranked")
        if _is_valid_order(proposed, topics, mastery):
            return proposed, rationale
        print("[PLANNER] LLM order violated prerequisites — keeping baseline")
    except Exception as e:
        print(f"[PLANNER] LLM re-rank failed ({e}) — keeping baseline")
    return baseline, "baseline (LLM fallback)"


# ── main node ─────────────────────────────────────────────────────────────────
def planner_node(state: LearnerState) -> LearnerState:
    state  = dict(state)
    topics = _load_graph(state["domain"])

    # ── Layer 1: pre-test bootstrap ───────────────────────────────────────────
    pretest_pct = state.get("pretest_score_pct")        # set by app.py after pretest
    mastery     = _pretest_bootstrap(
        dict(state.get("mastery", {})), topics, pretest_pct
    )
    # Only bootstrap once (clear the trigger so we don't re-run next session)
    state["mastery"] = mastery
    if pretest_pct is not None:
        state["pretest_score_pct"] = None

    # ── Layer 2 + 3: skip mastered, build baseline ────────────────────────────
    baseline = _topological_order(topics, mastery)

    # ── Layer 4: LLM re-rank with mastery gaps + misconceptions ──────────────
    misconceptions = state.get("misconceptions", {})
    new_sequence, rationale = _llm_rerank(
        state["domain"], baseline, topics, mastery, misconceptions
    )

    # ── Layer 5: annotate with Bloom levels ───────────────────────────────────
    bloom_plan = [
        {
            "topic":       t,
            "bloom_level": _bloom_level(mastery.get(t, 0.0)),
            "mastery_gap": round(1.0 - mastery.get(t, 0.0), 3),
        }
        for t in new_sequence
    ]

    # Store in state
    state["topic_sequence"] = new_sequence
    state["bloom_plan"]     = bloom_plan          # used by Content + Assessment
    state["topic_pointer"]  = new_sequence[0] if new_sequence else None
    state["replan_flag"]    = False

    state["session_history"] = list(state.get("session_history", [])) + [
        {
            "event":            "plan",
            "sequence":         new_sequence,
            "baseline":         baseline,
            "llm_reranked":     new_sequence != baseline,
            "rationale":        rationale,
            "pretest_used":     pretest_pct is not None,
            "topics_skipped":   len(topics) - len(new_sequence),
            "bloom_plan":       bloom_plan,
        }
    ]
    return state


planner_agent = log_node_call("planner")(planner_node)
