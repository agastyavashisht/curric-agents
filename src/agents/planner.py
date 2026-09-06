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
def _pretest_bootstrap(
    mastery: dict,
    topics: dict,
    pretest_score_pct: float | None,
    pretest_per_topic: dict | None = None,
) -> dict:
    """
    Layer 1 — Pre-test bootstrap.

    If per-topic scores are available (new sectioned test format), use them
    directly — each topic's mastery is set to its actual section score.
    This is the accurate path: a student who got 5/5 on loops gets mastery=1.0
    and skips it entirely; a student who got 2/5 gets mastery=0.40 and sees it.

    Falls back to positional estimation from overall score if per-topic data
    is not available (old flat test format).
    """
    if pretest_score_pct is None:
        return mastery

    mastery = dict(mastery)

    # ── ACCURATE PATH: per-topic scores available (new sectioned test) ────────
    if pretest_per_topic:
        for topic, scores in pretest_per_topic.items():
            if topic not in topics:
                continue
            if topic in mastery and mastery[topic] > 0:
                continue   # real history exists — don't overwrite
            pct = scores.get("pct", 0.0)
            # Direct mapping: topic score % → mastery estimate
            # 100% → 1.0, 80% → 0.85, 60% → 0.72, 40% → 0.52, 20% → 0.30, 0% → 0.05
            # Apply a slight conservative discount so students still touch topics
            # where they borderline passed rather than completely skipping them.
            raw = pct / 100.0
            # Slight conservative discount for borderline topics (60-80%):
            # we want 100% to skip (≥0.70), 80% to skip, 60% to borderline show
            mastery[topic] = round(raw * 0.95, 3)
        return mastery

    # ── FALLBACK PATH: only overall score available (old flat test) ───────────
    score      = max(0.0, min(pretest_score_pct / 100.0, 1.0))
    plain_topo = _topological_order(topics, {}, mastery_threshold=1.1)
    n          = len(plain_topo)

    for rank, topic in enumerate(plain_topo):
        if topic in mastery and mastery[topic] > 0:
            continue
        positional = 1.0 - (rank / max(n - 1, 1))
        initial    = round(score * (0.4 + 0.6 * positional), 3)
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
        return "remember"
    elif mastery_score < 0.45:
        return "understand"
    elif mastery_score < 0.65:
        return "apply"
    elif mastery_score < 0.80:
        return "analyze"
    else:
        return "evaluate"


# ── Paper Eq. 2: instructional-value heuristic ────────────────────────────────
# c* = argmax  w1·(1 − P(Lc)) + w2·recency(c) − w3·redundancy(c)
#
# w1 = mastery gap weight  (prioritise what the student doesn't know)
# w2 = recency weight      (prioritise topics not reviewed recently)
# w3 = redundancy penalty  (penalise re-teaching already mastered topics)
#
# recency(c)    = days_since_last_review / 14  capped at 1.0
# redundancy(c) = 1.0 if mastery >= MASTERY_THRESHOLD else 0.0

W1 = 0.6   # mastery-gap weight
W2 = 0.3   # recency weight
W3 = 0.5   # redundancy penalty


def _instructional_value(
    topic: str,
    mastery: dict,
    last_reviewed: dict,
    mastery_threshold: float = MASTERY_THRESHOLD,
) -> float:
    """
    Computes the instructional-value heuristic from paper Eq. 2.

    c* = argmax  w1·(1 − P(Lc)) + w2·recency(c) − w3·redundancy(c)

    Returns a scalar; higher = more valuable to teach next.
    Used to rank the topologically-valid baseline before LLM re-rank.
    """
    from datetime import datetime, timezone
    p_lc = mastery.get(topic, 0.0)
    mastery_gap = 1.0 - p_lc

    # recency: fraction of a 14-day window since last review (0 = reviewed today)
    ts = last_reviewed.get(topic)
    if ts:
        try:
            reviewed = datetime.fromisoformat(ts)
            if reviewed.tzinfo is None:
                reviewed = reviewed.replace(tzinfo=timezone.utc)
            days = (datetime.now(timezone.utc) - reviewed).total_seconds() / 86_400
            recency = min(days / 14.0, 1.0)
        except (ValueError, TypeError):
            recency = 0.5
    else:
        recency = 0.5   # never reviewed → neutral recency

    redundancy = 1.0 if p_lc >= mastery_threshold else 0.0

    return W1 * mastery_gap + W2 * recency - W3 * redundancy


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


def _iv_topological_sort(
    topics: dict,
    mastery: dict,
    iv_scores: dict,
    mastery_threshold: float = MASTERY_THRESHOLD,
) -> list:
    """
    Greedy topological sort that at each step picks the READY topic
    with the highest instructional value (paper Eq. 2).
    'Ready' = all prerequisites already placed in the output.
    Returns empty list on cycle (caller falls back to plain topo sort).
    """
    mastered  = {t for t in topics if mastery.get(t, 0.0) >= mastery_threshold}
    remaining = {
        t: set(info["prerequisites"])
        for t, info in topics.items()
        if t not in mastered
    }
    order     = []
    satisfied = set(mastered)

    while remaining:
        ready = [t for t, prereqs in remaining.items() if not (prereqs - satisfied)]
        if not ready:
            return []   # cycle — caller uses plain baseline
        # pick highest instructional value among ready topics
        best = max(ready, key=lambda t: iv_scores.get(t, 0.0))
        order.append(best)
        satisfied.add(best)
        del remaining[best]

    return order


def _llm_rerank(
    domain: str, baseline: List[str], topics: dict,
    mastery: dict, misconceptions: dict,
) -> tuple[List[str], str]:
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
    pretest_pct       = state.get("pretest_score_pct")
    pretest_per_topic = state.get("pretest_per_topic")   # per-topic scores from sectioned test
    mastery = _pretest_bootstrap(
        dict(state.get("mastery", {})), topics, pretest_pct, pretest_per_topic
    )
    state["mastery"] = mastery
    if pretest_pct is not None:
        state["pretest_score_pct"] = None

    # ── Layer 2 + 3: skip mastered, build topo baseline ──────────────────────
    baseline = _topological_order(topics, mastery)

    # ── Layer 3b: sort baseline by paper Eq. 2 instructional-value heuristic ──
    # Within the topologically-valid ordering, promote high-value topics.
    # We do a stable sort that only reorders topics whose prerequisites are
    # already satisfied at every position (preserves validity).
    last_reviewed  = state.get("last_reviewed", {})
    ablation_mode  = state.get("ablation_mode", "full")

    if ablation_mode in ("full", "planner_only"):
        # Apply Eq. 2 heuristic sort over the baseline
        iv_scores = {
            t: _instructional_value(t, mastery, last_reviewed)
            for t in baseline
        }
        # Re-sort: build a valid order greedily — at each step pick the
        # ready topic with the highest instructional value.
        sorted_baseline = _iv_topological_sort(topics, mastery, iv_scores)
        if sorted_baseline:
            baseline = sorted_baseline

    # ── Layer 4: LLM re-rank with mastery gaps + misconceptions ──────────────
    misconceptions = state.get("misconceptions", {})
    if ablation_mode in ("full", "planner_only"):
        new_sequence, rationale = _llm_rerank(
            state["domain"], baseline, topics, mastery, misconceptions
        )
    else:
        # Ablation: no adaptive planning — keep fixed topo order
        new_sequence = baseline
        rationale    = f"ablation_mode={ablation_mode}: fixed sequence"

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
            "event":          "plan",
            "sequence":       new_sequence,
            "baseline":       baseline,
            "llm_reranked":   new_sequence != baseline,
            "rationale":      rationale,
            "pretest_used":   pretest_pct is not None,
            "topics_skipped": len(topics) - len(new_sequence),
            "bloom_plan":     bloom_plan,
            "ablation_mode":  ablation_mode,
            "iv_scores":      {t: round(_instructional_value(t, mastery, last_reviewed), 4)
                               for t in new_sequence},
        }
    ]
    return state


planner_agent = log_node_call("planner")(planner_node)
