"""
Smoke test: personalised path logic.
Run: python -m tests.test_personalisation
"""
import sys, os
sys.path.insert(0, '.')

from src.agents.planner import (
    _pretest_bootstrap, _topological_order, _bloom_level, _load_graph, planner_node
)
from src.memory.store import load_state

PASS = []
FAIL = []

def check(name, cond, detail=''):
    if cond:
        PASS.append(name)
        print(f"  PASS  {name}")
    else:
        FAIL.append(name)
        print(f"  FAIL  {name}" + (f" -- {detail}" if detail else ""))

topics = _load_graph('python_programming')

# ── Test 1: Pre-test bootstrap 80% ──────────────────────────────────
print("\n=== Test 1: Pre-test bootstrap (80% score) ===")
m_80 = _pretest_bootstrap({}, topics, 80.0)
early_topics_high = any(v > 0.5 for v in m_80.values())
all_nonzero = all(v > 0 for v in m_80.values())
check("80% score → some topics > 0.5 mastery",  early_topics_high)
check("80% score → all topics get initial mastery", all_nonzero)
check("max mastery ≤ 1.0",  max(m_80.values()) <= 1.0)
print("  Sample mastery after 80% pretest:")
for t, v in list(m_80.items())[:4]:
    print(f"    {t}: {v:.3f}")

# ── Test 2: Pre-test bootstrap 20% ──────────────────────────────────
print("\n=== Test 2: Pre-test bootstrap (20% score) ===")
m_20 = _pretest_bootstrap({}, topics, 20.0)
skipped_20 = [t for t, v in m_20.items() if v >= 0.7]
check("20% score → no topics reach 0.7 mastery",  len(skipped_20) == 0,
      f"skipped: {skipped_20}")
check("20% score max < 80% score max",  max(m_20.values()) < max(m_80.values()))

# ── Test 3: Topics skipped based on pretest ──────────────────────────
print("\n=== Test 3: Topics skipped based on pretest ===")
order_zero = _topological_order(topics, {})
order_80   = _topological_order(topics, m_80)
check("0% student sees all topics",   len(order_zero) == len(topics))
check("80% student sees fewer topics", len(order_80) < len(order_zero),
      f"zero={len(order_zero)} vs 80pct={len(order_80)}")
print(f"  Full order ({len(order_zero)} topics): {order_zero[:4]}...")
print(f"  After 80% pretest ({len(order_80)} topics): {order_80[:4]}...")
print(f"  Topics skipped: {len(order_zero) - len(order_80)}")

# ── Test 4: Bloom level mapping ──────────────────────────────────────
print("\n=== Test 4: Bloom level mapping ===")
expected_blooms = [
    (0.05, "remember"),
    (0.30, "understand"),
    (0.50, "apply"),
    (0.70, "analyze"),
    (0.85, "evaluate"),
]
for mastery_val, expected_bloom in expected_blooms:
    got = _bloom_level(mastery_val)
    check(f"mastery={mastery_val} -> {expected_bloom}", got == expected_bloom,
          f"got '{got}'")

# ── Test 5: Full planner_node with pretest score ─────────────────────
print("\n=== Test 5: Full planner_node with pretest_score_pct=60% ===")
DB = "results/_plan_test.db"
state = load_state(DB, "S01", "python_programming")
state["pretest_score_pct"] = 60.0
state = planner_node(state)

check("topic_sequence generated",        len(state["topic_sequence"]) > 0)
check("topic_pointer set",               state["topic_pointer"] is not None)
check("bloom_plan populated",            len(state["bloom_plan"]) > 0)
check("bloom_plan matches sequence",     len(state["bloom_plan"]) == len(state["topic_sequence"]))
check("pretest_score_pct cleared",       state.get("pretest_score_pct") is None)
check("session_history has plan event",  any(e.get("event") == "plan"
                                           for e in state["session_history"]))
check("plan notes topics_skipped",       state["session_history"][-1].get("topics_skipped", -1) >= 0)

print("\n  First 4 topics in personalised plan:")
for b in state["bloom_plan"][:4]:
    print(f"    {b['topic']:35s}  bloom={b['bloom_level']:10s}  gap={b['mastery_gap']:.2f}")

os.remove(DB)

# ── Test 6: Misconception prioritisation ─────────────────────────────
print("\n=== Test 6: Misconception in state used by planner ===")
DB2 = "results/_plan_test2.db"
state2 = load_state(DB2, "S02", "python_programming")
# Give student decent mastery everywhere except one topic with a misconception
for t in topics:
    state2["mastery"][t] = 0.6
mid_topic = list(topics.keys())[3]
state2["mastery"][mid_topic] = 0.2
state2["misconceptions"][mid_topic] = ["confuses_assignment_with_comparison"]
state2 = planner_node(state2)
seq = state2["topic_sequence"]
check("misconception topic in plan",  mid_topic in seq,
      f"topic={mid_topic}, seq={seq}")
os.remove(DB2)

# ── Summary ────────────────────────────────────────────────────────────
print("\n" + "=" * 50)
print(f"  {len(PASS)} PASSED  |  {len(FAIL)} FAILED")
if FAIL:
    print("\n  FAILED:")
    for f in FAIL:
        print(f"    x {f}")
else:
    print("  All personalisation checks passed.")
print("=" * 50)
sys.exit(0 if not FAIL else 1)
