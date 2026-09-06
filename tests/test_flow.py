"""
Comprehensive test-flow checker.
Tests every aspect of the pre-test / post-test / session routing logic.
Run: python -m tests.test_flow
"""
import sys, os, json, copy
sys.path.insert(0, '.')

PASS = []
FAIL = []

def ok(name, detail=''):
    PASS.append(name)
    print(f"  OK    {name}" + (f"  ({detail})" if detail else ""))

def fail(name, detail=''):
    FAIL.append(name)
    print(f"  FAIL  {name}" + (f"  -- {detail}" if detail else ""))

def check(name, cond, detail=''):
    ok(name, detail) if cond else fail(name, detail)

# ══════════════════════════════════════════════════════════════════
print("\n=== 1. DATA FILES ===")

import json as _json

# Pretest / posttest structure
for fname, label in [
    ("data/pretest_python.json",  "pretest"),
    ("data/posttest_python.json", "posttest"),
]:
    with open(fname) as f:
        data = _json.load(f)

    has_sections = "sections" in data
    check(f"{label}: has 'sections' key",    has_sections)

    if has_sections:
        secs = data["sections"]
        check(f"{label}: 5 sections",        len(secs) == 5,        f"got {len(secs)}")
        for sec in secs:
            t  = sec["topic"]
            qs = sec["questions"]
            check(f"{label}: {t} has 5 questions", len(qs) == 5, f"got {len(qs)}")
            for q in qs:
                check(f"{label}: {t}/{q['id']} has correct_index 0-3",
                      0 <= q["correct_index"] <= 3, str(q["correct_index"]))
                check(f"{label}: {t}/{q['id']} has 4 options",
                      len(q["options"]) == 4, str(len(q["options"])))

# Pretest and posttest cover the same 5 topics
pre_topics  = {s["topic"] for s in _json.load(open("data/pretest_python.json"))["sections"]}
post_topics = {s["topic"] for s in _json.load(open("data/posttest_python.json"))["sections"]}
check("pre and post cover same topics", pre_topics == post_topics,
      f"pre={pre_topics} post={post_topics}")

# No duplicate question IDs within a test
for fname, label in [("data/pretest_python.json","pre"),("data/posttest_python.json","post")]:
    all_ids = []
    for sec in _json.load(open(fname))["sections"]:
        all_ids += [q["id"] for q in sec["questions"]]
    check(f"{label}: no duplicate question IDs", len(all_ids) == len(set(all_ids)),
          f"dupes: {[x for x in all_ids if all_ids.count(x)>1]}")

# Pretest and posttest have different question prompts (parallel forms)
pre_prompts  = {q["prompt"] for sec in _json.load(open("data/pretest_python.json"))["sections"] for q in sec["questions"]}
post_prompts = {q["prompt"] for sec in _json.load(open("data/posttest_python.json"))["sections"] for q in sec["questions"]}
overlap = pre_prompts & post_prompts
check("pre and post have no overlapping questions", len(overlap) == 0,
      f"{len(overlap)} shared: {list(overlap)[:2]}")

# Prerequisite graph has 5 topics matching test sections
with open("data/prerequisite_graph.json") as f:
    graph = _json.load(f)
graph_topics = set(graph["topics"].keys())
check("graph has exactly 5 topics",         len(graph_topics) == 5, str(graph_topics))
check("graph topics match test sections",   graph_topics == pre_topics,
      f"graph={graph_topics} test={pre_topics}")

# ══════════════════════════════════════════════════════════════════
print("\n=== 2. SCORING LOGIC ===")

# _score_sectioned
sys.path.insert(0, '.')

# simulate _score_sectioned manually (can't import app.py safely here)
def score_sectioned(sections, answers):
    n_correct, n_total, per_topic = 0, 0, {}
    for sec in sections:
        topic = sec["topic"]
        t_corr, t_tot = 0, len(sec["questions"])
        for q in sec["questions"]:
            if answers.get(q["id"]) == q["correct_index"]:
                t_corr += 1
                n_correct += 1
            n_total += 1
        per_topic[topic] = {"correct": t_corr, "total": t_tot,
                            "pct": round(t_corr/t_tot*100,1) if t_tot else 0.0}
    return n_correct, n_total, per_topic

pre_data = _json.load(open("data/pretest_python.json"))["sections"]

# All correct
answers_all_correct = {q["id"]: q["correct_index"]
                       for sec in pre_data for q in sec["questions"]}
nc, nt, pt = score_sectioned(pre_data, answers_all_correct)
check("all correct → 25/25",              nc == 25 and nt == 25, f"{nc}/{nt}")
check("all correct → 100% per topic",
      all(v["pct"] == 100.0 for v in pt.values()))

# All wrong
answers_all_wrong = {q["id"]: (q["correct_index"]+1) % 4
                     for sec in pre_data for q in sec["questions"]}
nc, nt, pt = score_sectioned(pre_data, answers_all_wrong)
check("all wrong → 0/25",                nc == 0 and nt == 25, f"{nc}/{nt}")
check("all wrong → 0% per topic",
      all(v["pct"] == 0.0 for v in pt.values()))

# One topic perfect, others zero
answers_v_only = {q["id"]: (q["correct_index"]
                  if sec["topic"] == "variables_datatypes"
                  else (q["correct_index"]+1) % 4)
                  for sec in pre_data for q in sec["questions"]}
nc, nt, pt = score_sectioned(pre_data, answers_v_only)
check("only variables correct → 5/25",   nc == 5, f"got {nc}")
check("variables topic 100%",            pt["variables_datatypes"]["pct"] == 100.0)
check("other topics 0%",
      all(pt[t]["pct"] == 0.0 for t in pt if t != "variables_datatypes"))

# ══════════════════════════════════════════════════════════════════
print("\n=== 3. PRETEST BOOTSTRAP (per-topic) ===")

from src.agents.planner import _pretest_bootstrap, _topological_order, MASTERY_THRESHOLD
from src.agents.planner import _load_graph

topics = _load_graph("python_programming")

# Test per-topic bootstrap: 100% on loops should give mastery ≥ 0.70
per_topic_perfect_loops = {
    "variables_datatypes": {"correct": 0, "total": 5, "pct": 0.0},
    "control_flow":        {"correct": 0, "total": 5, "pct": 0.0},
    "loops":               {"correct": 5, "total": 5, "pct": 100.0},
    "functions":           {"correct": 0, "total": 5, "pct": 0.0},
    "oop_basics":          {"correct": 0, "total": 5, "pct": 0.0},
}
m = _pretest_bootstrap({}, topics, 20.0, per_topic_perfect_loops)
check("loops 100% → mastery ≥ 0.70 (skip)",
      m.get("loops", 0) >= MASTERY_THRESHOLD,
      f"got {m.get('loops')}")
check("variables 0% → mastery < 0.70 (show)",
      m.get("variables_datatypes", 1.0) < MASTERY_THRESHOLD,
      f"got {m.get('variables_datatypes')}")

# 96% student (24/25): only 1 wrong on loops → loops shown, rest skipped
per_topic_96 = {
    "variables_datatypes": {"correct": 5, "total": 5, "pct": 100.0},
    "control_flow":        {"correct": 5, "total": 5, "pct": 100.0},
    "loops":               {"correct": 4, "total": 5, "pct": 80.0},
    "functions":           {"correct": 5, "total": 5, "pct": 100.0},
    "oop_basics":          {"correct": 5, "total": 5, "pct": 100.0},
}
m96 = _pretest_bootstrap({}, topics, 96.0, per_topic_96)
skipped = [t for t,v in m96.items() if v >= MASTERY_THRESHOLD]
shown   = [t for t,v in m96.items() if v < MASTERY_THRESHOLD]
check("96% student: variables skipped",  "variables_datatypes" in skipped, str(skipped))
check("96% student: functions skipped",  "functions" in skipped,           str(skipped))
check("96% student: loops shown (80%)",  "loops" in shown,                 str(shown))
print(f"    96% mastery: {m96}")

# 0% student → nothing skipped
m0 = _pretest_bootstrap({}, topics, 0.0, {t: {"correct":0,"total":5,"pct":0.0} for t in topics})
check("0% student: nothing skipped",    all(v < MASTERY_THRESHOLD for v in m0.values()),
      str(m0))

# ══════════════════════════════════════════════════════════════════
print("\n=== 4. SESSION ROUTING ===")

from src.memory.store import load_state, save_state
from src.db import save_pilot_score, get_pilot_scores

DB = "results/_test_routing.db"

# Helper: reset student
def reset(sid):
    import sqlite3, os
    if os.path.exists(DB):
        conn = sqlite3.connect(DB)
        conn.execute("DELETE FROM pilot_scores WHERE student_id=?", (sid,))
        conn.execute("DELETE FROM learner_state WHERE student_id=?", (sid,))
        conn.commit(); conn.close()

# Scenario 1: Brand new student — no scores → pretest
reset("R01")
scores = [s for s in get_pilot_scores(DB) if s["student_id"]=="R01"]
has_pre  = any(s["test_type"]=="pretest"  for s in scores)
has_post = any(s["test_type"]=="posttest" for s in scores)
check("new student: has_pretest=False",  not has_pre)
check("new student: has_posttest=False", not has_post)
# routing would go to "pretest" ✓

# Scenario 2: Pretest done, no posttest → gen_lesson
save_pilot_score("R02","python_programming","pretest",60.0,15,25,"experimental","full",DB)
scores2 = [s for s in get_pilot_scores(DB) if s["student_id"]=="R02"]
has_pre2  = any(s["test_type"]=="pretest"  for s in scores2)
has_post2 = any(s["test_type"]=="posttest" for s in scores2)
check("post-pretest: has_pretest=True",  has_pre2)
check("post-pretest: has_posttest=False", not has_post2)

# Scenario 3: Both done → practice_menu or done
save_pilot_score("R03","python_programming","pretest", 60.0,15,25,"experimental","full",DB)
save_pilot_score("R03","python_programming","posttest",75.0,19,25,"experimental","full",DB)
scores3 = [s for s in get_pilot_scores(DB) if s["student_id"]=="R03"]
has_pre3  = any(s["test_type"]=="pretest"  for s in scores3)
has_post3 = any(s["test_type"]=="posttest" for s in scores3)
check("post-posttest: has_pretest=True",  has_pre3)
check("post-posttest: has_posttest=True", has_post3)
# routing: check weak topics → practice_menu or done

# Scenario 4: Pretest cannot be retaken
save_pilot_score("R04","python_programming","pretest",50.0,12,25,"experimental","full",DB)
# Try saving again (different score) — ON CONFLICT update should keep/overwrite
save_pilot_score("R04","python_programming","pretest",99.0,24,25,"experimental","full",DB)
scores4 = [s for s in get_pilot_scores(DB) if s["student_id"]=="R04" and s["test_type"]=="pretest"]
check("pretest: ON CONFLICT updates (not duplicates)", len(scores4) == 1, f"got {len(scores4)}")
# Score should be 99.0 (latest)
check("pretest: latest score stored", scores4[0]["score_pct"] == 99.0, str(scores4[0]["score_pct"]))

# Cleanup
import os as _os; _os.remove(DB) if _os.path.exists(DB) else None

# ══════════════════════════════════════════════════════════════════
print("\n=== 5. DB SCHEMA: save_pilot_score signature ===")

import inspect
from src.db import save_pilot_score as sps
sig = inspect.signature(sps)
params = list(sig.parameters.keys())
check("save_pilot_score has ablation_mode param", "ablation_mode" in params, str(params))
check("save_pilot_score has db_path param",       "db_path"       in params)

# ══════════════════════════════════════════════════════════════════
print("\n=== 6. APP.PY TEST PHASES ===")

app = open("app.py", encoding="utf-8").read()

check("phase_pretest defined",           "def phase_pretest" in app)
check("phase_posttest defined",          "def phase_posttest" in app)
check("phase_practice_menu defined",     "def phase_practice_menu" in app)
check("pretest DB guard (already_done)", "already_done" in app)
check("posttest DB guard",               "post-test score already in DB" in app)
check("posttest requires pretest first", "Pre-test not found" in app)
check("_is_sectioned helper",            "def _is_sectioned" in app)
check("_score_sectioned helper",         "def _score_sectioned" in app)
check("per-topic scores saved to state", "pretest_per_topic" in app)
check("progress bar in pretest",         "st.progress" in app)
check("section expanders in pretest",    "st.expander" in app)
check("per-topic breakdown shown",       "Per-topic scores" in app)
check("practice_menu in PHASES",         '"practice_menu":  phase_practice_menu' in app)
check("practice_mode flag",              "practice_mode" in app)
check("done page shows weak topics",     "below mastery" in app)
check("MASTERY_THRESHOLD_PRACTICE",      "MASTERY_THRESHOLD_PRACTICE" in app)

# ══════════════════════════════════════════════════════════════════
print("\n=== 7. STATE SCHEMA ===")

from src.state import LearnerState, new_state
state = new_state("TEST", "python_programming")

required_fields = [
    "student_id", "domain", "topic_sequence", "topic_pointer",
    "bloom_plan", "pretest_score_pct", "pretest_per_topic",
    "ablation_mode", "current_material", "pending_item", "last_grade",
    "mastery", "last_reviewed", "misconceptions", "engagement",
    "replan_flag", "session_history", "step_count", "current_phase",
]
for field in required_fields:
    check(f"state has field: {field}", field in state, str(list(state.keys())))

# ══════════════════════════════════════════════════════════════════
# Summary
print("\n" + "=" * 55)
print(f"  TOTAL: {len(PASS)} PASSED  |  {len(FAIL)} FAILED")
if FAIL:
    print("\n  FAILED:")
    for f in FAIL:
        print(f"    ✗ {f}")
else:
    print("  ALL CHECKS PASSED ✓")
print("=" * 55)
sys.exit(0 if not FAIL else 1)
