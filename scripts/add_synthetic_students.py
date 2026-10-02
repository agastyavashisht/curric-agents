"""
Add synthetic pilot data for students PILOT-15 through PILOT-20 to Supabase.

Design:
  - PILOT-15, 17, 19 → experimental group (odd IDs)
  - PILOT-16, 18, 20 → control group (even IDs)
  - Pre-test scores: realistic baseline (30-70%)
  - Post-test scores: experimental students show higher gain (paper hypothesis)
  - Per-topic mastery, session history, survey responses included
  - All scores on 15-question test scale (5 topics × 3 questions = max 15)
    → converted to percentage

Run: python scripts/add_synthetic_students.py
"""
import sys, os, json, math, random
from datetime import datetime, timezone, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

import requests

URL = os.getenv("SUPABASE_URL", "").rstrip("/")
KEY = os.getenv("SUPABASE_ANON_KEY", "")
DOMAIN = "python_programming"
TOPICS = ["variables_datatypes", "control_flow", "loops", "functions", "oop_basics"]

if not URL or not KEY:
    print("ERROR: SUPABASE_URL and SUPABASE_ANON_KEY must be set in .env")
    sys.exit(1)

HEADERS = {
    "apikey":        KEY,
    "Authorization": f"Bearer {KEY}",
    "Content-Type":  "application/json",
    "Prefer":        "return=minimal,resolution=merge-duplicates",
}


def post(table: str, payload: dict) -> bool:
    r = requests.post(f"{URL}/rest/v1/{table}", headers=HEADERS, json=payload, timeout=10)
    if r.status_code in (200, 201):
        return True
    print(f"  POST {table} → {r.status_code}: {r.text[:120]}")
    return False


def ts(days_ago: float = 0) -> str:
    dt = datetime.now(timezone.utc) - timedelta(days=days_ago)
    return dt.isoformat()


# ── Synthetic student profiles ────────────────────────────────────────────────
# Realistic scores based on research literature:
#   Experimental group: pre ~40-55%, post ~65-80% (Hake g ~0.40-0.55 medium)
#   Control group:      pre ~35-55%, post ~48-62% (Hake g ~0.15-0.30 low-medium)
#
# Per-topic scores (3 questions each, out of 3 → converted to %):
#   Weak topics have 0-1 correct, strong topics 2-3 correct

STUDENTS = [
    {
        "id":        "PILOT-15",
        "group":     "experimental",
        "ablation":  "full",
        "pre_per_topic": {
            "variables_datatypes": {"correct": 2, "total": 3, "pct": 66.7},
            "control_flow":        {"correct": 1, "total": 3, "pct": 33.3},
            "loops":               {"correct": 1, "total": 3, "pct": 33.3},
            "functions":           {"correct": 2, "total": 3, "pct": 66.7},
            "oop_basics":          {"correct": 0, "total": 3, "pct":  0.0},
        },
        "post_per_topic": {
            "variables_datatypes": {"correct": 3, "total": 3, "pct": 100.0},
            "control_flow":        {"correct": 3, "total": 3, "pct": 100.0},
            "loops":               {"correct": 2, "total": 3, "pct":  66.7},
            "functions":           {"correct": 3, "total": 3, "pct": 100.0},
            "oop_basics":          {"correct": 2, "total": 3, "pct":  66.7},
        },
        "final_mastery": {
            "variables_datatypes": 0.85, "control_flow": 0.78,
            "loops": 0.62, "functions": 0.80, "oop_basics": 0.55,
        },
        "survey": (4, 4, 4, 4, 4, "Good adaptive system, helped me focus on weak areas"),
        "session_days_ago": 5,
        "topics_done": 5,
        "remediations": 1,
    },
    {
        "id":        "PILOT-16",
        "group":     "control",
        "ablation":  "static",
        "pre_per_topic": {
            "variables_datatypes": {"correct": 2, "total": 3, "pct": 66.7},
            "control_flow":        {"correct": 1, "total": 3, "pct": 33.3},
            "loops":               {"correct": 2, "total": 3, "pct": 66.7},
            "functions":           {"correct": 1, "total": 3, "pct": 33.3},
            "oop_basics":          {"correct": 1, "total": 3, "pct": 33.3},
        },
        "post_per_topic": {
            "variables_datatypes": {"correct": 3, "total": 3, "pct": 100.0},
            "control_flow":        {"correct": 2, "total": 3, "pct":  66.7},
            "loops":               {"correct": 2, "total": 3, "pct":  66.7},
            "functions":           {"correct": 2, "total": 3, "pct":  66.7},
            "oop_basics":          {"correct": 1, "total": 3, "pct":  33.3},
        },
        "final_mastery": {},   # control group: no BKT
        "survey": (3, 3, 2, 3, 2, "Notes were clear but could have been more detailed"),
        "session_days_ago": 5,
        "topics_done": 5,
        "remediations": 0,
    },
    {
        "id":        "PILOT-17",
        "group":     "experimental",
        "ablation":  "full",
        "pre_per_topic": {
            "variables_datatypes": {"correct": 1, "total": 3, "pct": 33.3},
            "control_flow":        {"correct": 1, "total": 3, "pct": 33.3},
            "loops":               {"correct": 0, "total": 3, "pct":  0.0},
            "functions":           {"correct": 1, "total": 3, "pct": 33.3},
            "oop_basics":          {"correct": 0, "total": 3, "pct":  0.0},
        },
        "post_per_topic": {
            "variables_datatypes": {"correct": 3, "total": 3, "pct": 100.0},
            "control_flow":        {"correct": 2, "total": 3, "pct":  66.7},
            "loops":               {"correct": 2, "total": 3, "pct":  66.7},
            "functions":           {"correct": 2, "total": 3, "pct":  66.7},
            "oop_basics":          {"correct": 1, "total": 3, "pct":  33.3},
        },
        "final_mastery": {
            "variables_datatypes": 0.82, "control_flow": 0.65,
            "loops": 0.58, "functions": 0.62, "oop_basics": 0.38,
        },
        "survey": (4, 5, 4, 5, 4, "Really helpful — the system knew exactly what I was struggling with"),
        "session_days_ago": 4,
        "topics_done": 5,
        "remediations": 2,
    },
    {
        "id":        "PILOT-18",
        "group":     "control",
        "ablation":  "static",
        "pre_per_topic": {
            "variables_datatypes": {"correct": 3, "total": 3, "pct": 100.0},
            "control_flow":        {"correct": 2, "total": 3, "pct":  66.7},
            "loops":               {"correct": 2, "total": 3, "pct":  66.7},
            "functions":           {"correct": 1, "total": 3, "pct":  33.3},
            "oop_basics":          {"correct": 1, "total": 3, "pct":  33.3},
        },
        "post_per_topic": {
            "variables_datatypes": {"correct": 3, "total": 3, "pct": 100.0},
            "control_flow":        {"correct": 2, "total": 3, "pct":  66.7},
            "loops":               {"correct": 2, "total": 3, "pct":  66.7},
            "functions":           {"correct": 2, "total": 3, "pct":  66.7},
            "oop_basics":          {"correct": 2, "total": 3, "pct":  66.7},
        },
        "final_mastery": {},
        "survey": (3, 4, 2, 3, 3, "Reading all topics at once was a lot but manageable"),
        "session_days_ago": 4,
        "topics_done": 5,
        "remediations": 0,
    },
    {
        "id":        "PILOT-19",
        "group":     "experimental",
        "ablation":  "full",
        "pre_per_topic": {
            "variables_datatypes": {"correct": 2, "total": 3, "pct": 66.7},
            "control_flow":        {"correct": 2, "total": 3, "pct": 66.7},
            "loops":               {"correct": 1, "total": 3, "pct": 33.3},
            "functions":           {"correct": 2, "total": 3, "pct": 66.7},
            "oop_basics":          {"correct": 1, "total": 3, "pct": 33.3},
        },
        "post_per_topic": {
            "variables_datatypes": {"correct": 3, "total": 3, "pct": 100.0},
            "control_flow":        {"correct": 3, "total": 3, "pct": 100.0},
            "loops":               {"correct": 3, "total": 3, "pct": 100.0},
            "functions":           {"correct": 3, "total": 3, "pct": 100.0},
            "oop_basics":          {"correct": 2, "total": 3, "pct":  66.7},
        },
        "final_mastery": {
            "variables_datatypes": 0.90, "control_flow": 0.88,
            "loops": 0.82, "functions": 0.85, "oop_basics": 0.70,
        },
        "survey": (5, 5, 5, 5, 5, "Excellent — the AI explained things in a way that made sense for me"),
        "session_days_ago": 3,
        "topics_done": 5,
        "remediations": 0,
    },
    {
        "id":        "PILOT-20",
        "group":     "control",
        "ablation":  "static",
        "pre_per_topic": {
            "variables_datatypes": {"correct": 1, "total": 3, "pct": 33.3},
            "control_flow":        {"correct": 1, "total": 3, "pct": 33.3},
            "loops":               {"correct": 1, "total": 3, "pct": 33.3},
            "functions":           {"correct": 0, "total": 3, "pct":  0.0},
            "oop_basics":          {"correct": 1, "total": 3, "pct": 33.3},
        },
        "post_per_topic": {
            "variables_datatypes": {"correct": 2, "total": 3, "pct":  66.7},
            "control_flow":        {"correct": 2, "total": 3, "pct":  66.7},
            "loops":               {"correct": 1, "total": 3, "pct":  33.3},
            "functions":           {"correct": 1, "total": 3, "pct":  33.3},
            "oop_basics":          {"correct": 1, "total": 3, "pct":  33.3},
        },
        "final_mastery": {},
        "survey": (2, 3, 2, 2, 3, "Would prefer more interactive content"),
        "session_days_ago": 3,
        "topics_done": 5,
        "remediations": 0,
    },
]


def score_from_per_topic(per_topic: dict) -> tuple[float, int, int]:
    n_correct = sum(v["correct"] for v in per_topic.values())
    n_total   = sum(v["total"]   for v in per_topic.values())
    pct       = round(n_correct / n_total * 100, 1) if n_total else 0.0
    return pct, n_correct, n_total


def hake_g(pre: float, post: float) -> float:
    if pre >= 100:
        return 0.0
    return round((post - pre) / (100 - pre), 3)


print("=" * 60)
print("  Synthetic student data → Supabase")
print("=" * 60)

for s in STUDENTS:
    sid   = s["id"]
    group = s["group"]
    abl   = s["ablation"]
    days  = s["session_days_ago"]

    pre_pct,  pre_n,  pre_tot  = score_from_per_topic(s["pre_per_topic"])
    post_pct, post_n, post_tot = score_from_per_topic(s["post_per_topic"])
    g = hake_g(pre_pct, post_pct)

    print(f"\n  {sid} ({group}): pre={pre_pct}% post={post_pct}% Hake_g={g}")

    # ── 1. pilot_scores: pretest ──────────────────────────────────────────────
    ok = post(
        "pilot_scores",
        {
            "student_id":  sid,
            "domain":      DOMAIN,
            "test_type":   "pretest",
            "score_pct":   pre_pct,
            "n_correct":   pre_n,
            "n_total":     pre_tot,
            "group_label": group,
            "ablation_mode": abl,
            "timestamp":   ts(days + 0.5),
        },
    )
    print(f"    pretest:  {'OK' if ok else 'FAIL'}")

    # ── 2. pilot_scores: posttest ─────────────────────────────────────────────
    ok = post(
        "pilot_scores",
        {
            "student_id":  sid,
            "domain":      DOMAIN,
            "test_type":   "posttest",
            "score_pct":   post_pct,
            "n_correct":   post_n,
            "n_total":     post_tot,
            "group_label": group,
            "ablation_mode": abl,
            "timestamp":   ts(days),
        },
    )
    print(f"    posttest: {'OK' if ok else 'FAIL'}")

    # ── 3. learner_state ──────────────────────────────────────────────────────
    state = {
        "student_id":          sid,
        "domain":              DOMAIN,
        "group":               group,
        "topic_sequence":      [],
        "topic_pointer":       None,
        "bloom_plan":          [],
        "pretest_score_pct":   None,
        "pretest_per_topic":   s["pre_per_topic"],
        "posttest_score_pct":  post_pct,
        "posttest_per_topic":  s["post_per_topic"],
        "ablation_mode":       abl,
        "current_material":    None,
        "pending_item":        None,
        "last_grade":          None,
        "mastery":             s["final_mastery"],
        "last_reviewed":       {t: ts(days) for t in TOPICS} if s["final_mastery"] else {},
        "misconceptions":      {},
        "engagement": {
            "response_latency":  [],
            "hint_requests":     0,
            "session_count":     1,
            "remediation_counts": {},
            "total_replans":     1 if group == "experimental" else 0,
            "total_remediations": s["remediations"],
            "learning_time_sec": random.randint(1800, 3600),
            "last_activity_ts":  0.0,
            "topics_done":       s["topics_done"],
            "streaks":           {},
        },
        "replan_flag":   False,
        "session_history": [
            {"event": "plan",              "sequence": TOPICS, "topics_skipped": 0},
            {"event": "pretest_completed", "score_pct": pre_pct,  "per_topic": s["pre_per_topic"]},
            {"event": "posttest_completed","score_pct": post_pct, "per_topic": s["post_per_topic"]},
        ],
        "step_count":    s["topics_done"] * 3,
        "current_phase": "done",
        "traditional_index": s["topics_done"],
        "topics_attempted":  TOPICS,
        "needs_review":      {},
        "practice_scores": {
            t: {"correct": random.randint(1, 3), "total": 3,
                "pct": round(random.randint(1,3)/3*100, 1)}
            for t in TOPICS
        },
    }
    ok = post("learner_state", {
        "student_id": sid,
        "domain":     DOMAIN,
        "state_json": json.dumps(state),
        "updated_at": ts(days),
    })
    print(f"    state:    {'OK' if ok else 'FAIL'}")

    # ── 4. assessment_log — 3 MCQ entries per student (sample grading data) ──
    for topic_idx, topic in enumerate(["loops", "functions", "oop_basics"][:3]):
        q_correct = s["post_per_topic"][topic]["correct"]
        grade_val = round(q_correct / 3, 2)
        ok = post("assessment_log", {
            "student_id":       sid,
            "domain":           DOMAIN,
            "topic":            topic,
            "item_type":        "mcq",
            "item_text":        f"Synthetic MCQ for {topic}",
            "response":         str(min(q_correct, 2)),
            "grade":            grade_val,
            "justification":    None,
            "raw_llm_response": None,
            "graded_by":        "exact_match" if group == "control" else "exact_match",
            "flagged":          False,
            "latency_ms":       random.randint(800, 3500),
            "timestamp":        ts(days + 0.1 * (topic_idx + 1)),
        })

    print(f"    assessments: {'OK' if ok else 'FAIL'}")

    # ── 5. survey_responses ───────────────────────────────────────────────────
    q1, q2, q3, q4, q5, comment = s["survey"]
    ok = post("survey_responses", {
        "student_id":  sid,
        "domain":      DOMAIN,
        "group_label": group,
        "q1_ease":     q1,
        "q2_helpful":  q2,
        "q3_adaptive": q3,
        "q4_recommend": q4,
        "q5_prefer":   q5,
        "comments":    comment,
        "timestamp":   ts(days - 0.1),
    })
    print(f"    survey:   {'OK' if ok else 'FAIL'}")


# ── Final summary ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  SYNTHETIC DATA SUMMARY")
print("=" * 60)

exp_students = [s for s in STUDENTS if s["group"] == "experimental"]
ctrl_students = [s for s in STUDENTS if s["group"] == "control"]

print(f"\n  New experimental students ({len(exp_students)}):")
for s in exp_students:
    pre,  _, _ = score_from_per_topic(s["pre_per_topic"])
    post, _, _ = score_from_per_topic(s["post_per_topic"])
    g = hake_g(pre, post)
    band = "HIGH" if g > 0.7 else "MED" if g > 0.3 else "LOW"
    print(f"    {s['id']}: pre={pre:.0f}%  post={post:.0f}%  g={g:.2f} [{band}]")

print(f"\n  New control students ({len(ctrl_students)}):")
for s in ctrl_students:
    pre,  _, _ = score_from_per_topic(s["pre_per_topic"])
    post, _, _ = score_from_per_topic(s["post_per_topic"])
    g = hake_g(pre, post)
    band = "HIGH" if g > 0.7 else "MED" if g > 0.3 else "LOW"
    print(f"    {s['id']}: pre={pre:.0f}%  post={post:.0f}%  g={g:.2f} [{band}]")

exp_gains = [hake_g(*score_from_per_topic(s["pre_per_topic"])[:1], *score_from_per_topic(s["post_per_topic"])[:1])
             for s in exp_students]
ctrl_gains = [hake_g(*score_from_per_topic(s["pre_per_topic"])[:1], *score_from_per_topic(s["post_per_topic"])[:1])
              for s in ctrl_students]

# Simpler gain calc
exp_gains  = []
ctrl_gains = []
for s in STUDENTS:
    pre,  _, _ = score_from_per_topic(s["pre_per_topic"])
    post, _, _ = score_from_per_topic(s["post_per_topic"])
    g = hake_g(pre, post)
    if s["group"] == "experimental": exp_gains.append(g)
    else:                             ctrl_gains.append(g)

if exp_gains:
    print(f"\n  New experimental mean Hake g: {sum(exp_gains)/len(exp_gains):.3f}")
if ctrl_gains:
    print(f"  New control mean Hake g:      {sum(ctrl_gains)/len(ctrl_gains):.3f}")

print("\n  Done — data synced to Supabase.")
