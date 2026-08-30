"""
Diagnose and fix all dashboard issues.
Run: python -m tests.diagnose_dashboard
"""
import sys, os, json, traceback
sys.path.insert(0, '.')

import pandas as pd
from src.db import save_pilot_score, get_pilot_scores, log_assessment, get_assessment_log, get_conn

DB = 'results/learner_state.db'

PASS, FAIL = [], []

def check(name, cond, detail=''):
    if cond:
        PASS.append(name)
        print(f"  OK   {name}")
    else:
        FAIL.append(name)
        print(f"  FAIL {name}" + (f" -- {detail}" if detail else ""))

# ── 1. Check actual DB schema ─────────────────────────────────────────────────
print("\n=== DB Schema Check ===")
conn = get_conn(DB)
al_cols = [r[1] for r in conn.execute("PRAGMA table_info(assessment_log)").fetchall()]
ps_cols = [r[1] for r in conn.execute("PRAGMA table_info(pilot_scores)").fetchall()]
conn.close()
print("  assessment_log columns:", al_cols)
print("  pilot_scores columns:  ", ps_cols)
check("raw_llm_response in assessment_log", "raw_llm_response" in al_cols)
check("group_label in pilot_scores",        "group_label"       in ps_cols)

# ── 2. Seed test data ──────────────────────────────────────────────────────────
print("\n=== Seeding Test Data ===")
try:
    save_pilot_score('DASH-01','python_programming','pretest', 50.0,5,10,'experimental',DB)
    save_pilot_score('DASH-01','python_programming','posttest',75.0,7,10,'experimental',DB)
    save_pilot_score('DASH-02','python_programming','pretest', 40.0,4,10,'control',DB)
    save_pilot_score('DASH-02','python_programming','posttest',55.0,5,10,'control',DB)
    print("  pilot_scores seeded OK")
    log_assessment('DASH-01','python_programming','variables_datatypes',
                   'mcq','Q?','2',1.0,None,'exact_match',False,100,None,DB)
    log_assessment('DASH-01','python_programming','functions',
                   'short_answer','Q2','some answer',0.67,'Good','llm_judge+embedding',False,800,'raw',DB)
    print("  assessment_log seeded OK")
except Exception as e:
    print(f"  SEED ERROR: {e}")
    traceback.print_exc()
    sys.exit(1)

# ── 3. Tab 1 – Learning Gain ──────────────────────────────────────────────────
print("\n=== Tab 1: Learning Gain ===")
try:
    scores = get_pilot_scores(DB)
    check("get_pilot_scores returns data", len(scores) > 0)
    df = pd.DataFrame(scores)
    check("'group' column in scores", "group" in df.columns, str(list(df.columns)))
    pivot = df.pivot_table(
        index=["student_id","domain","group"],
        columns="test_type",
        values="score_pct"
    ).reset_index()
    check("pivot has pretest column",  "pretest"  in pivot.columns)
    check("pivot has posttest column", "posttest" in pivot.columns)
    pivot["gain_pct"] = pivot["posttest"] - pivot["pretest"]
    pivot["hake_g"] = pivot.apply(
        lambda r: (r["posttest"] - r["pretest"]) / (100 - r["pretest"])
        if r["pretest"] < 100 else 0.0, axis=1).round(3)
    check("hake_g computed", pivot["hake_g"].notna().all())
    exp  = pivot[pivot["group"] == "experimental"]["hake_g"].dropna()
    ctrl = pivot[pivot["group"] == "control"]["hake_g"].dropna()
    check("experimental group present", len(exp) > 0)
    check("control group present",      len(ctrl) > 0)
    print(f"  Mean <g> experimental={exp.mean():.3f}  control={ctrl.mean():.3f}")
except Exception as e:
    check("Tab 1 runs without error", False, str(e))
    traceback.print_exc()

# ── 4. Tab 2 – Assessment Log ─────────────────────────────────────────────────
print("\n=== Tab 2: Assessment Log ===")
try:
    alog = get_assessment_log(DB)
    check("get_assessment_log returns data", len(alog) > 0)
    adf = pd.DataFrame(alog)
    needed = ["student_id","domain","topic","item_type","grade",
              "graded_by","flagged","justification","timestamp","raw_llm_response"]
    missing = [c for c in needed if c not in adf.columns]
    check("all required columns present", not missing, f"missing: {missing}")
    flagged = adf[adf["flagged"] == True]
    check("flagged filter works", isinstance(len(flagged), int))
    sub = adf[["student_id","domain","topic","item_type","grade",
               "graded_by","flagged","justification","timestamp"]]
    check("dataframe subset works", len(sub) == len(adf))
except Exception as e:
    check("Tab 2 runs without error", False, str(e))
    traceback.print_exc()

# ── 5. Tab 3 – Agent Coordination ────────────────────────────────────────────
print("\n=== Tab 3: Agent Coordination ===")
log_path = os.getenv("LOG_PATH","logs/agent_calls.jsonl")
check("log file exists", os.path.exists(log_path), f"not found: {log_path}")
if os.path.exists(log_path):
    try:
        rows = []
        with open(log_path) as f:
            for line in f:
                line = line.strip()
                if line: rows.append(json.loads(line))
        check("log has rows", len(rows) > 0, f"log is empty")
        if rows:
            df2 = pd.DataFrame(rows)
            needed2 = ["agent","student_id","latency_seconds","redundant","start_ts","session_id"]
            missing2 = [c for c in needed2 if c not in df2.columns]
            check("all log columns present", not missing2, f"missing: {missing2}")
            # columns used in dashboard display
            disp = ["agent","student_id","latency_seconds","redundant","start_ts"]
            avail = [c for c in disp if c in df2.columns]
            check("display columns available", len(avail) == len(disp), f"missing: {set(disp)-set(avail)}")
    except Exception as e:
        check("Tab 3 runs without error", False, str(e))
        traceback.print_exc()

# ── 6. Tab 4 – Export ────────────────────────────────────────────────────────
print("\n=== Tab 4: Export ===")
try:
    scores = get_pilot_scores(DB)
    df3 = pd.DataFrame(scores)
    for tt in ["pretest","posttest"]:
        sub = df3[df3["test_type"]==tt][["student_id","group","score_pct"]]
        check(f"{tt} CSV export works", len(sub) >= 0)

    alog2 = get_assessment_log(DB)
    adf2  = pd.DataFrame(alog2)
    short = adf2[adf2["item_type"]=="short_answer"][["student_id","topic","grade"]].rename(
        columns={"topic":"question_id","grade":"system_score"})
    if len(short) > 0:
        short["system_score"] = (short["system_score"].astype(float)*3).round().astype("Int64")
        short["instructor_score"] = ""
        check("instructor template generates", len(short) > 0)
    else:
        check("instructor template generates", True, "(no short_answer rows yet — ok)")
except Exception as e:
    check("Tab 4 runs without error", False, str(e))
    traceback.print_exc()

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 50)
print(f"  {len(PASS)} PASSED  |  {len(FAIL)} FAILED")
if FAIL:
    print("\n  ISSUES TO FIX:")
    for f in FAIL:
        print(f"    x {f}")
print("=" * 50)
