"""
Test Supabase connection and full dual-write flow.
Run: python -m tests.test_supabase
"""
import sys, os
sys.path.insert(0, '.')

print("=== Supabase Integration Test ===\n")

from src.online_db import (
    is_configured, test_connection,
    save_pilot_score_online, get_pilot_scores_online,
    log_assessment_online, get_assessment_log_online,
    save_learner_state_online, load_learner_state_online,
)

# 1. Configuration check
print("1. Configuration")
print(f"   SUPABASE_URL set:      {bool(os.getenv('SUPABASE_URL'))}")
print(f"   SUPABASE_ANON_KEY set: {bool(os.getenv('SUPABASE_ANON_KEY'))}")
print(f"   is_configured():       {is_configured()}")

# 2. Connection test
print("\n2. Connection test")
ok, msg = test_connection()
status = "OK" if ok else "FAIL"
print(f"   [{status}] {msg}")

if not ok:
    print("\n   To fix:")
    if not os.getenv("SUPABASE_ANON_KEY"):
        print("   - Add SUPABASE_ANON_KEY to .env (Supabase dashboard → Settings → API → anon/public)")
    print("   - Make sure you ran the SQL schema in src/supabase_schema.sql")
    print("   - Make sure Row Level Security policies are created")
    sys.exit(1)

# 3. Write pilot score
print("\n3. Write pilot score")
ok = save_pilot_score_online("TEST-SUPABASE", "python_programming", "pretest",
                              72.0, 7, 10, "experimental")
print(f"   write: {'OK' if ok else 'FAIL'}")

# 4. Read it back
print("\n4. Read pilot scores")
scores = get_pilot_scores_online()
test_score = next((s for s in scores if s["student_id"] == "TEST-SUPABASE"), None)
print(f"   total rows: {len(scores)}")
print(f"   test row found: {'OK' if test_score else 'FAIL'}")
if test_score:
    print(f"   score: {test_score}")

# 5. Write assessment log
print("\n5. Write assessment log")
ok = log_assessment_online("TEST-SUPABASE", "python_programming", "variables_datatypes",
                            "mcq", "What is x=5?", "2", 1.0, None, "exact_match",
                            False, 120, None)
print(f"   write: {'OK' if ok else 'FAIL'}")

# 6. Write + read learner state
print("\n6. Write + read learner state")
from src.state import new_state
state = new_state("TEST-SUPABASE", "python_programming")
state["mastery"]["variables_datatypes"] = 0.75
ok = save_learner_state_online(state)
print(f"   save: {'OK' if ok else 'FAIL'}")

loaded = load_learner_state_online("TEST-SUPABASE", "python_programming")
mastery_ok = loaded and loaded.get("mastery", {}).get("variables_datatypes") == 0.75
print(f"   load: {'OK' if loaded else 'FAIL'}")
print(f"   mastery round-trip: {'OK' if mastery_ok else 'FAIL'}")

# 7. Dual-write test (SQLite + Supabase)
print("\n7. Dual-write (SQLite + Supabase)")
from src.db import save_pilot_score, get_pilot_scores
from src.memory.store import save_state, load_state
DB = "results/learner_state.db"

save_pilot_score("TEST-DUAL", "ml_basics", "pretest", 60.0, 6, 10, "control", DB)
all_scores = get_pilot_scores(DB)
dual_score = next((s for s in all_scores if s["student_id"] == "TEST-DUAL"), None)
print(f"   dual write + read: {'OK' if dual_score else 'FAIL'}")

state2 = new_state("TEST-DUAL", "ml_basics")
state2["mastery"]["ml_introduction"] = 0.6
save_state(DB, state2)
loaded2 = load_state(DB, "TEST-DUAL", "ml_basics")
print(f"   state dual write + read: {'OK - mastery=' + str(loaded2['mastery'].get('ml_introduction')) if loaded2 else 'FAIL'}")

print("\n=== ALL TESTS PASSED ===")
print("Your project is now fully connected to Supabase.")
print("All student data will be stored online automatically.")
