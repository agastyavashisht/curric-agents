"""
Computes the three agent-coordination-efficiency numbers from
logs/agent_calls.jsonl (produced by src/logging_utils.py).

Groups by SESSION (session_id = student_id__domain__session_count) rather
than by student_id alone, so multi-session pilot data doesn't inflate
task_completion_rate by treating calls from different sessions as one cycle.

Usage:
    python -m eval.coordination_efficiency [--log logs/agent_calls.jsonl]
"""
import argparse
import json
import pandas as pd


def load_log(path: str) -> pd.DataFrame:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    # Back-compat: old logs may not have session_id — synthesise it
    if "session_id" not in df.columns:
        df["session_id"] = df["student_id"].astype(str) + "__legacy__1"
    return df


def compute(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"error": "log is empty — run at least one session first"}

    df = df.copy()
    df["start_ts"] = pd.to_datetime(df["start_ts"])
    df = df.sort_values(["session_id", "start_ts"])

    # ── handoff latency ───────────────────────────────────────────────────────
    # Time between consecutive agent calls within the same session
    df["next_start"] = df.groupby("session_id")["start_ts"].shift(-1)
    df["handoff_gap_s"] = (df["next_start"] - df["start_ts"]).dt.total_seconds()
    mean_handoff = df["handoff_gap_s"].dropna().mean()

    # ── redundant call rate ───────────────────────────────────────────────────
    redundant_rate = float(df["redundant"].mean()) if "redundant" in df.columns else 0.0

    # ── task completion rate ──────────────────────────────────────────────────
    # A "complete cycle" = planner → content → assessment → monitor all appear
    # in order at least once within a single session.
    required = ["planner", "content", "assessment", "monitor"]
    cycles   = []
    for session_id, group in df.groupby("session_id"):
        agents = list(group.sort_values("start_ts")["agent"])
        idx = 0
        for a in agents:
            if idx < len(required) and a == required[idx]:
                idx += 1
        cycles.append(idx == len(required))

    task_completion_rate = sum(cycles) / len(cycles) if cycles else 0.0

    return {
        "n_agent_calls":               len(df),
        "n_students":                  int(df["student_id"].nunique()),
        "n_sessions":                  int(df["session_id"].nunique()),
        "mean_agent_call_latency_s":   round(float(df["latency_seconds"].mean()), 4),
        "mean_handoff_gap_s":          round(float(mean_handoff), 4) if pd.notna(mean_handoff) else None,
        "redundant_call_rate":         round(redundant_rate, 4),
        "task_completion_rate":        round(task_completion_rate, 4),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", default="logs/agent_calls.jsonl")
    args = parser.parse_args()
    df     = load_log(args.log)
    result = compute(df)
    print(json.dumps(result, indent=2))
