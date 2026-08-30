"""
Runs ALL evaluation metrics and writes results/metrics_report.json.

Usage:
    python -m eval.run_all_metrics [--domain DOMAIN]

Arguments:
    --domain    Domain key to filter coherence/coordination by (default: all).
                Options: python_programming, ml_basics, signal_processing,
                         physiology_basics, data_analysis

Any metric whose required input file is missing is reported as "skipped"
rather than crashing the whole run — so you can run this at any stage of
the pilot and see partial results.

Output file: results/metrics_report.json
"""
import argparse
import json
import os
import traceback
from datetime import datetime, timezone

# ── safe wrapper ─────────────────────────────────────────────────────────────
def safe_run(label, fn):
    try:
        result = fn()
        return {"status": "ok", "result": result}
    except FileNotFoundError as e:
        return {"status": "skipped", "reason": f"input file not found: {e.filename or e}"}
    except Exception as e:
        return {
            "status": "error",
            "reason": str(e),
            "traceback": traceback.format_exc(limit=6),
        }


# ── helpers for exporting DB → CSV (so eval scripts can read them) ────────────
def _export_pilot_scores():
    """Write pretest/posttest CSVs from the SQLite DB for learning_gain eval."""
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

    from src.db import get_pilot_scores
    import pandas as pd
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv())
    db_path = os.getenv("DB_PATH", "results/learner_state.db")

    scores = get_pilot_scores(db_path)
    if not scores:
        raise FileNotFoundError("results/learner_state.db has no pilot scores yet")

    df = pd.DataFrame(scores)
    os.makedirs("results", exist_ok=True)

    for test_type in ["pretest", "posttest"]:
        sub = df[df["test_type"] == test_type][["student_id", "group", "score_pct"]]
        if not sub.empty:
            path = f"results/{test_type}_scores.csv"
            sub.to_csv(path, index=False)

    return {"pretest_rows": int((df["test_type"] == "pretest").sum()),
            "posttest_rows": int((df["test_type"] == "posttest").sum())}


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--domain", default=None,
        help="Filter coherence/coordination to a specific domain (default: all)"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  CurricAgents — Evaluation Metrics Runner")
    print(f"  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    if args.domain:
        print(f"  Domain filter: {args.domain}")
    print("=" * 60)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "domain_filter": args.domain or "all",
    }

    # ── 1. Export DB scores to CSVs ───────────────────────────────────────────
    print("\n[1/5] Exporting pilot scores from DB…")
    export_result = safe_run("export_scores", _export_pilot_scores)
    report["score_export"] = export_result
    _print_status("Score export", export_result)

    # ── 2. Learning gain ──────────────────────────────────────────────────────
    print("\n[2/5] Learning gain (Hake's <g>)…")
    from eval.learning_gain import analyze as lg_analyze
    lg_result = safe_run(
        "learning_gain",
        lambda: lg_analyze("results/pretest_scores.csv", "results/posttest_scores.csv"),
    )
    report["learning_gain"] = lg_result
    _print_status("Learning gain", lg_result)
    if lg_result["status"] == "ok":
        r = lg_result["result"]
        print(f"    Experimental mean <g>: {r.get('mean_gain_experimental', 'N/A')}")
        print(f"    Control mean <g>:      {r.get('mean_gain_control', 'N/A')}")
        if "t_test" in r:
            print(f"    t-test p-value:        {r['t_test']['p_value']}")
            print(f"    Cohen's d:             {r.get('cohens_d', 'N/A')}")

    # ── 3. Agent coordination efficiency ─────────────────────────────────────
    print("\n[3/5] Agent coordination efficiency…")
    from eval.coordination_efficiency import load_log, compute as ce_compute
    def _coord():
        df = load_log("logs/agent_calls.jsonl")
        if args.domain:
            # filter to rows involving this domain's students, if domain column exists
            if "domain" in df.columns:
                df = df[df["domain"] == args.domain]
        return ce_compute(df)
    coord_result = safe_run("coordination_efficiency", _coord)
    report["coordination_efficiency"] = coord_result
    _print_status("Coordination efficiency", coord_result)
    if coord_result["status"] == "ok":
        r = coord_result["result"]
        print(f"    Task completion rate:  {r.get('task_completion_rate', 'N/A')}")
        print(f"    Mean latency (s):      {r.get('mean_agent_call_latency_seconds', 'N/A')}")
        print(f"    Redundant call rate:   {r.get('redundant_call_rate', 'N/A')}")

    # ── 4. Curriculum coherence ───────────────────────────────────────────────
    print("\n[4/5] Curriculum coherence (objective alignment)…")
    from eval.curriculum_coherence import (
        objective_alignment, run_all_domains,
        edge_consistency_from_content_log, coherence_score,
    )
    def _coherence():
        log = "results/generated_content.jsonl"
        if args.domain and args.domain != "all":
            align = objective_alignment(log, domain_filter=args.domain)
            edge = edge_consistency_from_content_log(log, domain_filter=args.domain)
            combined = None
            if "mean_objective_alignment" in align and "mean_edge_consistency" in edge:
                combined = coherence_score(
                    {"edge_consistency": edge["mean_edge_consistency"]},
                    align,
                )
            return {**align, "edge_consistency": edge, "combined_coherence_score": combined}
        return run_all_domains(log)
    coherence_result = safe_run("curriculum_coherence", _coherence)
    report["curriculum_coherence"] = coherence_result
    _print_status("Curriculum coherence", coherence_result)
    if coherence_result["status"] == "ok":
        r = coherence_result["result"]
        if "mean_objective_alignment" in r:
            print(f"    Mean objective alignment: {r['mean_objective_alignment']}")
            print(f"    Combined coherence score: {r.get('combined_coherence_score', 'N/A')}")
        elif "all_domains" in r:
            print(f"    All-domain alignment: {r['all_domains'].get('mean_objective_alignment', 'N/A')}")
            print(f"    Mean edge consistency:    {r.get('edge_consistency', {}).get('mean_edge_consistency', 'N/A')}")
            print(f"    Combined coherence score: {r.get('combined_coherence_score', 'N/A')}")

    # ── 5. Grading reliability (QWK) ─────────────────────────────────────────
    print("\n[5/5] Grading reliability (QWK against instructor grades)…")
    from eval.grading_reliability import analyze as gr_analyze
    gr_result = safe_run(
        "grading_reliability",
        lambda: gr_analyze("results/instructor_grades.csv"),
    )
    report["grading_reliability"] = gr_result
    _print_status("Grading reliability", gr_result)
    if gr_result["status"] == "ok":
        r = gr_result["result"]
        print(f"    QWK:                   {r.get('quadratic_weighted_kappa', 'N/A')}")
        print(f"    Interpretation:        {r.get('interpretation', 'N/A')}")
        print(f"    Exact agreement:       {r.get('exact_agreement_rate', 'N/A')}")

    # ── Write report ──────────────────────────────────────────────────────────
    os.makedirs("results", exist_ok=True)
    out_path = "results/metrics_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    print("\n" + "=" * 60)
    statuses = [v["status"] for k, v in report.items() if isinstance(v, dict) and "status" in v]
    ok      = statuses.count("ok")
    skipped = statuses.count("skipped")
    errors  = statuses.count("error")
    print(f"  Done: {ok} ok  |  {skipped} skipped  |  {errors} errors")
    print(f"  Full report → {out_path}")
    print("=" * 60)


def _print_status(label: str, result: dict):
    icon = {"ok": "✓", "skipped": "~", "error": "✗"}.get(result["status"], "?")
    msg = result.get("reason", "")
    print(f"    [{icon}] {label}: {result['status']}" + (f" — {msg}" if msg else ""))


if __name__ == "__main__":
    main()
