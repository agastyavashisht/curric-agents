# Pilot Study — Operations Guide

This folder contains materials for running the 20-student pilot study.

## Pre-Pilot Checklist

- [ ] Ethics / IRB sign-off obtained from department (ask Dr. Mehra — allow 2–4 weeks)
- [ ] Consent form reviewed and dated (see `consent_form.md`)
- [ ] 20–25 volunteers recruited (over-recruit to buffer dropout)
- [ ] `student_id_map.csv` created locally — **never commit this file to git**
- [ ] Deployment host live and accessible (free-tier VM or spare laptop on static IP)
- [ ] Pre-test and post-test items locked (no edits after Week 0 starts)
- [ ] Support channel created (WhatsApp / Discord group for pilot participants)

## student_id_map.csv Format

This file maps real names to anonymised IDs. **It lives only on the researcher's local machine and is listed in `.gitignore`.**

```
real_name,study_id,email,group
Jane Smith,PILOT-01,jane@example.com,experimental
John Doe,PILOT-02,john@example.com,control
```

`group` is either `experimental` (uses the AI system) or `control` (traditional study materials).
Aim for ~10 per group.

## Pilot Timeline

| Week | Activity |
|------|----------|
| 0 | Pre-test (all participants, same sitting if possible) |
| 1–5 | Experimental group uses the AI system; control group uses traditional materials |
| 5 | Post-test (all participants) |
| 6 | Compute metrics (`python -m eval.run_all_metrics`) |

## Running a Participant Session

1. Give the participant their Study ID (from `student_id_map.csv`)
2. Direct them to `http://<deployment-host>:8501`
3. They log in with their Study ID — the group is auto-assigned from the ID
   (odd = experimental/AI, even = control/traditional) and the system handles the rest
4. Session state is saved automatically — they can resume at any time

### What each participant experiences

| Stage | Experimental (AI) | Control (Traditional) |
|-------|-------------------|------------------------|
| Pre-test | Same fixed 25-question test (5 per topic), no feedback | Same fixed 25-question test, no feedback |
| Learning | Planner picks the next topic; Content agent personalises the lesson; 5 AI-generated practice questions per topic; Monitor updates mastery and remediates weak topics (max 2 rounds, then the topic is marked *needs_review* and re-scheduled) | Fixed topic order (Variables → Control Flow → Loops → Functions → OOP); corpus notes for one topic at a time; 5 FIXED practice questions per topic from `data/practice_python.json` — scores never change the path |
| Post-test | Same fixed 25-question test, no feedback | Same fixed 25-question test, no feedback |

Both arms record `total_learning_time`, replans/remediations and per-topic scores
for the research comparison (export `research_summary.csv` from the Researcher Dashboard).

## Monitoring During the Pilot

- Check `logs/agent_calls.jsonl` weekly for errors or unusual latency
- Check `results/learner_state.db` — if a participant reports a broken session,
  you can inspect their state: `sqlite3 results/learner_state.db "SELECT * FROM learner_state WHERE student_id='PILOT-XX'"`
- Run `python -m eval.run_all_metrics` at the end of each week to see partial metrics

## Post-Pilot Analysis

```bash
# 1. Export all scores and compute metrics
python -m eval.run_all_metrics

# 2. Compute grading reliability (fill in instructor_score column first)
python -m eval.grading_reliability --file results/instructor_grades.csv

# 3. Check coherence per domain
python -m eval.curriculum_coherence --domain all --content_log results/generated_content.jsonl
```

Results are written to `results/metrics_report.json`.

## Files in This Folder

| File | Status |
|------|--------|
| `consent_form.md` | Template — fill in institution name, ethics ref, dates |
| `README.md` | This file |
| `student_id_map.csv` | **DO NOT COMMIT** — create locally only |
| `session_logs/` | **DO NOT COMMIT** — raw session exports if needed |
