# curric-agents — Prototype Spec

Multi-Agent LLM System for Autonomous Curriculum Planning and Adaptive Assessment.
This is the **minimum-viable prototype** scoped for evaluation, not a production app —
built to generate real data for the three metrics in the synopsis (agent coordination
efficiency, curriculum coherence, student learning gain) plus the grading-reliability
check, using a 20-student pilot.

---

## 1. What this prototype does (and deliberately doesn't)

**In scope for the prototype:**
- One domain: Python programming (12 topics, `data/prerequisite_graph.json`)
- All 4 agents wired into a real LangGraph loop: Planner → Content → Assessment → Monitor → (replan or advance)
- Persistent memory across sessions via SQLite (a learner who runs the CLI twice resumes where they left off)
- MCQ generation + exact-match grading
- Short-answer generation + LLM-as-judge grading, cross-checked with embedding similarity
- Every agent call logged automatically (this is what the coordination-efficiency metric is computed from)
- A CLI runner — no web UI

**Explicitly out of scope for the prototype** (add later if there's time, but none of this blocks evaluation):
- The other 4 domains (ML basics, Signal Processing, Physiology, Data Analysis) — same schema, just more JSON files once Python is validated
- Vector store / semantic memory (Chroma/FAISS) — SQLite alone is enough to prove persistence works
- Any web UI (Streamlit/FastAPI) — the CLI is sufficient to run a real pilot session
- Authentication, error recovery, rate-limit handling — fine to let it crash during dev; harden only if time allows

---

## 2. Requirements

**Functional**
| ID | Requirement |
|---|---|
| FR1 | System generates an ordered topic sequence respecting `prerequisite_graph.json` |
| FR2 | System generates an explanation + worked example per topic (Content agent) |
| FR3 | System generates an MCQ or short-answer item per topic (Assessment agent) |
| FR4 | System grades MCQs by exact match |
| FR5 | System grades short answers via LLM-as-judge + embedding cross-check, flags disagreement |
| FR6 | System updates a per-topic mastery score (EMA) after every graded item |
| FR7 | System triggers a replan when mastery drifts below threshold or a struggle streak occurs |
| FR8 | Learner state persists across separate CLI runs (same `student_id`) |
| FR9 | Every agent call is logged with timing and read/write info |

**Non-functional**
| ID | Requirement |
|---|---|
| NFR1 | LLM backend (OpenAI/Anthropic) switchable via `.env`, zero code changes |
| NFR2 | End-to-end run via a single CLI command |
| NFR3 | Deterministic logic (Planner ordering, Monitor mastery math) has no LLM dependency and is unit-testable offline |
| NFR4 | All 4 evaluation metrics are computable from files alone — no manual computation |

**Environment**
- Python 3.10+
- An OpenAI or Anthropic API key (at least one)
- See `requirements.txt` for exact packages

---

## 3. File structure

```
curric-agents/
├── README.md                    <- this file
├── requirements.txt
├── .env.example                 <- copy to .env and fill in your API key
├── data/
│   ├── prerequisite_graph.json  <- topic order + learning objectives (Python domain)
│   ├── pretest_python.json      <- Form A, 10 MCQs, one per topic
│   ├── posttest_python.json     <- Form B, parallel to Form A
│   └── rubrics/
│       └── python_rubrics.json  <- short-answer rubrics (3 sample topics — extend to all 12)
├── src/
│   ├── config.py                 <- .env loader + get_llm() backend switch
│   ├── state.py                  <- LearnerState schema (shared by all agents)
│   ├── logging_utils.py          <- @log_node_call decorator -> logs/agent_calls.jsonl
│   ├── memory/store.py           <- SQLite load_state/save_state (fully working)
│   ├── agents/
│   │   ├── planner.py            <- WORKING, no LLM (topological sort)
│   │   ├── content.py            <- needs LLM call (real LangChain code, tune the prompt)
│   │   ├── assessment.py         <- needs LLM call (generation + grading, tune the prompts)
│   │   └── monitor.py            <- WORKING, no LLM (EMA mastery update)
│   ├── graph.py                  <- LangGraph StateGraph wiring (Fig. 2 in the synopsis)
│   └── run_session.py            <- CLI entrypoint
├── eval/                          <- ALL WORKING, no LLM dependency, already tested
│   ├── coordination_efficiency.py
│   ├── curriculum_coherence.py
│   ├── learning_gain.py
│   ├── grading_reliability.py
│   └── run_all_metrics.py        <- run this one script once pilot data exists
├── logs/                          <- agent_calls.jsonl lands here automatically
├── results/                       <- pilot CSVs + sample synthetic data go here
│   ├── sample_pretest_scores.csv      <- fake data, to test the eval scripts today
│   ├── sample_posttest_scores.csv
│   └── sample_instructor_grades.csv
└── tests/
    └── test_smoke.py             <- 7 tests, all passing, run before every commit
```

---

## 4. Setup

```bash
git clone <repo>
cd curric-agents
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env      # then edit .env with a real API key
pytest tests/test_smoke.py -v     # confirms the deterministic logic works, no API key needed
```

## 5. Running a session

```bash
python -m src.run_session --student_id S01 --domain python_programming --max_topics 3
```

This will print explanations, ask MCQ/short-answer questions interactively in the
terminal, grade them, and save the updated mastery profile to
`results/learner_state.db`. Run it again with the same `--student_id` to confirm
persistence — it should skip topics already mastered and remember prior misconceptions.

---

## 6. What's already working vs. what the team needs to build

| Component | Status | What's left |
|---|---|---|
| Planner (topological sort) | ✅ Done, tested | Nothing — works as-is |
| Monitor (EMA mastery + replan) | ✅ Done, tested | Nothing — works as-is |
| SQLite persistence | ✅ Done, tested | Nothing — works as-is |
| Agent-call logging | ✅ Done | Nothing — works as-is |
| Content agent | 🔧 Real LLM code, untuned prompt | Run it, read the explanations it produces, iterate on `_PROMPT_TEMPLATE` in `content.py` |
| Assessment agent | 🔧 Real LLM code, untuned prompts | Same — iterate on `_MCQ_PROMPT` / `_GRADE_PROMPT` in `assessment.py`; **author rubrics for the other 9 topics** in `python_rubrics.json` (only 3 are done) |
| All 4 eval scripts | ✅ Done, tested against sample data | Nothing structural — just point them at real pilot CSVs when ready |
| Domains 2–5 | ❌ Not started | Copy `prerequisite_graph.json`'s schema for ML basics / Signal Processing / Physiology / Data Analysis, once Python is validated |

**Suggested split across 5 people:**
- Person A — owns `content.py` prompt quality + runs test sessions
- Person B — owns `assessment.py` prompt quality + finishes the rubric bank
- Person C — owns `graph.py` + `run_session.py`, handles any LangGraph wiring issues
- Person D — owns the pilot logistics: recruiting the 20 students, running sessions with them, collecting pre/post test CSVs in the right format
- Person E — owns `eval/` — already working, but owns turning `results/metrics_report.json` into the actual paper tables/figures

---

## 7. How to evaluate it

Once you have real data, everything reduces to one command:

```bash
python -m eval.run_all_metrics
```

This reads whatever's already in `logs/` and `results/` and writes
`results/metrics_report.json`. Anything missing is reported as `"skipped"` rather
than crashing, so you can run this at any point during the pilot, not just at the end.

**What each metric needs, specifically:**

| Metric | Reads from | You need to produce |
|---|---|---|
| Coordination efficiency | `logs/agent_calls.jsonl` | Nothing — fills automatically as sessions run |
| Curriculum coherence | `results/generated_content.jsonl` + `data/prerequisite_graph.json` | Nothing — fills automatically; run `eval/curriculum_coherence.py --sequence <comma-separated topics>` for the edge-consistency half using a session's final topic order |
| Student learning gain | `results/pretest_scores.csv`, `results/posttest_scores.csv` | **You create these** — columns `student_id,group,score_pct`, `group` is `experimental` or `control`. Use `data/pretest_python.json` / `posttest_python.json` as the actual test questions given to students; you just log the resulting percentage scores into the CSVs |
| Grading reliability | `results/instructor_grades.csv` | **You create this** — have the instructor blind-grade ~30-40 of the system's short-answer gradings; columns `student_id,question_id,system_score,instructor_score` |

Try it right now against the fake data already included, to see the exact output
shape before real numbers exist:

```bash
python -m eval.learning_gain --pretest results/sample_pretest_scores.csv --posttest results/sample_posttest_scores.csv
python -m eval.grading_reliability --file results/sample_instructor_grades.csv
```

**Reading the numbers for the paper:**
- Learning gain: report `mean_gain_experimental` vs `mean_gain_control`, the `t_test` p-value, and `cohens_d` (effect size — a significant p-value with tiny d is a weak result, say so honestly)
- Grading reliability: `quadratic_weighted_kappa` — below 0.4 is a real limitation worth discussing in the paper, not hiding
- Coordination efficiency: report all three numbers separately (latency, redundant-call rate, completion rate) rather than inventing a single blended score
- Curriculum coherence: report the two components (edge consistency, objective alignment) separately too — a 100% edge-consistency score is expected by construction here since the Planner uses a topological sort, so the more informative number is objective alignment

---

## 8. Known limitations to state honestly in the paper

- Grading agreement between the Assessment agent and the instructor is exactly the
  kind of gap the literature review already flags for GPT-4-class graders — don't be
  surprised or defensive if kappa comes back moderate rather than excellent.
- Curriculum coherence's edge-consistency component is close to guaranteed at 100%
  because Planner uses a deterministic topological sort rather than an LLM call — this
  is a design choice, not a finding, and should be described as such.
- A 20-student pilot (or fewer, if time-constrained) is underpowered for strong
  statistical claims — report effect sizes and confidence intervals, not just p-values,
  and don't overclaim significance.
