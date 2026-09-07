# CurricAgents — Multi-Agent LLM System for Adaptive Learning

**A Closed-Loop Multi-Agent LLM System for Autonomous Curriculum Planning and Adaptive Assessment in Higher Education**

Team: Agastya Vashisht · Shivansh Kutlehria · Mohit Kumar · Harinder Singh · Aryan Kohli  
Supervisor: Dr. Raghav Mehra · Chandigarh University, AIT-CSE

---

## What this system does

Two learning approaches are compared in a controlled experiment. Both groups take
the **same 25-question pre-test** and the **same 25-question post-test** — only the
learning between them differs:

```
                 BOTH GROUPS
Pre-test (25 Q, 5 per topic, no feedback)
        │
        ├── Traditional / Control ──► Fixed material in fixed order
        │                             (Variables → Control Flow → Loops →
        │                              Functions → OOP) + FIXED 25-question
        │                              practice bank (5 per topic).
        │                             Scores never change the path.
        │
        └── AI / Experimental ──────► PLANNER → CONTENT → ASSESSMENT → MONITOR
                                      closed loop. Mastery from the pre-test
                                      bootstraps the plan; weak topics are
                                      remediated with NEW material/questions
                                      (max 2 rounds per topic, then the topic
                                      is marked needs_review and re-scheduled).
        │
Post-test (same 25 Q, no feedback)
        │
learning_gain = post − pre   (plus Hake's normalised ⟨g⟩)
```

| Agent | Role |
|---|---|
| **Planner** | Selects next topic using BKT mastery + paper Eq. 2 heuristic + LLM re-rank |
| **Content** | Retrieves course notes (RAG) and generates a Bloom-level lesson; on remediation it generates a *different* lesson addressing the student's misconceptions |
| **Assessment** | Generates an MCQ or short-answer question; grades the response |
| **Monitor** | Updates mastery via Bayesian Knowledge Tracing (BKT, paper Eq. 1) + Ebbinghaus forgetting; triggers replan/remediation |

### Fairness controls (spec §15, §19)

* Pre-test and post-test are identical fixed files for both groups (`data/pretest_python.json` form A, `data/posttest_python.json` form B) — neither is AI-generated.
* Practice questions (AI-generated for the AI arm, fixed bank for the control arm) are separate from both tests.
* `total_learning_time`, `number_of_replans`, `number_of_remediations`, topic-wise scores and all practice answers are recorded per student (spec §17) and exportable as `research_summary.csv` from the Researcher Dashboard.

---

## Subjects (5 domains)

| Domain key | Display name | Topics |
|---|---|---|
| `python_programming` | Python Programming | 12 |
| `ml_basics` | ML Basics | 10 |
| `signal_processing` | Signal Processing | 10 |
| `physiology_basics` | Physiology Basics | 10 |
| `data_analysis` | Data Analysis | 10 |

---

## Quick start (local machine)

```powershell
# 1. Clone and activate the virtualenv
cd c:\Users\dg264\Desktop\Python\curric-agents
.\venv\Scripts\python.exe -m pip install -r requirements.txt   # if not already done

# 2. Copy and fill in .env
copy .env.example .env
# edit .env: set GROQ_API_KEY, PILOT_DOMAIN, RESEARCHER_PASSWORD

# 3. Start the web app
.\venv\Scripts\python.exe -m streamlit run app.py --server.address 0.0.0.0 --server.port 8501

# 4. Share the URL with students
# Local: http://localhost:8501
# LAN:   http://192.168.x.x:8501  (find IP with ipconfig)
```

---

## Pilot study setup

1. **Set `PILOT_DOMAIN`** in `.env` to lock all 20 students to one domain (e.g. `python_programming`).
2. **Student IDs**: assign `PILOT-01` through `PILOT-20`. Odd numbers = experimental group, even = control — auto-detected by the login page.
3. **Share the URL** — students only enter their ID and click Begin.
4. Students complete: Pre-test → Learning session (experimental) or Static notes (control) → Post-test → Survey.
5. After all sessions: go to the **Researcher Dashboard** (password in `.env`) to view results and export CSVs.

---

## Evaluation metrics

Run after pilot data is collected:

```powershell
.\venv\Scripts\python.exe -m eval.run_all_metrics
# writes results/metrics_report.json
```

| Metric | Script | Paper section |
|---|---|---|
| Learning gain (Hake's `<g>`, t-test, Cohen's d) | `eval/learning_gain.py` | §VI-B |
| Curriculum coherence (edge consistency + LLM judge) | `eval/curriculum_coherence.py` | §VI-B |
| Grading reliability (Accuracy, Precision, Recall, F1, QWK, Fleiss κ, Krippendorff α) | `eval/grading_reliability.py` | §VI-B, Table V |
| Agent coordination efficiency (latency, redundant calls, task completion rate) | `eval/coordination_efficiency.py` | §VI-B |

---

## .env reference

```env
# LLM provider (groq recommended — 14,400 free req/day)
MODEL_PROVIDER=groq
GROQ_API_KEY=gsk_...
GROQ_MODEL=openai/gpt-oss-120b

# Pilot settings
PILOT_DOMAIN=python_programming     # locks all students to this subject
PILOT_MAX_TOPICS=4                  # topics shown per session
PILOT_QPT=3                         # questions per topic
ABLATION_MODE=full                  # full | no_adaptive_assessment | no_adaptive_planning | planner_only | static
RESEARCHER_PASSWORD=changeme        # dashboard password

# Database
DB_PATH=results/learner_state.db

# Supabase (optional — enables multi-machine dashboard)
SUPABASE_URL=https://<ref>.supabase.co
SUPABASE_ANON_KEY=eyJ...
```

---

## File structure

```
app.py                          ← Streamlit web app (student-facing)
src/
  agents/
    planner.py                  ← Planner (BKT bootstrap + Eq.2 heuristic + LLM re-rank)
    content.py                  ← Content (RAG + Bloom-level generation, remediation-aware)
    assessment.py               ← Assessment (MCQ/SA generation + BKT grading)
    monitor.py                  ← Monitor (BKT Eq.1 + Ebbinghaus decay)
  remediation.py                ← Topic-end decision: remediate / advance / needs_review (spec §12)
  notes.py                      ← Fixed topic order + per-topic corpus notes (control arm)
  config.py                     ← LLM factory + retry proxy (30 s timeout)
  state.py                      ← LearnerState schema
  db.py                         ← SQLite + Supabase dual-write
  memory/store.py               ← load_state / save_state
  retrieval.py                  ← Pure-numpy TF-IDF RAG retriever
  utils.py                      ← tokenize, tfidf_cosine
  online_db.py                  ← Supabase REST API layer
  logging_utils.py              ← @log_node_call → logs/agent_calls.jsonl
  graph.py                      ← LangGraph StateGraph wiring
  supabase_schema.sql           ← SQL to run in Supabase SQL Editor
eval/
  learning_gain.py              ← Hake's <g>, t-test, Cohen's d (pure numpy)
  curriculum_coherence.py       ← Edge consistency + LLM judge + TF-IDF alignment
  coordination_efficiency.py    ← Latency, redundancy, task completion rate
  grading_reliability.py        ← QWK, Fleiss κ, Krippendorff α, F1 (pure numpy)
  run_all_metrics.py            ← Run all 4 metrics with one command
data/
  prerequisite_graph.json       ← Python domain (5 pilot topics)
  ml_basics_graph.json
  signal_processing_graph.json
  physiology_graph.json
  data_analysis_graph.json
  corpus/                       ← RAG source material (5 × .md files)
  pretest_*.json                ← Fixed 25-question (5/topic) pre-tests
  posttest_*.json               ← Fixed 25-question (5/topic) post-tests
  practice_python.json          ← Fixed 25-question practice bank (control arm)
  rubrics/                      ← Short-answer rubrics
pilot/
  consent_form.md               ← Fill in before recruiting students
  README.md                     ← Pilot logistics guide
results/                        ← DB, CSVs, metrics report land here
logs/                           ← agent_calls.jsonl lands here
tests/
  full_check.py                 ← 64-check automated health test
  test_personalisation.py       ← 20-check personalisation test
  test_smoke.py                 ← Deterministic no-API-key tests
  test_flow_spec.py             ← Traditional-vs-AI flow spec tests (no API key)
```
