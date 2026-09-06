# CurricAgents — Complete Project Working Document

**Title:** A Closed-Loop Multi-Agent LLM System for Autonomous Curriculum Planning and Adaptive Assessment in Higher Education  
**Team:** Agastya Vashisht · Shivansh Kutlehria · Mohit Kumar · Harinder Singh · Aryan Kohli  
**Supervisor:** Dr. Raghav Mehra · Chandigarh University, AIT-CSE

---

## 1. What the Project Does (One Line)

> An AI system with 4 agents that teaches each student in a personalised order, generates questions matched to their level, and automatically adapts when they struggle — then measures whether this works better than traditional study.

---

## 2. The Big Picture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        20 STUDENTS                                  │
│                                                                     │
│   10 Experimental          vs          10 Control                   │
│   (use the AI system)              (read static notes)              │
│                                                                     │
│   Pre-test → AI Learning → Post-test    Pre-test → Notes → Post-test│
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
                     Compare learning gains
                     Compute 4 research metrics
                     Write up results for paper
```

The experiment runs on **5 subjects:**
- Python Programming (12 topics)
- Basics of ML (10 topics)
- Signal Processing (10 topics)
- Physiology Basics (10 topics)
- Data Analysis (10 topics)

All 20 students in the pilot study use the **same subject** (set by PILOT_DOMAIN in .env).

---

## 3. The 4 Agents — Who Does What

### Agent 1: PLANNER
**File:** `src/agents/planner.py`  
**Uses AI:** YES (Groq LLM)  
**Job:** Decide which topic to teach next and in what order.

How it works:
```
Step 1 — Pre-test bootstrap
  Student scores 60% on pre-test
  → Easy topics get mastery ~0.60, hard topics get ~0.20
  → Any topic already at 0.70+ is SKIPPED entirely

Step 2 — Prerequisite graph (Kahn's algorithm)
  "You can't learn functions before control_flow"
  → Enforces the dependency chain strictly

Step 3 — Instructional value formula (Paper Eq. 2)
  value = 0.6×(gap) + 0.3×(not reviewed recently) − 0.5×(already mastered)
  → Topics with biggest gaps float to the top

Step 4 — LLM re-rank
  AI receives mastery scores + misconceptions
  → Re-orders the list to put weakest+misconception topics first
  → If AI gives invalid order (breaks prerequisites) → fallback to Step 3

Step 5 — Bloom level assignment
  mastery < 0.25 → remember  ("What is X?")
  mastery 0.25–0.45 → understand ("Explain why...")
  mastery 0.45–0.65 → apply   ("What does this code output?")
  mastery 0.65–0.80 → analyze  ("What is the difference?")
  mastery 0.80+    → evaluate  ("Which design is better?")
```

**Replan trigger:** After every question, Monitor checks — if mastery < 0.35 OR 2 wrong in a row → Planner runs again with updated scores.

---

### Agent 2: CONTENT
**File:** `src/agents/content.py`  
**Uses AI:** YES (Groq LLM)  
**Job:** Generate a personalised explanation + example for the current topic.

How it works:
```
1. RAG Retrieval (NO AI)
   Searches data/corpus/<domain>.md using TF-IDF similarity
   Finds the 3 most relevant paragraphs for the topic
   Example query: "variables data types Python assignment"
   Returns: relevant notes about variables

2. LLM generation (AI)
   Prompt to AI:
     "Topic: variables_datatypes
      Bloom level: apply (student mastery = 0.50)
      Retrieved notes: [3 paragraphs from corpus]
      Known misconceptions: ['confuses int and float']
      
      Write a 3-6 sentence explanation at 'apply' level.
      Address the misconception about int vs float.
      Include a worked code example."
   
   AI returns:
     explanation: "In Python, variables are labels attached to values..."
     example: "x = 5  # int\ny = 5.0  # float\nprint(type(x))  # <class 'int'>"
```

The explanation depth changes based on Bloom level — a beginner gets simpler language, an advanced student gets more technical depth.

---

### Agent 3: ASSESSMENT
**File:** `src/agents/assessment.py`  
**Uses AI:** YES for generation; NO for MCQ grading  
**Job:** Generate a question and grade the answer.

**MCQ generation (AI):**
```
Prompt to AI:
  "Topic: loops
   Bloom level: remember (mastery = 0.15)
   Write 1 MCQ. Make 3 wrong answers target real misconceptions."

AI returns:
  Q: "How many times does 'for i in range(3):' run?"
  A. 2   ← distractor: off-by-one error
  B. 3   ← CORRECT
  C. 4   ← distractor: includes endpoint
  D. Infinite ← distractor: confuses with while True
  correct_index: 1
```

**MCQ grading (NO AI):**
```
Student picks option 1 (B = 3) → compare to correct_index (1) → CORRECT
No AI needed. Instant. 100% accurate.
```

**Short-answer grading (AI):**
```
Student types: "A loop repeats code until condition is false"

Prompt to AI:
  "Question: What is a loop?
   Rubric: [3 criteria worth 1 point each]
   Reference answer: [ideal answer]
   Student answer: 'A loop repeats code until condition is false'
   
   Give a score 0-3 and one-sentence justification."

AI returns: score=2, justification="Correct concept but missing iteration over sequence"
Cross-check: TF-IDF similarity between student answer and reference = 0.71
If |AI_score - similarity| > 0.35 → flagged for instructor review
```

**Format alternation:**
- Even steps → MCQ (quick, objective)
- Odd steps → Short answer (deeper, graded by AI)

---

### Agent 4: MONITOR
**File:** `src/agents/monitor.py`  
**Uses AI:** NO — pure maths  
**Job:** Update mastery scores and decide if replanning is needed.

**Bayesian Knowledge Tracing (Paper Eq. 1):**
```
Parameters:
  P(T) = 0.30  — probability of learning per episode
  P(F) = 0.05  — probability of forgetting within session
  P(G) = 0.25  — probability of guessing correctly (1/4 for MCQ)
  P(S) = 0.10  — probability of slipping (knowing but wrong)

If student answered CORRECT:
  P(L | correct) = P(L) × (1 - P(S))
                 / [P(L)×(1-P(S)) + (1-P(L))×P(G)]

Then update:
  P(L_new) = P(L|obs) × (1-P(F)) + (1-P(L|obs)) × P(T)

Example:
  Prior mastery = 0.30, student answered CORRECT
  Posterior = 0.30×0.90 / (0.30×0.90 + 0.70×0.25) = 0.27/0.445 = 0.607
  New mastery = 0.607×0.95 + 0.393×0.30 = 0.694
  → Mastery jumps from 0.30 to 0.69 on one correct answer
```

**Ebbinghaus forgetting (between sessions):**
```
m_decayed = m × exp(−0.05 × days_since_review)
→ 14 days without review: mastery drops by ~50%
→ This means returning students don't restart at 0 but don't keep full credit either
```

**Replan triggers:**
```
1. mastery < 0.35 after update → student is drifting → REPLAN
2. 2 wrong answers in a row on same topic → student is struggling → REPLAN
```

---

## 4. Complete Student Journey

```
STUDENT OPENS: https://your-app.streamlit.app
                    │
                    ▼
            ┌──────────────┐
            │  LOGIN PAGE  │
            │              │
            │  Enter ID:   │
            │  PILOT-01    │
            │              │
            │  Subject:    │
            │  Python ✓    │
            │  (locked)    │
            │              │
            │  [Begin ▶]   │
            └──────┬───────┘
                   │
        ID is odd → Experimental group
        ID is even → Control group
                   │
        ┌──────────┴──────────┐
        │                     │
        ▼                     ▼
EXPERIMENTAL GROUP        CONTROL GROUP
        │                     │
        ▼                     ▼
┌──────────────┐      ┌──────────────┐
│  PRE-TEST    │      │  PRE-TEST    │
│  10 MCQs     │      │  10 MCQs     │
│  No feedback │      │  No feedback │
│  ~5 min      │      │  ~5 min      │
└──────┬───────┘      └──────┬───────┘
       │                     │
       ▼                     ▼
Score saved to DB         Score saved to DB
Planner bootstraps        Nothing happens
mastery from score
       │                     │
       ▼                     ▼
┌──────────────┐      ┌──────────────┐
│  AI LEARNING │      │  STATIC NOTES│
│  SESSION     │      │              │
│              │      │  Reads the   │
│  For each    │      │  course      │
│  topic:      │      │  corpus file │
│  1. Plan     │      │  All topics  │
│  2. Explain  │      │  Same for    │
│  3. Question │      │  everyone    │
│  4. Grade    │      │  No feedback │
│  5. Update   │      │              │
│  6. Replan?  │      │  ~20-30 min  │
│  ~40-60 min  │      └──────┬───────┘
└──────┬───────┘             │
       │                     │
       ▼                     ▼
┌──────────────┐      ┌──────────────┐
│  POST-TEST   │      │  POST-TEST   │
│  10 MCQs     │      │  10 MCQs     │
│  No feedback │      │  No feedback │
│  ~5 min      │      │  ~5 min      │
└──────┬───────┘      └──────┬───────┘
       │                     │
       ▼                     ▼
Score saved to DB         Score saved to DB
       │                     │
       ▼                     ▼
┌──────────────────────────────────────┐
│  RESULTS PAGE (both groups)          │
│                                      │
│  Pre-test score:  60%                │
│  Post-test score: 78%                │
│  Learning gain:   +18 percentage pts │
│  Hake's <g>:      0.45 (medium)      │
│                                      │
│  Mastery per topic (experimental):   │
│  variables    ████████░░ 80%         │
│  loops        █████░░░░░ 50%         │
│  functions    ██████░░░░ 60%         │
└──────────────────────────────────────┘
       │
       ▼
┌──────────────┐
│  SURVEY      │
│  5 questions │
│  1-5 scale   │
│  ~1 min      │
└──────────────┘
```

---

## 5. The 4 Research Metrics

### Metric 1: Learning Gain (Primary — for paper Table III)
```
Formula: Hake's <g> = (post_score - pre_score) / (100 - pre_score)

Example:
  pre = 40%, post = 70%
  <g> = (70-40)/(100-40) = 30/60 = 0.50  (medium gain)

Interpretation:
  <g> < 0.30  = low gain
  0.30-0.70   = medium gain
  >0.70       = high gain

Comparison:
  Experimental group mean <g>  vs  Control group mean <g>
  t-test: is the difference statistically significant? (p < 0.05)
  Cohen's d: how large is the effect?

Script: python -m eval.learning_gain
```

### Metric 2: Curriculum Coherence (for paper Section VIII-C)
```
Two components:

a) Edge consistency
   "Did the AI always teach prerequisites before dependent topics?"
   Score = (edges respected) / (total edges in graph)
   Target: > 0.95

b) Objective alignment
   "Did the AI's explanations match the learning objectives?"
   Method: TF-IDF cosine similarity between explanation and objective
   Target: > 0.60

c) LLM judge score (1-5)
   Another AI rates: "Was this sequence pedagogically sound?"
   Target: > 4.0

Script: python -m eval.curriculum_coherence --domain python_programming
```

### Metric 3: Grading Reliability (for paper Table V)
```
Process:
  1. Take 30-50 short-answer responses from the pilot
  2. AI grades them (already done automatically)
  3. One researcher also grades them by hand (instructor_grades.csv)
  4. Compare AI grades vs human grades

Metrics computed:
  Accuracy:             how often AI agrees exactly
  Precision/Recall/F1:  treating "pass" as positive class
  Cohen's Kappa (QWK):  agreement accounting for chance (target: > 0.60)
  Fleiss' Kappa:        multi-rater agreement
  Krippendorff's Alpha: ordinal scale agreement

Script: python -m eval.grading_reliability --file results/instructor_grades.csv
```

### Metric 4: Agent Coordination Efficiency (for paper Section VIII-E)
```
What it measures:
  - How many agent calls per learning episode? (target: 6-8)
  - Mean latency per agent call? (target: < 8 seconds)
  - What % of calls were redundant/wasted? (target: < 5%)
  - Task completion rate (full Planner→Content→Assessment→Monitor cycle)?

Data source: logs/agent_calls.jsonl
  Every agent call is automatically logged with timing

Script: python -m eval.coordination_efficiency
```

---

## 6. Data Flow — What Gets Saved Where

```
Student answers question
        │
        ├──► SQLite DB (local, /tmp on cloud)    ← session cache
        │    results/learner_state.db
        │
        └──► Supabase PostgreSQL (online)        ← permanent store
             Tables:
               learner_state      ← full mastery, history per student
               assessment_log     ← every question + grade + raw AI response
               pilot_scores       ← pre/post test scores
               survey_responses   ← Likert questionnaire answers

Agent calls
        │
        └──► logs/agent_calls.jsonl              ← coordination metrics
             {agent, student_id, latency, tokens, changed_keys, redundant}

Content generated
        │
        └──► results/generated_content.jsonl    ← coherence metrics
             {topic, objective, explanation, n_chunks_retrieved}
```

---

## 7. File Map — Every Important File

```
curric-agents/
│
├── app.py                        ← The web app students use
│   Phases: login → pretest → gen_lesson → answering
│           → gen_feedback → posttest → done → dashboard
│
├── src/
│   ├── agents/
│   │   ├── planner.py            ← Curriculum sequencing (AI + maths)
│   │   │   Functions: _pretest_bootstrap(), _topological_order(),
│   │   │              _instructional_value(), _iv_topological_sort(),
│   │   │              _llm_rerank(), _bloom_level(), planner_node()
│   │   │
│   │   ├── content.py            ← Lesson generation (AI + RAG)
│   │   │   Functions: content_node()
│   │   │
│   │   ├── assessment.py         ← Question generation + grading (AI)
│   │   │   Functions: _generate_mcq(), _grade_short_answer(),
│   │   │              prepare_item(), apply_grade()
│   │   │
│   │   └── monitor.py            ← Mastery update (pure maths)
│   │       Functions: _bkt_posterior(), _bkt_update(),
│   │                  _apply_forgetting_decay(), monitor_node()
│   │
│   ├── config.py                 ← LLM factory (Groq/Gemini/OpenAI)
│   │   Key: get_llm(), _secret(), FallbackLLMError, _RetryProxy
│   │
│   ├── state.py                  ← Shared data structure all agents use
│   │   Fields: mastery, bloom_plan, topic_sequence, last_grade,
│   │           misconceptions, current_phase, ablation_mode...
│   │
│   ├── db.py                     ← SQLite + Supabase dual write
│   │   Functions: log_assessment(), save_pilot_score(),
│   │              save_survey_response(), get_pilot_scores()...
│   │
│   ├── memory/store.py           ← Load/save student state
│   │   Functions: load_state(), save_state()
│   │   Priority: Supabase first → SQLite fallback
│   │
│   ├── retrieval.py              ← RAG (TF-IDF, no AI)
│   │   Functions: retrieve(), _build_tfidf(), _query_vec()
│   │
│   ├── online_db.py              ← Supabase REST API layer
│   │   Functions: save_pilot_score_online(), get_pilot_scores_online(),
│   │              save_learner_state_online(), test_connection()
│   │
│   ├── graph.py                  ← LangGraph wiring (Planner→Content→Assessment→Monitor)
│   │
│   ├── logging_utils.py          ← @log_node_call decorator
│   │   Logs: agent, latency, tokens_used, changed_keys, session_id
│   │
│   ├── utils.py                  ← tokenize(), tfidf_cosine()
│   │
│   └── supabase_schema.sql       ← Run this in Supabase SQL Editor
│
├── eval/
│   ├── learning_gain.py          ← Hake's <g>, t-test, Cohen's d
│   ├── curriculum_coherence.py   ← Edge consistency + LLM judge
│   ├── grading_reliability.py    ← QWK, Fleiss κ, Krippendorff α, F1
│   ├── coordination_efficiency.py← Latency, redundancy, completion rate
│   └── run_all_metrics.py        ← Run all 4 with one command
│
├── data/
│   ├── prerequisite_graph.json   ← Python: 12 topics + prerequisites
│   ├── ml_basics_graph.json      ← ML: 10 topics
│   ├── signal_processing_graph.json
│   ├── physiology_graph.json
│   ├── data_analysis_graph.json
│   │
│   ├── pretest_python.json       ← 10 fixed MCQs (same for all students)
│   ├── posttest_python.json      ← 10 fixed MCQs (parallel form)
│   ├── pretest_ml_basics.json    ← (and same for other 4 domains)
│   ├── posttest_ml_basics.json
│   ├── ... (all 5 domains × 2 tests = 10 files)
│   │
│   ├── corpus/
│   │   ├── python_programming.md ← Course notes (AI retrieves from here)
│   │   ├── ml_basics.md
│   │   ├── signal_processing.md
│   │   ├── physiology_basics.md
│   │   └── data_analysis.md
│   │
│   └── rubrics/
│       └── python_rubrics.json   ← Short-answer rubrics
│
├── pilot/
│   ├── consent_form.md           ← Give to students before study
│   └── README.md                 ← Pilot study operations guide
│
├── tests/
│   ├── full_check.py             ← 64 automated health checks
│   └── test_personalisation.py  ← 20 planner personalisation checks
│
├── .env                          ← API keys + pilot settings (never commit)
├── .env.example                  ← Template for teammates
├── .streamlit/
│   ├── config.toml               ← Streamlit server config
│   └── secrets.toml.example      ← Template for Streamlit Cloud secrets
└── requirements.txt              ← Python packages needed
```

---

## 8. API Calls — How Many Per Student Session

```
One full session (4 topics, 3 questions each):

  Planner call:     1   (at session start)
  Content calls:    4   (one per topic)
  MCQ gen calls:   ~9   (3 questions × ~3 topics in MCQ format)
  SA grading calls: ~3  (3 questions × ~1 topic in short-answer format)
  Replan calls:    0-2  (only if student struggles)
                 ────
  Total:          ~17-19 AI calls per student

At 14,400 free calls/day (Groq):
  20 students × 18 calls = 360 calls per day
  Well within the free tier (only 2.5% of daily limit)
```

---

## 9. How to Run After Pilot

```
After all 20 students complete their sessions:

1. Export data from dashboard
   → Researcher Dashboard → Tab 4 → Download CSVs

2. Fill in instructor grades for QWK
   → Download instructor_grades_template.csv
   → Add instructor_score column (0-3) for each short-answer
   → Save as results/instructor_grades.csv

3. Run all metrics
   python -m eval.run_all_metrics
   → writes results/metrics_report.json

4. Results to put in paper:
   learning_gain.json     → Table III (pre/post scores, Hake's <g>)
   grading_reliability    → Table V (QWK, F1, accuracy)
   coordination           → Section VIII-E (latency, completion rate)
   coherence              → Section VIII-C (edge consistency, LLM judge score)
```

---

## 10. What Makes This a Research Contribution

Most educational AI tools do ONE of these things:
- ChatGPT: answers questions (no memory, no curriculum)
- Quizlet: generates flashcards (no adaptive grading)
- Khan Academy: adaptive questions (no LLM content generation)
- Gradescope: grades answers (no curriculum planning)

**This project does ALL of them in a closed loop:**

```
Traditional approach (control group):
  Teacher makes syllabus → all students follow same path → test

This system (experimental group):
  AI plans path based on student → AI teaches at right level
  → AI tests at right difficulty → AI grades → AI replans if needed
  → repeat until post-test

The "closed loop" = assessment feeds back into planning
No other free/open system does this for higher education
```

This is exactly the research gap identified in the paper (Section II-F, Table I):
- EduPlanner: adaptive curriculum, but no persistent closed loop
- EducationQ: evaluates teaching, but no adaptive planning
- DK-PRACTICE: knowledge tracing, but no LLM generation
- **Proposed system: all of the above, fully integrated**
