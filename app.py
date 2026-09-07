"""
CurricAgents — Pilot Study Web App
===================================
Multi-Agent LLM System for Autonomous Curriculum Planning and Adaptive Assessment

Experimental design (Traditional vs AI):
  1. login        — enter student ID; group auto-assigned from ID parity
                    (odd = AI/experimental, even = Traditional/control)
  2. pretest      — SAME 25-question sectioned pre-test for both groups
                    (5 questions/topic, no feedback), score saved
  3. learning     — AI group:      planner → content → assessment → monitor
                                   closed loop; max 2 remediations per topic,
                                   then topic is marked needs_review and
                                   re-scheduled for a later review.
                    Traditional:   fixed topic order (Variables → Control Flow
                                   → Loops → Functions → OOP) with corpus notes
                                   + a FIXED 25-question practice bank
                                   (5 questions/topic) that never changes the
                                   learning path.
  4. posttest     — SAME 25-question post-test for both groups (no feedback)
  5. done         — mastery summary + learning gain shown

Both groups record total_learning_time, replans, remediations and per-topic
scores (spec §17) so the research comparison (learning gain §18) is fair (§19).

Run:
    streamlit run app.py
"""

import json
import re
import time
import os
import pandas as pd
import streamlit as st

try:
    import altair as alt
    _HAS_ALTAIR = True
except ImportError:
    _HAS_ALTAIR = False

from src.config import DB_PATH
from src.memory.store import load_state, save_state
from src.agents.planner import planner_agent, _load_graph
from src.agents.content import content_agent
from src.agents.monitor import monitor_agent
from src.agents.assessment import assessment_prepare_agent, apply_grade
from src.graph import advance_topic as _advance
from src.remediation import topic_end_decision, MAX_REMEDIATION_ATTEMPTS
from src.notes import topic_order, topic_notes
from src.db import log_assessment, save_pilot_score, get_pilot_scores, get_assessment_log, save_survey_response, get_survey_responses

# ── page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CurricAgents — Pilot Study",
    page_icon="🎓",
    layout="wide",
)

# ── Pilot study settings ─────────────────────────────────────────────────────
def _env(key: str, default: str = "") -> str:
    """Read from st.secrets (Streamlit Cloud) or os.environ (.env / local)."""
    try:
        import streamlit as _st
        val = _st.secrets.get(key, "")
        if val:
            return str(val)
    except Exception:
        pass
    return os.getenv(key, default)

PILOT_DOMAIN        = _env("PILOT_DOMAIN", "")
MASTERY_THRESHOLD_PRACTICE = 0.70   # topics below this are shown in practice mode
ABLATION_MODES = {
    "full":                   "Full system (adaptive planning + adaptive assessment)",
    "no_adaptive_assessment": "No adaptive assessment (fixed difficulty questions)",
    "no_adaptive_planning":   "No adaptive planning (fixed topic order)",
    "planner_only":           "Planner only (adaptive order, templated content/assessment)",
    "static":                 "Static baseline (fixed order + fixed questions)",
}


DOMAINS = {
    "python_programming":  "Python Programming",
    # Other domains available but excluded from current pilot (Python only)
    # "ml_basics":           "ML Basics",
    # "signal_processing":   "Signal Processing",
    # "physiology_basics":   "Physiology Basics",
    # "data_analysis":       "Data Analysis",
}

PRETEST_FILES = {
    "python_programming": "data/pretest_python.json",
    "ml_basics":          "data/pretest_ml_basics.json",
    "signal_processing":  "data/pretest_signal_processing.json",
    "physiology_basics":  "data/pretest_physiology_basics.json",
    "data_analysis":      "data/pretest_data_analysis.json",
}

POSTTEST_FILES = {
    "python_programming": "data/posttest_python.json",
    "ml_basics":          "data/posttest_ml_basics.json",
    "signal_processing":  "data/posttest_signal_processing.json",
    "physiology_basics":  "data/posttest_physiology_basics.json",
    "data_analysis":      "data/posttest_data_analysis.json",
}

# Fixed practice bank used ONLY by the Traditional/control group (spec §4).
# 25 questions (5 per topic), intentionally distinct from the pre-test and
# post-test questions — the control arm's practice never changes its path.
PRACTICE_FILES = {
    "python_programming": "data/practice_python.json",
    # Other domains default to Python practice — only python_programming is active
    # in the pilot. Future domains should have their own practice banks.
    "ml_basics":          "data/practice_python.json",
    "signal_processing":  "data/practice_python.json",
    "physiology_basics":  "data/practice_python.json",
    "data_analysis":      "data/practice_python.json",
}

# ── helpers ───────────────────────────────────────────────────────────────────
def _label(t: str) -> str:
    return t.replace("_", " ").title()


def _color(s: float) -> str:
    return "🟢" if s >= 0.7 else ("🟡" if s >= 0.35 else "🔴")


def _load_test(path: str) -> list[dict]:
    """Load test file. Supports both old flat format and new sectioned format."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    # New format: has "sections" with per-topic question groups
    if "sections" in data:
        return data["sections"]   # list of {topic, title, questions:[...]}
    # Old flat format: has "questions" at top level
    return data.get("questions", [])


def _is_sectioned(test_data: list) -> bool:
    """True if test data uses the new sectioned format (each item has 'questions' key)."""
    return bool(test_data) and "questions" in test_data[0]


def _score_sectioned(sections: list, answers: dict) -> tuple[int, int, dict]:
    """
    Score a sectioned test.
    Returns (n_correct, n_total, per_topic_scores).
    per_topic_scores = {topic: {"correct": int, "total": int, "pct": float}}
    """
    n_correct = 0
    n_total   = 0
    per_topic = {}
    for sec in sections:
        topic  = sec["topic"]
        t_corr = 0
        t_tot  = len(sec["questions"])
        for q in sec["questions"]:
            if answers.get(q["id"]) == q["correct_index"]:
                t_corr += 1
                n_correct += 1
            n_total += 1
        per_topic[topic] = {
            "correct": t_corr,
            "total":   t_tot,
            "pct":     round(t_corr / t_tot * 100, 1) if t_tot else 0.0,
        }
    return n_correct, n_total, per_topic


def _load_all_topics(domain: str) -> list[str]:
    """Ordered topic list from the domain's prerequisite graph."""
    from src.agents.planner import _load_graph
    try:
        return list(_load_graph(domain).keys())
    except Exception:
        return []


def _altair_chart(chart) -> None:
    st.altair_chart(chart, use_container_width=True)


def _chart_bars(df: pd.DataFrame, x: str, y: str, color: str | None = None,
                 title: str = "", y_title: str = "", y_domain=None, height: int = 320) -> None:
    """Render a bar chart (Altair when available, otherwise Streamlit native)."""
    if df is None or df.empty:
        st.info("No data yet for this chart.")
        return
    plot = df.copy()
    if _HAS_ALTAIR:
        y_enc = alt.Y(f"{y}:Q", title=y_title or y.replace("_", " ").title())
        if y_domain is not None:
            y_enc = alt.Y(
                f"{y}:Q",
                title=y_title or y.replace("_", " ").title(),
                scale=alt.Scale(domain=y_domain),
            )
        enc = {
            "x": alt.X(f"{x}:N", title=x.replace("_", " ").title(), sort=None),
            "y": y_enc,
            "tooltip": list(plot.columns),
        }
        if color and color in plot.columns:
            enc["color"] = alt.Color(f"{color}:N", title=color.replace("_", " ").title())
        chart = alt.Chart(plot).mark_bar().encode(**enc).properties(height=height, title=title)
        _altair_chart(chart)
        return
    kwargs = {"x": x, "y": y, "use_container_width": True}
    if color and color in plot.columns:
        kwargs["color"] = color
    if title:
        st.caption(title)
    st.bar_chart(plot, **kwargs)


def _chart_pre_post_grouped(pivot: pd.DataFrame) -> None:
    """Grouped pre vs post bars per student (shows even with one row)."""
    long = pivot.melt(
        id_vars=[c for c in ["student_id", "group", "domain"] if c in pivot.columns],
        value_vars=[c for c in ["pretest", "posttest"] if c in pivot.columns],
        var_name="test",
        value_name="score_pct",
    ).dropna(subset=["score_pct"])
    if long.empty:
        st.info("No pre/post scores to plot yet.")
        return
    long["student"] = long["student_id"].astype(str)
    if "domain" in long.columns:
        long["student"] = long["student"] + " (" + long["domain"].astype(str) + ")"
    if _HAS_ALTAIR:
        try:
            chart = (
                alt.Chart(long)
                .mark_bar()
                .encode(
                    x=alt.X("student:N", title="Student", sort=None),
                    y=alt.Y("score_pct:Q", title="Score (%)", scale=alt.Scale(domain=[0, 100])),
                    color=alt.Color(
                        "test:N",
                        title="Test",
                        scale=alt.Scale(domain=["pretest", "posttest"], range=["#4C78A8", "#54A24B"]),
                    ),
                    xOffset="test:N",
                    tooltip=["student_id", "test", "score_pct"]
                    + (["group"] if "group" in long.columns else []),
                )
                .properties(height=340, title="Pre-test vs post-test (same student)")
            )
            _altair_chart(chart)
            return
        except Exception:
            pass
    wide = long.pivot_table(index="student", columns="test", values="score_pct", aggfunc="first")
    st.bar_chart(wide, use_container_width=True)


def _chart_group_means(pivot: pd.DataFrame) -> None:
    """Experimental vs control mean pre, post, and Hake ⟨g⟩."""
    rows = []
    for gname, sub in pivot.groupby("group"):
        rows.append({
            "group": gname,
            "mean_pretest": sub["pretest"].mean() if "pretest" in sub else None,
            "mean_posttest": sub["posttest"].mean() if "posttest" in sub else None,
            "mean_hake_g": sub["hake_g"].mean() if "hake_g" in sub else None,
        })
    summary = pd.DataFrame(rows)
    if summary.empty:
        return
    long = summary.melt(id_vars=["group"], var_name="metric", value_name="value").dropna()
    if long.empty:
        return
    labels = {
        "mean_pretest": "Mean pre-test %",
        "mean_posttest": "Mean post-test %",
        "mean_hake_g": "Mean Hake ⟨g⟩",
    }
    long["metric"] = long["metric"].map(labels).fillna(long["metric"])
    _chart_bars(
        long, x="metric", y="value", color="group",
        title="Experimental vs control (paper comparison)",
        height=320,
    )


def _save(ls: dict, phase: str) -> None:
    """Save state and record the current phase for session resume.

    Also accumulates wall-clock learning time (spec §19): every save adds the
    time elapsed since the previous save, which gives a good approximation of
    how long the student actually spent on the current screen.
    """
    ls = dict(ls)
    ls["current_phase"] = phase
    eng = dict(ls.get("engagement", {}))
    now = time.time()
    last = eng.get("last_activity_ts")
    if last:
        eng["learning_time_sec"] = (
            eng.get("learning_time_sec", 0.0) + (now - float(last))
        )
    eng["last_activity_ts"] = now
    ls["engagement"] = eng
    st.session_state.ls = ls
    save_state(DB_PATH, ls)


def _ensure_engagement(ls: dict) -> dict:
    """Back-fill engagement research counters on states saved before they existed."""
    eng = dict(ls.get("engagement", {}))
    for k, v in {
        "remediation_counts": {},
        "total_replans": 0,
        "total_remediations": 0,
        "learning_time_sec": 0.0,
        "last_activity_ts": 0.0,
    }.items():
        eng.setdefault(k, v)
    ls["engagement"] = eng
    return ls


def _grade(state: dict, answer: str) -> dict:
    """Grade then persist to assessment_log for QWK / ASAG evaluation."""
    item = state.get("pending_item") or {}
    t0 = time.time()
    state = apply_grade(state, answer)
    latency_ms = int((time.time() - t0) * 1000)
    grade = state.get("last_grade") or {}
    grade_val = grade.get("score")
    if item.get("format") == "mcq":
        graded_by = "exact_match"
    elif grade.get("embedding_similarity") is not None:
        graded_by = "llm_judge+embedding"
    else:
        graded_by = "llm_judge"
    log_assessment(
        student_id=state["student_id"],
        domain=state["domain"],
        topic=state["topic_pointer"],
        item_type=item.get("format", "unknown"),
        item_text=item.get("prompt") or item.get("question", ""),
        response=answer,
        grade=grade_val,
        justification=grade.get("justification"),
        raw_llm_response=grade.get("raw_response"),
        graded_by=graded_by,
        flagged=bool(grade.get("flagged")),
        latency_ms=latency_ms,
        db_path=DB_PATH,
    )
    return state


# ── sidebar ───────────────────────────────────────────────────────────────────
def _sidebar():
    st.sidebar.title("🎓 CurricAgents")
    phase = st.session_state.get("phase", "login")

    if phase in ("pretest", "traditional", "traditional_practice", "gen_lesson",
                 "answering", "gen_feedback", "posttest", "done"):
        ls = st.session_state.get("ls")
        if ls:
            domain_label = DOMAINS.get(ls["domain"], ls["domain"])
            st.sidebar.markdown(f"**Student:** `{ls['student_id']}`")
            st.sidebar.markdown(f"**Domain:** {domain_label}")
            st.sidebar.markdown(f"**Group:** `{st.session_state.get('group', 'experimental')}`")
            st.sidebar.markdown(f"**Session:** {ls['engagement'].get('session_count', 1)}")
            st.sidebar.divider()

            all_topics = _load_all_topics(ls["domain"])
            mastery    = ls.get("mastery", {})
            bloom_plan = {b["topic"]: b for b in ls.get("bloom_plan", [])}
            current    = ls.get("topic_pointer")
            is_ctrl    = ls.get("group") == "control"
            trad_idx   = int(ls.get("traditional_index", 0))
            trad_order = []
            if is_ctrl:
                try:
                    from src.notes import topic_order as _topic_order
                    trad_order = _topic_order(ls["domain"])
                except Exception:
                    pass

            for i, t in enumerate(all_topics):
                s      = mastery.get(t, 0.0)
                bar    = "█" * int(s * 10) + "░" * (10 - int(s * 10))
                prefix = "▶ " if t == current else "  "
                b_info = bloom_plan.get(t)
                bloom  = f" _{b_info['bloom_level']}_" if b_info else ""

                if is_ctrl:
                    # Control group: show topic completion status, not mastery %
                    if trad_order and i < len(trad_order):
                        trad_pos = trad_order.index(t) if t in trad_order else -1
                        if trad_pos < trad_idx:
                            label  = f"~~{_label(t)}~~ ✓"
                            prefix = "  "
                        elif trad_pos == trad_idx:
                            label  = f"**{_label(t)}** ← current"
                            prefix = "▶ "
                        else:
                            label  = f"{_label(t)}"
                    else:
                        label = f"{_label(t)}"
                    st.sidebar.markdown(f"{prefix} {label}")
                else:
                    # Experimental group: show mastery bars
                    if t not in (ls.get("topic_sequence", []) + ([current] if current else [])) \
                            and s >= 0.7:
                        label = f"~~{_label(t)}~~ ✓"
                    else:
                        label = f"**{_label(t)}**"
                    st.sidebar.markdown(
                        f"{prefix}{_color(s)} {label}{bloom}  \n`{bar}` {s:.0%}"
                    )

            st.sidebar.divider()
            is_control = ls.get("group") == "control"
            if is_control:
                # Control arm shows fixed topic-order progress (spec §16)
                order = topic_order(ls["domain"])
                done  = min(int(ls.get("traditional_index", 0)), len(order))
                total = max(len(order), 1)
                st.sidebar.progress(
                    min(done / total, 1.0),
                    text=f"Topics {done}/{total}  •  Fixed path"
                )
            else:
                done = st.session_state.get("topics_done", 0)
                total = st.session_state.get("max_topics", 5)
                q_done = st.session_state.get("q_done", 0)
                q_total = st.session_state.get("q_per_topic", 5)
                st.sidebar.progress(
                    min(done / max(total, 1), 1.0),
                    text=f"Topics {done}/{total}  •  Q {q_done}/{q_total}"
                )

    # Researcher dashboard link
    st.sidebar.divider()
    if st.sidebar.button("📊 Researcher Dashboard"):
        st.session_state.phase = "dashboard"
        st.rerun()
    # Confirmation guard — prevents accidental navigation during survey/session
    phase = st.session_state.get("phase", "login")
    if phase not in ("login", "done", "dashboard"):
        if st.sidebar.button("🏠 Home (⚠ ends session)"):
            st.session_state["_confirm_home"] = True
        if st.session_state.get("_confirm_home"):
            st.sidebar.warning("This will end your current session.")
            c1, c2 = st.sidebar.columns(2)
            if c1.button("Yes, exit", key="confirm_home_yes"):
                st.session_state["_confirm_home"] = False
                st.session_state.phase = "login"
                st.rerun()
            if c2.button("Cancel", key="confirm_home_no"):
                st.session_state["_confirm_home"] = False
                st.rerun()
    else:
        if st.sidebar.button("🏠 Home"):
            st.session_state.phase = "login"
            st.rerun()


def _show_history():
    for role, text in st.session_state.get("history", []):
        with st.chat_message(role):
            st.markdown(text)


# ════════════════════════════════════════════════════════════════
#  PHASE HANDLERS
# ════════════════════════════════════════════════════════════════

def phase_login():
    # ── Page layout ───────────────────────────────────────────────────────────
    st.markdown("""
        <style>
        .login-card {
            max-width: 480px;
            margin: 2rem auto;
            padding: 2.5rem 2rem 2rem 2rem;
            border: 1px solid #e0e0e0;
            border-radius: 12px;
            background: #fafafa;
        }
        .login-title { text-align: center; font-size: 2rem; margin-bottom: 0.2rem; }
        .login-sub   { text-align: center; color: #555; margin-bottom: 1.5rem; }
        </style>
    """, unsafe_allow_html=True)

    # ── Centred card ──────────────────────────────────────────────────────────
    _, card_col, _ = st.columns([1, 2, 1])
    with card_col:
        st.markdown('<p class="login-title">🎓 CurricAgents</p>', unsafe_allow_html=True)
        st.markdown(
            '<p class="login-sub">Adaptive Learning System — Pilot Study<br>'
            '<small>Chandigarh University · AIT-CSE</small></p>',
            unsafe_allow_html=True,
        )

        # ── Student ID ────────────────────────────────────────────────────────
        sid = st.text_input(
            "Your Pilot Study ID",
            placeholder="e.g. PILOT-01",
            max_chars=20,
            help="You received this ID in the study consent form. Contact your researcher if unsure.",
        )

        # ── Domain — locked to PILOT_DOMAIN if set ────────────────────────────
        if PILOT_DOMAIN and PILOT_DOMAIN in DOMAINS:
            domain = PILOT_DOMAIN
            st.info(f"📌 **Subject:** {DOMAINS[domain]}")
        else:
            domain = st.selectbox(
                "Select your subject",
                list(DOMAINS.keys()),
                format_func=lambda k: DOMAINS[k],
            )

        # ── Group — auto-assigned from Student ID suffix, not shown to student ─
        # Odd-numbered IDs (PILOT-01, 03 ...) → experimental
        # Even-numbered IDs (PILOT-02, 04 ...) → control
        nums = re.findall(r"\d+", sid.strip())
        if nums:
            group = "experimental" if int(nums[-1]) % 2 == 1 else "control"
        else:
            group = "experimental"

    # ── Ablation — always "full" for pilot students (researcher sets in .env/secrets)
        ablation = _env("ABLATION_MODE", "full")

        # ── How it works ──────────────────────────────────────────────────────
        with st.expander("ℹ️ How this study works"):
            st.markdown("""
**You will:**
1. Answer a **25-question pre-test** (5 questions per topic, no feedback)
2. Complete a **learning session** — the system teaches all 5 Python topics and
   quizzes you with **5 practice questions per topic**
3. Answer the **same 25-question post-test** (no feedback)
4. Fill in a **short questionnaire** (about 1 minute)

**Total time:** approximately 45–60 minutes.

**Your data:** only your anonymised Pilot ID is stored — never your name.
You can stop at any time without penalty.
""")

        st.divider()
        go = st.button("Begin Session ▶", type="primary", use_container_width=True)

        if not sid.strip():
            st.caption("Enter your Pilot Study ID to begin.")

        # ── Researcher back-door (collapsed, password-gated) ──────────────────
        with st.expander("🔬 Researcher / Admin access"):
            r_pass = st.text_input("Researcher password", type="password", key="r_pass")
            RESEARCHER_PASS = _env("RESEARCHER_PASSWORD", "admin123")
            if r_pass == RESEARCHER_PASS:
                st.success("Researcher mode unlocked")
                r_domain   = st.selectbox("Domain override", list(DOMAINS.keys()),
                                          format_func=lambda k: DOMAINS[k], key="r_dom")
                r_group    = st.radio("Group override", ["experimental", "control"], key="r_grp")
                r_ablation = st.selectbox(
                    "Ablation mode", list(ABLATION_MODES.keys()),
                    format_func=lambda k: ABLATION_MODES[k], key="r_abl"
                )
                r_topics = st.slider("Max topics", 1, 10, 5, key="r_top")
                r_qpt    = st.slider("Questions per topic", 1, 5, 5, key="r_qpt")
                if st.button("Start as Researcher ▶", key="r_go"):
                    _start_session(
                        sid.strip() or "RESEARCHER",
                        r_domain, r_group, r_ablation, r_topics, r_qpt
                    )
            elif r_pass:
                st.error("Incorrect password")

    # ── Normal student start ──────────────────────────────────────────────────
    if go and sid.strip():
        _start_session(
            sid.strip(), domain, group, ablation,
            max_topics=int(_env("PILOT_MAX_TOPICS", "5")),
            q_per_topic=int(_env("PILOT_QPT", "5")),
        )


def _start_session(
    sid: str,
    domain: str,
    group: str,
    ablation: str,
    max_topics: int = 5,
    q_per_topic: int = 5,
) -> None:
    """Initialise session state and navigate to the correct phase."""
    from src.memory.store import load_state
    ls = load_state(DB_PATH, sid, domain)
    ls = dict(ls)
    ls["ablation_mode"] = ablation
    ls["group"]         = group
    ls                  = _ensure_engagement(ls)
    # New session -> fresh remediation budget for this session (spec §12 cap is
    # per session; cumulative totals stay in total_replans/total_remediations).
    ls["engagement"]["remediation_counts"] = {}
    ls["engagement"]["session_count"] = ls["engagement"].get("session_count", 0) + 1
    ls["engagement"]["last_activity_ts"] = time.time()   # start the learning-time clock
    if "traditional_index" not in ls:
        ls["traditional_index"] = 0

    # ── Determine what the student should do this session ─────────────────────
    saved_phase = ls.get("current_phase", "pretest")

    # Check DB for existing pre/post test scores
    stored_scores = get_pilot_scores(DB_PATH)
    student_scores = [s for s in stored_scores
                      if s["student_id"] == sid and s["domain"] == domain]
    has_pretest  = any(s["test_type"] == "pretest"  for s in student_scores)
    has_posttest = any(s["test_type"] == "posttest" for s in student_scores)

    if has_posttest:
        # Research study is complete (pre+post done).
        # Control group: mastery is never updated via BKT so skip the mastery check.
        # Experimental group: offer practice if weak topics remain.
        if group == "control":
            resume_phase = "done"
        else:
            weak_topics        = [t for t, m in ls.get("mastery", {}).items()
                                  if m < MASTERY_THRESHOLD_PRACTICE]
            remaining_sequence = ls.get("topic_sequence", [])
            if remaining_sequence or weak_topics:
                resume_phase = "practice_menu"
            else:
                resume_phase = "done"

    elif has_pretest and saved_phase in ("gen_lesson", "answering",
                                         "gen_feedback", "traditional",
                                         "traditional_practice"):
        # Pre-test done, still in learning phase — resume exactly where left off
        resume_phase = saved_phase

    elif has_pretest and saved_phase == "posttest":
        # Was about to do post-test — resume there
        resume_phase = "posttest"

    elif has_pretest and saved_phase in ("done", "pretest", ""):
        # Pre-test done but session 1 ended before post-test (e.g. ran out of topics quota)
        # Start a fresh learning session continuing from remaining topics
        resume_phase = "gen_lesson" if group == "experimental" else "traditional"

    else:
        # First time — start with pre-test
        resume_phase = "pretest"

    # Run planner for experimental group when starting fresh learning
    if group == "experimental" and ablation != "static":
        needs_plan = (
            (resume_phase in ("pretest", "gen_lesson") and not has_pretest)
            or (resume_phase == "gen_lesson" and has_pretest)
        )
        if needs_plan:
            try:
                # Planner makes an LLM call — catch all errors so login never crashes
                ls = planner_agent(ls)
            except Exception as e:
                # Log but don't block the student — they can still do the pretest
                print(f"[PLANNER] _start_session planner call failed: {e}")
                # Set a minimal safe plan so the state is usable
                from src.agents.planner import _load_graph, _topological_order
                try:
                    topics = _load_graph(domain)
                    topo   = _topological_order(topics, ls.get("mastery", {}))
                    ls["topic_sequence"] = topo
                    ls["topic_pointer"]  = topo[0] if topo else None
                    ls["bloom_plan"]     = []
                except Exception:
                    pass

    # topics_done: restore from engagement or reset for new learning chunk
    topics_done = ls["engagement"].get("topics_done", 0) if resume_phase != "pretest" else 0

    st.session_state.ls           = ls
    st.session_state.domain       = domain
    st.session_state.group        = group
    st.session_state.ablation     = ablation
    st.session_state.max_topics   = max_topics
    st.session_state.q_per_topic  = q_per_topic
    st.session_state.topics_done  = topics_done
    st.session_state.q_done       = 0
    st.session_state.history      = []
    st.session_state.test_answers = {}
    st.session_state.survey_submitted = False
    st.session_state.phase        = resume_phase

    # Show the student what's happening
    if resume_phase == "done":
        st.toast("✅ Your study session is complete. See your results below.", icon="🎉")
    elif resume_phase != "pretest":
        st.toast("✅ Welcome back! Resuming from where you left off.", icon="🎓")

    st.rerun()


# ── pre-test ──────────────────────────────────────────────────────────────────
def phase_pretest():
    ls     = st.session_state.ls
    domain = ls["domain"]
    _sidebar()

    # ── Hard guard: never retake ──────────────────────────────────────────────
    stored = get_pilot_scores(DB_PATH)
    already_done = any(
        s["student_id"] == ls["student_id"]
        and s["domain"] == domain
        and s["test_type"] == "pretest"
        for s in stored
    )
    if already_done:
        group       = st.session_state.get("group", "experimental")
        stored_post = any(
            s["student_id"] == ls["student_id"]
            and s["domain"] == domain
            and s["test_type"] == "posttest"
            for s in stored
        )
        if stored_post:
            st.session_state.phase = "done"
        elif group == "control":
            st.session_state.phase = "traditional"
        else:
            st.session_state.phase = "gen_lesson"
        st.rerun()

    # ── Load test data ────────────────────────────────────────────────────────
    test_data = _load_test(PRETEST_FILES[domain])
    sectioned = _is_sectioned(test_data)
    answers   = st.session_state.get("test_answers", {})

    st.title(f"📋 Pre-Test — {DOMAINS[domain]}")

    if sectioned:
        # ── New sectioned format: one collapsible section per topic ──────────
        st.info(
            f"This test has **{len(test_data)} sections** — one per topic — "
            f"with **5 questions each** ({len(test_data) * 5} questions total).  \n"
            "Answer every question. **No feedback is given.** Submit when done."
        )

        # Progress bar — how many questions answered so far
        total_qs     = sum(len(s["questions"]) for s in test_data)
        answered_qs  = sum(1 for s in test_data
                           for q in s["questions"] if q["id"] in answers)
        st.progress(
            answered_qs / total_qs if total_qs else 0,
            text=f"Answered {answered_qs} / {total_qs} questions"
        )

        with st.form("pretest_form"):
            for sec_idx, section in enumerate(test_data):
                topic_label = section.get("title", section["topic"].replace("_", " ").title())
                answered_in_sec = sum(1 for q in section["questions"] if q["id"] in answers)
                sec_done = answered_in_sec == len(section["questions"])
                icon = "✅" if sec_done else "📝"

                with st.expander(f"{icon} {topic_label}  ({answered_in_sec}/{len(section['questions'])} answered)", expanded=not sec_done):
                    for q_idx, q in enumerate(section["questions"]):
                        st.markdown(f"**Q{q_idx+1}. {q['prompt']}**")
                        # Use None as default so unanswered questions are not silently submitted
                        current_answer = answers.get(q["id"])
                        radio_opts     = list(range(len(q["options"])))
                        answers[q["id"]] = st.radio(
                            f"Answer for {q['id']}",
                            options=radio_opts,
                            format_func=lambda i, opts=q["options"]: f"{chr(65+i)}. {opts[i]}",
                            key=f"pre_{q['id']}",
                            index=current_answer if current_answer is not None else None,
                            label_visibility="collapsed",
                        )
                        if q_idx < len(section["questions"]) - 1:
                            st.markdown("---")

            st.divider()
            st.caption(f"Make sure all {total_qs} questions are answered before submitting.")
            already_submitted = st.session_state.get("_pretest_submitted", False)
            submitted = st.form_submit_button(
                "Submit Pre-Test ✓", type="primary",
                use_container_width=True,
                disabled=already_submitted,
            )
            if submitted:
                st.session_state["_pretest_submitted"] = True

        if submitted:
            n_correct, n_total, per_topic = _score_sectioned(test_data, answers)
            score_pct = round(n_correct / n_total * 100, 1) if n_total else 0.0

            # Show per-topic breakdown before proceeding
            st.success(f"✅ Pre-test submitted! Overall score: **{score_pct}%** ({n_correct}/{n_total})")
            with st.expander("Per-topic scores"):
                for topic, s in per_topic.items():
                    label = topic.replace("_", " ").title()
                    bar   = "█" * s["correct"] + "░" * (s["total"] - s["correct"])
                    st.markdown(f"**{label}**: {s['correct']}/{s['total']}  `{bar}`  {s['pct']}%")

            save_pilot_score(
                student_id=ls["student_id"], domain=domain, test_type="pretest",
                score_pct=score_pct, n_correct=n_correct, n_total=n_total,
                group_label=st.session_state.get("group", "experimental"),
                ablation_mode=st.session_state.get("ablation", "full"),
                db_path=DB_PATH,
            )
            # Save per-topic pretest scores into state for planner bootstrap
            ls = dict(ls)
            ls["pretest_score_pct"] = score_pct
            ls["pretest_per_topic"] = per_topic
            st.session_state.pretest_score = score_pct
            st.session_state.test_answers  = {}
            st.session_state.ls = ls

            if st.session_state.get("group") == "control":
                ls = dict(ls)
                ls["traditional_index"] = 0   # start the fixed learning path
                st.session_state.practice_answers = {}
                _save(ls, "traditional")
                st.session_state.phase = "traditional"
            else:
                ls = planner_agent(ls)
                _save(ls, "gen_lesson")
                st.session_state.ls = ls
                st.session_state.phase = "gen_lesson"
            st.rerun()

    else:
        # ── Old flat format (fallback) ────────────────────────────────────────
        st.info("Answer all questions. **No feedback is given during the test.** Submit when done.")
        with st.form("pretest_form"):
            for q in test_data:
                st.markdown(f"**{q['id'].upper()}. {q['prompt']}**")
                answers[q["id"]] = st.radio(
                    f"Answer for question {q['id'].upper()}",
                    options=list(range(len(q["options"]))),
                    format_func=lambda i, opts=q["options"]: f"{i}. {opts[i]}",
                    key=f"pre_{q['id']}",
                    index=answers.get(q["id"], 0),
                    label_visibility="collapsed",
                )
                st.markdown("---")
            submitted = st.form_submit_button("Submit Pre-Test ✓", type="primary")

        if submitted:
            n_correct = sum(1 for q in test_data if answers.get(q["id"]) == q["correct_index"])
            n_total   = len(test_data)
            score_pct = round(n_correct / n_total * 100, 1)
            save_pilot_score(
                student_id=ls["student_id"], domain=domain, test_type="pretest",
                score_pct=score_pct, n_correct=n_correct, n_total=n_total,
                group_label=st.session_state.get("group", "experimental"),
                ablation_mode=st.session_state.get("ablation", "full"),
                db_path=DB_PATH,
            )
            st.session_state.pretest_score = score_pct
            st.session_state.test_answers  = {}
            ls = dict(ls)
            ls["pretest_score_pct"] = score_pct
            st.session_state.ls = ls
            if st.session_state.get("group") == "control":
                ls = dict(ls)
                ls["traditional_index"] = 0   # start the fixed learning path
                st.session_state.practice_answers = {}
                _save(ls, "traditional")
                st.session_state.phase = "traditional"
            else:
                ls = planner_agent(ls)
                _save(ls, "gen_lesson")
                st.session_state.ls = ls
                st.session_state.phase = "gen_lesson"
            st.rerun()


def phase_traditional():
    """
    Control arm — fixed per-topic study materials (spec §3).
    The student studies ONE topic at a time in the fixed order
    (Variables → Control Flow → Loops → Functions → OOP Basics). The pre-test
    score NEVER changes this order — no skipping, no personalisation.
    """
    ls     = st.session_state.ls
    domain = ls["domain"]
    _sidebar()

    # ── Guard: if post-test already done, skip straight to done ───────────────
    stored = get_pilot_scores(DB_PATH)
    if any(s["student_id"] == ls["student_id"] and s["domain"] == domain
           and s["test_type"] == "posttest" for s in stored):
        st.session_state.phase = "done"
        st.rerun()

    order = topic_order(domain)
    done  = len(order)
    idx   = int(ls.get("traditional_index", 0))

    if not order:
        st.warning("Topic order not configured for this domain. Moving to post-test.")
        _save(ls, "posttest")
        st.session_state.phase = "posttest"
        st.rerun()
        return

    if idx >= done:
        # All topics studied + practised — move to the post-test.
        st.success("✅ You have finished studying and practising all topics.")
        if st.button("Continue to the Post-Test ▶", type="primary"):
            _save(ls, "posttest")
            st.session_state.phase = "posttest"
            st.rerun()
        return

    topic = order[idx]
    st.title(f"📚 Topic {idx + 1} of {done}: {_label(topic)}")
    st.progress(idx / done if done else 0,
                text=f"Topics studied: {idx}/{done}")
    st.caption("Study the material below carefully. After studying you will "
               "answer 5 practice questions on this topic.")

    notes = topic_notes(domain, topic)
    if notes:
        st.markdown("---")
        st.markdown(notes)
    else:
        st.warning("Study material not found for this topic.")

    st.divider()
    if st.button("I have studied this topic — continue to practice ▶", type="primary"):
        _save(ls, "traditional_practice")
        st.session_state.phase = "traditional_practice"
        st.rerun()

def phase_traditional_practice():
    """
    Control arm — fixed practice assessment (spec §4).
    5 fixed MCQs per topic from data/practice_python.json. Answers and scores
    are recorded for the research evaluation but NEVER change the next topic:
    the student always proceeds through the fixed topic order regardless of
    performance.
    """
    ls     = st.session_state.ls
    domain = ls["domain"]
    _sidebar()

    # ── Guard: if post-test already done, skip straight to done ───────────────
    stored = get_pilot_scores(DB_PATH)
    if any(s["student_id"] == ls["student_id"] and s["domain"] == domain
           and s["test_type"] == "posttest" for s in stored):
        st.session_state.phase = "done"
        st.rerun()

    order = topic_order(domain)
    done  = len(order)
    idx   = int(ls.get("traditional_index", 0))

    if not order:
        st.warning("Topic order not configured. Moving to post-test.")
        _save(ls, "posttest")
        st.session_state.phase = "posttest"
        st.rerun()
        return

    if idx >= done:
        _save(ls, "posttest")
        st.session_state.phase = "posttest"
        st.rerun()

    topic = order[idx]

    # Load the fixed practice bank for this topic
    test_data = _load_test(PRACTICE_FILES[domain])
    section   = next((s for s in test_data if s["topic"] == topic), None)
    if section is None:
        # No practice bank for this topic — skip to next topic gracefully
        st.info(f"No practice questions available for *{_label(topic)}* — moving to the next topic.")
        ls = dict(ls)
        idx += 1
        ls["traditional_index"] = idx
        next_phase = "posttest" if idx >= done else "traditional"
        _save(ls, next_phase)
        st.session_state.ls = ls
        st.session_state.phase = next_phase
        st.rerun()
        return
    qs = section["questions"]

    st.title(f"📝 Practice — {_label(topic)} ({idx + 1}/{done})")
    st.info(
        f"Answer the **{len(qs)} practice questions** for *{_label(topic)}*.  \n"
        "Your answers are recorded for the study but **do not change the next "
        "topic** — the path is the same for every student in this group."
    )

    # ── Review screen: score + correct answers + Continue ────────────────────────
    # Shown after submitting. Feedback parity with the AI group — but the score
    # never alters the fixed learning path.
    review = st.session_state.get("practice_review")
    if review and review.get("topic") != topic:
        review = None                      # stale review from another topic/session
        st.session_state.practice_review = None
    if review:
        st.success(f"Practice complete: **{review['correct']}/{review['total']}** "
                   f"correct for *{_label(review['topic'])}* ({review['pct']}%).")
        with st.expander("Review your answers", expanded=True):
            for qi, q in enumerate(review["qs"]):
                chosen   = review["answers"].get(q["id"], -1)
                is_right = chosen == q["correct_index"]
                icon     = "✅" if is_right else "❌"
                st.markdown(f"{icon} **Q{qi + 1}.** {q['prompt']}")
                if not is_right:
                    ci   = q["correct_index"]
                    opts = q["options"]
                    st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;Correct answer: "
                                f"**{chr(65 + ci)}. {opts[ci]}**")
        if st.button("Continue ▶", type="primary", use_container_width=True):
            st.session_state.practice_review = None
            st.session_state.phase = review["next_phase"]
            st.rerun()
        return

    answers = st.session_state.get("practice_answers", {})
    with st.form("traditional_practice_form"):
        for qi, q in enumerate(qs):
            st.markdown(f"**Q{qi + 1}. {q['prompt']}**")
            answers[q["id"]] = st.radio(
                f"Answer for {q['id']}",
                options=list(range(len(q["options"]))),
                format_func=lambda i, opts=q["options"]: f"{chr(65 + i)}. {opts[i]}",
                key=f"tp_{q['id']}",
                index=answers.get(q["id"], 0),
                label_visibility="collapsed",
            )
            if qi < len(qs) - 1:
                st.markdown("---")
        st.divider()
        submitted = st.form_submit_button("Submit Practice Answers ✓", type="primary",
                                          use_container_width=True)

    if submitted:
        n_correct, n_total, topic_score = _score_sectioned([section], answers)
        pct = round(n_correct / n_total * 100, 1) if n_total else 0.0

        # Log every answer to assessment_log (research data: practice_questions)
        for q in qs:
            chosen   = answers.get(q["id"], -1)
            is_right = chosen == q["correct_index"]
            log_assessment(
                student_id=ls["student_id"], domain=domain, topic=topic,
                item_type="practice_bank_mcq", item_text=q["prompt"],
                response=str(chosen), grade=1.0 if is_right else 0.0,
                justification="",
                raw_llm_response=None, graded_by="fixed_bank",
                flagged=False, latency_ms=None, db_path=DB_PATH,
            )

        # Persist per-topic practice scores + attempted topics (spec §17)
        ls = dict(ls)
        practice_scores = dict(ls.get("practice_scores", {}))
        practice_scores[topic] = {"correct": n_correct, "total": n_total, "pct": pct}
        ls["practice_scores"] = practice_scores
        attempted = list(ls.get("topics_attempted", []))
        if topic not in attempted:
            attempted.append(topic)
        ls["topics_attempted"] = attempted
        ls["session_history"] = list(ls.get("session_history", [])) + [{
            "event": "traditional_practice_completed",
            "topic": topic,
            "correct": n_correct,
            "total": n_total,
            "pct": pct,
        }]
        st.session_state.practice_answers = {}

        # The score NEVER changes the path (spec §4): always advance through
        # the fixed topic order, whatever the result.
        idx += 1
        ls["traditional_index"] = idx
        next_phase = "posttest" if idx >= done else "traditional"
        _save(ls, next_phase)            # persisted for session resume
        st.session_state.ls = ls

        # Stash the review (score + correct answers) and rerun so the review
        # screen replaces the form.
        st.session_state.practice_review = {
            "topic": topic, "correct": n_correct, "total": n_total,
            "pct": pct, "qs": qs, "answers": dict(answers),
            "next_phase": next_phase,
        }
        st.rerun()



# ── post-test ─────────────────────────────────────────────────────────────────
def phase_posttest():
    ls     = st.session_state.ls
    domain = ls["domain"]
    _sidebar()

    # ── Hard guard: if post-test score already in DB, never show it again ─────
    stored = get_pilot_scores(DB_PATH)
    if any(s["student_id"] == ls["student_id"] and s["domain"] == domain
           and s["test_type"] == "posttest" for s in stored):
        st.session_state.phase = "done"
        st.rerun()

    # ── Pre-test must exist before post-test ──────────────────────────────────
    if not any(s["student_id"] == ls["student_id"] and s["domain"] == domain
               and s["test_type"] == "pretest" for s in stored):
        st.warning("Pre-test not found. Please complete the pre-test first.")
        st.session_state.phase = "pretest"
        st.rerun()

    # ── Load test data ────────────────────────────────────────────────────────
    test_data = _load_test(POSTTEST_FILES[domain])
    sectioned = _is_sectioned(test_data)
    answers   = st.session_state.get("test_answers", {})

    st.title(f"📋 Post-Test — {DOMAINS[domain]}")

    if sectioned:
        # ── Per-topic section UI ──────────────────────────────────────────────
        total_qs    = sum(len(s["questions"]) for s in test_data)
        answered_qs = sum(1 for s in test_data
                          for q in s["questions"] if q["id"] in answers)

        st.info(
            f"This test has **{len(test_data)} sections** — one per topic — "
            f"with **5 questions each** ({total_qs} questions total).  \n"
            "Answer every question. **No feedback is given.** Submit when done."
        )
        st.progress(
            answered_qs / total_qs if total_qs else 0,
            text=f"Answered {answered_qs} / {total_qs} questions"
        )

        with st.form("posttest_form"):
            for sec_idx, section in enumerate(test_data):
                topic_label     = section.get("title", section["topic"].replace("_", " ").title())
                answered_in_sec = sum(1 for q in section["questions"] if q["id"] in answers)
                sec_done        = answered_in_sec == len(section["questions"])
                icon            = "✅" if sec_done else "📝"

                with st.expander(f"{icon} {topic_label}  ({answered_in_sec}/{len(section['questions'])} answered)", expanded=not sec_done):
                    for q_idx, q in enumerate(section["questions"]):
                        st.markdown(f"**Q{q_idx+1}. {q['prompt']}**")
                        answers[q["id"]] = st.radio(
                            f"Answer for {q['id']}",
                            options=list(range(len(q["options"]))),
                            format_func=lambda i, opts=q["options"]: f"{chr(65+i)}. {opts[i]}",
                            key=f"post_{q['id']}",
                            index=answers.get(q["id"], 0),
                            label_visibility="collapsed",
                        )
                        if q_idx < len(section["questions"]) - 1:
                            st.markdown("---")

            st.divider()
            st.caption(f"Make sure all {total_qs} questions are answered before submitting.")
            # Guard against double-submit: disable button if already submitted this render
            already_submitted = st.session_state.get("_posttest_submitted", False)
            submitted = st.form_submit_button(
                "Submit Post-Test ✓", type="primary",
                use_container_width=True,
                disabled=already_submitted,
            )
            if submitted:
                st.session_state["_posttest_submitted"] = True

        if submitted:
            n_correct, n_total, per_topic = _score_sectioned(test_data, answers)
            score_pct = round(n_correct / n_total * 100, 1) if n_total else 0.0

            # Show per-topic breakdown
            st.success(f"✅ Post-test submitted! Overall score: **{score_pct}%** ({n_correct}/{n_total})")
            with st.expander("Per-topic scores"):
                for topic, s in per_topic.items():
                    label = topic.replace("_", " ").title()
                    bar   = "█" * s["correct"] + "░" * (s["total"] - s["correct"])
                    st.markdown(f"**{label}**: {s['correct']}/{s['total']}  `{bar}`  {s['pct']}%")

            save_pilot_score(
                student_id=ls["student_id"], domain=domain, test_type="posttest",
                score_pct=score_pct, n_correct=n_correct, n_total=n_total,
                group_label=st.session_state.get("group", "experimental"),
                ablation_mode=st.session_state.get("ablation", "full"),
                db_path=DB_PATH,
            )
            st.session_state.posttest_score = score_pct
            st.session_state.test_answers   = {}
            # Persist topic-level post-test scores for the research data (spec §17)
            ls = dict(ls)
            ls["posttest_per_topic"]  = per_topic
            ls["posttest_score_pct"]  = score_pct
            st.session_state.ls = ls
            _save(ls, "done")
            st.session_state.phase = "done"
            st.rerun()

    else:
        # ── Old flat format (fallback) ────────────────────────────────────────
        st.info("Answer all questions. **No feedback is given.** Submit when done.")
        with st.form("posttest_form"):
            for q in test_data:
                st.markdown(f"**{q['id'].upper()}. {q['prompt']}**")
                answers[q["id"]] = st.radio(
                    f"Answer for question {q['id'].upper()}",
                    options=list(range(len(q["options"]))),
                    format_func=lambda i, opts=q["options"]: f"{i}. {opts[i]}",
                    key=f"post_{q['id']}",
                    index=answers.get(q["id"], 0),
                    label_visibility="collapsed",
                )
                st.markdown("---")
            submitted = st.form_submit_button("Submit Post-Test ✓", type="primary")

        if submitted:
            n_correct = sum(1 for q in test_data if answers.get(q["id"]) == q["correct_index"])
            n_total   = len(test_data)
            score_pct = round(n_correct / n_total * 100, 1)
            save_pilot_score(
                student_id=ls["student_id"], domain=domain, test_type="posttest",
                score_pct=score_pct, n_correct=n_correct, n_total=n_total,
                group_label=st.session_state.get("group", "experimental"),
                ablation_mode=st.session_state.get("ablation", "full"),
                db_path=DB_PATH,
            )
            st.session_state.posttest_score = score_pct
            ls = dict(ls)
            ls["posttest_score_pct"] = score_pct
            st.session_state.ls = ls
            _save(ls, "done")
            st.session_state.phase = "done"
            st.rerun()


# ── learning loop ─────────────────────────────────────────────────────────────
def phase_gen_lesson():
    ls = st.session_state.ls
    _sidebar()
    topic = ls.get("topic_pointer")

    # Guard: no topic means all topics are complete
    if not topic:
        if st.session_state.get("practice_mode"):
            st.session_state.practice_mode = False
            st.session_state.phase = "practice_menu"
        else:
            st.info("🎉 All topics complete! Moving to post-test.")
            _save(ls, "posttest")
            st.session_state.phase = "posttest"
        st.rerun()

    bloom_plan  = {b["topic"]: b for b in ls.get("bloom_plan", [])}
    b_info      = bloom_plan.get(topic, {})
    bloom_badge = f" — _{b_info.get('bloom_level','').title()} level_" if b_info else ""
    mastery_pct = ls.get("mastery", {}).get(topic, 0.0)

    st.title(f"📘 {_label(topic)} — {DOMAINS[ls['domain']]}{bloom_badge}")
    st.caption(f"Current mastery: {mastery_pct:.0%}  |  Gap: {1-mastery_pct:.0%}  |  "
               f"Question style: **{b_info.get('bloom_level','standard')}** level")

    # Remediation banner — the student is re-learning a topic that didn't stick
    rem_attempt = (ls.get("engagement", {})
                   .get("remediation_counts", {}).get(topic, 0))
    if rem_attempt:
        st.warning(
            f"🔄 **Remediation round {rem_attempt}/{MAX_REMEDIATION_ATTEMPTS}** — "
            f"let's look at *{_label(topic)}* from a different angle with new examples."
        )
    _show_history()

    from src.config import FallbackLLMError
    try:
        with st.spinner("Retrieving corpus and generating lesson…"):
            ls = content_agent(ls)
    except FallbackLLMError as e:
        st.error(f"⚠️ The AI is temporarily unavailable (API error). Please wait 30 seconds and refresh the page.\n\n*{e}*")
        _save(ls, "gen_lesson")
        st.stop()

    mat    = ls.get("current_material", {})
    text   = mat.get("explanation", "") or "*(Lesson content could not be generated — please refresh.)*"
    ex     = mat.get("example", "")
    lesson = text + (f"\n\n**Example:**\n```\n{ex}\n```" if ex else "")
    st.session_state.history.append(("assistant", lesson))

    try:
        with st.spinner("Preparing question…"):
            ls = assessment_prepare_agent(ls)
    except FallbackLLMError as e:
        st.error(f"⚠️ Could not generate a question (API error). Please refresh.\n\n*{e}*")
        _save(ls, "gen_lesson")
        st.stop()

    _save(ls, "answering")
    st.session_state.phase = "answering"
    st.rerun()


def phase_answering():
    ls = st.session_state.ls
    _sidebar()
    topic = ls.get("topic_pointer") or ""
    st.title(f"📘 {_label(topic)} — {DOMAINS[ls['domain']]}")
    _show_history()

    item    = ls.get("pending_item") or {}
    q_done  = st.session_state.q_done
    q_total = st.session_state.q_per_topic

    # Guard: item missing or empty — regenerate
    if not item or "format" not in item:
        st.warning("Question not available — regenerating…")
        _save(ls, "gen_lesson")
        st.session_state.phase = "gen_lesson"
        st.rerun()

    if item.get("format") == "mcq":
        options = item.get("options", [])
        prompt  = item.get("prompt") or item.get("question") or "Answer this question:"
        with st.chat_message("assistant"):
            st.markdown(f"**Question {q_done+1}/{q_total}:** {prompt}")
            chosen = st.radio(
                "Select your answer:",
                options=list(range(len(options))),
                format_func=lambda i: f"{i}. {options[i]}",
                key=f"radio_{ls.get('step_count',0)}_{q_done}",
            )
            submitted = st.button(
                "Submit Answer ✓", type="primary",
                key=f"sub_{ls.get('step_count',0)}_{q_done}"
            )
        if submitted:
            st.session_state.history.append(
                ("user", f"My answer: {chosen}. {options[chosen] if options else ''}"))
            st.session_state.pending_answer = str(chosen)
            st.session_state.phase = "gen_feedback"
            st.rerun()
    else:
        question = item.get("question") or item.get("prompt") or "Answer this question:"
        with st.chat_message("assistant"):
            st.markdown(f"**Question {q_done+1}/{q_total}:** {question}")
        answer = st.chat_input("Type your answer…",
                               key=f"sa_{ls.get('step_count',0)}_{q_done}")
        if answer:
            st.session_state.history.append(("user", answer))
            st.session_state.pending_answer = answer
            st.session_state.phase = "gen_feedback"
            st.rerun()


def phase_gen_feedback():
    ls     = st.session_state.ls
    answer = st.session_state.pop("pending_answer", "")
    topic  = ls.get("topic_pointer") or ""

    # Guard: if no pending item exists, skip grading
    if not ls.get("pending_item"):
        _save(ls, "gen_lesson")
        st.session_state.phase = "gen_lesson"
        st.rerun()

    from src.config import FallbackLLMError
    try:
        with st.spinner("Grading…"):
            ls = _grade(ls, answer)
            ls = monitor_agent(ls)
    except FallbackLLMError as e:
        st.error(f"⚠️ Grading failed (API error). Your answer was recorded but could not be scored.\n\n*{e}*")
        ls = dict(ls)
        ls["last_grade"] = {"correct": None, "score": 0.5, "flagged": True,
                            "justification": "Auto-grading failed — partial credit awarded."}
        # Persist the partial-credit grade to session state AND DB
        st.session_state.ls = ls
        _save(ls, "gen_lesson")

    grade   = ls.get("last_grade", {})
    correct = grade.get("correct", False)
    new_m   = ls["mastery"].get(topic, 0.0)
    item    = ls.get("pending_item", {})

    if correct is True:
        fb = f"✅ **Correct!** Mastery of *{_label(topic)}* → **{new_m:.0%}**"
    elif correct is False:
        fb = f"❌ **Incorrect.** Mastery of *{_label(topic)}* → **{new_m:.0%}**"
        if item.get("format") == "mcq":
            ci  = item.get("correct_index", 0)
            ops = item.get("options", [])
            if ops:
                fb += f"\n\nCorrect answer: **{ci}. {ops[ci]}**"
        just = grade.get("justification", "")
        if just:
            fb += f"\n\n*Feedback:* {just}"
    else:
        fb = f"📝 *Answer recorded.* Mastery of *{_label(topic)}* → **{new_m:.0%}**"
        just = grade.get("justification", "")
        if just:
            fb += f"\n\n*Feedback:* {just}"

    if ls.get("replan_flag") and st.session_state.q_done >= st.session_state.q_per_topic - 1:
        fb += "\n\n⚠️ *Low mastery detected — the system will adapt your learning path.*"

    if correct is True:
        st.session_state.q_correct = st.session_state.get("q_correct", 0) + 1

    st.session_state.history.append(("assistant", fb))
    _save(ls, "gen_feedback")

    st.session_state.q_done += 1
    more_qs = st.session_state.q_done < st.session_state.q_per_topic

    if more_qs:
        # Stay on the same topic. The Monitor's replan flag is evaluated at the
        # topic boundary only (spec §9: 5 practice questions per topic), so a
        # mid-topic struggle never cuts the assessment short.
        try:
            with st.spinner("Preparing next question…"):
                ls = assessment_prepare_agent(ls)
        except FallbackLLMError:
            pass   # fall through to next topic if question generation fails
        _save(ls, "answering")
        st.session_state.phase = "answering"
    else:
        # ── Topic boundary: adapt, remediate, or advance (spec §11–12) ────────
        finished_topic = ls.get("topic_pointer")
        decision, ls   = topic_end_decision(ls, bool(ls.get("replan_flag")))

        if decision == "remediate":
            # Same topic, all-new material and new questions. The Content agent
            # reads remediation_counts and produces a DIFFERENT lesson; the
            # Assessment agent generates a fresh non-repeated question.
            st.session_state.q_done    = 0
            st.session_state.q_correct = 0
            ls = dict(ls)
            ls["engagement"]["topics_done"] = st.session_state.topics_done
            _save(ls, "gen_lesson")
            st.session_state.ls = ls
            st.session_state.phase = "gen_lesson"
        else:
            # Record the AI arm's per-topic practice score (research data §17)
            q_correct = st.session_state.get("q_correct", 0)
            q_total   = st.session_state.q_per_topic
            ls = dict(ls)
            practice_scores = dict(ls.get("practice_scores", {}))
            practice_scores[finished_topic] = {
                "correct": q_correct, "total": q_total,
                "pct": round(q_correct / q_total * 100, 1) if q_total else 0.0,
            }
            ls["practice_scores"] = practice_scores
            attempted = list(ls.get("topics_attempted", []))
            if finished_topic and finished_topic not in attempted:
                attempted.append(finished_topic)
            ls["topics_attempted"] = attempted

            ls = _advance(ls)
            if decision == "advance_needs_review" and ls.get("topic_sequence"):
                try:
                    ls = planner_agent(ls)
                except FallbackLLMError:
                    ls = dict(ls)
                    ls["replan_flag"] = False   # clear flag so we don't loop

            st.session_state.topics_done += 1
            st.session_state.q_done      = 0
            st.session_state.q_correct   = 0

            # Persist topics_done into engagement so session resume knows progress
            ls["engagement"]["topics_done"] = st.session_state.topics_done
            _save(ls, "gen_lesson")
            st.session_state.ls = ls

            no_topics = not ls.get("topic_sequence")
            quota     = st.session_state.topics_done >= st.session_state.max_topics
            if no_topics or quota:
                _save(ls, "posttest")
                # If in practice mode, go back to practice menu instead of posttest
                if st.session_state.get("practice_mode"):
                    st.session_state.practice_mode = False
                    st.session_state.phase = "practice_menu"
                else:
                    st.session_state.phase = "posttest"
            else:
                st.session_state.phase = "gen_lesson"

    st.rerun()


# ── done ──────────────────────────────────────────────────────────────────────
def phase_done():
    ls = st.session_state.ls
    _sidebar()
    st.title("🎉 Session Complete!")
    st.markdown(f"**Student:** `{ls['student_id']}` | **Domain:** {DOMAINS[ls['domain']]}")

    # ── Issue 6: read scores from DB as fallback (works after browser refresh) ─
    pre  = st.session_state.get("pretest_score")
    post = st.session_state.get("posttest_score")
    if pre is None or post is None:
        stored = get_pilot_scores(DB_PATH)
        for s in stored:
            if s["student_id"] == ls["student_id"] and s["domain"] == ls["domain"]:
                if s["test_type"] == "pretest"  and pre  is None:
                    pre  = s["score_pct"]
                if s["test_type"] == "posttest" and post is None:
                    post = s["score_pct"]
    if pre is not None and post is not None:
        gain = (post - pre) / (100 - pre) if pre < 100 else 0.0
        col1, col2, col3 = st.columns(3)
        col1.metric("Pre-test", f"{pre:.1f}%")
        col2.metric("Post-test", f"{post:.1f}%", delta=f"{post-pre:+.1f}%")
        col3.metric("Hake's Gain ⟨g⟩", f"{gain:.2f}",
                    help="⟨g⟩ < 0.3 = low, 0.3–0.7 = medium, > 0.7 = high")

        st.subheader("Learning gain")
        score_df = pd.DataFrame({
            "test": ["Pre-test", "Post-test"],
            "score_pct": [float(pre), float(post)],
        })
        _chart_bars(
            score_df, x="test", y="score_pct",
            title="This student: pre-test vs post-test",
            y_title="Score (%)", y_domain=[0, 100], height=280,
        )
        gain_df = pd.DataFrame({"metric": ["Hake ⟨g⟩"], "value": [float(gain)]})
        _chart_bars(
            gain_df, x="metric", y="value",
            title="Normalised learning gain ⟨g⟩ (0 = none, 1 = max possible)",
            y_title="⟨g⟩", y_domain=[0, 1], height=220,
        )
        st.divider()

    st.subheader("Topic Mastery")
    mastery = ls.get("mastery", {})
    all_topics = _load_all_topics(ls["domain"])
    m_rows = []
    for t in all_topics:
        s = float(mastery.get(t, 0.0))
        m_rows.append({
            "Topic": _label(t),
            "mastery_pct": round(s * 100, 1),
            "Mastery": f"{s:.0%}",
            "Status": (
                "✅ Mastered" if s >= 0.7
                else "🟡 Progressing" if s >= 0.35
                else "🔴 Needs work"
            ),
        })
    if m_rows:
        m_df = pd.DataFrame(m_rows)
        _chart_bars(
            m_df, x="Topic", y="mastery_pct",
            title="Mastery by topic (0–100%)",
            y_title="Mastery (%)", y_domain=[0, 100], height=320,
        )
        st.dataframe(m_df[["Topic", "Mastery", "Status"]], use_container_width=True, hide_index=True)
    else:
        st.info("No mastery yet (typical for the control group, which only studies notes).")

    eng = ls.get("engagement", {})
    if eng.get("learning_time_sec"):
        minutes = eng["learning_time_sec"] / 60.0
        st.caption(f"⏱ Total time spent in this study: **{minutes:.0f} min**")

    st.divider()
    # ── Issue 9: clear completion message for students ────────────────────────
    st.success(
        "✅ **Your session is now complete.** Thank you for participating!\n\n"
        "Please fill in the short questionnaire below, then let the researcher know you are done."
    )

    # ── Offer practice mode if weak topics remain ─────────────────────────────
    mastery    = ls.get("mastery", {})
    all_topics = _load_all_topics(ls["domain"])
    weak       = [t for t in all_topics if mastery.get(t, 0.0) < MASTERY_THRESHOLD_PRACTICE]
    if weak:
        with st.expander(f"📚 You have {len(weak)} topic(s) below mastery — want to keep practising?"):
            for t in weak:
                m = mastery.get(t, 0.0)
                bar = "█" * int(m * 10) + "░" * (10 - int(m * 10))
                st.markdown(f"**{_label(t)}**: `{bar}` {m:.0%}")
            if st.button("🔄 Go to Practice Mode", type="primary"):
                st.session_state.phase = "practice_menu"
                st.rerun()
    # ── Gap 6: Post-study Likert survey (paper §VII-C) ────────────────────────
    if not st.session_state.get("survey_submitted"):
        st.subheader("📋 Post-Study Questionnaire")
        st.markdown(
            "Please rate your experience on a scale of **1 (strongly disagree) "
            "to 5 (strongly agree)**. This takes ~1 minute and helps us improve the system."
        )
        group = st.session_state.get("group", "experimental")
        with st.form("survey_form"):
            if group == "experimental":
                q1 = st.slider("The system was easy to use.",                           1, 5, 3)
                q2 = st.slider("The explanations and examples were helpful.",            1, 5, 3)
                q3 = st.slider("The system felt personalised to my level.",             1, 5, 3)
                q4 = st.slider("I would recommend this system to a classmate.",        1, 5, 3)
                q5 = st.slider("I preferred this system over studying notes alone.",   1, 5, 3)
            else:
                q1 = st.slider("The study materials were easy to follow.",              1, 5, 3)
                q2 = st.slider("The course notes were helpful.",                        1, 5, 3)
                q3 = st.slider("The materials were appropriate for my level.",         1, 5, 3)
                q4 = st.slider("I would recommend these materials to a classmate.",    1, 5, 3)
                q5 = st.slider("I preferred studying these notes over an AI system.",  1, 5, 3)
            comments = st.text_area("Any other comments? (optional)", height=80)
            submit_survey = st.form_submit_button("Submit Survey ✓", type="primary")

        if submit_survey:
            save_survey_response(
                student_id=ls["student_id"],
                domain=ls["domain"],
                group_label=group,
                q1_ease=q1, q2_helpful=q2, q3_adaptive=q3,
                q4_recommend=q4, q5_prefer=q5,
                comments=comments,
                db_path=DB_PATH,
            )
            st.session_state.survey_submitted = True
            st.success("✅ Survey submitted. Thank you for participating!")
            st.rerun()
    else:
        st.success("✅ Survey already submitted. Thank you!")

    st.divider()
    if st.button("New Session / New Domain", type="primary"):
        for k in ["ls", "phase", "history", "topics_done", "q_done",
                  "max_topics", "q_per_topic", "pending_answer",
                  "pretest_score", "posttest_score", "test_answers",
                  "survey_submitted"]:
            st.session_state.pop(k, None)
        st.rerun()


# ── researcher dashboard ──────────────────────────────────────────────────────
def phase_dashboard():
    _sidebar()
    st.title("📊 Researcher Dashboard")
    st.caption("Live metrics from the pilot database — refreshes on page reload.")

    # ── Supabase connection status ────────────────────────────────────────────
    from src.online_db import test_connection, is_configured
    if is_configured():
        ok, msg = test_connection()
        if ok:
            st.success(f"🟢 Supabase connected — reading live data from all students | {msg}")
        else:
            st.warning(f"🟡 Supabase offline — showing local data only | {msg}")
        if st.button("🔌 Run full Supabase connection test"):
            st.session_state.phase = "supabase_test"
            st.rerun()
    else:
        with st.expander("⚪ Supabase not configured — click to set up online database"):
            st.markdown("""
**Steps to connect:**

1. Go to your Supabase dashboard → **SQL Editor**
2. Paste the contents of `src/supabase_schema.sql` and click **Run**
3. Go to **Settings → API** → copy the **anon/public** key
4. Add these two lines to your `.env` file:
```
SUPABASE_URL=https://zmhadthvmuxkivxeehvz.supabase.co
SUPABASE_ANON_KEY=<paste your anon key here>
```
5. Restart the app

Until configured, all data is stored locally in `results/learner_state.db`.
""")
            if st.button("🔄 Test connection now (after adding keys)"):
                st.rerun()

    import pandas as pd

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Learning Gain", "📝 Assessment Log", "🤖 Agent Coordination",
        "⬇ Export CSVs", "📋 Survey Responses"
    ])

    # ── tab 1: learning gain ──────────────────────────────────────────────────
    with tab1:
        st.subheader("Pre/Post Test Scores & Hake's Gain")
        scores = get_pilot_scores(DB_PATH)
        if not scores:
            st.info("No test scores yet. Students must complete at least one pre-test.")
        else:
            df = pd.DataFrame(scores)

            # pivot: one row per student×domain, columns = pretest / posttest
            pivot = df.pivot_table(
                index=["student_id", "domain", "group"],
                columns="test_type",
                values="score_pct",
                aggfunc="first",
            ).reset_index()
            pivot.columns.name = None  # remove the "test_type" header label

            # ensure both columns exist (students may have only done pretest so far)
            for col in ["pretest", "posttest"]:
                if col not in pivot.columns:
                    pivot[col] = float("nan")

            # compute gain where both scores exist
            both = pivot["pretest"].notna() & pivot["posttest"].notna()
            pivot["hake_g"] = float("nan")
            pivot.loc[both, "hake_g"] = pivot.loc[both].apply(
                lambda r: round(
                    (r["posttest"] - r["pretest"]) / (100.0 - r["pretest"]), 3
                ) if r["pretest"] < 100 else 0.0,
                axis=1,
            )
            pivot["gain_pct"] = pivot["posttest"] - pivot["pretest"]

            # summary metrics
            exp  = pivot[pivot["group"] == "experimental"]["hake_g"].dropna()
            ctrl = pivot[pivot["group"] == "control"]["hake_g"].dropna()

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Students", len(pivot))
            c2.metric("Completed both tests", int(both.sum()))
            c3.metric("Mean ⟨g⟩ Experimental",
                      f"{exp.mean():.3f}" if len(exp) else "—",
                      help="Hake's normalised gain. >0.7=high, 0.3-0.7=medium, <0.3=low")
            c4.metric("Mean ⟨g⟩ Control",
                      f"{ctrl.mean():.3f}" if len(ctrl) else "—")

            st.subheader("Graphs")
            _chart_pre_post_grouped(pivot)
            _chart_group_means(pivot)

            hake_plot = pivot.loc[both, ["student_id", "group", "hake_g"]].dropna()
            if not hake_plot.empty:
                hake_plot = hake_plot.copy()
                hake_plot["student"] = (
                    hake_plot["student_id"].astype(str) + " (" + hake_plot["group"].astype(str) + ")"
                )
                _chart_bars(
                    hake_plot, x="student", y="hake_g", color="group",
                    title="Hake's ⟨g⟩ per student",
                    y_title="⟨g⟩", height=320,
                )
            else:
                st.caption("Hake ⟨g⟩ bars appear after at least one student finishes **both** pre-test and post-test.")

            st.subheader("Score table")
            display_cols = ["student_id", "domain", "group", "pretest", "posttest",
                            "gain_pct", "hake_g"]
            st.dataframe(
                pivot[display_cols].sort_values(["group", "student_id"]),
                use_container_width=True,
            )

    # ── tab 2: assessment log ─────────────────────────────────────────────────
    with tab2:
        st.subheader("Assessment Log (for ASAG / QWK)")
        alog = get_assessment_log(DB_PATH)
        if not alog:
            st.info("No assessments logged yet. Start a learning session first.")
        else:
            adf = pd.DataFrame(alog)

            c1, c2, c3 = st.columns(3)
            c1.metric("Total graded items",   len(adf))
            c2.metric("Unique students",       adf["student_id"].nunique())
            c3.metric("Flagged for review",    int(adf["flagged"].sum()))

            flagged_df = adf[adf["flagged"] == True]
            if len(flagged_df):
                st.warning(
                    f"⚠️ {len(flagged_df)} items flagged — "
                    "LLM grade and embedding similarity disagreed. "
                    "Check the instructor_grades_template.csv in Export tab."
                )

            # filter controls
            col_f1, col_f2 = st.columns(2)
            with col_f1:
                domain_filter = st.selectbox(
                    "Filter by domain",
                    ["All"] + sorted(adf["domain"].unique().tolist()),
                )
            with col_f2:
                type_filter = st.selectbox(
                    "Filter by item type",
                    ["All", "mcq", "short_answer"],
                )

            view = adf.copy()
            if domain_filter != "All":
                view = view[view["domain"] == domain_filter]
            if type_filter != "All":
                view = view[view["item_type"] == type_filter]

            show_cols = ["student_id", "domain", "topic", "item_type",
                         "grade", "graded_by", "flagged", "justification", "timestamp"]
            st.dataframe(view[show_cols], use_container_width=True)

            grade_view = view.dropna(subset=["grade"]).copy()
            if not grade_view.empty:
                st.subheader("Graphs")
                by_type = (
                    grade_view.groupby("item_type", as_index=False)["grade"]
                    .mean()
                    .rename(columns={"grade": "mean_grade"})
                )
                _chart_bars(
                    by_type, x="item_type", y="mean_grade",
                    title="Mean grade by item type (0–1)",
                    y_title="Mean grade", y_domain=[0, 1], height=260,
                )
                by_topic = (
                    grade_view.groupby("topic", as_index=False)["grade"]
                    .mean()
                    .rename(columns={"grade": "mean_grade"})
                )
                by_topic["topic_label"] = by_topic["topic"].map(_label)
                _chart_bars(
                    by_topic, x="topic_label", y="mean_grade",
                    title="Mean grade by topic",
                    y_title="Mean grade", y_domain=[0, 1], height=320,
                )

    # ── tab 3: agent coordination ─────────────────────────────────────────────
    with tab3:
        st.subheader("Agent Coordination Efficiency")
        log_path = os.getenv("LOG_PATH", "logs/agent_calls.jsonl")

        if not os.path.exists(log_path):
            st.info("No agent calls logged yet. Run at least one experimental session.")
        else:
            rows = []
            with open(log_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            rows.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue

            if not rows:
                st.info("Log file exists but is empty.")
            else:
                df = pd.DataFrame(rows)

                # compute metrics using eval script
                try:
                    from eval.coordination_efficiency import compute
                    metrics = compute(df)

                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Total Agent Calls",  metrics.get("n_agent_calls", "—"))
                    c2.metric("Sessions",           metrics.get("n_sessions",    "—"))
                    c3.metric("Mean Latency (s)",   metrics.get("mean_agent_call_latency_s", "—"))
                    c4.metric("Redundant Call Rate",
                              f"{metrics.get('redundant_call_rate', 0):.1%}")

                    st.metric("Task Completion Rate (full Planner→Content→Assessment→Monitor cycle)",
                              f"{metrics.get('task_completion_rate', 0):.1%}")
                except Exception as e:
                    st.warning(f"Could not compute coordination metrics: {e}")

                st.subheader("Graphs")
                if "latency_seconds" in df.columns and "agent" in df.columns:
                    lat = (
                        df.groupby("agent", as_index=False)["latency_seconds"]
                        .mean()
                        .rename(columns={"latency_seconds": "mean_latency_s"})
                    )
                    _chart_bars(
                        lat, x="agent", y="mean_latency_s",
                        title="Mean agent call latency (seconds)",
                        y_title="Seconds", height=280,
                    )
                    counts = df["agent"].value_counts().reset_index()
                    counts.columns = ["agent", "n_calls"]
                    _chart_bars(
                        counts, x="agent", y="n_calls",
                        title="Number of calls per agent",
                        y_title="Calls", height=280,
                    )

                # recent calls table
                st.subheader("Recent Agent Calls (last 50)")
                disp_cols = [c for c in
                             ["agent", "student_id", "session_id", "domain",
                              "latency_seconds", "redundant", "start_ts"]
                             if c in df.columns]
                st.dataframe(df[disp_cols].tail(50), use_container_width=True)

    # ── tab 4: export ─────────────────────────────────────────────────────────
    with tab4:
        st.subheader("Export Data for Eval Scripts")
        st.markdown("Download CSVs to run `python -m eval.run_all_metrics`.")

        col1, col2 = st.columns(2)

        # pretest / posttest scores
        scores = get_pilot_scores(DB_PATH)
        with col1:
            st.markdown("**Pre/Post Test Scores**")
            if scores:
                df_s = pd.DataFrame(scores)
                for test_type in ["pretest", "posttest"]:
                    sub = df_s[df_s["test_type"] == test_type][
                        ["student_id", "group", "score_pct"]
                    ]
                    if not sub.empty:
                        st.download_button(
                            f"⬇ {test_type}_scores.csv  ({len(sub)} rows)",
                            data=sub.to_csv(index=False),
                            file_name=f"{test_type}_scores.csv",
                            mime="text/csv",
                        )
            else:
                st.info("No scores yet.")

        # instructor grading template
        with col2:
            st.markdown("**Instructor Grading Template (for QWK)**")
            alog = get_assessment_log(DB_PATH)
            if alog:
                adf = pd.DataFrame(alog)
                short = adf[adf["item_type"] == "short_answer"][
                    ["student_id", "topic", "grade"]
                ].rename(columns={"topic": "question_id", "grade": "system_score"})
                if not short.empty:
                    short = short.copy()
                    short["system_score"] = (
                        short["system_score"].astype(float) * 3
                    ).round().astype("Int64")
                    short["instructor_score"] = ""
                    st.download_button(
                        f"⬇ instructor_grades_template.csv  ({len(short)} rows)",
                        data=short.to_csv(index=False),
                        file_name="instructor_grades_template.csv",
                        mime="text/csv",
                        help="Fill in instructor_score column (0-3), then run eval/grading_reliability.py",
                    )
                else:
                    st.info("No short-answer items graded yet.")
            else:
                st.info("No assessment log yet.")

        st.divider()
        st.markdown("**Survey responses (see Survey tab for full view):**")
        sdata = get_survey_responses(DB_PATH)
        if sdata:
            st.download_button(
                f"⬇ survey_responses.csv  ({len(sdata)} rows)",
                data=pd.DataFrame(sdata).to_csv(index=False),
                file_name="survey_responses.csv",
                mime="text/csv",
            )
        st.markdown("**Research summary (spec §17 — one row per student):**")
        try:
            research_rows = []
            loaded_keys = set()
            for s in (scores or []):
                key = (s["student_id"], s["domain"])
                if key in loaded_keys:
                    continue
                loaded_keys.add(key)
                stt = load_state(DB_PATH, s["student_id"], s["domain"])
                eng = stt.get("engagement", {})
                research_rows.append({
                    "student_id":   s["student_id"],
                    "domain":       s["domain"],
                    "group":        stt.get("group", s.get("group", "")),
                    "pre_test_score":        stt.get("pretest_score_pct"),
                    "pre_test_topic_scores": json.dumps(stt.get("pretest_per_topic") or {}),
                    "post_test_score":       stt.get("posttest_score_pct"),
                    "post_test_topic_scores": json.dumps(stt.get("posttest_per_topic") or {}),
                    "topics_attempted":      json.dumps(stt.get("topics_attempted") or []),
                    "needs_review":          json.dumps(stt.get("needs_review") or {}),
                    "practice_scores":       json.dumps(stt.get("practice_scores") or {}),
                    "number_of_replans":     eng.get("total_replans", 0),
                    "number_of_remediations": eng.get("total_remediations", 0),
                    "max_remediation_attempts": MAX_REMEDIATION_ATTEMPTS,
                    "total_learning_time_sec": round(eng.get("learning_time_sec", 0.0), 1),
                    "n_sessions": eng.get("session_count", 0),
                    "final_mastery":         json.dumps(stt.get("mastery") or {}),
                })
            if research_rows:
                st.download_button(
                    f"⬇ research_summary.csv  ({len(research_rows)} rows)",
                    data=pd.DataFrame(research_rows).to_csv(index=False),
                    file_name="research_summary.csv", mime="text/csv",
                    help="All spec §17 fields flattened per student for the research evaluation.",
                )
            else:
                st.info("No learner state yet.")
        except Exception as e:
            st.warning(f"Could not build research summary: {e}")
        st.divider()
        st.markdown("**Run all metrics from terminal:**")
        st.code("python -m eval.run_all_metrics", language="bash")
        st.markdown("Or individually:")
        st.code(
            "python -m eval.coordination_efficiency --log logs/agent_calls.jsonl\n"
            "python -m eval.learning_gain --pretest results/pretest_scores.csv "
            "--posttest results/posttest_scores.csv\n"
            "python -m eval.grading_reliability --file results/instructor_grades.csv",
            language="bash",
        )

    # ── tab 5: survey responses ───────────────────────────────────────────────
    with tab5:
        st.subheader("Post-Study Survey Responses")
        sdata = get_survey_responses(DB_PATH)
        if not sdata:
            st.info("No survey responses yet. Students submit the survey at the end of phase_done.")
        else:
            import pandas as pd
            sdf = pd.DataFrame(sdata)
            c1, c2, c3 = st.columns(3)
            c1.metric("Responses", len(sdf))
            c2.metric("Experimental", int((sdf["group"] == "experimental").sum()))
            c3.metric("Control",      int((sdf["group"] == "control").sum()))

            # Mean Likert scores per question
            q_cols = {
                "q1_ease":      "Q1: Ease of use",
                "q2_helpful":   "Q2: Helpful",
                "q3_adaptive":  "Q3: Felt personalised",
                "q4_recommend": "Q4: Would recommend",
                "q5_prefer":    "Q5: Preferred over traditional",
            }
            means = {label: round(sdf[col].mean(), 2)
                     for col, label in q_cols.items() if col in sdf.columns}
            if means:
                st.subheader("Mean Likert scores (1=strongly disagree, 5=strongly agree)")
                mean_df = pd.DataFrame(
                    list(means.items()), columns=["Question", "Mean (1–5)"]
                )
                st.dataframe(mean_df, use_container_width=True, hide_index=True)
                _chart_bars(mean_df, x="Question", y="Mean (1–5)",
                            title="Survey mean scores", y_domain=[1, 5], height=300)

            st.subheader("All responses")
            show_s = ["student_id", "group", "q1_ease", "q2_helpful",
                      "q3_adaptive", "q4_recommend", "q5_prefer", "comments", "timestamp"]
            st.dataframe(sdf[[c for c in show_s if c in sdf.columns]],
                         use_container_width=True)

            # Export
            st.download_button(
                f"⬇ survey_responses.csv  ({len(sdf)} rows)",
                data=sdf.to_csv(index=False),
                file_name="survey_responses.csv",
                mime="text/csv",
            )


def phase_practice_menu():
    """
    Shown after a student completes the full study (pre-test + learning + post-test).
    Lets them keep practising their weak topics without retaking tests.
    This is separate from the research study — it's bonus learning.
    """
    ls = st.session_state.ls
    _sidebar()

    st.title("📚 Practice Mode")
    st.success(
        "✅ **Your study session is complete** — pre-test, learning, and post-test are all done.  \n"
        "Your results are saved. Below are the topics you can still improve."
    )

    # Find weak topics (below mastery threshold)
    mastery      = ls.get("mastery", {})
    all_topics   = _load_all_topics(ls["domain"])
    weak         = [(t, mastery.get(t, 0.0)) for t in all_topics
                    if mastery.get(t, 0.0) < MASTERY_THRESHOLD_PRACTICE]
    strong       = [(t, mastery.get(t, 0.0)) for t in all_topics
                    if mastery.get(t, 0.0) >= MASTERY_THRESHOLD_PRACTICE]

    if strong:
        st.markdown("**Already mastered ✅**")
        cols = st.columns(len(strong))
        for i, (t, m) in enumerate(strong):
            cols[i].metric(_label(t), f"{m:.0%}")

    if not weak:
        st.balloons()
        st.success("🎉 You have mastered all topics! Nothing left to practise.")
        if st.button("View Results"):
            st.session_state.phase = "done"
            st.rerun()
        return

    st.divider()
    st.markdown("**Topics that need more practice** — click one to start:")

    for topic, m in sorted(weak, key=lambda x: x[1]):  # weakest first
        bar   = "█" * int(m * 10) + "░" * (10 - int(m * 10))
        label = _label(topic)
        bloom = _bloom_level_label(m)
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"**{label}**  `{bar}` {m:.0%}  —  _{bloom} level_")
        with col2:
            if st.button(f"Practise ▶", key=f"prac_{topic}"):
                # Set this topic as the only one in the practice sequence
                ls = dict(ls)
                ls["topic_sequence"] = [topic]
                ls["topic_pointer"]  = topic
                ls["bloom_plan"] = [{
                    "topic":       topic,
                    "bloom_level": bloom,
                    "mastery_gap": round(1.0 - m, 3),
                }]
                ls["replan_flag"] = False
                st.session_state.ls          = ls
                st.session_state.topics_done = 0
                st.session_state.q_done      = 0
                st.session_state.history     = []
                # Track which topics have been practised for the re-test
                practised = st.session_state.get("practised_topics", [])
                if topic not in practised:
                    practised.append(topic)
                st.session_state.practised_topics = practised
                # After practising this topic, come back to practice menu
                st.session_state.practice_mode = True
                _save(ls, "gen_lesson")
                st.session_state.phase = "gen_lesson"
                st.rerun()

    st.divider()
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("📊 View My Results"):
            st.session_state.phase = "done"
            st.rerun()
    with col2:
        # After practising, offer a mini re-test on the practised topics
        practised = st.session_state.get("practised_topics", [])
        if practised:
            if st.button(f"📝 Re-test on {len(practised)} practised topic(s)", type="primary"):
                st.session_state.phase = "practice_test"
                st.rerun()
    with col3:
        if st.button("✅ I'm done practising"):
            st.session_state.phase = "done"
            st.rerun()


def phase_practice_test():
    """
    Mini re-test after practice mode.
    Shows only the sections (topics) the student actually practised.
    Uses questions from the posttest (Form B) so they are different from the pretest.
    Score saved as test_type='practice_test' — separate from the main post-test.
    This lets the paper report: pre → post → practice → practice_post improvement.
    """
    ls     = st.session_state.ls
    domain = ls["domain"]
    _sidebar()

    practised_topics = st.session_state.get("practised_topics", [])
    if not practised_topics:
        st.info("No topics practised yet. Go back and practise some topics first.")
        if st.button("← Back to Practice"):
            st.session_state.phase = "practice_menu"
            st.rerun()
        return

    # Load posttest and filter to only practised topics
    post_data  = _load_test(POSTTEST_FILES[domain])
    sectioned  = _is_sectioned(post_data)
    if sectioned:
        sections = [s for s in post_data if s["topic"] in practised_topics]
    else:
        sections = []

    if not sections:
        st.warning("Could not find test questions for practised topics.")
        if st.button("← Back to Practice"):
            st.session_state.phase = "practice_menu"
            st.rerun()
        return

    st.title("📋 Practice Re-Test")
    st.info(
        f"This mini re-test covers the **{len(sections)} topic(s) you just practised**.  \n"
        "It uses the same post-test questions so your improvement is directly comparable.  \n"
        "**No feedback** is given. Submit when done."
    )

    answers   = st.session_state.get("practice_test_answers", {})
    total_qs  = sum(len(s["questions"]) for s in sections)
    answered  = sum(1 for s in sections for q in s["questions"] if q["id"] in answers)

    st.progress(
        answered / total_qs if total_qs else 0,
        text=f"Answered {answered} / {total_qs} questions"
    )

    with st.form("practice_test_form"):
        for section in sections:
            topic_label = section.get("title", _label(section["topic"]))
            m_before = ls.get("mastery", {}).get(section["topic"], 0.0)

            # Show mastery before practice for comparison
            st.subheader(f"📘 {topic_label}")
            st.caption(f"Mastery before this practice: {m_before:.0%}")

            for q_idx, q in enumerate(section["questions"]):
                st.markdown(f"**Q{q_idx+1}. {q['prompt']}**")
                answers[q["id"]] = st.radio(
                    f"Answer for {q['id']}",
                    options=list(range(len(q["options"]))),
                    format_func=lambda i, opts=q["options"]: f"{chr(65+i)}. {opts[i]}",
                    key=f"ptest_{q['id']}",
                    index=answers.get(q["id"], 0),
                    label_visibility="collapsed",
                )
                if q_idx < len(section["questions"]) - 1:
                    st.markdown("---")
            st.divider()

        submitted = st.form_submit_button("Submit Re-Test ✓", type="primary",
                                          use_container_width=True)

    if submitted:
        n_correct, n_total, per_topic = _score_sectioned(sections, answers)
        score_pct = round(n_correct / n_total * 100, 1) if n_total else 0.0

        # Save as practice_test score (separate from main post-test)
        save_pilot_score(
            student_id=ls["student_id"],
            domain=domain,
            test_type="practice_test",
            score_pct=score_pct,
            n_correct=n_correct,
            n_total=n_total,
            group_label=st.session_state.get("group", "experimental"),
            ablation_mode=st.session_state.get("ablation", "full"),
            db_path=DB_PATH,
        )

        # Compare with post-test score on same topics
        stored = get_pilot_scores(DB_PATH)
        post_score = next((s["score_pct"] for s in stored
                          if s["student_id"] == ls["student_id"]
                          and s["domain"] == domain
                          and s["test_type"] == "posttest"), None)

        st.success(f"✅ Practice re-test complete!  Overall: **{score_pct}%** ({n_correct}/{n_total})")

        # Show per-topic improvement
        st.subheader("📈 Your improvement")
        for topic, s in per_topic.items():
            # Get original mastery from pretest bootstrap
            pre_topic_score = ls.get("pretest_per_topic", {}).get(topic, {}).get("pct", 0)
            new_pct = s["pct"]
            gain    = new_pct - pre_topic_score
            arrow   = "⬆" if gain > 0 else ("⬇" if gain < 0 else "→")
            bar_new  = "█" * s["correct"] + "░" * (s["total"] - s["correct"])
            st.markdown(
                f"**{_label(topic)}**: "
                f"Pre-test {pre_topic_score:.0f}% → Re-test **{new_pct:.0f}%**  "
                f"`{bar_new}`  {arrow} {abs(gain):.0f}pp"
            )

        st.session_state.practice_test_answers = {}
        st.session_state.practised_topics = []

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Practise more weak topics"):
                st.session_state.phase = "practice_menu"
                st.rerun()
        with col2:
            if st.button("📊 View Full Results"):
                st.session_state.phase = "done"
                st.rerun()


def _bloom_level_label(mastery_score: float) -> str:
    """Human-readable Bloom level label."""
    if mastery_score < 0.25:   return "remember"
    elif mastery_score < 0.45: return "understand"
    elif mastery_score < 0.65: return "apply"
    elif mastery_score < 0.80: return "analyze"
    else:                       return "evaluate"


def phase_supabase_test():
    """Hidden diagnostic page — go to ?phase=supabase_test in browser."""
    st.title("🔌 Supabase Connection Test")

    from src.online_db import (
        is_configured, test_connection,
        save_pilot_score_online, get_pilot_scores_online,
        save_learner_state_online, load_learner_state_online,
    )

    # Config check
    st.subheader("1. Configuration")
    url  = os.getenv("SUPABASE_URL", "")
    key  = os.getenv("SUPABASE_ANON_KEY", "")
    st.write(f"SUPABASE_URL:      `{url or '❌ not set'}`")
    st.write(f"SUPABASE_ANON_KEY: `{'✅ set (' + key[:20] + '...)' if key else '❌ not set'}`")
    st.write(f"is_configured():   `{is_configured()}`")

    # Connection test
    st.subheader("2. Connection Test")
    with st.spinner("Connecting to Supabase..."):
        ok, msg = test_connection()
    if ok:
        st.success(f"✅ {msg}")
    else:
        st.error(f"❌ {msg}")
        st.markdown("""
**To fix:**
1. Go to Supabase → **SQL Editor** → paste `src/supabase_schema.sql` → **Run**
2. Go to **Settings → API** → copy the **anon/public** key
3. Add `SUPABASE_ANON_KEY=eyJ...` to your `.env` file
4. Restart the app
""")
        if st.button("Back to Dashboard"):
            st.session_state.phase = "dashboard"
            st.rerun()
        return

    # Write test
    st.subheader("3. Write Test")
    with st.spinner("Writing test data to Supabase..."):
        w1 = save_pilot_score_online("TEST-UI", "python_programming", "pretest",
                                      72.0, 7, 10, "experimental")
        from src.state import new_state
        s = new_state("TEST-UI", "python_programming")
        s["mastery"]["variables_datatypes"] = 0.75
        w2 = save_learner_state_online(s)

    st.write(f"pilot_scores write:    {'✅ OK' if w1 else '❌ FAIL'}")
    st.write(f"learner_state write:   {'✅ OK' if w2 else '❌ FAIL'}")

    # Read test
    st.subheader("4. Read Test")
    with st.spinner("Reading back from Supabase..."):
        scores   = get_pilot_scores_online()
        loaded_s = load_learner_state_online("TEST-UI", "python_programming")

    test_row    = next((r for r in scores if r["student_id"] == "TEST-UI"), None)
    mastery_ok  = loaded_s and loaded_s.get("mastery", {}).get("variables_datatypes") == 0.75

    st.write(f"pilot_scores read:     {'✅ found row' if test_row else '❌ row not found'}")
    st.write(f"learner_state read:    {'✅ mastery=0.75 round-trip OK' if mastery_ok else '❌ mastery not found'}")
    st.write(f"Total scores in DB:    {len(scores)} rows")

    if w1 and w2 and test_row and mastery_ok:
        st.balloons()
        st.success("🎉 Supabase is fully working! All student data will sync online automatically.")
    else:
        st.warning("Some tests failed. Check the errors above.")

    st.divider()
    if st.button("← Back to Dashboard"):
        st.session_state.phase = "dashboard"
        st.rerun()


# ════════════════════════════════════════════════════════════════
#  ROUTER
# ════════════════════════════════════════════════════════════════
PHASES = {
    "login":           phase_login,
    "pretest":         phase_pretest,
    "traditional":     phase_traditional,
    "traditional_practice": phase_traditional_practice,
    "gen_lesson":      phase_gen_lesson,
    "answering":       phase_answering,
    "gen_feedback":    phase_gen_feedback,
    "posttest":        phase_posttest,
    "done":            phase_done,
    "practice_menu":   phase_practice_menu,
    "practice_test":   phase_practice_test,
    "dashboard":       phase_dashboard,
    "supabase_test":   phase_supabase_test,
}

if "phase" not in st.session_state:
    st.session_state.phase = "login"

PHASES[st.session_state.phase]()
