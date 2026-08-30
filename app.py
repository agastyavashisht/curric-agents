"""
CurricAgents — Pilot Study Web App
===================================
Multi-Agent LLM System for Autonomous Curriculum Planning and Adaptive Assessment

Flow per student per domain:
  1. login        — enter student ID, pick domain & group
  2. pretest      — 10-MCQ timed pre-test (no feedback), score saved to DB
  3. gen_lesson   — Content agent retrieves corpus + calls LLM → explanation
  4. answering    — student answers MCQ or short-answer question
  5. gen_feedback — Assessment agent grades, Monitor updates mastery, replan if needed
  6. posttest     — 10-MCQ post-test (no feedback), score saved
  7. done         — mastery summary + learning gain shown

Sidebar shows live mastery bars and agent activity log.

Run:
    streamlit run app.py
"""

import json
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
from src.agents.planner import planner_agent
from src.agents.content import content_agent
from src.agents.monitor import monitor_agent
from src.agents.assessment import assessment_prepare_agent, apply_grade
from src.graph import advance_topic as _advance
from src.db import log_assessment, save_pilot_score, get_pilot_scores, get_assessment_log

# ── page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CurricAgents — Pilot Study",
    page_icon="🎓",
    layout="wide",
)

# ── domain registry ──────────────────────────────────────────────────────────
DOMAINS = {
    "python_programming":  "Python Programming",
    "ml_basics":           "ML Basics",
    "signal_processing":   "Signal Processing",
    "physiology_basics":   "Physiology Basics",
    "data_analysis":       "Data Analysis",
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

# ── helpers ───────────────────────────────────────────────────────────────────
def _label(t: str) -> str:
    return t.replace("_", " ").title()


def _color(s: float) -> str:
    return "🟢" if s >= 0.7 else ("🟡" if s >= 0.35 else "🔴")


def _load_test(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)["questions"]


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

    if phase in ("pretest", "traditional", "gen_lesson", "answering", "gen_feedback", "posttest", "done"):
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
            for t in all_topics:
                s      = mastery.get(t, 0.0)
                bar    = "█" * int(s * 10) + "░" * (10 - int(s * 10))
                prefix = "▶ " if t == current else "  "
                b_info = bloom_plan.get(t)
                bloom  = f" _{b_info['bloom_level']}_" if b_info else ""
                # strike through mastered topics not in the current plan
                if t not in (ls.get("topic_sequence", []) + ([current] if current else [])) \
                        and s >= 0.7:
                    label = f"~~{_label(t)}~~ ✓"
                else:
                    label = f"**{_label(t)}**"
                st.sidebar.markdown(
                    f"{prefix}{_color(s)} {label}{bloom}  \n`{bar}` {s:.0%}"
                )

            st.sidebar.divider()
            done = st.session_state.get("topics_done", 0)
            total = st.session_state.get("max_topics", 3)
            q_done = st.session_state.get("q_done", 0)
            q_total = st.session_state.get("q_per_topic", 3)
            st.sidebar.progress(
                min(done / max(total, 1), 1.0),
                text=f"Topics {done}/{total}  •  Q {q_done}/{q_total}"
            )

    # Researcher dashboard link
    st.sidebar.divider()
    if st.sidebar.button("📊 Researcher Dashboard"):
        st.session_state.phase = "dashboard"
        st.rerun()
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
    st.title("🎓 CurricAgents — Pilot Study")
    st.markdown(
        "Multi-Agent LLM System for Autonomous Curriculum Planning and Adaptive Assessment  \n"
        "*Each session: Pre-test → Adaptive Learning → Post-test*"
    )
    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        sid    = st.text_input("Student ID (anonymised)", "S01", max_chars=20)
        domain = st.selectbox("Subject Domain", list(DOMAINS.keys()),
                              format_func=lambda k: DOMAINS[k])
        group  = st.radio("Study Group", ["experimental", "control"],
                          help="Experimental = uses the AI system. Control = traditional study.")
    with col2:
        st.markdown("**Session Settings**")
        topics = st.slider("Max topics this session", 1, 10, 4)
        qpt    = st.slider("Questions per topic", 1, 5, 3)
        st.info(
            "**Flow:**\n"
            "1. 10-question pre-test (no feedback)\n"
            "2. Experimental: adaptive multi-agent lessons. Control: traditional corpus study.\n"
            "3. 10-question post-test (no feedback)\n"
            "4. Mastery (experimental) + learning gain summary"
        )
        go = st.button("Start Session ▶", type="primary", use_container_width=True)

    if go and sid.strip():
        ls = load_state(DB_PATH, sid.strip(), domain)
        ls["engagement"]["session_count"] = ls["engagement"].get("session_count", 0) + 1
        if group == "experimental":
            ls = planner_agent(ls)

        st.session_state.ls           = ls
        st.session_state.domain       = domain
        st.session_state.group        = group
        st.session_state.max_topics   = topics
        st.session_state.q_per_topic  = qpt
        st.session_state.topics_done  = 0
        st.session_state.q_done       = 0
        st.session_state.history      = []
        st.session_state.test_answers = {}
        st.session_state.phase        = "pretest"
        st.rerun()


# ── pre-test ──────────────────────────────────────────────────────────────────
def phase_pretest():
    ls = st.session_state.ls
    domain = ls["domain"]
    _sidebar()
    st.title(f"📋 Pre-Test — {DOMAINS[domain]}")
    st.info("Answer all 10 questions. **No feedback is given during the test.** Submit when done.")

    questions = _load_test(PRETEST_FILES[domain])
    answers = st.session_state.get("test_answers", {})

    with st.form("pretest_form"):
        for q in questions:
            st.markdown(f"**{q['id'].upper()}. {q['prompt']}**")
            answers[q["id"]] = st.radio(
                f"",
                options=list(range(len(q["options"]))),
                format_func=lambda i, opts=q["options"]: f"{i}. {opts[i]}",
                key=f"pre_{q['id']}",
                index=answers.get(q["id"], 0),
                label_visibility="collapsed",
            )
            st.markdown("---")
        submitted = st.form_submit_button("Submit Pre-Test ✓", type="primary")

    if submitted:
        n_correct = sum(
            1 for q in questions
            if answers.get(q["id"]) == q["correct_index"]
        )
        score_pct = round(n_correct / len(questions) * 100, 1)
        save_pilot_score(
            student_id=ls["student_id"],
            domain=domain,
            test_type="pretest",
            score_pct=score_pct,
            n_correct=n_correct,
            n_total=len(questions),
            group_label=st.session_state.get("group", "experimental"),
            db_path=DB_PATH,
        )
        st.session_state.pretest_score = score_pct
        st.session_state.test_answers  = {}
        # Pass score into state so Planner can bootstrap mastery (personalised start)
        ls = dict(ls)
        ls["pretest_score_pct"] = score_pct
        st.session_state.ls = ls
        if st.session_state.get("group") == "control":
            st.session_state.phase = "traditional"
        else:
            ls = planner_agent(ls)   # re-run planner now that pretest_score_pct is set
            st.session_state.ls = ls
            st.session_state.phase = "gen_lesson"
        st.rerun()


def phase_traditional():
    """Control arm: static corpus study materials (no Planner/Content/Assessment/Monitor)."""
    ls = st.session_state.ls
    domain = ls["domain"]
    _sidebar()
    st.title(f"📚 Traditional study — {DOMAINS[domain]}")
    st.info(
        "You are in the **control** group. Study the course notes below "
        "(same corpus the AI retrieves from). When you are ready, continue to the post-test."
    )
    corpus_path = os.path.join("data", "corpus", f"{domain}.md")
    if os.path.exists(corpus_path):
        with open(corpus_path, encoding="utf-8") as f:
            notes = f.read()
        st.markdown(notes)
    else:
        st.warning("Course notes file is missing for this domain.")
    if st.button("I have finished studying — continue to post-test ▶", type="primary"):
        save_state(DB_PATH, ls)
        st.session_state.phase = "posttest"
        st.rerun()


# ── post-test ─────────────────────────────────────────────────────────────────
def phase_posttest():
    ls = st.session_state.ls
    domain = ls["domain"]
    _sidebar()
    st.title(f"📋 Post-Test — {DOMAINS[domain]}")
    st.info("Answer all 10 questions. **No feedback is given.** Submit when done.")

    questions = _load_test(POSTTEST_FILES[domain])
    answers = st.session_state.get("test_answers", {})

    with st.form("posttest_form"):
        for q in questions:
            st.markdown(f"**{q['id'].upper()}. {q['prompt']}**")
            answers[q["id"]] = st.radio(
                f"",
                options=list(range(len(q["options"]))),
                format_func=lambda i, opts=q["options"]: f"{i}. {opts[i]}",
                key=f"post_{q['id']}",
                index=answers.get(q["id"], 0),
                label_visibility="collapsed",
            )
            st.markdown("---")
        submitted = st.form_submit_button("Submit Post-Test ✓", type="primary")

    if submitted:
        n_correct = sum(
            1 for q in questions
            if answers.get(q["id"]) == q["correct_index"]
        )
        score_pct = round(n_correct / len(questions) * 100, 1)
        save_pilot_score(
            student_id=ls["student_id"],
            domain=domain,
            test_type="posttest",
            score_pct=score_pct,
            n_correct=n_correct,
            n_total=len(questions),
            group_label=st.session_state.get("group", "experimental"),
            db_path=DB_PATH,
        )
        st.session_state.posttest_score = score_pct
        st.session_state.phase = "done"
        st.rerun()


# ── learning loop ─────────────────────────────────────────────────────────────
def phase_gen_lesson():
    ls = st.session_state.ls
    _sidebar()
    topic = ls["topic_pointer"]
    # Show Bloom level badge
    bloom_plan  = {b["topic"]: b for b in ls.get("bloom_plan", [])}
    b_info      = bloom_plan.get(topic, {})
    bloom_badge = f" — _{b_info.get('bloom_level','').title()} level_" if b_info else ""
    mastery_pct = ls.get("mastery", {}).get(topic, 0.0)

    st.title(f"📘 {_label(topic)} — {DOMAINS[ls['domain']]}{bloom_badge}")
    st.caption(f"Current mastery: {mastery_pct:.0%}  |  Gap: {1-mastery_pct:.0%}  |  "
               f"Question style: **{b_info.get('bloom_level','standard')}** level")
    _show_history()

    with st.spinner("Retrieving corpus and generating lesson…"):
        ls = content_agent(ls)

    mat    = ls.get("current_material", {})
    text   = mat.get("explanation", "")
    ex     = mat.get("example", "")
    lesson = text + (f"\n\n**Example:**\n```\n{ex}\n```" if ex else "")
    st.session_state.history.append(("assistant", lesson))

    with st.spinner("Preparing question…"):
        ls = assessment_prepare_agent(ls)

    st.session_state.ls    = ls
    st.session_state.phase = "answering"
    st.rerun()


def phase_answering():
    ls = st.session_state.ls
    _sidebar()
    st.title(f"📘 {_label(ls['topic_pointer'])} — {DOMAINS[ls['domain']]}")
    _show_history()

    item    = ls.get("pending_item", {})
    q_done  = st.session_state.q_done
    q_total = st.session_state.q_per_topic

    if item.get("format") == "mcq":
        options = item.get("options", [])
        with st.chat_message("assistant"):
            st.markdown(f"**Question {q_done+1}/{q_total}:** {item['prompt']}")
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
                ("user", f"My answer: {chosen}. {options[chosen]}"))
            st.session_state.pending_answer = str(chosen)
            st.session_state.phase = "gen_feedback"
            st.rerun()
    else:
        with st.chat_message("assistant"):
            st.markdown(f"**Question {q_done+1}/{q_total}:** {item['question']}")
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
    topic  = ls["topic_pointer"]

    # Guard: if no pending item exists, skip grading and go straight to next lesson
    if not ls.get("pending_item"):
        st.session_state.phase = "gen_lesson"
        st.rerun()

    with st.spinner("Grading…"):
        ls = _grade(ls, answer)
        ls = monitor_agent(ls)

    grade   = ls.get("last_grade", {})
    correct = grade.get("correct", False)
    new_m   = ls["mastery"].get(topic, 0.0)
    item    = ls.get("pending_item", {})

    if correct:
        fb = f"✅ **Correct!** Mastery of *{_label(topic)}* → **{new_m:.0%}**"
    else:
        fb = f"❌ **Incorrect.** Mastery of *{_label(topic)}* → **{new_m:.0%}**"
        if item.get("format") == "mcq":
            ci  = item.get("correct_index", 0)
            ops = item.get("options", [])
            if ops:
                fb += f"\n\nCorrect answer: **{ci}. {ops[ci]}**"
        just = grade.get("justification", "")
        if just:
            fb += f"\n\n*Feedback:* {just}"
    if ls.get("replan_flag"):
        fb += "\n\n⚠️ *Low mastery detected — re-sequencing topics.*"

    st.session_state.history.append(("assistant", fb))
    save_state(DB_PATH, ls)

    st.session_state.q_done += 1
    more_qs = st.session_state.q_done < st.session_state.q_per_topic
    replan  = ls.get("replan_flag", False)

    if more_qs and not replan:
        with st.spinner("Preparing next question…"):
            ls = assessment_prepare_agent(ls)
        st.session_state.ls    = ls
        st.session_state.phase = "answering"
    else:
        ls = _advance(ls)
        if replan and ls.get("topic_sequence"):
            # Keep replan_flag=True so planner's session_history records it was a replan.
            # planner_node itself clears the flag at the end.
            ls = planner_agent(ls)
        st.session_state.ls          = ls
        st.session_state.topics_done += 1
        st.session_state.q_done      = 0

        no_topics = not ls.get("topic_sequence")
        quota     = st.session_state.topics_done >= st.session_state.max_topics
        if no_topics or quota:
            save_state(DB_PATH, ls)
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

    pre  = st.session_state.get("pretest_score")
    post = st.session_state.get("posttest_score")
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

    st.divider()
    if st.button("New Session / New Domain", type="primary"):
        for k in ["ls", "phase", "history", "topics_done", "q_done",
                  "max_topics", "q_per_topic", "pending_answer",
                  "pretest_score", "posttest_score", "test_answers"]:
            st.session_state.pop(k, None)
        st.rerun()


# ── researcher dashboard ──────────────────────────────────────────────────────
def phase_dashboard():
    _sidebar()
    st.title("📊 Researcher Dashboard")
    st.caption("Live metrics from the pilot database — refreshes on page reload.")

    import pandas as pd

    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Learning Gain", "📝 Assessment Log", "🤖 Agent Coordination", "⬇ Export CSVs"
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


# ════════════════════════════════════════════════════════════════
#  ROUTER
# ════════════════════════════════════════════════════════════════
PHASES = {
    "login":        phase_login,
    "pretest":      phase_pretest,
    "traditional":  phase_traditional,
    "gen_lesson":   phase_gen_lesson,
    "answering":    phase_answering,
    "gen_feedback": phase_gen_feedback,
    "posttest":     phase_posttest,
    "done":         phase_done,
    "dashboard":    phase_dashboard,
}

if "phase" not in st.session_state:
    st.session_state.phase = "login"

PHASES[st.session_state.phase]()
