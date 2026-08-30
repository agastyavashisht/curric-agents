"""
Assessment Agent.

Generates one item per call (MCQ / short-answer by step parity) and grades
the student's response:
  - MCQ: exact match against the correct option (no LLM needed to grade).
  - Short-answer: LLM-as-judge against a topic-specific rubric, cross-
    checked with a sentence-embedding similarity score against a
    reference answer. Disagreement beyond THRESHOLD gets flagged for
    instructor review rather than silently trusted.

`prepare_item` and `apply_grade` are the Streamlit-safe split (no input()).
`assessment_node` is the CLI LangGraph node that still uses input().
"""
import json

from src.state import LearnerState
from src.logging_utils import log_node_call
from src.config import get_llm, extract_content, _strip_fences

SIMILARITY_DISAGREEMENT_THRESHOLD = 0.35  # |llm_score - embedding_score| above this -> flag


def _choose_format(state: LearnerState) -> str:
    return "mcq" if state.get("step_count", 0) % 2 == 0 else "short_answer"


def _load_rubric(domain: str, topic: str):
    candidates = [
        f"data/rubrics/{domain}_rubrics.json",
        "data/rubrics/python_rubrics.json",
    ]
    for path in candidates:
        try:
            with open(path, "r", encoding="utf-8") as f:
                rubrics = json.load(f)["rubrics"]
            if topic in rubrics:
                return rubrics[topic]
        except FileNotFoundError:
            continue
    return None


_MCQ_PROMPT = """Write one multiple-choice question testing this learning objective
at the specified Bloom cognitive level.

Topic: {topic}
Learning objective: {objective}
Bloom level: {bloom_level}

Bloom level guidance for question style:
- remember: "What is...?" / "Which term describes...?"
- understand: "Why does...?" / "What does ... mean in context?"
- apply: "What will this code output?" / "Which approach correctly solves...?"
- analyze: "What is the key difference between X and Y?" / "Which code has a bug and why?"
- evaluate: "Which design is better and why?" / "What is the trade-off of using...?"

Write 4 options. Exactly one must be correct. The three incorrect options should each
target a plausible misconception a learner at the {bloom_level} level might have.

Respond ONLY as JSON:
{{"prompt": "...", "options": ["...", "...", "...", "..."],
  "correct_index": 0, "bloom_level": "{bloom_level}",
  "distractor_misconceptions": ["tag1", "tag2", "tag3"]}}
No markdown fences.
"""

_GRADE_PROMPT = """Grade this student's short answer using the rubric below.

Question: {question}
Rubric points (each worth 1 point, {max_score} points total):
{rubric_points}

Reference answer (for your reference, not to be copied verbatim in feedback): {reference_answer}

Student's answer: {student_answer}

Respond ONLY as JSON with keys: "score" (float, 0 to {max_score}),
"justification" (1-2 sentences), "misconception_tag" (a short tag naming
the main gap if score is not full marks, else "none").
No markdown fences.
"""


def _generate_mcq(llm, topic, objective, bloom_level="understand"):
    raw = ""
    try:
        resp = llm.invoke(_MCQ_PROMPT.format(
            topic=topic, objective=objective, bloom_level=bloom_level))
        raw = _strip_fences(extract_content(resp))
        if raw:
            return json.loads(raw)
        raise ValueError("Empty response from LLM")
    except Exception as e:
        print(f"[WARNING] MCQ generation failed for '{topic}': {e}")
        print(f"[WARNING] Raw response: {raw[:300]}")
        return {
            "prompt": f"Which of the following best describes '{topic}'?",
            "options": [
                objective[:80] if len(objective) <= 80 else objective[:77] + "...",
                "An unrelated concept",
                "A different unrelated topic",
                "None of the above are correct"
            ],
            "correct_index": 0,
            "bloom_level": bloom_level,
            "distractor_misconceptions": ["generic_1", "generic_2", "generic_3"]
        }


def _grade_short_answer(llm, rubric, student_answer):
    resp = llm.invoke(_GRADE_PROMPT.format(
        question=rubric["question"],
        rubric_points="\n".join(f"- {p}" for p in rubric["rubric_points"]),
        max_score=rubric["max_score"],
        reference_answer=rubric["reference_answer"],
        student_answer=student_answer,
    ))
    raw = extract_content(resp)
    raw = _strip_fences(raw)

    llm_result = json.loads(raw)
    llm_result["raw_response"] = raw

    # Embedding cross-check using pure-numpy TF-IDF cosine (no scipy/sentence-transformers)
    from src.utils import tfidf_cosine
    similarity = tfidf_cosine(student_answer, rubric["reference_answer"])
    llm_score_normalized = llm_result["score"] / rubric["max_score"]
    disagreement = abs(similarity - llm_score_normalized)
    llm_result["embedding_similarity"] = round(similarity, 4)
    llm_result["flagged"] = disagreement > SIMILARITY_DISAGREEMENT_THRESHOLD
    return llm_result


def prepare_item(state: LearnerState) -> LearnerState:
    """Generate an MCQ or short-answer item for the current topic. No I/O."""
    state = dict(state)
    topic = state["topic_pointer"]
    if topic is None:
        return state

    from src.agents.planner import _load_graph
    graph = _load_graph(state["domain"])
    objective = graph[topic]["objective"]

    fmt = _choose_format(state)
    gen_llm = get_llm("assessment_gen")

    # Look up Bloom level for this topic from the planner's bloom_plan
    bloom_plan  = state.get("bloom_plan", [])
    bloom_entry = next((b for b in bloom_plan if b["topic"] == topic), None)
    bloom_level = bloom_entry["bloom_level"] if bloom_entry else "understand"

    if fmt == "short_answer":
        rubric = _load_rubric(state["domain"], topic)
        if rubric is not None:
            state["pending_item"] = {
                "format": "short_answer",
                "question": rubric["question"],
                "rubric": rubric,
            }
            return state

    item = _generate_mcq(gen_llm, topic, objective, bloom_level)
    state["pending_item"] = {"format": "mcq", **item}
    return state


def apply_grade(state: LearnerState, answer: str) -> LearnerState:
    """Grade `answer` against pending_item. No I/O."""
    state = dict(state)
    item = state.get("pending_item") or {}
    topic = state.get("topic_pointer")
    if not item or topic is None:
        return state

    if item.get("format") == "mcq":
        try:
            chosen = int(str(answer).strip())
        except ValueError:
            chosen = -1
        correct = chosen == item.get("correct_index")
        state["last_grade"] = {
            "correct": correct,
            "score": 1.0 if correct else 0.0,
            "flagged": False,
            "justification": "",
        }
    else:
        rubric = item.get("rubric") or _load_rubric(state["domain"], topic)
        if rubric is None:
            state["last_grade"] = {
                "correct": None, "score": None, "flagged": True,
                "misconception_tag": "no_rubric_available",
            }
            return state
        grade_llm = get_llm("assessment_grade")
        result = _grade_short_answer(grade_llm, rubric, answer)
        state["last_grade"] = {
            "correct": result["score"] >= rubric["max_score"] * 0.6,
            "score": result["score"] / rubric["max_score"],
            "flagged": result["flagged"],
            "misconception_tag": result.get("misconception_tag", "none"),
            "justification": result.get("justification", ""),
            "raw_response": result.get("raw_response"),
            "embedding_similarity": result.get("embedding_similarity"),
        }

    state["session_history"] = state["session_history"] + [
        {
            "event": "assessment",
            "topic": topic,
            "format": item.get("format"),
            "grade": {
                k: v for k, v in (state["last_grade"] or {}).items()
                if k != "raw_response"
            },
        }
    ]
    return state


def assessment_node(state: LearnerState) -> LearnerState:
    """CLI LangGraph node: generate item, collect answer via input(), grade."""
    state = prepare_item(state)
    topic = state.get("topic_pointer")
    item = state.get("pending_item") or {}
    if topic is None or not item:
        return state

    if item.get("format") == "mcq":
        print(f"\n[{topic}] {item['prompt']}")
        for i, opt in enumerate(item["options"]):
            print(f"  {i}. {opt}")
        answer = input("Your answer (option number): ").strip()
    else:
        print(f"\n[{topic}] {item['question']}")
        answer = input("Your answer: ").strip()

    return apply_grade(state, answer)


assessment_agent = log_node_call("assessment")(assessment_node)
assessment_prepare_agent = log_node_call("assessment")(prepare_item)
