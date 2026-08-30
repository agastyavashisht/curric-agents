"""
Content Agent.

Retrieves relevant passages from the domain corpus (via sentence-transformer
similarity search), then calls the LLM with the retrieved context in-prompt.
This satisfies FR-3: "Content retrieves from the corpus and explains the
current topic" — the LLM must cite/use a retrieved chunk, not generate
purely from parametric memory.
"""
import json

from src.state import LearnerState
from src.logging_utils import log_node_call
from src.config import get_llm, extract_content, _strip_fences

_PROMPT_TEMPLATE = """You are an expert instructor writing a short lesson for one student.

Topic: {topic}
Domain: {domain}
Learning objective: {objective}
Student mastery (0=none, 1=mastered): {mastery}
Recommended Bloom level for this student: {bloom_level}
Known misconceptions to address: {misconceptions}

Bloom level guidance:
- remember: define terms, recall key facts — keep it simple and concrete
- understand: explain concepts in plain language with analogies
- apply: show how to USE the concept with a worked example
- analyze: compare alternatives, explain trade-offs, show edge cases
- evaluate: discuss design choices, pros/cons, when to use vs. not use

Reference material retrieved from the course corpus (use this as the factual basis):
---
{context}
---

Write:
1. A {bloom_level}-level explanation of the topic in 3-6 sentences grounded in the
   reference material above. If misconceptions are listed, address them directly.
2. One worked example appropriate for a {bloom_level}-level learner (code snippet
   for programming/data topics; calculation or diagram description for science topics).

Respond ONLY as JSON with exactly two keys: "explanation" (string) and "example" (string).
No markdown fences, no extra keys.
"""
# NOTE: No student_id or personally-identifying data in this prompt (FR-15).


def content_node(state: LearnerState) -> LearnerState:
    state = dict(state)
    topic = state["topic_pointer"]
    if topic is None:
        return state

    from src.agents.planner import _load_graph
    graph = _load_graph(state["domain"])
    objective = graph[topic]["objective"]

    # RAG: retrieve relevant corpus passages for this topic
    from src.retrieval import retrieve
    query = f"{topic}: {objective}"
    retrieved_chunks = retrieve(state["domain"], query, top_k=3)
    context = "\n\n".join(retrieved_chunks) if retrieved_chunks else \
              f"No corpus available — generate from your own knowledge of: {objective}"

    # Get Bloom level for this topic from the plan
    bloom_plan = state.get("bloom_plan", [])
    bloom_entry = next((b for b in bloom_plan if b["topic"] == topic), None)
    bloom_level = bloom_entry["bloom_level"] if bloom_entry else "understand"

    llm = get_llm("content")
    prompt = _PROMPT_TEMPLATE.format(
        topic=topic,
        domain=state["domain"],
        objective=objective,
        mastery=state["mastery"].get(topic, 0.0),
        bloom_level=bloom_level,
        misconceptions=state["misconceptions"].get(topic, []),
        context=context,
    )

    response = llm.invoke(prompt)
    raw = _strip_fences(extract_content(response))

    try:
        material = json.loads(raw)
    except json.JSONDecodeError:
        material = {"explanation": raw, "example": ""}

    state["current_material"] = material

    # Log generated content for curriculum_coherence evaluation
    from src.logging_utils import _append
    _append("results/generated_content.jsonl", {
        "student_id": state["student_id"],
        "domain": state["domain"],
        "topic": topic,
        "objective": objective,
        "explanation": material.get("explanation", ""),
        "n_retrieved_chunks": len(retrieved_chunks),
    })

    state["session_history"] = state["session_history"] + [
        {"event": "content_delivered", "topic": topic, "domain": state["domain"]}
    ]
    return state


content_agent = log_node_call("content")(content_node)
