"""
Static study-material helpers for the Traditional / control group (spec §3).

topic_order(domain)  — returns the FIXED topic sequence for the control group.
                       The pre-test score MUST NOT change this order (spec §3).

topic_notes(domain, topic) — returns the corpus text for one topic, extracted
                             from data/corpus/<domain>.md.
                             Returns None if the file or section is missing.

Design note:
  The control group studies one topic at a time in the fixed order defined here.
  Each topic's notes come from the SAME corpus that the AI group's Content agent
  retrieves from — this ensures both groups receive equivalent source material.
  The only difference is HOW it is delivered:
    Control:      raw corpus text, fixed order, no personalisation
    Experimental: AI-generated explanation + example, adaptive order, Bloom-calibrated
"""

from __future__ import annotations
import os
import re

# ── Fixed topic orders (spec §3 — NEVER reorder based on performance) ─────────
_TOPIC_ORDERS: dict[str, list[str]] = {
    "python_programming": [
        "variables_datatypes",
        "control_flow",
        "loops",
        "functions",
        "oop_basics",
    ],
    "ml_basics": [
        "ml_introduction",
        "data_preprocessing",
        "linear_regression",
        "logistic_regression",
        "model_evaluation",
    ],
    "signal_processing": [
        "signals_systems",
        "sampling_theorem",
        "fourier_series",
        "fourier_transform",
        "filtering_basics",
    ],
    "physiology_basics": [
        "cell_biology",
        "homeostasis",
        "nervous_system",
        "cardiovascular_system",
        "respiratory_system",
    ],
    "data_analysis": [
        "descriptive_statistics",
        "data_visualization",
        "probability_basics",
        "hypothesis_testing",
        "correlation_regression",
    ],
}

# Human-readable section headings to search for inside the corpus markdown.
# The corpus files use headings like "## Variables and Data Types" or
# "## Variables & Data Types".  We match case-insensitively.
_TOPIC_HEADING_ALIASES: dict[str, list[str]] = {
    "variables_datatypes":  ["variables", "data types", "datatypes"],
    "control_flow":         ["control flow", "conditionals", "if", "elif"],
    "loops":                ["loops", "iteration", "for loop", "while loop"],
    "functions":            ["functions", "def ", "parameters", "return"],
    "oop_basics":           ["oop basics", "oop", "object-oriented", "classes", "class"],
    "ml_introduction":      ["machine learning", "ml introduction", "introduction"],
    "data_preprocessing":   ["preprocessing", "data cleaning", "normalization"],
    "linear_regression":    ["linear regression"],
    "logistic_regression":  ["logistic regression"],
    "model_evaluation":     ["model evaluation", "cross-validation", "accuracy"],
    "signals_systems":      ["signals", "systems", "signal"],
    "sampling_theorem":     ["sampling", "nyquist"],
    "fourier_series":       ["fourier series"],
    "fourier_transform":    ["fourier transform"],
    "filtering_basics":     ["filter", "low-pass", "high-pass"],
    "cell_biology":         ["cell", "membrane", "nucleus"],
    "homeostasis":          ["homeostasis", "feedback"],
    "nervous_system":       ["nervous system", "neuron", "action potential"],
    "cardiovascular_system":["cardiovascular", "heart", "cardiac"],
    "respiratory_system":   ["respiratory", "lungs", "breathing"],
    "descriptive_statistics":["descriptive statistics", "mean", "median", "variance"],
    "data_visualization":   ["visualization", "chart", "histogram", "boxplot"],
    "probability_basics":   ["probability", "conditional"],
    "hypothesis_testing":   ["hypothesis", "t-test", "p-value"],
    "correlation_regression":["correlation", "regression", "pearson"],
}


def topic_order(domain: str) -> list[str]:
    """
    Returns the FIXED topic sequence for the control group.
    Pre-test score MUST NOT change this order (spec §3).
    Falls back to alphabetical sort if the domain is not listed.
    """
    return list(_TOPIC_ORDERS.get(domain, []))


def topic_notes(domain: str, topic: str) -> str | None:
    """
    Returns the corpus text relevant to `topic` from data/corpus/<domain>.md.

    Strategy:
      1. Split the corpus on markdown H2 headings (## ...).
      2. Find the section whose heading best matches the topic's aliases.
      3. Return that section's text (heading + body).
      4. If no match found, return the full corpus with a note.
      5. Return None if the corpus file is missing entirely.
    """
    corpus_path = os.path.join("data", "corpus", f"{domain}.md")
    if not os.path.exists(corpus_path):
        return None

    with open(corpus_path, "r", encoding="utf-8") as f:
        raw = f.read()

    # Split on H2 headings
    parts = re.split(r"\n(?=## )", raw)

    # Try to match the best section for this topic
    # Use whole-word or start-of-word matching to avoid "oop" matching "loops"
    aliases = _TOPIC_HEADING_ALIASES.get(topic, [topic.replace("_", " ")])
    best_section: str | None = None

    for part in parts:
        heading_line = part.splitlines()[0].lower() if part.strip() else ""
        for alias in aliases:
            al = alias.lower()
            # Require the alias to appear as a whole word or phrase
            # (not as a substring of another word, e.g. "oop" inside "loops")
            pattern = r'\b' + re.escape(al)
            if re.search(pattern, heading_line):
                best_section = part.strip()
                break
        if best_section:
            break

    if best_section:
        return best_section

    # Fallback: return the full corpus with a note
    return (
        f"> *Note: a specific section for '{topic.replace('_', ' ').title()}' "
        f"was not found — the full course notes are shown below.*\n\n{raw}"
    )
