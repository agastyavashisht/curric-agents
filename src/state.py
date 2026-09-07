"""
The single shared state object every agent reads from and writes to.
Single-writer rule (enforced by convention, not by code): each field below
is written by exactly one agent - see the comment next to each field.
"""
from typing import TypedDict, List, Dict, Optional, Any


class LearnerState(TypedDict):
    student_id: str
    domain: str
    group: str                      # "experimental" | "control"  (persisted arm assignment)

    # --- written by Planner ---
    topic_sequence: List[str]        # ordered list of topic ids remaining to teach
    topic_pointer: Optional[str]     # the topic currently being taught/assessed
    bloom_plan: List[Dict[str, Any]] # [{topic, bloom_level, mastery_gap}, ...] — from Planner
    pretest_score_pct: Optional[float]  # set after pretest so Planner can bootstrap mastery
    pretest_per_topic: Optional[Dict[str, Any]]  # {topic: {correct,total,pct}} from sectioned test
    ablation_mode: str               # "full"|"no_adaptive_assessment"|"no_adaptive_planning"|"planner_only"|"static"

    # --- written by Content ---
    current_material: Optional[Dict[str, Any]]  # {"explanation": str, "example": str}

    # --- written by Assessment ---
    pending_item: Optional[Dict[str, Any]]   # the MCQ / short-answer item just generated
    last_grade: Optional[Dict[str, Any]]     # {"correct": bool/None, "score": float, "flagged": bool, ...}

    # --- written by Monitor ---
    mastery: Dict[str, float]            # topic_id -> mastery estimate in [0, 1]
    last_reviewed: Dict[str, str]        # topic_id -> ISO timestamp of last assessment
    misconceptions: Dict[str, List[str]]  # topic_id -> list of tags
    engagement: Dict[str, Any]           # response_latency, hint_requests, session_count, ...
    replan_flag: bool                    # set True when Monitor detects drift/struggle

    # --- session bookkeeping ---
    session_history: List[Dict[str, Any]]  # append-only log of what happened this session
    step_count: int
    current_phase: str   # last known Streamlit phase — used to resume interrupted sessions

    # --- research experiment fields (spec §17) ---
    traditional_index: int               # Traditional/control group: index into the fixed topic order
    topics_attempted: List[str]          # topics the student actually studied/answered this experiment
    needs_review: Dict[str, bool]        # topic_id -> True once remediation cap is hit (scheduled later review)
    practice_scores: Dict[str, Any]      # topic_id -> {"correct", "total", "pct"} per-topic practice assessment
    posttest_per_topic: Optional[Dict[str, Any]] # topic-level post-test scores (correct/total/pct)
    posttest_score_pct: Optional[float]  # overall post-test percentage


def new_state(student_id: str, domain: str, group: str = "experimental") -> LearnerState:
    """A fresh state for a student who has no prior history in this domain."""
    return LearnerState(
        student_id=student_id,
        domain=domain,
        group=group,
        topic_sequence=[],
        topic_pointer=None,
        bloom_plan=[],
        pretest_score_pct=None,
        pretest_per_topic=None,
        ablation_mode="full",
        current_material=None,
        pending_item=None,
        last_grade=None,
        mastery={},
        last_reviewed={},
        misconceptions={},
        engagement={
            "response_latency": [], "hint_requests": 0, "session_count": 0,
            # research counters (spec §17) — remediation_counts reset per session
            "remediation_counts": {}, "total_replans": 0, "total_remediations": 0,
            "learning_time_sec": 0.0, "last_activity_ts": 0.0,
        },
        replan_flag=False,
        session_history=[],
        step_count=0,
        current_phase="pretest",
        traditional_index=0,
        topics_attempted=[],
        needs_review={},
        practice_scores={},
        posttest_per_topic=None,
        posttest_score_pct=None,
    )
