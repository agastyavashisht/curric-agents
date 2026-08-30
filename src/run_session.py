"""
CLI entrypoint. Run one learner session end to end:

    python -m src.run_session --student_id S01 --domain python_programming --max_topics 3

Loads the student's saved state (or creates a fresh one), runs the
graph for up to --max_topics topic-cycles, then saves state back to
SQLite so the next session picks up where this one left off.
"""
import argparse

from src.config import DB_PATH
from src.memory.store import load_state, save_state
from src.graph import build_graph


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--student_id", required=True)
    parser.add_argument("--domain", default="python_programming")
    parser.add_argument("--max_topics", type=int, default=3,
                         help="Stop after this many topics are covered, so one CLI session doesn't run forever.")
    args = parser.parse_args()

    state = load_state(DB_PATH, args.student_id, args.domain)
    state["engagement"]["session_count"] = state["engagement"].get("session_count", 0) + 1

    graph = build_graph()

    print(f"\n=== Session start: {args.student_id} / {args.domain} ===")

    # graph.invoke() runs the whole Planner->Content->Assessment->Monitor
    # loop internally until the topic queue is empty (END) or a replan
    # loops back. Each topic-cycle is up to 6 node visits in the worst case
    # (planner replan + content + assessment + monitor + advance_topic + content again),
    # so we use a generous multiplier with a fixed buffer to avoid hitting the cap.
    recursion_limit = 10 + args.max_topics * 10
    state = graph.invoke(state, config={"recursion_limit": recursion_limit})

    save_state(DB_PATH, state)

    print(f"\n=== Session end ===")
    print("Mastery snapshot:")
    for topic, score in state["mastery"].items():
        print(f"  {topic}: {score}")
    print(f"State saved to {DB_PATH}")


if __name__ == "__main__":
    main()
