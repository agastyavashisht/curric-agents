"""
Computes the curriculum-coherence score:
  (a) prerequisite edge-consistency of a generated topic sequence
      against the domain's prerequisite graph
  (b) mean embedding similarity between each topic's stated learning
      objective and what Content actually generated, read from
      results/generated_content.jsonl

Supports all 5 domains via --domain flag.

Usage:
    python -m eval.curriculum_coherence --domain python_programming --sequence variables_datatypes,strings,operators_expressions
    python -m eval.curriculum_coherence --domain ml_basics --content_log results/generated_content.jsonl
    python -m eval.curriculum_coherence --domain all --content_log results/generated_content.jsonl
"""
import argparse
import json
import os


DOMAIN_GRAPH_FILES = {
    "python_programming": "data/prerequisite_graph.json",
    "ml_basics":          "data/ml_basics_graph.json",
    "signal_processing":  "data/signal_processing_graph.json",
    "physiology_basics":  "data/physiology_graph.json",
    "data_analysis":      "data/data_analysis_graph.json",
}


def _graph_path(domain: str) -> str:
    if domain not in DOMAIN_GRAPH_FILES:
        raise ValueError(
            f"Unknown domain '{domain}'. "
            f"Valid options: {list(DOMAIN_GRAPH_FILES.keys())} or 'all'"
        )
    path = DOMAIN_GRAPH_FILES[domain]
    if not os.path.exists(path):
        raise FileNotFoundError(f"Graph file not found: {path}")
    return path


def edge_consistency(sequence: list, domain: str = "python_programming") -> dict:
    """
    Checks that every prerequisite edge in the graph is respected by
    the given topic ordering.  Only edges whose both endpoints appear
    in the sequence are counted.
    """
    graph_path = _graph_path(domain)
    with open(graph_path, "r", encoding="utf-8") as f:
        topics = json.load(f)["topics"]

    position = {t: i for i, t in enumerate(sequence)}
    total_edges = 0
    respected_edges = 0

    for topic, info in topics.items():
        for prereq in info.get("prerequisites", []):
            if topic not in position or prereq not in position:
                continue   # can't judge — skip
            total_edges += 1
            if position[prereq] < position[topic]:
                respected_edges += 1

    fraction = respected_edges / total_edges if total_edges > 0 else 1.0
    return {
        "domain": domain,
        "edges_checked": total_edges,
        "edges_respected": respected_edges,
        "edge_consistency": round(fraction, 4),
    }


def objective_alignment(
    content_log_path: str = "results/generated_content.jsonl",
    domain_filter: str | None = None,
) -> dict:
    """
    For each logged content item, embeds the generated explanation and
    the topic's learning objective, then reports mean cosine similarity.
    Optionally filters to a single domain.
    """
    try:
        from sentence_transformers import SentenceTransformer, util
    except ImportError:
        return {"error": "sentence-transformers not installed"}

    if not os.path.exists(content_log_path):
        return {"error": f"content log not found: {content_log_path}"}

    rows = []
    with open(content_log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if domain_filter and rec.get("domain") != domain_filter:
                continue
            rows.append(rec)

    if not rows:
        return {"error": "no content rows found (check domain filter or run a session first)"}

    model = SentenceTransformer("all-MiniLM-L6-v2")
    similarities = []
    for row in rows:
        if not row.get("explanation") or not row.get("objective"):
            continue
        emb = model.encode([row["explanation"], row["objective"]], convert_to_tensor=True)
        sim = float(util.cos_sim(emb[0], emb[1])[0][0])
        similarities.append(sim)

    if not similarities:
        return {"error": "no valid explanation/objective pairs found"}

    return {
        "domain_filter": domain_filter or "all",
        "n_content_items": len(similarities),
        "mean_objective_alignment": round(sum(similarities) / len(similarities), 4),
        "min_alignment": round(min(similarities), 4),
        "max_alignment": round(max(similarities), 4),
    }


def coherence_score(
    edge_result: dict,
    alignment_result: dict,
    weight_edges: float = 0.5,
    weight_alignment: float = 0.5,
) -> float | None:
    ec = edge_result.get("edge_consistency")
    oa = alignment_result.get("mean_objective_alignment")
    if ec is None or oa is None:
        return None
    return round(weight_edges * ec + weight_alignment * oa, 4)


def sequences_from_content_log(content_log_path: str) -> dict:
    """
    Recover taught topic order per (student_id, domain) from Content logs.
    Used for prerequisite edge-consistency without a separately typed sequence.
    """
    sequences = {}
    if not os.path.exists(content_log_path):
        return sequences
    with open(content_log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            key = (rec.get("student_id"), rec.get("domain"))
            topic = rec.get("topic")
            if not key[0] or not key[1] or not topic:
                continue
            seq = sequences.setdefault(key, [])
            if topic not in seq:
                seq.append(topic)
    return sequences


def edge_consistency_from_content_log(
    content_log_path: str = "results/generated_content.jsonl",
    domain_filter: str | None = None,
) -> dict:
    sequences = sequences_from_content_log(content_log_path)
    per = []
    for (sid, domain), seq in sequences.items():
        if domain_filter and domain != domain_filter:
            continue
        if len(seq) < 2:
            continue
        r = edge_consistency(seq, domain=domain)
        r["student_id"] = sid
        per.append(r)
    if not per:
        return {"error": "not enough multi-topic sequences in content log"}
    mean_ec = sum(p["edge_consistency"] for p in per) / len(per)
    return {
        "n_sequences": len(per),
        "mean_edge_consistency": round(mean_ec, 4),
        "per_sequence": per,
    }


def run_all_domains(content_log: str = "results/generated_content.jsonl") -> dict:
    """Run objective_alignment for every domain and return combined results."""
    results = {}
    for domain in DOMAIN_GRAPH_FILES:
        results[domain] = objective_alignment(content_log, domain_filter=domain)
    results["all_domains"] = objective_alignment(content_log, domain_filter=None)
    results["edge_consistency"] = edge_consistency_from_content_log(content_log)
    align = results["all_domains"]
    edge = results["edge_consistency"]
    if "mean_objective_alignment" in align and "mean_edge_consistency" in edge:
        results["combined_coherence_score"] = coherence_score(
            {"edge_consistency": edge["mean_edge_consistency"]},
            align,
        )
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Curriculum coherence evaluation (all 5 domains supported)"
    )
    parser.add_argument(
        "--domain",
        default="python_programming",
        help=f"Domain key or 'all'. Options: {list(DOMAIN_GRAPH_FILES.keys())}",
    )
    parser.add_argument(
        "--sequence",
        default=None,
        help="Comma-separated topic IDs to check edge consistency for",
    )
    parser.add_argument(
        "--content_log",
        default="results/generated_content.jsonl",
    )
    args = parser.parse_args()

    result = {}

    if args.domain == "all":
        result["objective_alignment_by_domain"] = run_all_domains(args.content_log)
    else:
        if args.sequence:
            result["edge_consistency"] = edge_consistency(
                args.sequence.split(","), domain=args.domain
            )
        align = objective_alignment(args.content_log, domain_filter=args.domain)
        result["objective_alignment"] = align

        if args.sequence and "edge_consistency" in result:
            result["combined_coherence_score"] = coherence_score(
                result["edge_consistency"], align
            )

    print(json.dumps(result, indent=2))
