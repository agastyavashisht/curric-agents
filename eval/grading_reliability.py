"""
Computes agreement between the Assessment Agent's short-answer grades
and an instructor's independent blind grading of the same sample.

Expects a CSV with columns:
    student_id, question_id, system_score, instructor_score
Both scores on the same integer scale (e.g. 0-3).

Usage:
    python -m eval.grading_reliability --file results/instructor_grades.csv

NOTE: sklearn is intentionally not used — QWK is implemented with pure
numpy so the script runs under App Control policies that block sklearn DLLs.
"""
import argparse
import json
import numpy as np
import pandas as pd


def quadratic_weighted_kappa(y1: np.ndarray, y2: np.ndarray) -> float:
    """
    Cohen's quadratic-weighted kappa — pure numpy.
    y1, y2: integer arrays with the same set of possible ratings.
    """
    min_rating = int(min(y1.min(), y2.min()))
    max_rating = int(max(y1.max(), y2.max()))
    n_ratings = max_rating - min_rating + 1
    n = len(y1)

    # weight matrix: w[i,j] = (i-j)^2 / (n_ratings-1)^2
    denom = (n_ratings - 1) ** 2 if n_ratings > 1 else 1
    weights = np.array(
        [[(i - j) ** 2 / denom for j in range(n_ratings)] for i in range(n_ratings)],
        dtype=float,
    )

    # observed confusion matrix
    O = np.zeros((n_ratings, n_ratings), dtype=float)
    for a, b in zip(y1 - min_rating, y2 - min_rating):
        O[int(a), int(b)] += 1
    O /= O.sum()

    # expected matrix (outer product of marginals)
    hist1 = np.bincount(y1 - min_rating, minlength=n_ratings).astype(float)
    hist2 = np.bincount(y2 - min_rating, minlength=n_ratings).astype(float)
    E = np.outer(hist1, hist2)
    E /= E.sum()

    num = (weights * O).sum()
    den = (weights * E).sum()
    return 1.0 - num / den if den != 0 else 1.0


def _interpret_kappa(k: float) -> str:
    if k < 0:
        return "poor (worse than chance)"
    elif k < 0.2:
        return "slight"
    elif k < 0.4:
        return "fair"
    elif k < 0.6:
        return "moderate"
    elif k < 0.8:
        return "substantial"
    return "almost perfect"


def analyze(path: str) -> dict:
    df = pd.read_csv(path)

    system = df["system_score"].round().astype(int).values
    instructor = df["instructor_score"].round().astype(int).values

    kappa = quadratic_weighted_kappa(system, instructor)
    exact_agreement_rate = float((system == instructor).mean())
    mean_abs_diff = float(np.abs(df["system_score"] - df["instructor_score"]).mean())

    # per-question breakdown
    per_q = (
        df.groupby("question_id")
        .apply(lambda g: round(float(np.abs(g["system_score"] - g["instructor_score"]).mean()), 3))
        .to_dict()
    )

    return {
        "n_graded_samples": len(df),
        "quadratic_weighted_kappa": round(float(kappa), 4),
        "exact_agreement_rate": round(exact_agreement_rate, 4),
        "mean_absolute_score_difference": round(mean_abs_diff, 4),
        "interpretation": _interpret_kappa(kappa),
        "mean_abs_diff_per_question": per_q,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default="results/instructor_grades.csv")
    args = parser.parse_args()
    print(json.dumps(analyze(args.file), indent=2))
