"""
Computes agreement between the Assessment Agent's automated short-answer
grades and an instructor's independent blind grading of the same sample.

Metrics computed (paper §VI-B, Table V):
  - Accuracy, Precision, Recall, F1 (binary: system_score >= threshold)
  - Cohen's Quadratic-Weighted Kappa (QWK)
  - Fleiss' Kappa (multi-rater agreement — uses system + instructor as two raters)
  - Krippendorff's Alpha (ordinal scale)
  - Mean Absolute Score Difference

Expects a CSV with columns:
    student_id, question_id, system_score, instructor_score
Both scores on the same integer scale (e.g. 0-3).

Usage:
    python -m eval.grading_reliability --file results/instructor_grades.csv
    python -m eval.grading_reliability --file results/instructor_grades.csv --threshold 2

NOTE: All statistics are implemented with pure numpy — no sklearn, no scipy.
"""
import argparse
import json
import numpy as np
import pandas as pd


# ── Cohen's Quadratic-Weighted Kappa ─────────────────────────────────────────
def quadratic_weighted_kappa(y1: np.ndarray, y2: np.ndarray) -> float:
    min_r = int(min(y1.min(), y2.min()))
    max_r = int(max(y1.max(), y2.max()))
    n_r   = max_r - min_r + 1
    denom = (n_r - 1) ** 2 if n_r > 1 else 1

    weights = np.array(
        [[(i - j) ** 2 / denom for j in range(n_r)] for i in range(n_r)],
        dtype=float,
    )
    O = np.zeros((n_r, n_r), dtype=float)
    for a, b in zip(y1 - min_r, y2 - min_r):
        O[int(a), int(b)] += 1
    O /= O.sum()

    h1 = np.bincount(y1 - min_r, minlength=n_r).astype(float)
    h2 = np.bincount(y2 - min_r, minlength=n_r).astype(float)
    E  = np.outer(h1, h2) / (h1.sum() * h2.sum())

    num = (weights * O).sum()
    den = (weights * E).sum()
    return 1.0 - num / den if den != 0 else 1.0


# ── Fleiss' Kappa (2 raters = system + instructor) ────────────────────────────
def fleiss_kappa(y1: np.ndarray, y2: np.ndarray) -> float:
    """
    Fleiss' Kappa for exactly 2 raters over n items.
    Each item has 2 votes distributed over the possible categories.
    """
    min_r = int(min(y1.min(), y2.min()))
    max_r = int(max(y1.max(), y2.max()))
    cats  = list(range(min_r, max_r + 1))
    n     = len(y1)
    k     = len(cats)
    r     = 2   # number of raters

    # Build ratings matrix: n_items × n_categories
    mat = np.zeros((n, k), dtype=float)
    for i, (a, b) in enumerate(zip(y1, y2)):
        mat[i, cats.index(int(a))] += 1
        mat[i, cats.index(int(b))] += 1

    # P_i per item
    P_i = ((mat ** 2).sum(axis=1) - r) / (r * (r - 1))
    P_bar = P_i.mean()

    # P_j proportions per category
    p_j = mat.sum(axis=0) / (n * r)
    P_e = (p_j ** 2).sum()

    if P_e == 1.0:
        return 1.0
    return (P_bar - P_e) / (1.0 - P_e)


# ── Krippendorff's Alpha (ordinal) ────────────────────────────────────────────
def krippendorff_alpha(y1: np.ndarray, y2: np.ndarray, level: str = "ordinal") -> float:
    """
    Krippendorff's Alpha for 2 raters, ordinal or interval metric.
    Uses the standard coincidence-matrix formulation.
    """
    n    = len(y1)
    vals = sorted(set(y1.tolist() + y2.tolist()))
    k    = len(vals)
    v2i  = {v: i for i, v in enumerate(vals)}

    # Build coincidence matrix
    C = np.zeros((k, k), dtype=float)
    for a, b in zip(y1, y2):
        i, j = v2i[int(a)], v2i[int(b)]
        C[i, j] += 1
        if i != j:
            C[j, i] += 1

    n_c = C.sum()
    n_k = C.sum(axis=1)   # marginal for each category

    # Observed disagreement D_o
    D_o = 0.0
    for i in range(k):
        for j in range(k):
            if level == "ordinal":
                # ordinal distance: (sum of n_g between i and j, inclusive)^2
                lo, hi = min(i, j), max(i, j)
                d = (n_k[lo:hi + 1].sum() - (n_k[i] + n_k[j]) / 2.0) ** 2
            else:
                d = (vals[i] - vals[j]) ** 2
            D_o += C[i, j] * d
    D_o /= n_c

    # Expected disagreement D_e
    D_e = 0.0
    for i in range(k):
        for j in range(k):
            if level == "ordinal":
                lo, hi = min(i, j), max(i, j)
                d = (n_k[lo:hi + 1].sum() - (n_k[i] + n_k[j]) / 2.0) ** 2
            else:
                d = (vals[i] - vals[j]) ** 2
            D_e += n_k[i] * n_k[j] * d
    D_e /= (n_c * (n_c - 1)) if n_c > 1 else 1.0

    if D_e == 0.0:
        return 1.0
    return 1.0 - D_o / D_e


# ── Binary classification metrics ─────────────────────────────────────────────
def _binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Accuracy, Precision, Recall, F1 for binary labels."""
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    n  = tp + fp + fn + tn

    accuracy  = (tp + tn) / n if n else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall    = tp / (tp + fn) if (tp + fn) else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) else 0.0)
    return {
        "accuracy":  round(accuracy, 4),
        "precision": round(precision, 4),
        "recall":    round(recall, 4),
        "f1":        round(f1, 4),
    }


def _interpret_kappa(k: float) -> str:
    if k < 0:      return "poor (worse than chance)"
    elif k < 0.20: return "slight"
    elif k < 0.40: return "fair"
    elif k < 0.60: return "moderate"
    elif k < 0.80: return "substantial"
    return "almost perfect"


# ── Main analysis function ────────────────────────────────────────────────────
def analyze(path: str, pass_threshold: int = 2) -> dict:
    """
    Compute all grading-agreement metrics from the instructor grades CSV.

    pass_threshold: integer score at or above which an item is considered
                    'correct' for binary Accuracy/Precision/Recall/F1.
                    Default = 2 (on a 0-3 scale, ≥ 2 = passing).
    """
    df        = pd.read_csv(path)
    system    = df["system_score"].round().astype(int).values
    instructor = df["instructor_score"].round().astype(int).values

    # ── ordinal / weighted metrics ─────────────────────────────────────────────
    qwk   = quadratic_weighted_kappa(system, instructor)
    f_kap = fleiss_kappa(system, instructor)
    k_alp = krippendorff_alpha(system, instructor, level="ordinal")

    # ── binary metrics ─────────────────────────────────────────────────────────
    sys_bin = (system    >= pass_threshold).astype(int)
    ins_bin = (instructor >= pass_threshold).astype(int)
    bin_m   = _binary_metrics(ins_bin, sys_bin)   # instructor = ground truth

    # ── other agreement stats ──────────────────────────────────────────────────
    exact_agreement = float((system == instructor).mean())
    mean_abs_diff   = float(np.abs(system - instructor).mean())

    per_q = (
        df.groupby("question_id")
        .apply(lambda g: round(
            float(np.abs(g["system_score"] - g["instructor_score"]).mean()), 3
        ))
        .to_dict()
    )

    return {
        "n_graded_samples":              len(df),
        "pass_threshold":                pass_threshold,

        # Binary classification (paper Table V)
        "accuracy":                      bin_m["accuracy"],
        "precision":                     bin_m["precision"],
        "recall":                        bin_m["recall"],
        "f1_score":                      bin_m["f1"],

        # Agreement statistics (paper Table V)
        "cohens_kappa_qwk":              round(float(qwk),   4),
        "fleiss_kappa":                  round(float(f_kap), 4),
        "krippendorffs_alpha":           round(float(k_alp), 4),

        # Supporting stats
        "exact_agreement_rate":          round(exact_agreement, 4),
        "mean_absolute_score_difference": round(mean_abs_diff, 4),
        "interpretation_qwk":            _interpret_kappa(qwk),
        "mean_abs_diff_per_question":    per_q,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file",      default="results/instructor_grades.csv")
    parser.add_argument("--threshold", type=int, default=2,
                        help="Score threshold for binary pass/fail (default 2 on 0-3 scale)")
    args = parser.parse_args()
    print(json.dumps(analyze(args.file, pass_threshold=args.threshold), indent=2))
