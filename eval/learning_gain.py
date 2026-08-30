"""
Computes Hake's normalized gain per student and compares experimental
vs. control groups.

Expects two CSVs with columns: student_id, group, score_pct
  group is "experimental" or "control"
  score_pct is 0-100

Usage:
    python -m eval.learning_gain --pretest results/pretest_scores.csv
                                 --posttest results/posttest_scores.csv

NOTE: scipy is intentionally not used here — all statistics are
implemented with numpy so the script runs under App Control policies
that block compiled scipy DLLs.
"""
import argparse
import json
import math
import numpy as np
import pandas as pd


def normalized_gain(pre: float, post: float) -> float:
    """Hake's <g>: fraction of possible gain actually achieved."""
    if pre >= 100:
        return 0.0
    return (post - pre) / (100.0 - pre)


def cohens_d(a: pd.Series, b: pd.Series) -> float:
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return 0.0
    pooled_var = ((n1 - 1) * a.std(ddof=1) ** 2 + (n2 - 1) * b.std(ddof=1) ** 2) / (n1 + n2 - 2)
    pooled_std = math.sqrt(pooled_var) if pooled_var > 0 else 0.0
    return (a.mean() - b.mean()) / pooled_std if pooled_std else 0.0


def _welch_ttest(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """
    Welch's t-test (unequal variances) — pure numpy implementation.
    Returns (t_statistic, two-tailed p_value).
    """
    m1, m2 = a.mean(), b.mean()
    v1, v2 = a.var(ddof=1), b.var(ddof=1)
    n1, n2 = len(a), len(b)
    se = math.sqrt(v1 / n1 + v2 / n2)
    if se == 0:
        return 0.0, 1.0
    t = (m1 - m2) / se
    # Welch–Satterthwaite degrees of freedom
    df = (v1 / n1 + v2 / n2) ** 2 / (
        (v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1)
    )
    # two-tailed p via regularised incomplete beta (pure math, no scipy)
    p = _t_pvalue(abs(t), df)
    return round(t, 6), round(p, 6)


def _t_pvalue(t: float, df: float) -> float:
    """Approximate two-tailed p-value for t-distribution using the
    continued-fraction expansion of the regularised incomplete beta.
    Accurate to ~4 decimal places for df >= 2 and |t| up to ~10."""
    x = df / (df + t * t)
    p_one_tail = _betainc(df / 2.0, 0.5, x) / 2.0
    return min(2.0 * p_one_tail, 1.0)


def _betainc(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta I_x(a,b) via Lentz continued fraction."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(math.log(x) * a + math.log(1 - x) * b - lbeta) / a
    # use the symmetry relation for better convergence
    if x > (a + 1) / (a + b + 2):
        return 1.0 - _betainc(b, a, 1.0 - x)
    # Lentz's algorithm for continued fraction
    TINY = 1e-30
    f = TINY
    C = f
    D = 0.0
    for m in range(200):
        for j in range(2):
            if m == 0 and j == 0:
                num = 1.0
            elif j == 0:
                num = m * (b - m) * x / ((a + 2 * m - 1) * (a + 2 * m))
            else:
                num = -(a + m) * (a + b + m) * x / ((a + 2 * m) * (a + 2 * m + 1))
            D = 1.0 + num * D
            if abs(D) < TINY:
                D = TINY
            C = 1.0 + num / C
            if abs(C) < TINY:
                C = TINY
            D = 1.0 / D
            delta = C * D
            f *= delta
            if abs(delta - 1.0) < 1e-10:
                return front * f
    return front * f


def _mann_whitney_u(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """
    Mann-Whitney U test — pure numpy.
    Returns (U_statistic, two-tailed p_value) using the normal approximation
    (valid for n >= 8 per group; for smaller samples reports exact U only).
    """
    n1, n2 = len(a), len(b)
    combined = np.concatenate([a, b])
    ranks = np.argsort(np.argsort(combined)) + 1.0
    R1 = ranks[:n1].sum()
    U1 = R1 - n1 * (n1 + 1) / 2.0
    U2 = n1 * n2 - U1
    U = min(U1, U2)
    if n1 < 8 or n2 < 8:
        return round(U, 4), float("nan")
    mu_U = n1 * n2 / 2.0
    sigma_U = math.sqrt(n1 * n2 * (n1 + n2 + 1) / 12.0)
    z = (U - mu_U) / sigma_U
    p = 2.0 * (1.0 - _norm_cdf(abs(z)))
    return round(U, 4), round(p, 6)


def _norm_cdf(z: float) -> float:
    """Standard normal CDF via math.erf."""
    return (1.0 + math.erf(z / math.sqrt(2.0))) / 2.0


def analyze(pretest_path: str, posttest_path: str) -> dict:
    pre = pd.read_csv(pretest_path)
    post = pd.read_csv(posttest_path)

    df = pre.merge(post, on=["student_id", "group"], suffixes=("_pre", "_post"))
    df["gain"] = df.apply(
        lambda r: normalized_gain(r["score_pct_pre"], r["score_pct_post"]), axis=1
    )
    df["gain_band"] = df["gain"].apply(
        lambda g: "low" if g < 0.3 else ("medium" if g < 0.7 else "high")
    )

    exp = df[df["group"] == "experimental"]["gain"]
    ctrl = df[df["group"] == "control"]["gain"]

    result = {
        "n_experimental": len(exp),
        "n_control": len(ctrl),
        "mean_gain_experimental": round(float(exp.mean()), 4) if len(exp) else None,
        "mean_gain_control": round(float(ctrl.mean()), 4) if len(ctrl) else None,
        "per_student_gain": df[["student_id", "group", "gain", "gain_band"]].to_dict(orient="records"),
    }

    if len(exp) >= 2 and len(ctrl) >= 2:
        t_stat, p_value = _welch_ttest(exp.values, ctrl.values)
        result["t_test"] = {"t_statistic": t_stat, "p_value": p_value}
        result["cohens_d"] = round(cohens_d(exp, ctrl), 4)
        u_stat, p_u = _mann_whitney_u(exp.values, ctrl.values)
        result["mann_whitney_u"] = {"u_statistic": u_stat, "p_value": p_u}
    else:
        result["note"] = "fewer than 2 students in one group - significance test skipped"

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretest", default="results/pretest_scores.csv")
    parser.add_argument("--posttest", default="results/posttest_scores.csv")
    args = parser.parse_args()
    print(json.dumps(analyze(args.pretest, args.posttest), indent=2))
