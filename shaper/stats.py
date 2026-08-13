"""Statistical machinery (spec A.12, paper 3.9).

Confirmatory unit: SEEDS. Seed-level paired effects + Holm adjustment govern
confirmatory claims; users are NEVER treated as independent training runs.
User-level paired Wilcoxon, rank-biserial / Cliff's delta, and user
bootstraps are descriptive conditional-on-model analyses.

Planning half-width (Appendix J):
    h = t_(.975, S-1) * sqrt(sigma_seed^2 / S + sigma_user^2 / N_game)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats as sps


def seed_summary(values: Sequence[float], level: float = 0.95) -> Dict[str, Any]:
    """Per-seed aggregate statistics over the confirmatory seed set."""
    arr = np.asarray(values, dtype=np.float64)
    S = len(arr)
    mean = float(arr.mean())
    sd = float(arr.std(ddof=1)) if S > 1 else 0.0
    se = sd / np.sqrt(S) if S > 1 else np.nan
    if S > 1:
        half = sps.t.ppf(1 - (1 - level) / 2, S - 1) * se
        ci = (mean - half, mean + half)
    else:
        ci = (mean, mean)
    return {
        "n_seeds": S,
        "mean": mean,
        "sd": sd,
        "se": se,
        "ci_level": level,
        "ci": (float(ci[0]), float(ci[1])),
        "values": [float(v) for v in arr],
    }


def paired_seed_effects(
    a: Sequence[float], b: Sequence[float], level: float = 0.95
) -> Dict[str, Any]:
    """Seed-level paired difference a - b (the confirmatory unit)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if len(a) != len(b):
        raise ValueError("paired effects require equal-length seed series")
    diff = a - b
    S = len(diff)
    if S < 2:
        return {"n_seeds": S, "mean_diff": float(diff.mean()), "note": "too few seeds for a paired test"}
    t_stat, p_value = sps.ttest_rel(a, b)
    half = sps.t.ppf(1 - (1 - level) / 2, S - 1) * diff.std(ddof=1) / np.sqrt(S)
    return {
        "n_seeds": S,
        "mean_diff": float(diff.mean()),
        "per_seed_diff": [float(d) for d in diff],
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "ci_level": level,
        "ci": (float(diff.mean() - half), float(diff.mean() + half)),
    }


def holm_adjust(p_values: Sequence[float]) -> List[float]:
    """Holm-Bonferroni step-down adjusted p-values.

    adjusted_(i) = min(1, max_{j <= i} (m - j + 1) * p_(j)) on the sorted
    p-values, mapped back to the original order.
    """
    p = np.asarray(p_values, dtype=np.float64)
    m = len(p)
    order = np.argsort(p)
    sorted_p = p[order]
    adjusted_sorted = np.minimum(1.0, np.maximum.accumulate(sorted_p * np.arange(m, 0, -1)))
    out = np.empty(m)
    out[order] = adjusted_sorted
    return [float(x) for x in out]


def wilcoxon_paired(a: Sequence[float], b: Sequence[float]) -> Dict[str, Any]:
    """User-level paired Wilcoxon (DESCRIPTIVE, conditional-on-model)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    stat, p = sps.wilcoxon(a, b)
    return {"statistic": float(stat), "p_value": float(p), "n_pairs": len(a)}


def rank_biserial(a: Sequence[float], b: Sequence[float]) -> float:
    """Rank-biserial correlation for paired differences (positive = a > b)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    diff = a - b
    nonzero = diff[diff != 0]
    n = len(nonzero)
    if n == 0:
        return 0.0
    ranks = sps.rankdata(np.abs(nonzero))
    w = ranks[nonzero > 0].sum()
    r_minus = ranks[nonzero < 0].sum()
    return float((w - r_minus) / (w + r_minus))


def cliffs_delta(a: Sequence[float], b: Sequence[float]) -> float:
    """Cliff's delta for paired samples (descriptive effect size)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    diff = a - b
    n = len(diff)
    if n == 0:
        return 0.0
    wins = float((diff > 0).sum())
    losses = float((diff < 0).sum())
    return (wins - losses) / n


def bootstrap_ci(
    values: Sequence[float], n_bootstrap: int = 2000, seed: int = 0, level: float = 0.95
) -> Dict[str, Any]:
    """User-level bootstrap CI (descriptive)."""
    arr = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        means[i] = rng.choice(arr, size=len(arr), replace=True).mean()
    lo, hi = np.percentile(means, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return {
        "mean": float(arr.mean()),
        "ci": (float(lo), float(hi)),
        "n_bootstrap": n_bootstrap,
        "level": level,
    }


def planning_half_width(
    S: int, sigma_seed: float, sigma_user: float, N_game: int, level: float = 0.95
) -> float:
    """Conservative Appendix-J planning half-width
    h = t_(.975, S-1) * sqrt(sigma_seed^2 / S + sigma_user^2 / N_game)."""
    t = sps.t.ppf(1 - (1 - level) / 2, S - 1) if S > 1 else float("inf")
    return float(t * np.sqrt(sigma_seed**2 / S + sigma_user**2 / N_game))


def hierarchical_bootstrap_ci(
    seed_values: Sequence[Sequence[float]], n_bootstrap: int = 2000, seed: int = 0, level: float = 0.95
) -> Dict[str, Any]:
    """Hierarchical (seed -> user) bootstrap of the seed mean."""
    rng = np.random.default_rng(seed)
    S = len(seed_values)
    means = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        seeds_idx = rng.integers(0, S, size=S)
        acc = 0.0
        n = 0
        for s in seeds_idx:
            arr = np.asarray(seed_values[s], dtype=np.float64)
            users = rng.integers(0, len(arr), size=len(arr))
            acc += arr[users].sum()
            n += len(arr)
        means[i] = acc / n
    lo, hi = np.percentile(means, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return {"mean": float(np.mean(means)), "ci": (float(lo), float(hi)), "n_bootstrap": n_bootstrap}


def half_width_of(ci: Tuple[float, float]) -> float:
    return (ci[1] - ci[0]) / 2.0


__all__ = [
    "seed_summary",
    "paired_seed_effects",
    "holm_adjust",
    "wilcoxon_paired",
    "rank_biserial",
    "cliffs_delta",
    "bootstrap_ci",
    "planning_half_width",
    "hierarchical_bootstrap_ci",
    "half_width_of",
]
