"""Tables and figures for the paper (paper: "PLANNED FIGURES AND TABLES").

All writers take realized result dictionaries; nothing here invents data.
Figures use the Agg backend (headless-safe).
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def table_to_markdown(headers: Sequence[str], rows: Sequence[Sequence[Any]], caption: str = "") -> str:
    lines = []
    if caption:
        lines += [caption, ""]
    lines.append("| " + " | ".join(str(h) for h in headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for row in rows:
        lines.append("| " + " | ".join(_fmt(v) for v in row) + " |")
    return "\n".join(lines) + "\n"


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def coalition_value_table(
    value_tables: Sequence[Dict[str, float]], players: Sequence[str], metric: str = "ndcg"
) -> str:
    """Table 3: complete exact primary Game-A coalition values per seed."""
    headers = ["coalition"] + [f"seed {i + 1}" for i in range(len(value_tables))] + ["mean"]
    coalitions = sorted(value_tables[0].keys(), key=lambda c: (len(c), tuple(c)))
    rows = []
    for coalition in coalitions:
        vals = [t[coalition] for t in value_tables]
        name = "+".join(coalition) if coalition else "empty"
        rows.append([name] + [f"{v:.6f}" for v in vals] + [f"{sum(vals) / len(vals):.6f}"])
    return table_to_markdown(headers, rows, caption=f"**Complete exact Game-A coalition values ({metric} uplift over empty)**")


def shapley_table(
    per_seed_phi: Sequence[Dict[str, float]],
    players: Sequence[str],
    efficiency_residuals: Optional[Sequence[float]] = None,
) -> str:
    """Table 4: exact Game-A raw Shapley values per seed + mean."""
    headers = ["player"] + [f"seed {i + 1}" for i in range(len(per_seed_phi))] + ["mean"]
    rows = []
    for p in players:
        vals = [s[p] for s in per_seed_phi]
        rows.append([p] + [f"{v:.6f}" for v in vals] + [f"{sum(vals) / len(vals):.6f}"])
    if efficiency_residuals is not None:
        rows.append(["efficiency residual"] + [f"{r:.3e}" for r in efficiency_residuals] + [""])
    return table_to_markdown(headers, rows, caption="**Exact Game-A raw Shapley values (NDCG@10)**")


def interaction_table(
    interaction_means: Dict[Any, float],
    interaction_cis: Dict[Any, Any],
    declarations: Dict[Any, str],
) -> str:
    """Table 5 rows for pair interactions."""
    rows = []
    for pair, mean in interaction_means.items():
        ci = interaction_cis.get(pair, ("", ""))
        rows.append(
            [f"{pair[0]}-{pair[1]}", f"{mean:.6f}", f"({ci[0]:.6f}, {ci[1]:.6f})", declarations.get(pair, "")]
        )
    return table_to_markdown(
        ["pair", "interaction", "95% CI", "declaration"], rows,
        caption="**Exact K=3 Grabisch-Roubens pair interactions**",
    )


def segment_table(
    mu: Dict[str, Dict[str, float]], se: Dict[str, Dict[str, float]], players: Sequence[str]
) -> str:
    """Table 6: raw segment Shapley credit by Q1-Q4."""
    headers = ["player"] + [f"Q{m}" for m in range(1, 5)]
    rows = []
    for p in players:
        rows.append([p] + [f"{mu[str(m)][p]:.6f} (±{se[str(m)][p]:.6f})" for m in range(1, 5)])
    return table_to_markdown(headers, rows, caption="**Raw segment Shapley credit (mean ± SE)**")


def unconditional_transfer_table(rows: Sequence[Dict[str, Any]]) -> str:
    """Table 7A: unconditional transfer (reported regardless of activation)."""
    headers = ["method", "NDCG@10", "HR@10", "MRR@10", "training cost", "note"]
    out = [[r.get("method", ""), r.get("ndcg", ""), r.get("hr", ""), r.get("mrr", ""),
            r.get("cost", ""), r.get("note", "")] for r in rows]
    return table_to_markdown(headers, out, caption="**Table 7A — Unconditional transfer (test)**")


def activation_conditioned_table(row: Dict[str, Any]) -> str:
    """Table 7B: the model actually selected by the locked activation rule."""
    headers = ["deployment", "NDCG@10", "HR@10", "MRR@10", "status"]
    out = [[row.get("deployment", ""), row.get("ndcg", ""), row.get("hr", ""),
            row.get("mrr", ""), row.get("status", "")]]
    return table_to_markdown(headers, out, caption="**Table 7B — Activation-conditioned deployment (test)**")


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------

def figure_shapley_bars(
    per_seed_phi: Sequence[Dict[str, float]],
    players: Sequence[str],
    path: str,
    title: str = "Exact Game-A Shapley by dataset",
) -> str:
    """Figure 2-style bars with seed spread."""
    import numpy as np

    means = {p: np.mean([s[p] for s in per_seed_phi]) for p in players}
    sds = {p: np.std([s[p] for s in per_seed_phi], ddof=1) if len(per_seed_phi) > 1 else 0.0 for p in players}
    fig, ax = plt.subplots(figsize=(6, 4))
    xs = np.arange(len(players))
    ax.bar(xs, [means[p] for p in players], yerr=[sds[p] for p in players], capsize=4, color="tab:blue", alpha=0.85)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels(players)
    ax.set_ylabel("raw Shapley value (NDCG@10)")
    ax.set_title(title)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def figure_alpha_curve(
    alphas: Sequence[float], scores: Sequence[float], selected: Optional[float], path: str
) -> str:
    """Figure 6: V_select alpha path with the locked selection marked."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(alphas, scores, marker="o")
    if selected is not None:
        idx = list(alphas).index(selected)
        ax.plot(selected, scores[idx], "r*", markersize=14, label="locked alpha")
    ax.set_xlabel("alpha")
    ax.set_ylabel("V_select NDCG@10")
    ax.set_title("Coarse alpha path (test reports only the locked setting)")
    if selected is not None:
        ax.legend()
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def figure_interaction_heatmap(
    pairs: Sequence[Any], values: Sequence[float], players: Sequence[str], path: str
) -> str:
    """Figure 4-style interaction matrix."""
    import numpy as np

    mat = np.zeros((len(players), len(players)))
    for (p, q), v in zip(pairs, values):
        i, j = players.index(p), players.index(q)
        mat[i, j] = mat[j, i] = v
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(mat, cmap="coolwarm", vmin=-max(abs(mat).max(), 1e-9), vmax=max(abs(mat).max(), 1e-9))
    ax.set_xticks(range(len(players)))
    ax.set_yticks(range(len(players)))
    ax.set_xticklabels(players)
    ax.set_yticklabels(players)
    fig.colorbar(im, ax=ax)
    ax.set_title("Context-averaged pair interactions")
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def figure_segment_lines(
    mu: Dict[str, Dict[str, float]], players: Sequence[str], path: str
) -> str:
    """Figure 5: raw view credit by Q1-Q4 segment."""
    fig, ax = plt.subplots(figsize=(6, 4))
    for p in players:
        ax.plot(range(1, 5), [mu[str(m)][p] for m in range(1, 5)], marker="o", label=p)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("training-history length quartile")
    ax.set_ylabel("raw Shapley credit (NDCG@10)")
    ax.set_xticks(range(1, 5))
    ax.legend()
    ax.set_title("Segment profiles (exact Game A)")
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def figure_removal_curve(
    orders: Dict[str, List[float]], path: str, title: str = "Removal faithfulness"
) -> str:
    """Figure 3: removal displays for Shapley/LOO/random orderings (few
    points; small-game diagnostic, not a scalable removal curve)."""
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, vals in orders.items():
        ax.plot(range(1, len(vals) + 1), vals, marker="o", label=name)
    ax.set_xlabel("views removed")
    ax.set_ylabel("test NDCG@10")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


__all__ = [
    "table_to_markdown",
    "coalition_value_table",
    "shapley_table",
    "interaction_table",
    "segment_table",
    "unconditional_transfer_table",
    "activation_conditioned_table",
    "figure_shapley_bars",
    "figure_alpha_curve",
    "figure_interaction_heatmap",
    "figure_segment_lines",
    "figure_removal_curve",
]
