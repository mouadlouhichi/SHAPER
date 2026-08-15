"""SHAPLEY / LOO / INTERACTIONS stages plus the frozen NLL/cosine severity
diagnostics.

Shapley: exact allocation from complete per-seed coalition tables (K=3),
efficiency residuals, per-user decomposition (npz), grand-LOO, contextual
marginals, monotonicity violations, coalition spread vs seed noise.

Interactions: exact Grabisch-Roubens for K=3 only; epsilon_pq proximity;
declaration rule (95% CI excludes zero AND |mean| >= delta_interaction).

Severity: frozen rec-only checkpoints only — canonical vs grid NLL increases
and cosine displacements, 10% matching rule, partial-alignment labels.
No additional coalition Shapley sweep is trained (locked scope).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from shaper import MAIN_PLAYERS  # noqa: E402
from shaper.augment import apply_view  # noqa: E402
from shaper.config import load_run_config  # noqa: E402
from shaper.data import FrozenData  # noqa: E402
from shaper.game import (  # noqa: E402
    CoalitionValueRecord,
    coalition_spread,
    contextual_marginals,
    epsilon_pq,
    grand_loo,
    load_coalition_table,
    monotonicity_violations,
    per_user_values,
)
from shaper.interactions import (  # noqa: E402
    all_pair_interactions,
    declare_interaction,
)
from shaper.shapley import (  # noqa: E402
    efficiency_residual,
    exact_shapley,
    per_user_shapley,
    shapley_of_mean_table,
)
from shaper.stats import seed_summary  # noqa: E402


def load_seed_tables(run, policy: str, seeds: List[int]) -> Dict[int, List[CoalitionValueRecord]]:
    tables: Dict[int, List[CoalitionValueRecord]] = {}
    for seed in seeds:
        path = os.path.join(run.root, "coalition_tables", f"{policy}_seed{seed}.json")
        if not os.path.exists(path):
            continue
        tables[seed] = load_coalition_table(path)
    return tables


def run_shapley(cfg, data, run, args, device):
    seeds = cfg.seeds["confirmatory_game_a"]
    out = {}
    for policy, policy_seeds in (("game_a", seeds), ("game_b", cfg.seeds["game_b"])):
        tables = load_seed_tables(run, policy, policy_seeds)
        if not tables:
            out[policy] = {"status": "NOT_EXECUTED", "reason": "no coalition tables found"}
            continue
        value_tables = []
        phi_per_seed = []
        residuals = []
        per_user_tables = []
        loo_per_seed = []
        marginals_per_seed = []
        violations = []
        for seed in policy_seeds:
            if seed not in tables:
                continue
            records = tables[seed]
            values = {rec.coalition: rec.value_ndcg for rec in records}
            value_tables.append(values)
            phi = exact_shapley(values, MAIN_PLAYERS)
            phi_per_seed.append(phi)
            residuals.append(efficiency_residual(phi, values[tuple(MAIN_PLAYERS)]))
            loo_per_seed.append(grand_loo(values, MAIN_PLAYERS))
            marginals_per_seed.append(
                {f"{p}|{'+'.join(c) or 'empty'}": v for (p, c), v in contextual_marginals(values, MAIN_PLAYERS).items()}
            )
            violations.append(monotonicity_violations(values, MAIN_PLAYERS))
            from shaper.game import load_per_user_table

            pu = load_per_user_table(
                os.path.join(run.root, "coalition_tables", f"{policy}_seed{seed}.json")
            )
            if pu is not None:
                per_user_tables.append(pu)
        phi_mean = {p: float(np.mean([s[p] for s in phi_per_seed])) for p in MAIN_PLAYERS}
        phi_of_mean = shapley_of_mean_table(value_tables, MAIN_PLAYERS)
        if per_user_tables:
            per_user_phi = per_user_shapley(per_user_tables, MAIN_PLAYERS)
            per_user_consistency = {
                p: float(np.mean(per_user_phi[p])) for p in MAIN_PLAYERS
            }
        else:
            per_user_consistency = {p: None for p in MAIN_PLAYERS}
        out[policy] = {
            "label": "exact Shapley (complete K=3 table)" if policy == "game_a"
            else "exact Game-B Shapley (policy sensitivity, seeds 2001-2003)",
            "seeds": list(tables.keys()),
            "per_seed_phi": phi_per_seed,
            "mean_phi": phi_mean,
            "shapley_of_mean_table": phi_of_mean,
            "max_efficiency_residual": float(max(residuals)),
            "efficiency_residuals": residuals,
            "per_user_consistency": per_user_consistency,
            "max_per_user_consistency_gap": {
                p: (abs(phi_mean[p] - per_user_consistency[p]) if per_user_consistency[p] is not None else None)
                for p in MAIN_PLAYERS
            },
            "loo_per_seed": loo_per_seed,
            "contextual_marginals_per_seed": marginals_per_seed,
            "monotonicity_violations_per_seed": violations,
            "coalition_spread": [coalition_spread(t) for t in value_tables],
            "note": ("Game B never replaces Game A, is never pooled with it, and "
                     "never drives RQ4 or seed expansion"),
        }
        run.write_json("shapley", f"{policy}_shapley.json", out[policy])
    return out


def run_loo(cfg, data, run, args, device):
    seeds = cfg.seeds["confirmatory_game_a"]
    tables = load_seed_tables(run, "game_a", seeds)
    out = {}
    for seed, records in tables.items():
        values = {rec.coalition: rec.value_ndcg for rec in records}
        loo = grand_loo(values, MAIN_PLAYERS)
        marginals = contextual_marginals(values, MAIN_PLAYERS)
        out[seed] = {
            "grand_loo": loo,
            "contextual_marginals": {f"{p}|{'+'.join(c) or 'empty'}": v for (p, c), v in marginals.items()},
            "note": "LOO is a grand-coalition-context baseline; never equated with Shapley",
        }
    run.write_json("shapley", "loo.json", out)
    return out


def run_interactions(cfg, data, run, args, device):
    seeds = cfg.seeds["confirmatory_game_a"]
    tables = load_seed_tables(run, "game_a", seeds)
    out = {}
    if not tables:
        out["status"] = "NOT_EXECUTED"
        return out
    pairs = [(p, q) for i, p in enumerate(MAIN_PLAYERS) for q in MAIN_PLAYERS[i + 1:]]
    per_seed = {seed: {} for seed in tables}
    for seed, records in tables.items():
        values = {rec.coalition: rec.value_ndcg for rec in records}
        for pair in pairs:
            per_seed[seed][pair] = all_pair_interactions(values, MAIN_PLAYERS)[pair]
    declarations = {}
    means = {}
    cis = {}
    for pair in pairs:
        vals = [per_seed[s][pair] for s in tables]
        summ = seed_summary(vals)
        means[pair] = summ["mean"]
        cis[pair] = summ["ci"]
        declarations[pair] = declare_interaction(
            vals, cfg.statistics["thresholds"]["delta_interaction"], ci95=summ["ci"]
        )
    eps = {}
    for seed, records in tables.items():
        values = {rec.coalition: rec.value_ndcg for rec in records}
        for pair in pairs:
            eps.setdefault(pair, []).append(epsilon_pq(values, MAIN_PLAYERS, *pair))
    out = {
        "label": "exact Grabisch-Roubens (K=3 complete tables)",
        "pairs": [list(p) for p in pairs],
        "per_seed": {str(s): {f"{p}-{q}": v for (p, q), v in d.items()} for s, d in per_seed.items()},
        "mean_interactions": {f"{p}-{q}": v for (p, q), v in means.items()},
        "ci95": {f"{p}-{q}": list(v) for (p, q), v in cis.items()},
        "declarations": {f"{p}-{q}": d for (p, q), d in declarations.items()},
        "epsilon_pq": {f"{p}-{q}": float(np.mean(v)) for (p, q), v in eps.items()},
        "delta_interaction": cfg.statistics["thresholds"]["delta_interaction"],
    }
    run.write_json("interactions", "exact_k3_interactions.json", out)
    return out


# --------------------------------------------------------------------------
# Frozen severity diagnostics (NLL / cosine) — no coalition Shapley sweep
# --------------------------------------------------------------------------

def _corrupt_prefixes(prefix: torch.Tensor, view: str, mask_id: int, rng_seed: int,
                      gamma: float = 0.2, beta: float = 0.2, eta: Optional[float] = None,
                      protect_last: int = 0) -> torch.Tensor:
    import random as _random

    from shaper.provenance import key_int

    out = prefix.clone()
    for i in range(prefix.shape[0]):
        seq = [int(x) for x in prefix[i].tolist() if x != 0]
        rng = _random.Random(key_int(rng_seed, i, view, gamma, beta, eta, protect_last, salt="severity"))
        aug, _ = apply_view(view, seq, rng, mask_id=mask_id, gamma=gamma, beta=beta, eta=eta,
                            protect_last=protect_last)
        padded = [0] * (prefix.shape[1] - len(aug)) + aug
        out[i] = torch.tensor(padded)
    return out


def _frozen_nll(model, inputs, device, k):
    from shaper.metrics import evaluate_model

    res = evaluate_model(model, inputs, k, device=device)
    return res.nll


def _frozen_cosine(model, inputs, corrupted, device):
    from shaper.backbone import build_model  # noqa: F401

    model.eval()
    model.to(device)
    with torch.no_grad():
        z_orig = model.contrastive_forward(inputs["prefix"].to(device))
        z_corr = model.contrastive_forward(corrupted.to(device))
        cos = torch.nn.functional.cosine_similarity(z_orig, z_corr, dim=-1)
    return float(cos.mean().item())


def severity_calibration(cfg, data, run, checkpoint_paths, device):
    """Frozen rec-only NLL/cosine severity calibration on V_tune."""
    from shaper.backbone import build_model
    from shaper.checkpoints import load_coalition_checkpoint
    from shaper.metrics import evaluate_model

    grids = {
        "crop": cfg.statistics["severity"]["crop_eta_lows"],
        "mask": cfg.statistics["severity"]["mask_gammas"],
        "reorder": cfg.statistics["severity"]["reorder_betas"],
    }
    canonical = {"crop": 0.5, "mask": 0.2, "reorder": 0.2}
    mask_id = data.n_items + 1
    inputs = data.eval_inputs("tune")

    models = []
    for path in checkpoint_paths:
        payload = load_coalition_checkpoint(path)
        model = build_model(cfg, data.n_items)
        model.load_state_dict(payload["model"])
        model.to(device)
        models.append(model)

    def param_setting(view: str, value: float) -> Dict[str, Optional[float]]:
        if view == "crop":
            return {"eta": value}
        if view == "mask":
            return {"gamma": value}
        return {"beta": value}

    out = {}
    for metric in ("nll", "cosine"):
        base = []
        for model in models:
            if metric == "nll":
                base.append(_frozen_nll(model, inputs, device, cfg.evaluation["k"]))
            else:
                base.append(_frozen_cosine(model, inputs, inputs["prefix"], device))
        base_mean = float(np.mean(base))
        view_results = {}
        for view in ("crop", "mask", "reorder"):
            canonical_vals = []
            for model in models:
                corrupted = _corrupt_prefixes(inputs["prefix"], view, mask_id, rng_seed=901,
                                              gamma=canonical["mask"], beta=canonical["reorder"],
                                              eta=canonical["crop"])
                if metric == "nll":
                    canonical_vals.append(_frozen_nll(model, {"prefix": corrupted, "target": inputs["target"], "exclusion": inputs["exclusion"]}, device, cfg.evaluation["k"]) - base_mean)
                else:
                    canonical_vals.append(base_mean - _frozen_cosine(model, inputs, corrupted, device))
            d_canonical = float(np.mean(canonical_vals))
            grid_vals = {}
            for value in grids[view]:
                vals = []
                for model in models:
                    p = param_setting(view, value)
                    corrupted = _corrupt_prefixes(
                        inputs["prefix"], view, mask_id, rng_seed=901,
                        gamma=p.get("gamma", canonical["mask"]),
                        beta=p.get("beta", canonical["reorder"]),
                        eta=p.get("eta", canonical["crop"]),
                    )
                    if metric == "nll":
                        vals.append(_frozen_nll(model, {"prefix": corrupted, "target": inputs["target"], "exclusion": inputs["exclusion"]}, device, cfg.evaluation["k"]) - base_mean)
                    else:
                        vals.append(base_mean - _frozen_cosine(model, inputs, corrupted, device))
                grid_vals[value] = float(np.mean(vals))
            target = d_canonical
            # closest grid setting; ties toward the canonical parameter
            best = min(grids[view], key=lambda v: (abs(grid_vals[v] - target),
                                                   0 if v == canonical[view] else 1))
            residual = abs(grid_vals[best] - target) / max(abs(target), 1e-8)
            view_results[view] = {
                "canonical_increase": d_canonical,
                "grid_increases": grid_vals,
                "selected": best,
                "achieved": grid_vals[best],
                "residual": residual,
            }
        target_median = float(np.median([v["canonical_increase"] for v in view_results.values()]))
        ok = all(v["residual"] <= cfg.statistics["severity"]["match_tolerance"] for v in view_results.values())
        label = ("NLL-matched corruption severity" if metric == "nll" else "cosine-matched displacement")
        if not ok:
            label = cfg.statistics["severity"]["partial_label_nll" if metric == "nll" else "partial_label_cosine"]
        out[metric] = {
            "target": target_median,
            "views": view_results,
            "matched": ok,
            "label": label,
            "residuals": {v: view_results[v]["residual"] for v in view_results},
            "note": "frozen rec-only checkpoints; no additional coalition Shapley sweep",
        }
    # protect-last-1/2 sweeps: frozen rec-only diagnostics only (locked scope)
    protect_diag: Dict[str, Any] = {}
    for protect in (0, 1, 2):
        vals_nll = []
        vals_cos = []
        for model in models:
            corrupted = _corrupt_prefixes(
                inputs["prefix"], "mask", mask_id, rng_seed=901,
                gamma=canonical["mask"], beta=canonical["reorder"],
                eta=canonical["crop"], protect_last=protect,
            )
            vals_nll.append(_frozen_nll(model, {"prefix": corrupted, "target": inputs["target"], "exclusion": inputs["exclusion"]}, device, cfg.evaluation["k"]) - base_mean)
            vals_cos.append(base_mean - _frozen_cosine(model, inputs, corrupted, device))
        protect_diag[f"protect_last_{protect}"] = {
            "nll_increase_mean": float(np.mean(vals_nll)),
            "cosine_displacement_mean": float(np.mean(vals_cos)),
        }
    out["protect_last_diagnostics"] = {
        "sweep": protect_diag,
        "canonical": "protect_last_0 (main protocol does NOT protect the last two positions)",
        "scope": "frozen rec-only NLL/cosine only; no coalition retraining",
    }
    run.write_json("metrics", "severity_diagnostics.json", out)
    return out


def main() -> int:
    from _cli import base_parser, frozen_data, resolve_device

    p = base_parser("SHAPER game aggregation: Shapley / LOO / interactions / severity")
    p.add_argument("--stage", required=True, choices=["shapley", "loo", "interactions", "severity"])
    p.add_argument("--rec-only-checkpoints", default=None,
                   help="comma-separated checkpoint paths for the frozen severity diagnostics")
    args = p.parse_args()

    cfg = load_run_config(args.dataset)
    data = frozen_data(cfg)
    device = resolve_device(args)

    from shaper.artifacts import RunDirectory

    run_id = args.run_id or f"game-{cfg.dataset}-{int(time.time())}"
    run = RunDirectory(run_id, results_root=cfg.paths["results"]).create(cfg, resume=True)
    t0 = time.time()
    stage_name = {"shapley": "SHAPLEY", "loo": "LOO", "interactions": "INTERACTIONS",
                  "severity": "SEVERITY_DIAGNOSTICS"}[args.stage]
    run.logger.stage_start(stage_name, dataset=cfg.dataset)
    if args.stage == "shapley":
        out = run_shapley(cfg, data, run, args, device)
    elif args.stage == "loo":
        out = run_loo(cfg, data, run, args, device)
    elif args.stage == "interactions":
        out = run_interactions(cfg, data, run, args, device)
    else:
        paths = (args.rec_only_checkpoints or "").split(",")
        if not paths or not paths[0]:
            raise SystemExit("--rec-only-checkpoints required for severity diagnostics")
        out = severity_calibration(cfg, data, run, paths, device)
    run.stage_status(stage_name, "completed", wall_seconds=round(time.time() - t0, 2))
    run.logger.stage_end(stage_name, dataset=cfg.dataset)
    print(json.dumps(out, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
