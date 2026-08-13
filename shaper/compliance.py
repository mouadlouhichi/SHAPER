"""Spec-compliance registry and report generation (prompt sections 66-67).

Every protocol requirement is mapped to its source section, implementing
file, verifying test, and a status resolver. Statuses are NEVER fabricated:
PASS requires evidence (a passing test or a recorded run artifact);
NOT_EXECUTED marks requirements whose verification needs study execution
that has not happened; NOT_APPLICABLE marks items outside this checkout.

Engineering-detail module (like shaper.config / shaper.cost); holds no
scientific logic.
"""

from __future__ import annotations

import json
import os
from typing import Any, Callable, Dict, List, Optional

# (requirement, source section, implementation file, test, status_resolver)
# status_resolver: None -> PASS iff test collected/passed; otherwise a
# callable(evidence)->status.
REQUIREMENTS: List[Dict[str, Any]] = [
    # ---- determinism / environment -----------------------------------------
    {"req": "torch.use_deterministic_algorithms(True) with cuDNN benchmarking disabled and deterministic cuDNN enabled",
     "source": "spec A.2", "file": "shaper/schedules.py::configure_determinism", "test": "tests/protocol/test_preflight_and_misc.py"},
    {"req": "deterministic exception recorded, never silently suppressed; nondeterministic kernels labeled",
     "source": "spec A.2", "file": "shaper/schedules.py::configure_determinism", "test": "tests/protocol/test_preflight_and_misc.py"},
    {"req": "nondeterminism floor: repeat canonical Game-A grand coalition seed 2001; marginals below the floor labeled indistinguishable from execution nondeterminism",
     "source": "spec A.2", "file": "scripts/run_game.py", "test": "tests/unit/test_shapley.py"},
    {"req": "environment record: versions, lock file, OS, CPU, GPU, CUDA, torch, commit, dirty state, deterministic flags, peak GPU memory",
     "source": "spec A.2", "file": "shaper/provenance.py::environment_record", "test": "tests/unit/test_provenance_logging_artifacts_cost_report.py"},

    # ---- seeds ---------------------------------------------------------------
    {"req": "fixed seed registry 901-903 / 1001-1002 / 2001-2005 / 2006-2010 / 2001-2003 / 3001-3005 / 4001",
     "source": "spec A.2", "file": "configs/seeds.yaml", "test": "tests/protocol/test_protocol_locks.py"},
    {"req": "no unlisted confirmatory seed without a timestamped amendment",
     "source": "spec A.2", "file": "configs/seeds.yaml", "test": "tests/protocol/test_protocol_locks.py"},

    # ---- data ----------------------------------------------------------------
    {"req": "ML-1M: rating>=4 positive; iterative 5-core to fixed point; never report raw 1,000,209 count as processed",
     "source": "spec A.3", "file": "shaper/data.py", "test": "tests/unit/test_data.py"},
    {"req": "Beauty: every retained review positive; iterative 5-core",
     "source": "spec A.3", "file": "shaper/data.py", "test": "tests/unit/test_data.py"},
    {"req": "temporal leave-one-out; deterministic ties by raw row order; salted-hash robustness ordering persisted",
     "source": "spec A.3", "file": "shaper/data.py::split_leave_one_out", "test": "tests/unit/test_data.py"},
    {"req": "quartile edges frozen on pre-truncation training length over the full validation-eligible pool BEFORE role assignment",
     "source": "spec A.3", "file": "shaper/data.py::build_dataset_artifact", "test": "tests/unit/test_data.py"},
    {"req": "20/60/20 V_tune/V_game/V_select via stable persisted salted hash, stratified by dataset and quartiles",
     "source": "spec A.3", "file": "shaper/data.py::assign_roles", "test": "tests/unit/test_data.py"},
    {"req": "frozen artifact: maps, raw_row_id, tensors, lengths, quartiles, hash algorithm + salt, roles, exclusion masks, config/data hashes",
     "source": "spec A.3", "file": "shaper/data.py::build_dataset_artifact", "test": "tests/unit/test_data.py"},
    {"req": "no downstream stage reconstructs splits",
     "source": "spec A.3", "file": "shaper/data.py::FrozenData", "test": "tests/unit/test_data.py"},
    {"req": "full-catalog evaluation; prefix filtering; repeated targets retained and counted; ties by item ID",
     "source": "spec A.3", "file": "shaper/metrics.py", "test": "tests/unit/test_metrics.py"},
    {"req": "DatasetStats emitted from finalized artifacts",
     "source": "spec A.3", "file": "shaper/data.py::DatasetStats", "test": "tests/unit/test_data.py"},

    # ---- augmentation --------------------------------------------------------
    {"req": "crop: U[0.5,1.0], keep>=2, n>=3 -> at most n-1, forced non-identity where possible",
     "source": "spec A.4", "file": "shaper/augment.py::augment_crop", "test": "tests/unit/test_augment.py"},
    {"req": "mask: gamma=0.2, valid positions only, no last-two protection in main protocol",
     "source": "spec A.4", "file": "shaper/augment.py::augment_mask", "test": "tests/unit/test_augment.py"},
    {"req": "reorder: beta=0.2, local span, min span 2, non-identity, identity retried",
     "source": "spec A.4", "file": "shaper/augment.py::augment_reorder", "test": "tests/unit/test_augment.py"},
    {"req": "dropout player: two extra keyed stochastic forwards; ordinary dropout active in every model; player never toggles it",
     "source": "spec A.4", "file": "shaper/augment.py::contrastive_dropout_pair", "test": "tests/unit/test_augment.py"},
    {"req": "no-op accounting: a_p=B_p/B; B_p<8 -> zero loss + event; no redistribution; denominators |C| (A) and K (B)",
     "source": "spec A.4", "file": "shaper/contrast.py::per_view_cl_loss", "test": "tests/unit/test_contrast.py"},
    {"req": "effective coefficient mass logged (mean/SD/P10/P50/P90 + gradient norm); never called total gradient magnitude",
     "source": "spec A.4", "file": "shaper/augment.py::effective_mass", "test": "tests/unit/test_contrast.py"},
    {"req": "crop starts uniform over inclusive support 0..n-keep",
     "source": "spec A.4", "file": "shaper/augment.py", "test": "tests/unit/test_augment.py"},

    # ---- backbone / training -------------------------------------------------
    {"req": "SASRec: padding 0, MASK=n_items+1, two causal blocks d=64/2 heads/FFN 256/dropout 0.2, final valid hidden state, projection 64-64-64 ReLU",
     "source": "spec A.5", "file": "shaper/backbone.py", "test": "tests/unit/test_backbone.py"},
    {"req": "recommendation: original history only, BCE next-item, one unseen negative per positive, full-catalog dot-product ranking, projection discarded",
     "source": "spec A.5", "file": "shaper/backbone.py", "test": "tests/unit/test_backbone.py"},
    {"req": "batch sizes 128 (ML-1M) / 256 (Beauty); one user history per example; every user once per epoch permutation",
     "source": "spec A.5", "file": "configs/*.yaml, shaper/data.py::train_batches", "test": "tests/protocol/test_schedules_protocol.py"},
    {"req": "AdamW(0.9,0.98) eps 1e-8 wd 1e-4, clip 1.0, 10% linear warmup, cosine decay to 0.1*lr; no per-coalition early stopping",
     "source": "spec A.5/A.6", "file": "shaper/training.py", "test": "tests/unit/test_training.py"},
    {"req": "common initialization cloned into every coalition; independent optimizers; never warm-started from grand",
     "source": "spec A.5/A.6", "file": "shaper/training.py::common_initialization", "test": "tests/protocol/test_coalition_hygiene.py"},
    {"req": "keyed RNG schedules: augmentation (seed,step,example,occurrence,view); negatives (seed,step,user,position); epoch (seed,epoch,dataset_hash); dropout (seed,step,purpose,view,pass)",
     "source": "spec A.4/A.5", "file": "shaper/schedules.py", "test": "tests/protocol/test_schedules_protocol.py"},
    {"req": "symmetric NT-Xent batch-normalized over 2B representations; rec negatives never in contrastive denominator",
     "source": "spec A.5/A.6", "file": "shaper/contrast.py::symmetric_nt_xent", "test": "tests/unit/test_contrast.py"},

    # ---- games ---------------------------------------------------------------
    {"req": "primary Game A: L_cl=|C|^-1 sum L_p; v(C)=NDCG_game(C)-NDCG_game(empty); v(empty)=0; 8 coalitions x 5 seeds",
     "source": "spec A.6/A.8", "file": "shaper/game.py, shaper/contrast.py", "test": "tests/unit/test_shapley.py"},
    {"req": "secondary Game B: L_cl=K^-1 sum L_p; exact only for seeds 2001-2003; only 6 intermediate coalitions; empty/grand reused; no expansion; never drives RQ4",
     "source": "spec A.6", "file": "shaper/game.py", "test": "tests/protocol/test_protocol_locks.py"},
    {"req": "exact Shapley from complete K=3 tables; efficiency sum(phi)=v(P) within tolerance; maximum residual reported",
     "source": "spec A.8", "file": "shaper/shapley.py", "test": "tests/unit/test_shapley.py"},
    {"req": "LOO_p = v(P)-v(P\\p); all contextual marginals persisted; LOO never equated with Shapley",
     "source": "spec A.8", "file": "shaper/game.py", "test": "tests/unit/test_shapley.py"},
    {"req": "perfect substitutability: zero grand-LOO, equal Shapley, corrected 3-player behavior; rejected half-pair-removal formula never used",
     "source": "spec A.8 / paper 3.5", "file": "shaper/interactions.py::perfect_substitutability_game", "test": "tests/unit/test_interactions.py"},
    {"req": "exact Grabisch-Roubens interactions; directional only if 95% CI excludes zero AND |mean|>=0.003 else INCONCLUSIVE",
     "source": "spec A.8", "file": "shaper/interactions.py", "test": "tests/unit/test_interactions.py"},
    {"req": "per-user utilities stored; linear decomposition phi=(1/|U|)sum phi_u; individual explanations not interpreted",
     "source": "spec A.8", "file": "shaper/game.py::per_user_values, shaper/shapley.py::per_user_shapley", "test": "tests/unit/test_shapley.py"},
    {"req": "NLL/cosine severity diagnostics frozen rec-only; no additional coalition Shapley sweep",
     "source": "spec A.4", "file": "scripts/run_game.py::severity_diagnostics", "test": "tests/protocol/test_protocol_locks.py"},
    {"req": "value-table cache keyed by dataset/seed/policy/coalition/config/data/recipe hashes; valid cache reused, stale rejected",
     "source": "spec section 30/53", "file": "shaper/checkpoints.py::validate_cache_keys", "test": "tests/unit/test_checkpoints.py"},

    # ---- K=4 MC --------------------------------------------------------------
    {"req": "K=4 Beauty Game A with dropout; seeds 3001-3005; antithetic permutation MC; <=10 unique models per seed; loud failure at #11",
     "source": "spec A.8", "file": "shaper/monte_carlo.py", "test": "tests/unit/test_monte_carlo.py"},
    {"req": "MC telescoping: sum of path marginals == v(P)-v(empty) within precision; antithetic averaging preserves it",
     "source": "spec A.8", "file": "shaper/monte_carlo.py::telescoping_check", "test": "tests/unit/test_monte_carlo.py"},
    {"req": "K=4 labeled Monte-Carlo approximate; ordering only if 95% MC half-width <= delta_phi else INCONCLUSIVE; no models added to rescue",
     "source": "spec A.8", "file": "shaper/monte_carlo.py::AntitheticPermutationMCShapley.aggregate", "test": "tests/unit/test_monte_carlo.py"},
    {"req": "no exact K=4 interactions from the incomplete table",
     "source": "spec A.8", "file": "shaper/interactions.py", "test": "tests/unit/test_interactions.py"},
    {"req": "permutation is frozen run state; resume never resamples it",
     "source": "spec section 52", "file": "shaper/monte_carlo.py::K4SeedResult", "test": "tests/resume/test_resume_cases.py"},

    # ---- RQ3 -------------------------------------------------------------------
    {"req": "RQ3 uses only exact K=3 Game-A; Q1-Q4 frozen pre-truncation length segments",
     "source": "spec A.11", "file": "shaper/segments.py", "test": "tests/protocol/test_protocol_locks.py"},
    {"req": "studentized mask trend + omnibus profile statistic; 10,000 user-level label permutations, same map across seeds",
     "source": "spec A.11/A.12", "file": "shaper/segments.py::label_permutation_test", "test": "tests/unit/test_segments.py"},

    # ---- RQ4 -------------------------------------------------------------------
    {"req": "SHAPER-Weight: positive transform, uniform fallback, w=(1-a)/K+a*q, alpha in {0,.25,.5,.75,1}, no additional denominator",
     "source": "spec A.10", "file": "shaper/adaptive.py, shaper/contrast.py::weighted_cl_loss", "test": "tests/unit/test_adaptive.py"},
    {"req": "activation: grand uplift CI above zero; highest-weight view positive >=4/5 (8/10); positively weighted views >=80% seeds; else NOT_ACTIVATED + rec-only fallback",
     "source": "spec A.12", "file": "shaper/adaptive.py::weight_activation_rule", "test": "tests/unit/test_adaptive.py"},
    {"req": "SHAPER-Select: sort by mean raw Game-A Shapley, tie order crop<mask<reorder<dropout, delta_phi no-action rule, reuse coalition model, never test-selected",
     "source": "spec A.10", "file": "shaper/adaptive.py::select_decision", "test": "tests/unit/test_adaptive.py"},
    {"req": "fair controls: rec-only, uniform, LOO weights, learned gates (LR 1e-2, entropy {0,.01,.1}, training-only), drop-lowest-LOO, random removal, 15-point simplex search, ten Dirichlet candidates, random controls; budgets recorded",
     "source": "spec A.10", "file": "shaper/baselines.py", "test": "tests/unit/test_baselines.py"},
    {"req": "final test only after all decisions locked; all eligible users; training+validation prefix; repeated-target rate reported",
     "source": "spec A.3/A.10", "file": "shaper/metrics.py, shaper/data.py::test_inputs", "test": "tests/protocol/test_preflight_and_misc.py"},
    {"req": "Tables 7A unconditional and 7B activation-conditioned always produced; fallback not counted as adaptive success",
     "source": "spec A.10", "file": "shaper/report.py", "test": "tests/unit/test_provenance_logging_artifacts_cost_report.py"},

    # ---- statistics -----------------------------------------------------------
    {"req": "seeds are the confirmatory unit; hierarchical intervals; Holm on seed-level paired effects; users never independent runs",
     "source": "spec A.12", "file": "shaper/stats.py", "test": "tests/unit/test_stats.py"},
    {"req": "descriptive user analyses: Wilcoxon, rank-biserial, Cliff's delta, bootstrap",
     "source": "spec A.12", "file": "shaper/stats.py", "test": "tests/unit/test_stats.py"},
    {"req": "practical thresholds fixed at 0.003 (phi/interaction/action); never scaled by grand uplift; changes only pre-confirmatory and direction-independent",
     "source": "spec A.12", "file": "configs/statistics.yaml", "test": "tests/protocol/test_protocol_locks.py"},
    {"req": "direction-blind extension rules: Game A (phi CI or grand-uplift CI half-width), Weight-vs-uniform; Game B/diagnostics/K=4 never expand",
     "source": "spec A.12", "file": "shaper/power.py", "test": "tests/unit/test_power.py"},
    {"req": "locked Holm families: 6 intervention comparisons; 6 mask tests (NDCG+NLL); 3 interaction pairs per dataset",
     "source": "spec A.12", "file": "configs/statistics.yaml::holm", "test": "tests/protocol/test_protocol_locks.py"},
    {"req": "Appendix J planning half-width formula + hierarchical bootstrap; pilot-informed MDE tables",
     "source": "spec A.12", "file": "shaper/power.py", "test": "tests/unit/test_power.py"},

    # ---- infrastructure ---------------------------------------------------------
    {"req": "atomic checkpoints (temp -> flush -> fsync -> rename); checkpoint_manifest.json; corrupted checkpoint rejected and previous valid restored",
     "source": "spec sections 49-51", "file": "shaper/checkpoints.py", "test": "tests/unit/test_checkpoints.py"},
    {"req": "resume cases: crashed seed, crashed mid-coalition, crashed between stages, partial K=4 (same permutation), corrupted checkpoint",
     "source": "spec section 51", "file": "shaper/checkpoints.py, shaper/monte_carlo.py", "test": "tests/resume/test_resume_cases.py"},
    {"req": "worker invariance: num_workers=0 vs >0 -> identical augmentation hashes, negative schedules, epoch hashes, results within declared nondeterminism",
     "source": "spec section 55", "file": "shaper/schedules.py", "test": "tests/integration/test_worker_invariance.py"},
    {"req": "coalition-order invariance: shuffled enumeration -> unchanged schedules and results",
     "source": "spec section 56", "file": "shaper/schedules.py", "test": "tests/integration/test_order_invariance.py"},
    {"req": "budget estimator before expensive execution; safe CLI; no full-study default",
     "source": "spec sections 60-61", "file": "shaper/cost.py, scripts/run_all.py", "test": "tests/unit/test_provenance_logging_artifacts_cost_report.py"},
    {"req": "structured JSONL logging with canonical event schema",
     "source": "spec section 58", "file": "shaper/logging_utils.py", "test": "tests/unit/test_provenance_logging_artifacts_cost_report.py"},
    {"req": "compute tracking: measured wall/GPU time, memory, models, steps, cache hits/misses, Shapley/eval time",
     "source": "spec section 59", "file": "shaper/artifacts.py::ComputeTracker", "test": "tests/unit/test_provenance_logging_artifacts_cost_report.py"},
    {"req": "synthetic end-to-end: data->recipe->Game A->Shapley->Game B->K=4 MC->LOO->interactions->segments->Weight->Select->statistics->report->checkpoint->crash->resume",
     "source": "spec section 65", "file": "tests/integration/test_synthetic_end_to_end.py", "test": "tests/integration/test_synthetic_end_to_end.py"},
    {"req": "failure injection: crash, corrupt checkpoint, config-hash change, data-manifest change, worker count, coalition order; invalid artifacts rejected, completed work preserved",
     "source": "spec section 54", "file": "tests/integration/test_failure_injection.py", "test": "tests/integration/test_failure_injection.py"},
    {"req": "paper/code consistency document generated and checked",
     "source": "spec section 66", "file": "docs/paper_implementation_consistency.md, scripts/validate_run.py", "test": "tests/protocol/test_protocol_locks.py"},
    {"req": "results directory layout and reports (summary, compliance, reproducibility)",
     "source": "spec section 57", "file": "shaper/artifacts.py::RunDirectory", "test": "tests/unit/test_provenance_logging_artifacts_cost_report.py"},
    {"req": "notebook run_all.ipynb exists, calls package functions, restartable/resume-aware",
     "source": "spec sections 62-63", "file": "notebooks/run_all.ipynb", "test": "tests/protocol/test_preflight_and_misc.py"},

    # ---- no-go conditions ---------------------------------------------------------
    {"req": "no-go: identical coalition rankings / efficiency failure / per-user consistency failure / split leakage / excluded-view leakage / projection-only surrogate / grand warm-start / unequal budgets / test affects selection / invalid redundancy formula",
     "source": "spec section 47 / A.9", "file": "scripts/validate_run.py", "test": "tests/integration/test_synthetic_end_to_end.py"},

    # ---- study execution gates (status resolved from run manifests) ---------------
    {"req": "staged preregistration: PREFLIGHT->DATA->VALIDATE->RECIPE->PILOTS->AMENDMENT->ARCHIVE FREEZE->CONFIRMATORY; no confirmatory model before the archive",
     "source": "spec B.7", "file": "scripts/run_all.py", "test": None, "execution_gate": True,
     "gate_stage": "ARCHIVE_FREEZE"},
    {"req": "primary Game A executed (8 coalitions x 5 seeds x 2 datasets)",
     "source": "spec A.15", "file": "scripts/train_coalitions.py", "test": None, "execution_gate": True,
     "gate_stage": "PRIMARY_GAME_A"},
    {"req": "Game B executed (6 intermediate x 3 seeds x 2 datasets)",
     "source": "spec A.15", "file": "scripts/train_coalitions.py", "test": None, "execution_gate": True,
     "gate_stage": "SECONDARY_GAME_B"},
    {"req": "K=4 Beauty MC executed (<=10 models x 5 seeds)",
     "source": "spec A.15", "file": "scripts/train_coalitions.py", "test": None, "execution_gate": True,
     "gate_stage": "K4_BEAUTY_MC"},
    {"req": "final intervention test executed on locked test users",
     "source": "spec A.10", "file": "scripts/run_interventions.py", "test": None, "execution_gate": True,
     "gate_stage": "FINAL_INTERVENTION_TEST"},
]


def resolve_status(
    req: Dict[str, Any],
    test_results: Dict[str, bool],
    stage_manifest: Optional[Dict[str, Any]] = None,
) -> str:
    """PASS / FAIL / NOT_EXECUTED / NOT_APPLICABLE — never fabricated."""
    if req.get("execution_gate"):
        if stage_manifest is None:
            return "NOT_EXECUTED"
        # find any stage marker for the gates
        gate = req.get("gate_stage")
        if gate and gate in stage_manifest:
            return "PASS" if stage_manifest[gate].get("status") in ("completed", "executed") else "FAIL"
        return "NOT_EXECUTED"
    test = req.get("test")
    if test is None:
        return "NOT_EXECUTED"
    status = test_results.get(test)
    if status is None:
        return "NOT_EXECUTED"
    return "PASS" if status else "FAIL"


def generate_compliance_report(
    test_results: Dict[str, bool],
    run_id: str,
    stage_manifest: Optional[Dict[str, Any]] = None,
    evidence: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    evidence = evidence or {}
    rows = []
    for i, req in enumerate(REQUIREMENTS):
        status = resolve_status(req, test_results, stage_manifest)
        rows.append(
            {
                "id": f"R{i + 1:03d}",
                "requirement": req["req"],
                "source_section": req["source"],
                "implementation_file": req["file"],
                "test": req.get("test"),
                "status": status,
                "evidence": evidence.get(req["req"], req.get("test") or ""),
            }
        )
    counts = {s: sum(1 for r in rows if r["status"] == s) for s in ("PASS", "FAIL", "NOT_EXECUTED", "NOT_APPLICABLE")}
    return {
        "run_id": run_id,
        "generated_at": None,
        "counts": counts,
        "requirements": rows,
        "note": (
            "PASS requires executed evidence; NOT_EXECUTED separates implementation "
            "verification from experiment execution; hypotheses are never claimed "
            "confirmed merely because software passes."
        ),
    }


def render_compliance_markdown(report: Dict[str, Any]) -> str:
    lines = ["# SHAPER spec-compliance report", "", f"- run_id: `{report['run_id']}`",
             f"- counts: " + ", ".join(f"{k}={v}" for k, v in report["counts"].items()), ""]
    lines.append("| id | requirement | source | implementation | test | status | evidence |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in report["requirements"]:
        lines.append(
            f"| {r['id']} | {r['requirement']} | {r['source_section']} | {r['implementation_file']} "
            f"| {r['test'] or ''} | {r['status']} | {r['evidence']} |"
        )
    lines.append("")
    lines.append(report["note"])
    return "\n".join(lines) + "\n"


def write_compliance_report(
    report: Dict[str, Any], json_path: str, md_path: str
) -> None:
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write(render_compliance_markdown(report))


__all__ = [
    "REQUIREMENTS",
    "generate_compliance_report",
    "render_compliance_markdown",
    "write_compliance_report",
]
