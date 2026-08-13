# Paper ↔ Implementation Consistency Report

> Generated against the frozen protocol documents
> `specs/SHAPER_Implementation_Spec.md` and `specs/SHAPER_Paper_Structure.md`.
> This file is checked automatically by `tests/protocol/test_protocol_locks.py`.
> Status: **implementation verified** (synthetic end-to-end + full test suite).
> Experiment execution status is recorded per stage in `results/runs/<run_id>/manifest.json`.

The paper and the code describe the SAME experiment. Every locked quantity below
exists in one config file, is enforced by one code module, and is pinned by one
test. Where the code appears to deviate from the paper, the difference is an
engineering detail documented in the "Documented engineering choices" section
at the bottom.

## 1. Primary estimand — exact K=3 Game A

| Item | Paper | Code | Test |
|---|---|---|---|
| Players | crop, mask, reorder | `shaper/__init__.py::MAIN_PLAYERS` | `tests/protocol/test_protocol_locks.py` |
| Coalitions per seed | all 8 (complete 2^3 table) | `shaper/game.py::all_coalitions` | `tests/protocol/test_protocol_locks.py::test_game_b_scope_six_intermediate_only` |
| Objective | `L_cl(C) = |C|^{-1} sum_p L_cl^p` | `shaper/contrast.py::coalition_cl_loss` (policy `game_a`) | `tests/unit/test_contrast.py::test_budget_policy_identity` |
| Confirmatory seeds | 2001–2005 | `configs/seeds.yaml` | `tests/protocol/test_protocol_locks.py::test_seed_registry_locked` |
| Conditional extension | 2006–2010, direction-blind rule only | `shaper/power.py::evaluate_game_a_extension_trigger` | `tests/unit/test_power.py` |
| Characteristic function | `v(C) = NDCG@10_game(C) − NDCG@10_game(empty)`, `v(∅)=0` | `shaper/game.py::apply_baseline_subtraction` | `tests/unit/test_game.py` (via `tests/unit/test_shapley.py`) |
| Allocation | exact Shapley over the complete table | `shaper/shapley.py::exact_shapley` | `tests/unit/test_shapley.py` |
| Efficiency | `sum(phi) = v(P)` within tolerance; max residual reported | `shaper/shapley.py::efficiency_residual` | `tests/unit/test_shapley.py::test_efficiency_on_random_tables` |

## 2. Secondary Game B (policy sensitivity — NOT co-primary)

| Item | Paper | Code |
|---|---|---|
| Objective | `L_cl(C) = K^{-1} sum_p L_cl^p` | `shaper/contrast.py::coalition_cl_loss` (policy `game_b`) |
| Seeds | 2001–2003 ONLY (no expansion ever) | `configs/seeds.yaml::game_b` |
| Coalitions | only the 6 distinct intermediate coalitions; empty/grand REUSED from Game A | `scripts/train_coalitions.py::stage_game_b` |
| Usage | descriptive sensitivity only; never pooled with A; never drives RQ4 | `scripts/run_game.py::run_shapley` (separate output block), `scripts/run_interventions.py` consumes `game_a_shapley.json` only |

## 3. K=4 Beauty extension (Monte-Carlo approximate)

| Item | Paper | Code |
|---|---|---|
| Dataset/policy | Beauty, Game A only | `configs/scope.yaml::beauty_k4_game_a_mc` |
| Fourth player | dropout (two extra keyed stochastic forwards; ordinary dropout active in every model) | `shaper/augment.py::contrastive_dropout_pair` |
| Seeds | 3001–3005 | `configs/seeds.yaml` |
| Procedure | one uniform permutation per seed + its reverse (antithetic); all unique prefix coalitions on both paths; all four P\{p} grand-LOO coalitions; deduplicated | `shaper/monte_carlo.py::required_coalitions` |
| Hard cap | ≤ 10 unique trained models per seed; LOUD failure at #11 | `shaper/monte_carlo.py::MAX_UNIQUE_MODELS_PER_SEED` |
| Telescoping | `sum_p marginal_p = v(P) − v(empty)` preserved by antithetic averaging | `shaper/monte_carlo.py::telescoping_check` |
| Label | "Monte-Carlo approximate" — never exact | `AntitheticPermutationMCShapley.aggregate` |
| Ordering rule | interpret only if 95% MC half-width ≤ `delta_phi`; else INCONCLUSIVE; no models added to rescue | `aggregate(..., delta_phi)` |
| Interactions | exact K=4 Grabisch–Roubens FORBIDDEN (incomplete table raises) | `shaper/interactions.py::grabisch_roubens_interaction` |

## 4. Frozen severity diagnostics (no Shapley sweep)

| Item | Paper | Code |
|---|---|---|
| NLL/cosine calibration | frozen rec-only checkpoints (seeds 901–903) on V_tune | `scripts/run_game.py::severity_calibration` |
| Grids | crop η ∈ {0.5,0.6,0.7,0.8} (upper 1.0); mask γ ∈ {0.1..0.4}; reorder β ∈ {0.1..0.4} | `configs/statistics.yaml::severity` |
| Matching rule | every selected view within 10% of the target; else "partial NLL severity alignment" / "partial cosine alignment" | `severity_calibration` |
| Ranking confirmation | pilot seed 1002, three matched singletons + matched grand only | `configs/scope.yaml::ranking_confirmation_aligned_severity` |

## 5. Seeds, roles, thresholds

| Item | Paper | Code |
|---|---|---|
| Full seed registry | recipe/alpha 901, 902, 903; pilots 1001, 1002; confirmatory Game A 2001, 2002, 2003, 2004, 2005; extension 2006, 2007, 2008, 2009, 2010; Game B + cosine diagnostic 2001, 2002, 2003; K=4 Beauty MC 3001, 3002, 3003, 3004, 3005; cached-adapter 4001 | `configs/seeds.yaml` (hash-pinned in `tests/regression`) |
| Validation roles | 20%/60%/20% (V_tune / V_game / V_select), stable persisted salted hash stratified by dataset and frozen quartiles | `shaper/data.py::assign_roles` |
| Quartile basis | pre-truncation training-history length, frozen before role assignment | `shaper/data.py::build_dataset_artifact` |
| Thresholds | `delta_phi = delta_interaction = delta_action = 0.003` (never scaled by grand uplift) | `configs/statistics.yaml::thresholds` |

## 6. Model, training, data

| Item | Paper | Code |
|---|---|---|
| Backbone | SASRec-style; padding id 0; `[MASK] = n_items+1`; two causal blocks, d=64, 2 heads, FFN=256, dropout=0.2; final valid hidden state; projection 64→64→64 (ReLU), discarded for ranking | `shaper/backbone.py` |
| Recommendation | original history only; BCE next-item, all valid positions, mean per user then across users; one uniformly sampled unseen negative per positive; full-catalog dot-product ranking | `shaper/backbone.py::recommendation_loss`, `shaper/metrics.py` |
| Batch sizes | ML-1M 128, Beauty 256; one user history per example; every user once per keyed epoch permutation | `configs/ml1m.yaml`, `configs/beauty.yaml`, `shaper/data.py::train_batches` |
| Optimizer | AdamW(betas=(0.9,0.98), eps=1e-8, wd=1e-4), clip 1.0, 10% linear warmup, cosine decay to 0.1·lr, fixed step budget, NO per-coalition early stopping | `shaper/training.py` |
| Common init | one seed-specific initial state cloned into every coalition; independent optimizers; never warm-started from the grand coalition | `shaper/training.py::common_initialization` |
| Data | ML-1M rating≥4 positive; Beauty all positive; iterative 5-core to fixed point; temporal LOO; ties by raw row order (primary) / salted hash (robustness); raw counts never reported as processed counts | `shaper/data.py` |

## 7. RNG schedules (worker/order-invariant)

| Key | Paper | Code |
|---|---|---|
| Augmentation | (seed, optimizer_step, global_example_id, occurrence, view) | `shaper/schedules.py::augmentation_rng` |
| Recommendation negatives | (seed, optimizer_step, global_user_id, target_position) | `shaper/schedules.py::recommendation_negatives` |
| Epoch | (seed, epoch, dataset_hash) | `shaper/schedules.py::epoch_permutation` |
| Dropout | (seed, optimizer_step, purpose, view, pass_index) | `shaper/schedules.py::dropout_generator` |

## 8. No-op semantics and effective mass

| Item | Paper | Code |
|---|---|---|
| `a_p = B_p / B`; `B_p < 8 → L_cl^p = 0` + insufficient-pair event; no redistribution; denominators `|C|` (A) and `K` (B) unchanged | `shaper/contrast.py::per_view_cl_loss` |
| Effective mass `m_C^A = sum a_p/|C|`, `m_C^B = sum a_p/K`; mean/SD/P10/P50/P90 + gradient norm; never called total gradient magnitude | `shaper/augment.py::effective_mass`, `shaper/training.py::_mass_summary` |

## 9. RQ3 (segments)

| Item | Paper | Code |
|---|---|---|
| Confirmatory inference | exact K=3 Game-A per-user values on V_game ONLY (Game B and K=4 MC excluded) | `shaper/segments.py`, `scripts/run_segments.py` |
| Tests | studentized mask trend `T_mask = Σ_m (m−2.5)·mu[m,mask]`; omnibus `T_all = Σ (mu−grand)²/(SE²+eps)`; 10,000 user-level label permutations, SAME permuted map across all seeds | `shaper/segments.py::label_permutation_test` |
| Confirmatory claims | (1) mask credit increases Q1→Q4; (2) mask more valuable on ML-1M than Beauty; crop/reorder exploratory | `configs/statistics.yaml::segments` |

## 10. RQ4 (Weight / Select / controls / final test)

| Item | Paper | Code |
|---|---|---|
| Weight target | mean exact Game-A seed Shapley on V_game; positive transform with uniform fallback; `w = (1−α)/K + α·q`; `α ∈ {0,.25,.5,.75,1}`; `L = L_rec + λ Σ w_p L_p` with NO additional denominator | `shaper/adaptive.py`, `shaper/contrast.py::weighted_cl_loss` |
| Activation | grand-uplift 95% CI above zero AND highest-weight view positive in ≥4/5 seeds (8/10 after extension) AND every positively weighted view ≥80% positive; else NOT_ACTIVATED + rec-only fallback (never counted as adaptive success) | `shaper/adaptive.py::weight_activation_rule` |
| Select | rank by mean raw Game-A Shapley; tie order crop<mask<reorder<dropout; `phi_(2)−phi_(1) < delta_phi → no action`; else remove lowest, REUSING the enumerated coalition model; test metrics never used | `shaper/adaptive.py::select_decision` |
| Controls | rec-only, uniform (grand reuse), LOO-derived weights, learned dataset-level gates (LR 1e-2, entropy {0,.01,.1}, training-only), drop-lowest-LOO, random removal, 15-point simplex direct search (one calibration init + five-seed confirmation), ten Dirichlet reference candidates; budgets recorded | `shaper/baselines.py`, `scripts/run_interventions.py::stage_controls` |
| Final test | all eligible users; prefix = training history + validation item; target = test item; deterministic ties; repeated-target rate reported | `shaper/data.py::test_inputs`, `shaper/metrics.py` |
| Tables | 7A unconditional (always) + 7B activation-conditioned (rec-only + `not activated` if the rule fails) | `shaper/report.py` |

## 11. Statistics

| Item | Paper | Code |
|---|---|---|
| Confirmatory unit | seeds (seed-level paired effects + Holm); users are NEVER independent training runs | `shaper/stats.py` |
| Descriptive user analyses | Wilcoxon, rank-biserial, Cliff's delta, bootstrap | `shaper/stats.py` |
| Planning formula | `h = t_(.975,S−1)·sqrt(σ_seed²/S + σ_user²/N_game)` | `shaper/stats.py::planning_half_width`, `shaper/power.py` |
| Holm families | 6 intervention comparisons; 6 exact-Game-A mask tests (3 NDCG + 3 NLL); 3 interaction pairs per dataset | `configs/statistics.yaml::holm` |
| Extension rules | Game A: any phi CI half-width > delta_phi OR grand-uplift CI > delta_action → 2006–2010; Weight-vs-uniform: CI half-width > delta_action; Game B / diagnostics / K=4 NEVER expand | `shaper/power.py` |

## 12. Determinism, checkpointing, cache

| Item | Paper | Code |
|---|---|---|
| `torch.use_deterministic_algorithms(True)`; cuDNN benchmark off, deterministic cuDNN on; exceptions recorded never suppressed | `shaper/schedules.py::configure_determinism` |
| Nondeterminism floor | identical-configuration repeat of the Game-A grand coalition, seed 2001; marginals below the floor labeled indistinguishable from execution nondeterminism | `scripts/train_coalitions.py::stage_nondet_floor` |
| Atomic checkpoints | temp → flush → fsync → rename; `checkpoint_manifest.json`; corrupted checkpoint REJECTED, previous valid restored | `shaper/checkpoints.py` |
| Cache validity | dataset/seed/policy/coalition/config-hash/data-hash/recipe-hash + valid checkpoint; stale cache never reused | `shaper/checkpoints.py::validate_cache_keys` |
| Resume | crashed seed / crashed mid-coalition / crash between stages / partial K=4 with the SAME permutation / corrupted checkpoint | `tests/resume/test_resume_cases.py` |

## Documented engineering choices (software-engineering details not fixed by the spec)

1. **Positional table covers `max_len + 1` positions.** Training histories are
   truncated to `max_len`; the test prefix (training history + validation item)
   needs `max_len + 1` positions (spec A.3: "Test input: training history
   followed by validation item").
2. **Attention masking:** queries attend causally; real queries never attend to
   padding keys; padding queries (whose outputs are unused) keep a valid
   softmax to avoid NaN gradients. Standard SASRec implementations differ on
   this detail; the choice is documented here and in `shaper/backbone.py`.
3. **Left padding.** Sequences are left-padded so the final valid position is
   the last token; `final_valid_hidden` reads the last position.
4. **Negative-sampling universe** is the user's frozen (truncated) training
   history — the exact tensor the model consumes — plus the positive item.
5. **Warm-up starts at LR 0** (`lr·step/warmup_steps`), the common convention.
6. **Alpha tie-break** on `V_select` (absent from the spec): smaller alpha
   (closer to uniform), the more conservative candidate; documented in
   `shaper/adaptive.py::calibrate_alpha`.
7. **15-point simplex design** (K=3): all compositions of 6 into three
   nonnegative parts, sorted by Euclidean distance from uniform, first 15 —
   deterministic and prespecified (`shaper/baselines.py::simplex_design_15`).
8. **Dropout-player applicability `a_p = 1`**: the model-level dropout player
   applies to every example (it is not an input transformation).
9. **Salted robustness tie keys** use the full 128-bit hash integer (object
   dtype), avoiding int64 truncation.
10. **`shaper/config.py`, `shaper/cost.py`, `shaper/compliance.py`,
    `scripts/_cli.py`** are engineering-detail modules beyond the spec's
    module list (configuration loading, budget estimation, compliance
    registry, CLI helpers); they hold no scientific logic.
11. **Reference environment.** The registered environment is Python 3.12 with
    SciPy ≥ 1.18. This repository's verification suite was executed on Python
    3.11 with SciPy 1.17.1 (exact versions in `requirements.lock` and every
    run's `environment.json`); `requirements.txt` declares the spec ranges.
