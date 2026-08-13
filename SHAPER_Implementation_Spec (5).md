# SHAPER — Revised Technical Implementation Specification and Registered Hypotheses

**Companion to:** `SHAPER_Paper_Structure.md`  
**Status:** pre-implementation, revised on **2026-08-13 before experiments** after design review.  
**Important:** Part B contains hypotheses, not results. Exactness refers to enumeration of all view coalitions; model training and augmentation remain stochastic.

## Revision record

This revision makes five validity-critical changes:

1. **Every coalition trains a ranking-relevant recommender.** The primary game no longer freezes a grand-coalition backbone and refits an inference-irrelevant projection head.
2. **No excluded-view leakage.** Within a seed, all coalitions start from the same initial model state; none starts from an all-view checkpoint.
3. **The redundancy property is corrected.** Redundancy implies zero grand-coalition LOO and equal Shapley credit, but not the previously claimed half-pair-removal formula without an additional context-independence assumption.
4. **Validation constructs the method; test only evaluates locked decisions.** Test outcomes never determine Shapley weights, segments, hyperparameters, or pruning rules.
5. **Per-position denoising is removed from this study.** The second transfer probe is view selection, which is directly supported by the exact coalition sweep.

### Second-review refinements

6. Validation users are reallocated to 20%/60%/20% for tuning/game/selection precision.
7. Every coalition receives a fixed optimizer-step budget; per-coalition early stopping is removed.
8. Canonical Games A/B are co-primary; NLL-matched Game A and cosine/K=4 diagnostics have locked supporting scopes.
9. A direction-blind precision/seed-extension rule, smooth NLL game, studentized segment test, and fixed Holm family are preregistered.
10. NT-Xent, projection architecture, filtering, RNG semantics, alpha calibration, control budgets, and realistic 110–190 GPU-hour planning are specified.
11. Fixed-nominal-budget and fixed-per-view-dose games are co-primary and mapped to different decisions.
12. No-op examples contribute zero loss without changing either game’s denominator.
13. “Matched difficulty” is relabeled NLL-matched corruption severity, with a cosine-displacement diagnostic.
14. RQ3 is narrowed to two confirmatory mask hypotheses; crop/reorder trends are exploratory.
15. Perfect substitutability, contextual marginals, interaction thresholds, pilot-informed MDE tables, gradient-flow checks, and deterministic worker-invariant schedules are added.
16. Batch construction, epoch reshuffling, keyed negatives, AdamW, clipping, warm-up/decay, and single-pass recipe calibration are locked.
17. Every extra experiment receives an explicit scope budget; Beauty K=4 Game A becomes the required RQ2 extension.
18. Precision triggers name the exact intervals, NLL becomes co-primary for RQ3, and Holm uses seeds—not users—as the confirmatory unit.
19. Learned gates, SHAPER-Select no-action/tie rules, deterministic preflight, and seed integers are fixed.
20. NLL/cosine alignment has a 10% success criterion and an explicit partial-alignment failure label.
21. Game A is renamed fixed **nominal coefficient** budget; applicability-adjusted effective mass is logged per coalition/seed/segment.
22. RQ4 reporting is split into unconditional transfer and activation-conditioned deployment tables.
23. The interaction threshold is tied to Appendix J’s common 0.003 utility-scale precision target; K=4 Game-A-only scope is explicit.
24. Policy-consistent empty notation, persisted `raw_row_id`, and `B_min` firing rates are locked across paper, code, and manifests.

---

# PART A — IMPLEMENTATION

## A.1 Repository layout

```text
SHAPER/code/
├── requirements.txt
├── configs/
│   ├── ml1m.yaml
│   ├── beauty.yaml
│   └── manifest_freeze.yaml # seeds, roles, scope, deltas, hashes, selected recipe
├── data/
│   ├── raw/                 # downloaded archives; checksums, never committed if licensing forbids
│   ├── processed/           # frozen maps/splits/tensors keyed by config hash
│   └── manifests/           # counts, role assignments, hash salts, provenance
├── shaper/
│   ├── __init__.py
│   ├── data.py              # loaders, iterative core filtering, temporal split
│   ├── augment.py           # crop, mask, reorder, no-op diagnostics
│   ├── backbone.py          # SASRec-style ranking model
│   ├── contrast.py          # NT-Xent and coalition-specific joint training
│   ├── game.py              # value tables, exact Shapley, interactions, LOO
│   ├── segments.py          # prespecified behavioural segments; exploratory clusters
│   ├── adaptive.py          # SHAPER-Weight and SHAPER-Select
│   ├── baselines.py         # uniform, LOO, learned gates, direct-search controls
│   ├── metrics.py           # full-catalog NDCG/HR/MRR and beyond-accuracy metrics
│   ├── stats.py             # seed-aware and user-aware uncertainty
│   ├── power.py             # pilot-informed MDE/CI-width table generator
│   └── report.py            # LaTeX tables and figures
├── scripts/
│   ├── build_data.py        # data, splits, tensors, fixed random schedules
│   ├── train_coalitions.py  # train all 2^K coalition-specific recommenders
│   ├── run_game.py          # values, Shapley, LOO, interactions
│   ├── run_segments.py      # behavioural heterogeneity and cluster stability
│   ├── run_power.py         # provisional/final Appendix J precision tables
│   ├── run_interventions.py # Weight, Select, and prespecified controls
│   └── run_all.py
├── tests/
└── results/{raw,tables,figures}/
```

A cached **ranking-adapter** surrogate may be implemented only as an appendix experiment. It is never the primary characteristic function; a projection-head-only surrogate is invalid.

## A.2 Environment

```text
python = 3.12
torch >= 2.4
numpy >= 2.4, < 2.5
scipy >= 1.18
scikit-learn >= 1.6
pandas >= 2.2
matplotlib >= 3.9
pyyaml, tqdm, pytest
```

Preflight `torch.use_deterministic_algorithms(True)` on engineering seed `1001`, with cuDNN benchmarking disabled and deterministic cuDNN mode enabled. If every required operation is supported, retain deterministic mode. If a named operation fails, record the exception and run with deterministic data/RNG schedules but nondeterministic kernels; never silently fall back. Regardless of mode, repeat the canonical Game-A grand coalition on confirmatory seed `2001` with identical configuration and report the absolute utility difference as the nondeterminism floor. Any contextual marginal whose magnitude does not exceed that floor is labeled indistinguishable from execution nondeterminism.

Record the exact resolved lock file, operating system, CPU/GPU, CUDA version, deterministic flags, peak allocated/reserved GPU memory (`torch.cuda.max_memory_allocated/reserved`), baseline repository commit, and literature/baseline freeze date in every run manifest. The target is execution on a single 8-GB GPU, but only measured peak memory appears in the paper. The ranges above are authoring constraints, not the reproducibility artifact; CI must exercise the exact lock.

### Seed registry

| Purpose | Fixed seeds |
|---|---|
| Recipe and alpha calibration | `901, 902, 903` |
| Excluded full-game pilots | `1001, 1002` |
| Confirmatory canonical/NLL games and final interventions | `2001, 2002, 2003, 2004, 2005` |
| Direction-blind extension | `2006, 2007, 2008, 2009, 2010` |
| Cosine-matched diagnostic | `2001, 2002, 2003` |
| Required Beauty K=4 Game-A extension | `3001, 3002, 3003, 3004, 3005` |
| Cached-adapter appendix | `4001` |

Seed `1001` is also the engineering seed. The identical-configuration repeat uses seed `2001` with an additional run suffix, not a new seed. No unlisted seed enters a confirmatory table without a timestamped amendment.

## A.3 Data layer (`data.py`)

| Decision | MovieLens-1M | Amazon-Beauty |
|---|---|---|
| Source | GroupLens `ml-1m.zip` | Amazon Reviews 2018 Beauty 5-core |
| Implicit conversion | rating ≥ 4 is positive | every retained review is positive |
| Filtering | iterative 5-core to fixed point **after** positivity conversion | iterative 5-core to fixed point |
| Main maximum length | 200 | 50 |
| Split | temporal leave-one-out | temporal leave-one-out |
| Evaluation | full retained-item catalog | full retained-item catalog |

Implementation requirements:

1. **Never report raw MovieLens counts as processed counts.** The 1,000,209-row figure is the raw rating count and is incompatible with the rating-≥4 conversion. Table 2 is generated only from the finalized artifact.
2. **Iterate 5-core to convergence.** Record counts after positivity conversion, after each filtering iteration, and after temporal holdout.
3. **Sort deterministically.** Last interaction is test, second-last is validation, and all earlier interactions are training. Break timestamp ties by original row order.
4. **Freeze segment edges before role assignment.** Compute Q1–Q4 edges from pre-truncation training-history length over the full validation-eligible user pool, persist them, and never recompute them on `V_game` or from attribution outcomes. Then assign users by a stable, persisted hash to `V_tune` (20%), `V_game` (60%), and `V_select` (20%), stratified by dataset and those frozen quartiles. `V_tune` selects the common optimization recipe and fixed update budget; `V_game` defines the coalition game and Shapley vector; `V_select` selects alpha and other intervention hyperparameters. The 20/60/20 allocation prioritizes precision for the paper’s main game and segment analysis. Test targets are not used in any of these roles.
5. **Freeze data artifacts.** Persist user/item maps, original `raw_row_id` for every retained interaction, split tensors, validation-role assignments, the hash algorithm and salt, exclusion masks, and a config hash. No downstream stage re-derives splits.
6. **Report both pre-truncation and effective lengths.** Behavioural length segments use pre-truncation training-history length; model tensors use the configured maximum length.
7. **Pad safely.** Padding ID is 0; all attention and loss functions ignore padding.
8. **Use a common evaluation population within each estimand.** Every coalition game is evaluated on the same `V_game` users. Never exclude users separately by view or coalition.
9. Emit `DatasetStats`: finalized users/items/interactions, density, mean/median/quantiles of training length, truncated fraction, validation-role counts, and augmentation applicability rates.

### Evaluation histories and roles

- Validation input: training history; target: validation item.
- `V_tune`: common architecture, optimization settings, and fixed training-step budget only.
- `V_game`: coalition values, Shapley values, interactions, and confirmatory segment decomposition.
- `V_select`: selection of alpha, direct-search settings, and other intervention hyperparameters after the `V_game` weights are fixed.
- Test input: training history followed by validation item; target: test item. Final test tables use **all eligible users** after every decision is locked.
- At evaluation, exclude every item occurring in the available prefix. If the ground-truth target also occurs earlier in the prefix, retain that target as a candidate and report the repeated-target rate.
- Rank against the full retained catalog. Break score ties by item ID.
- For Amazon-Beauty, report the fraction of users with same-day timestamp ties at validation/test. Primary ordering uses original row order; the prespecified robustness ordering sorts tied events by a fixed salted hash of `(user_id, item_id, timestamp, raw_row_id)`. Persist the salt and compare all headline metrics.
- Report that iterative core eligibility is computed before temporal holdout and therefore uses the full interaction record; treat this standard transductive choice as a limitation.
- With one relevant target, HR@10 and Recall@10 are identical; report HR@10 only.

## A.4 Perturbation views (`augment.py`)

All views are pure input transformations used only in the contrastive branch. Recommendation loss is calculated on the original sequence. A view draw is keyed by `(seed, optimizer_step, user, view)` so that a view receives the same random draw whenever it occurs in different coalitions.

| Player | Operator | Main parameters | Main implementation |
|---|---|---|---|
| crop | contiguous subsequence | keep ratio η ~ U[0.5, 1.0] | keep at least 2 items and, for n≥3, at most n−1 |
| mask | position masking | γ=0.2 | mask valid positions; main protocol does **not** protect the last two positions |
| reorder | local non-identity permutation | β=0.2 | permute a span of at least 2; resample identity permutations |
| dropout | contrastive model-level view | p=0.2 | required Beauty K=4 RQ2 extension; two extra stochastic forwards, while base dropout remains active in every model |

```python
def augment_crop(seq, rng, eta=None):
    seq = list(seq)
    n = len(seq)
    if n < 3:
        return seq, False
    eta = rng.uniform(0.5, 1.0) if eta is None else eta
    keep = min(n - 1, max(2, int(n * eta)))
    # rng is Python random.Random; randrange upper bound is exclusive.
    start = rng.randrange(n - keep + 1)
    out = seq[start:start + keep]
    return out, out != seq


def augment_mask(seq, rng, mask_id, gamma=0.2):
    seq = list(seq)
    n = len(seq)
    if n == 0:
        return seq, False
    k = min(n, max(1, int(round(n * gamma))))
    idx = rng.sample(range(n), k)
    out = seq.copy()
    for i in idx:
        out[i] = mask_id
    return out, out != seq


def augment_reorder(seq, rng, beta=0.2, max_attempts=10):
    seq = list(seq)
    n = len(seq)
    if n < 2:
        return seq, False
    span = min(n, max(2, int(round(n * beta))))
    for _ in range(max_attempts):
        start = rng.randrange(n - span + 1)
        original = seq[start:start + span]
        if len(set(original)) < 2:
            continue
        permuted = original.copy()
        rng.shuffle(permuted)
        if permuted != original:
            out = seq[:start] + permuted + seq[start + span:]
            return out, True
    return seq, False
```

Required diagnostics:

- exact no-op/applicability rate and `B_p<B_min` zero-term rate by view × dataset × behavioural segment;
- edit fraction, crop suffix-deletion rate, mask final-position rate, and rec-only log-likelihood drop;
- positive-pair cosine displacement;
- gradient norm contributed by each view;
- a unit test that crop starts are uniform over the inclusive support `0..n-keep`.

`rng` is explicitly Python `random.Random` in the reference implementation; an alternative counter-based generator must pass the same distribution tests. Random draws are precomputed or counter-keyed by `(seed, optimizer_step, global_example_id, occurrence, view)`, so worker count and coalition enumeration order cannot change an augmentation. `changed` is a Boolean tensor of shape `[B]`; padding-only/empty inputs are always `False`. Unchanged examples contribute **zero** view loss but remain in the batch denominator: if `B_p` examples changed, compute NT-Xent on those valid pairs and set `L_cl^p = (B_p/B) * mean_valid_loss`; if `B_p<B_min`, set `L_cl^p=0` and log an insufficient-pair event, with `B_min=8` fixed before execution to avoid near-degenerate NT-Xent denominators. The coalition denominator remains exactly `|C|` in Game A and `K` in Game B. This prevents an inapplicable view from reallocating its nominal share to remaining views under either policy, but it does **not** hold realized contrastive signal constant. Applicability is part of the player.

For every step, log `a_p=B_p/B` and the applicability-adjusted coefficient mass

- Game A: `m_C^A = sum_{p∈C} a_p / |C|`;
- Game B: `m_C^B = sum_{p∈C} a_p / K`.

Report mean, standard deviation, and 10/50/90th percentiles of `a_p` and `m_C` by coalition, seed, dataset, and behavioural segment, alongside view-specific contrastive gradient norms. Call `m_C` effective coefficient mass, not total gradient magnitude.

The two co-primary budget games use the canonical augmentation parameters. A **required main-text NLL-matched corruption-severity companion under fixed-nominal-budget Game A** calibrates η, γ, and β on `V_tune`. Measure every grid point with the frozen rec-only checkpoints from recipe seeds `{901,902,903}`; no model is retrained per severity setting. Average target-NLL increases over users and those three checkpoints, then set target `d_NLL*` to the median canonical increase. For each view, select the grid setting minimizing `|d_p-d_NLL*|`, breaking ties toward the canonical parameter. Use crop lower keep-ratio bounds `{0.5,0.6,0.7,0.8}` with upper bound 1.0, mask ratios `{0.1,0.2,0.3,0.4}`, and reorder span ratios `{0.1,0.2,0.3,0.4}`. Archive settings, achieved distances, and ties before coalition training. Matching succeeds only if every selected view satisfies `|d_p-d_NLL*| / max(|d_NLL*|,1e-8) <= 0.10`. Otherwise label the result **partial NLL severity alignment**, report residuals, and do not claim type and severity were separated. Call successful calibration **NLL-matched corruption severity**, not general perturbation difficulty.

A contrastive-specific diagnostic repeats the calibration using mean cosine displacement between original and perturbed sequence representations from the frozen rec-only calibration encoder. Set `d_cos*` to the median canonical displacement, select from the same grids, and require the same 10% relative-residual rule; otherwise label it partial cosine alignment. Run fixed-nominal-budget Game A for three prespecified seeds per dataset. This cosine-matched diagnostic cannot replace either co-primary canonical budget game or the NLL-matched companion after results are observed.

Report canonical, NLL-matched, and cosine-matched conclusions explicitly. Prespecified η/γ/β and protect-last-1/2 sweeps use only frozen rec-only `V_tune` NLL/cosine diagnostics; they do not launch additional coalition games. Pilot seed `1002` provides the only exploratory singleton/grand ranking confirmation at selected severity extremes, as locked in A.15.

## A.5 Backbone and recommendation objective (`backbone.py`)

Use one SASRec-style architecture in the main paper:

- item embedding: `nn.Embedding(n_items + 2, d, padding_idx=0)`;
- `[MASK] = n_items + 1`;
- learnable positional embeddings up to dataset maximum length;
- two causal Transformer blocks, `d=64`, two heads, feed-forward size 256, dropout 0.2;
- sequence representation: final valid hidden state;
- contrastive projection MLP: `Linear(64,64) → ReLU → Linear(64,64)`, included in the common seed-specific initial state and trained jointly in every nonempty coalition;
- recommendation training: SASRec binary cross-entropy at valid positions with one uniformly sampled unseen negative per positive;
- recommendation evaluation: full-catalog dot-product ranking from the backbone representation and item embeddings; the projection MLP is discarded.

Within each seed, persist:

- one common initial `state_dict` cloned into every coalition model;
- one common sequence/batch order;
- one common recommendation-negative schedule;
- keyed augmentation draws for each view.

This matched design reduces variance without letting excluded players influence a coalition.

### Batch, recommendation-loss, and optimizer protocol

- Batch size is `128` for ML-1M and `256` for Beauty, subject only to a pre-archive reduction if the engineering memory test exceeds 8 GB; any reduction is applied to every coalition and all schedules are regenerated before pilots.
- One training example is one user’s full truncated training history. Each user appears once per deterministic epoch permutation; the loader does not sample arbitrary windows.
- At every optimizer step, recommendation BCE uses all valid next-item positions in each sequence, averages over valid positions per user, then averages over users. Each view NT-Xent is normalized to a batch mean as defined in A.4, so `lambda_cl` combines one batch-mean recommendation term with one batch-mean contrastive term.
- Generate a new epoch permutation with a counter-based key `(seed, epoch, dataset_hash)` and reuse it across every coalition/policy. Report the locked optimizer steps and their equivalent full-user epochs for each dataset. `drop_last=False`; a final batch with fewer than `B_min` valid contrastive pairs still contributes recommendation loss and zero for the affected view.
- Recommendation negatives are keyed by `(seed, optimizer_step, global_user_id, target_position)`, drawn uniformly from retained items absent from that user’s **training history** and different from the positive item, and reused across coalitions.
- Use AdamW with `betas=(0.9,0.98)`, `eps=1e-8`, weight decay `1e-4`, global gradient-norm clipping at `1.0`, linear warm-up for the first `10%` of locked steps, and cosine decay to `0.1 × learning_rate`. These settings are fixed; only the initial learning rate is selected from the declared grid.
- SASRec backbone dropout `0.2` is active in every coalition, including empty. Every stochastic forward uses a counter/forked RNG key `(seed, optimizer_step, purpose, view, pass_index)`, so adding a view cannot shift recommendation or incumbent-view dropout masks. The dropout **player** is a separate SimCSE-style contrastive procedure formed by two additional keyed stochastic forwards; it does not switch ordinary backbone dropout on or off.

## A.6 Coalition training and co-primary budget games (`contrast.py`)

For every coalition `C`, train the **full ranking model** from the common seed-specific initialization:

\[
\mathcal L_C = \mathcal L_{rec} + \lambda\mathcal L_{cl}(C),
\qquad
\mathcal L_{cl}^{A}(C)=
\begin{cases}
0,&C=\emptyset,\\
|C|^{-1}\sum_{p\in C}\mathcal L_{cl}^{p},&C\ne\emptyset,
\end{cases}
\qquad
\mathcal L_{cl}^{B}(C)=K^{-1}\sum_{p\in C}\mathcal L_{cl}^{p}.
\]

Game A fixes the sum of nominal view coefficients; Game B fixes each included view’s coefficient. For each view, encode the original and perturbed histories with the current coalition model, pass them through the common-initialized projection MLP, and L2-normalize. Use symmetric NT-Xent: for a batch of `B` valid original/augmented pairs, each of the `2B` representations uses its paired representation as the positive and the remaining `2B−2` representations as in-batch negatives. Calculate a separate symmetric loss for every view and average the applicable view losses according to the coalition policy. Recommendation negatives are not inserted into the contrastive denominator. Gradients update the Transformer and item embeddings, so coalition membership changes recommendation rankings.

```python
def train_coalition(base_state, coalition, loaders, schedules, cfg):
    model = SASRecWithProjection(cfg)
    model.load_state_dict(base_state)
    optimizer = make_optimizer(model, cfg)
    epoch = 0
    iterator = iter(loaders.train_for_epoch(key=(cfg.seed, epoch, cfg.data_hash)))

    # cfg.train_steps is fixed once per dataset using V_tune.
    for step in range(cfg.train_steps):
        try:
            batch = next(iterator)
        except StopIteration:
            epoch += 1
            iterator = iter(loaders.train_for_epoch(key=(cfg.seed, epoch, cfg.data_hash)))
            batch = next(iterator)
        rec_loss = model.recommendation_loss(
            batch, negatives=schedules.rec_neg(step, batch)
        )
        view_losses = []
        for p in coalition:
            augmented, changed = schedules.apply_view(p, step, batch)
            if changed.sum() >= cfg.min_contrastive_pairs:  # fixed at 8
                valid_mean = model.symmetric_nt_xent(
                    batch[changed], augmented[changed], cfg.tau
                )
                view_loss = changed.float().mean() * valid_mean
            else:
                view_loss = rec_loss.new_zeros(())
            view_losses.append(view_loss)
        summed = torch.stack(view_losses).sum() if coalition else rec_loss.new_zeros(())
        if cfg.budget_policy == "fixed_nominal":       # Game A
            cl_loss = summed / len(coalition) if coalition else summed
        elif cfg.budget_policy == "fixed_per_view": # Game B
            cl_loss = summed / cfg.K
        else:
            raise ValueError(cfg.budget_policy)
        loss = rec_loss + cfg.lambda_cl * cl_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    save_checkpoint(
        model,
        optimizer=optimizer,
        step=cfg.train_steps,
        rng_states=capture_all_rng_states(),
        config_hash=cfg.hash,
        data_manifest_hash=cfg.data_manifest_hash,
    )
    return model
```

### Fixed training budget and hyperparameter comparability

- **Primary protocol: fixed optimizer steps, not per-coalition early stopping.** Use this single-pass calibration order on `V_tune`:
  1. On recipe seed `901`, train empty-coalition models for 5,000 steps at initial learning rates `{1e-4, 5e-4, 1e-3}` and select the rate with lowest full-catalog target NLL; tie-break toward the smaller rate.
  2. With that rate and provisional `(lambda_cl,tau)=(0.1,0.1)`, train empty and grand models to 10,000 steps on recipe seeds `{901,902,903}`, checkpoint at `{2_500,5_000,10_000}`, and choose the step count minimizing the arithmetic mean empty/grand `V_tune` target NLL; tie-break toward fewer steps.
  3. At the locked step count, evaluate all nine `(lambda_cl,tau)` pairs from `{0.05,0.1,0.2} × {0.07,0.1,0.2}` on seed `901`; advance the best two pairs by NLL to seeds `{902,903}`, then select the lower mean-NLL pair, tie-breaking toward smaller `lambda_cl` then larger `tau`.
  4. Freeze learning rate, step count, `lambda_cl`, and `tau` for every declared coalition game.
- The recipe optimizes a deployment configuration, not each subset. Co-primary values are evaluated on disjoint `V_game`, and a bounded singleton stress test evaluates `lambda_cl ∈ {0.05,0.1,0.2}` on `V_tune` for one calibration seed without changing the main estimand.
- Fix one `lambda_cl` and one `tau` across all nonempty coalitions in every declared game. This defines the estimand as view value **inside one deployment-tuned training recipe**, not the best separately tuned model for each subset. Report a small singleton `lambda_cl` sensitivity on `V_tune` as a diagnostic.
- Every baseline receives a declared, comparable validation budget.
- **Co-primary Game A — fixed nominal coefficient budget:** `L_cl(C)=|C|^{-1} sum_{p∈C} L_cl^p`. Adding a player introduces its signal and reallocates nominal budget away from incumbents. This game directly supports SHAPER-Weight and SHAPER-Select, whose weights also sum to one.
- **Co-primary Game B — fixed per-view dose:** `L_cl(C)=K^{-1} sum_{p∈C} L_cl^p`. Adding a player leaves incumbent coefficients unchanged and increases total contrastive dose from `|C|λ/K` to `(|C|+1)λ/K`. This game supports the narrower question of whether a view adds value at a fixed per-view coefficient.
- Because `|P|=K`, Games A/B share the same empty and grand objectives; only intermediate coalition coefficients differ. Reuse the identical trained empty/grand models within a seed, run all distinct intermediate models for the full confirmatory seed set, and never pool policy-specific intermediate values or describe one as a robustness rerun of the other. If their conclusions differ, report policy dependence as a primary result. RQ4 actions are derived only from Game A.

### Optional cached surrogate

An appendix may train a rec-only backbone, freeze lower layers, and fit ranking-path adapters per coalition. It must start from the rec-only checkpoint, include recommendation loss, and alter item rankings. Report rank correlation and error against the full-retraining Shapley vector. A contrastive head outside the ranking path is prohibited.

## A.7 Representation normalization

Do not apply the earlier per-vector z-normalization imported from source-attribution work. It is nonstandard for this contrastive setting and is not needed to define the game.

- Apply `torch.nn.functional.normalize(z, dim=-1, eps=1e-12)` after the projection head for cosine NT-Xent.
- Log near-zero projection norms and NaN/Inf events.
- LayerNorm remains part of the Transformer architecture.
- Recommendation scores use the ranking representation specified in A.5, not the contrastive projection head.

## A.8 The games (`game.py`)

### Seed-specific calibration game

For seed `s` and budget policy `g ∈ {A,B}`, let `M_game^g(C,s)` be full-catalog validation NDCG@10 on fixed `V_game` users. Define

\[
v_s^{g}(C)=M_{game}^{g}(C,s)-M_{game}^{g}(\emptyset,s),
\qquad v_s^{g}(\emptyset)=0.
\]

The rec-only empty model is shared by Games A/B within a seed. `V_tune` users—not `V_game` users—select the common fixed step budget and hyperparameters. Coalition-specific early stopping is not used. Calculate every Shapley value and interaction separately by policy; suppress `g` in formulas only when unambiguous.

Calculate exact Shapley values from all coalitions:

\[
\varphi_{p,s}=\sum_{C\subseteq\mathcal P\setminus\{p\}}
\frac{|C|!(K-|C|-1)!}{K!}
[v_s(C\cup\{p\})-v_s(C)].
\]

- `K=3`: eight fully trained models per dataset and seed.
- `K=4`: sixteen coalitions for the required Beauty fixed-nominal-budget RQ2 extension on seeds `3001–3005`; the dropout player is a contrastive stochastic-forward procedure, not ordinary backbone dropout.
- Exactness is conditional on the realized seed-specific value table.
- Report one Shapley vector per seed, then aggregate with uncertainty.
- By linearity, `Shapley(mean_s v_s) = mean_s Shapley(v_s)`. Thus the mean seed-specific vector is exactly the Shapley vector of the mean observed game; finite-seed uncertainty remains.

### Test use

Shapley values averaged over the prespecified `V_game` seed runs construct one dataset-level SHAPER-Weight target and SHAPER-Select ordering. `V_select` users choose alpha and any direct-search setting. After all decisions are locked, prespecified models are evaluated on test. A test coalition table may be reported as post-hoc descriptive evidence, but test-derived attributions never feed back into training or selection.

### Per-user and segment utilities

Store per-user metric values for every coalition. Because NDCG is averaged over a common user set,

\[
v_s(C)=|\mathcal U|^{-1}\sum_u v_{u,s}(C),
\quad
\varphi_{p,s}=|\mathcal U|^{-1}\sum_u\varphi_{p,u,s}.
\]

This requires no additional model fits, but aggregation costs `O(|U| K 2^K)`. Per-user values are coarse and are not interpreted individually.

### LOO and interaction diagnostics

\[
\operatorname{LOO}^{grand}_p=v(\mathcal P)-v(\mathcal P\setminus\{p\}).
\]

This baseline uses only the grand-coalition context. Persist and report all contextual marginals `v(C ∪ {p}) - v(C)` separately.

For empirical proximity to perfect substitutability, compute `epsilon_pq` as the maximum across contexts of `|v(Cp)-v(Cq)|`, `|v(Cpq)-v(Cp)|`, and `|v(Cpq)-v(Cq)|`. This is a descriptive tolerance diagnostic, not proof that natural views satisfy the exact property.

Use the context-averaged Grabisch–Roubens Shapley interaction index:

\[
I_{pq}=\sum_{S\subseteq\mathcal P\setminus\{p,q\}}
\frac{|S|!(K-|S|-2)!}{(K-1)!}
\Delta_{pq}v(S),
\]

\[
\Delta_{pq}v(S)=v(S\cup\{p,q\})-v(S\cup\{p\})-v(S\cup\{q\})+v(S).
\]

Negative interaction is consistent with substitutability; positive interaction is consistent with complementarity. Declare a practically directional interaction only if its 95% interval excludes zero and `abs(mean(I_pq)) >= delta_interaction`, fixed at `0.003` NDCG@10 before confirmatory runs. Otherwise label it inconclusive. Representation similarity is secondary evidence only.

As an approximation audit—not a competing estimator—sample `{2,4,8,16,32}` random player permutations with replacement from the exact value table and report error against exact Shapley. Discuss GraSP and Beta Shapley as different example/data-valuation estimands rather than forcing them into the three-view baseline table.

## A.9 Test suite (`tests/`)

Correctness tests:

1. **Efficiency:** `sum(phi) == v(P)` within numerical tolerance for every seed/game; report the maximum absolute residual across all games as a numerical audit.
2. **Per-user consistency:** mean user Shapley equals aggregate Shapley on the identical user set.
3. **Empty coalition:** `v(empty) == 0` exactly after baseline subtraction.
4. **Symmetry:** a deterministic hand-built symmetric value table gives equal Shapley values.
5. **Dummy:** a synthetic player satisfying `v(C ∪ {d}) = v(C)` receives exactly zero.
6. **Known game:** compare against analytically calculated Shapley values, including a three-player redundancy counterexample.
7. **Order invariance:** coalition enumeration order does not alter Shapley values, and hashes of keyed augmentation draws and model-dropout masks are identical when coalitions are enumerated in different orders or loaded with different worker counts.
8. **Coalition isolation and gradient flow:** an excluded view is never generated or accessed; for every included view, a synthetic contrastive backward pass produces nonzero gradients in at least one Transformer parameter and the item-embedding table, not only the projection MLP.
9. **Common initialization and budget:** every coalition in a seed begins from an identical saved state and receives exactly the same number of optimizer updates.
10. **Split hygiene:** `V_tune`, `V_game`, and `V_select` roles are disjoint and frozen; test targets are never accessed by training, game construction, weight selection, or segmentation.
11. **Metric and tie identity:** full-catalog ranking, repeated-target handling, seen-item filtering, primary raw-row ties, and salted-hash robustness ties match hand calculations; processed artifacts retain `raw_row_id`.
12. **Augmentation invariants:** padding is untouched, `mask_id` is valid and required, reorder is non-identity when applicable, crop-start support includes both endpoints, and no-op rates are logged.
13. **Weighted loss identity:** SHAPER-Weight uses `sum_p w_p L_p` exactly once, with nonnegative weights summing to one.
14. **No-op budget accounting:** replacing one view’s entire batch by no-ops sets that view loss to zero without changing the `|C|` or `K` coalition denominator or any incumbent coefficient.
15. **Budget-policy identity:** hand-specified view losses produce `sum(L_p)/|C|` in Game A and `sum(L_p)/K` in Game B, including the empty coalition.
16. **Epoch schedule:** every eligible training user appears exactly once per epoch permutation, epoch hashes match across coalitions/workers, and fixed-step cycling generates a new keyed permutation rather than replaying one cached order.
17. **Recommendation negatives:** every sampled negative differs from the positive, is absent from the user’s training history, and is hash-identical across coalitions for the same `(seed,step,user,position)`.
18. **Effective-mass logging:** synthetic applicability tensors produce exactly `sum(a_p)/|C|` in Game A and `sum(a_p)/K` in Game B, with unchanged results across batching/worker order.

Interpretation diagnostics—not unit tests—include coalition spread versus seed noise, uplift confidence intervals, augmentation no-op rates, stopping-budget sensitivity, and an identical-configuration repeat of the grand coalition. Also report every monotonicity violation `v(C∪{p}) < v(C)` and its uncertainty. Such a violation can indicate a harmful view, negative interaction, dose-policy effect, or training instability; it is a finding to diagnose, not automatically a failed run. The seed-2001 repeat is always reported; exact identity confirms deterministic execution, while any difference defines the empirical nondeterminism floor. Failing a precision diagnostic yields an underpowered or null result, not an implementation failure.

## A.10 SHAPER interventions (`adaptive.py`)

### SHAPER-Weight

Raw signed Shapley values are reported unchanged. Training weights use a separate positive-credit transformation:

```python
positive = np.maximum(phi_validation, 0.0)
if positive.sum() <= eps:
    q = np.full(K, 1.0 / K)
else:
    q = positive / positive.sum()
weights = (1 - alpha) * np.full(K, 1.0 / K) + alpha * q
```

- Form one dataset-level `phi_validation` by averaging the **canonical fixed-nominal-budget Game-A** seed-specific Shapley values on `V_game`. NLL/cosine severity-control-derived weights are exploratory and cannot replace this target after outcomes are observed.
- Train the exact weighted objective `L = L_rec + lambda_cl * sum_p weights[p] * L_cl^p`; do not apply an additional `1/K` or `1/|C|` factor.
- Sweep the coarse path `alpha ∈ {0.0, 0.25, 0.5, 0.75, 1.0}` on `V_select` using the same three prespecified recipe-calibration initializations; they are not used for final seed-level uncertainty.
- Lock one alpha per dataset before evaluating test, then train five final evaluation seeds from their seed-specific common initializations.
- If every raw Shapley value is nonpositive, `q` is numerically set to uniform. Activation and rec-only fallback follow A.12’s locked grand-uplift/sign-stability rule; uniform contrast remains a reported baseline and is never relabeled as adaptive success.
- Clipping breaks the efficiency interpretation; call these **Shapley-derived training weights**, not an exact utility decomposition.

### SHAPER-Select

Use validation attributions to select a lower-cost augmentation set:

1. if canonical Game-A grand uplift does not satisfy A.12’s positive-uncertainty activation rule, mark Select not activated and use rec-only as the deployment fallback;
2. otherwise rank views by mean raw canonical Game-A `V_game` Shapley value, using fixed display tie order `crop < mask < reorder < dropout`;
3. let `phi_(1) <= phi_(2)` be the two smallest values; if `phi_(2)-phi_(1) < delta_phi`, take **no action** and retain the grand coalition;
4. otherwise set `C_select = P \ {argmin_p phi_p}` and reuse the already trained coalition model;
5. evaluate the locked action/no-action decision on test;
6. compare small-game removal orderings from Shapley, grand-LOO, and random order.

This directly tests whether attribution supports an actionable augmentation decision and has no recommendation-time cost.

### Weighting controls

- uniform weighting;
- LOO-derived nonnegative weights using the same clipping, coarse alpha grid, three calibration initializations, and five final seeds as SHAPER-Weight;
- learned **dataset-level** three-logit softmax gates shared by all users/batches, initialized at zero (uniform), optimized jointly with a separate gate learning rate `1e-2`, and never used at inference. Softmax weights cannot be exactly zero. Select entropy coefficient from `{0,0.01,0.1}` on `V_select` using seeds `{901,902,903}`, then train locked final seeds with the same backbone recipe and step budget;
- drop-lowest-LOO and random removal, both using already enumerated coalition models;
- a prespecified 15-point simplex calibration on one declared calibration initialization per dataset, followed by five-seed evaluation of only the selected vector;
- ten Dirichlet random-weight candidates on the same calibration initialization, reported as a `V_select` reference distribution rather than ten test-tuned methods.

Do not claim Shapley weights are optimal. The question is whether an axiomatic small-game attribution produces competitive, interpretable weights.

### Locked RQ4 reporting contract

Always report two views of RQ4 after all decisions are frozen:

1. **Unconditional scientific table:** rec-only, uniform contrast, the locked forced Weight candidate, the forced lowest-Shapley removal candidate, LOO controls, learned gates, and direct search—regardless of whether deployment activation passed.
2. **Activation-conditioned deployment table:** the model actually selected by the preregistered activation/no-action rules. If activation fails, this row is rec-only and is labeled `not activated`; it is not counted as a Weight/Select win.

The unconditional table tests transfer; the conditioned table describes what the deployment rule would do. Neither table influences activation after test access.

## A.11 Segmentation and heterogeneity (`segments.py`)

### Confirmatory behavioural segments

- Define Q1–Q4 from **pre-truncation training-history length** only.
- Freeze boundaries before looking at Shapley values.
- Use the same `V_game` users across coalitions.
- Average each user’s raw Shapley vector across confirmatory seeds. For the one-sided confirmatory mask trend, use `T_mask = sum_{m=1}^4 (m-2.5) * mu_mask[m]`; larger values indicate increasing credit. For the omnibus profile test, use the studentized statistic `T_all = sum_{m,p} (mu[m,p] - mu[p])**2 / (SE[m,p]**2 + eps)`. Use 10,000 permutations of frozen user-to-segment labels; one permuted user map applies to all seed records.
- Report raw segment Shapley values and uplift before normalized shares.
- Suppress percentage shares when segment uplift is near zero or changes sign.

### Exploratory attribution clusters

Clustering on the same Shapley vectors cannot be validated by shuffling the labels produced by that clustering. Instead report:

- bootstrap and seed stability;
- held-out silhouette/gap diagnostics;
- consensus after label alignment;
- predictability from pre-outcome behavioural variables.

Attribution clusters are explanatory and are not used as deployment rules unless a separate pre-outcome assignment model is validated.

## A.12 Statistical analysis (`stats.py`)

- Calculate coalition values and Shapley vectors separately for each matched seed.
- Report per-seed aggregate metrics and uncertainty.
- Primary confirmatory decisions use seed-hierarchical confidence intervals and the preregistered practical-effect thresholds. A claim requires the appropriate interval to exclude zero and reach the practical threshold unless explicitly labeled descriptive.
- Seed-level paired effects are the sampling unit for confirmatory tests and Holm adjustment. With five seeds, report their low power rather than substituting users as independent training runs.
- User-level paired Wilcoxon tests, rank-biserial/Cliff’s delta, and user bootstraps are descriptive conditional-on-model analyses; they do not govern confirmatory claims.
- Do not rely on a high-dimensional plug-in covariance matrix as the sole Shapley CI. Report absolute differences, relative differences, hierarchical intervals, seed distributions, and effect sizes. P-values remain secondary.
- For RQ3, use a studentized between-segment statistic and apply the same user-to-segment permutation across all seeds; quartile labels are never permuted independently by seed.

### Precision and seed-count plan

- Run two full pilot seeds per dataset after code/tests are frozen; the engineering seed may be one of them. Pilot outcomes are excluded from confirmatory estimates and the final archive is labeled **pilot-informed**.
- Before confirmatory training, generate Appendix J’s MDE/CI-width table from finalized `V_game` sizes, pilot variance estimates, and the prespecified variance grid. The conservative planning half-width is `h=t_(.975,S-1)*sqrt(sigma_seed^2/S + sigma_user^2/N_game)`; also report a hierarchical-bootstrap estimate without pretending two pilot seeds precisely identify seed variance.
- The smallest practically relevant absolute effects are fixed at `delta_phi = 0.003`, `delta_interaction = 0.003`, and `delta_action = 0.003` NDCG@10. Shapley and interactions share the same utility scale; values below 0.003 are treated as too small for directional/actionable interpretation. Appendix J reports whether five/ten seeds can resolve this threshold. Do not scale it by observed `v(P)`, which is unstable near zero. Any change must occur in the timestamped pilot-informed amendment before confirmatory seeds and be justified independently of effect direction.
- Provisional planning illustration for `N_game=3,600` (not an empirical result):

| `sigma_user` | `sigma_seed` | 5-seed 95% half-width | 10-seed 95% half-width |
|---:|---:|---:|---:|
| 0.05 | 0.001 | 0.0026 | 0.0020 |
| 0.05 | 0.003 | 0.0044 | 0.0029 |
| 0.10 | 0.001 | 0.0048 | 0.0038 |
| 0.10 | 0.003 | 0.0059 | 0.0043 |

Appendix J regenerates this table from actual counts and pilot-informed variance scenarios.
- If even the optimistic prespecified five-seed scenario exceeds the relevant delta, the pilot-informed amendment starts that dataset at all ten confirmatory seeds rather than planning a predictable mid-run extension.
- After five coalition seeds, extend canonical Games A/B for a dataset to seeds `2006–2010` if **any** canonical raw `phi_p` interval has half-width above `delta_phi` or the shared grand-uplift interval has half-width above `delta_action`. The NLL-matched Game-A companion follows the resulting Game-A seed count but never triggers expansion independently; cosine and Beauty K=4 diagnostics remain at their locked counts.
- After alpha is locked and five final Weight/uniform intervention models are evaluated on `V_select` without test access, extend final intervention seeds to `2006–2010` if the Weight-minus-uniform interval half-width exceeds `delta_action`.
- SHAPER-Weight activates only if the Game-A grand-uplift 95% interval lies above zero, the highest-weight view has the same positive sign in at least 4/5 seeds (8/10 after extension), and every view receiving positive clipped mass is positive in at least 80% of seeds. Otherwise report non-activation and use rec-only as the deployment fallback. Never expand or activate because an effect merely looks promising.

### Prespecified smooth secondary game

Evaluate full-catalog target negative log-likelihood on `V_game` with a deterministic item catalog and define

\[
v^{NLL}(C)= -\operatorname{NLL}(C)+\operatorname{NLL}(\emptyset).
\]

This smooth game is a prespecified secondary outcome for RQ1/RQ2 and a **co-primary outcome for RQ3 segment profiles**, where one-target NDCG is especially quantized. NDCG remains the sole primary outcome for RQ4. If NDCG- and NLL-based segment conclusions disagree, report metric dependence; neither may replace the other after results are seen.

### Holm family

The confirmatory intervention family is fixed as: (1) Weight vs uniform, (2) Select vs uniform, (3) Weight vs LOO-derived weights, (4) Select vs drop-lowest-LOO, (5) Weight vs learned gates, and (6) Weight vs selected direct-search weights. The confirmatory mask family contains 12 tests: the six dataset/budget tests (four Q1→Q4 trends plus two cross-dataset contrasts) under each of NDCG and NLL. Separately, interaction intervals are Holm-adjusted within each dataset across the six canonical K=3 pair-by-budget-game tests. NLL/cosine severity-control interactions and Beauty K=4 pair interactions are descriptive stress tests unless a separate family is archived before execution. Other comparisons are descriptive.

## A.13 Runtime budget

Every declared coalition game requires full ranking-model training. Estimated ranges must be replaced by measured values in the paper.

| Stage | ML-1M | Beauty |
|---|---:|---:|
| Data preparation | < 2 min | < 2 min |
| Target peak GPU memory | ≤ 8 GB design target; report measured peak | ≤ 8 GB design target; report measured peak |
| One coalition model, GPU | 15–30 min | 8–15 min |
| Locked recipe calibration | 8–12 hr | 4–7 hr |
| Two excluded pilot games | 4–8 hr | 2–4 hr |
| One eight-coalition game, one seed | 2–4 hr | 1–2 hr |
| Canonical Game A, five seeds | 10–20 hr | 5–10 hr |
| Canonical Game B additional intermediate models, five seeds | 8–15 hr | 4–8 hr |
| NLL-matched corruption-severity Game A, five seeds | 10–20 hr | 5–10 hr |
| Cosine-matched diagnostic Game A, three seeds | 6–12 hr | 3–6 hr |
| Required Beauty K=4 Game A, five seeds | — | 11–20 hr |
| Shapley/LOO/interactions | seconds | seconds |
| Segment aggregation | < 5 min | < 5 min |
| Alpha calibration + final Weight seeds | 8–18 hr across both datasets | included in combined estimate |
| LOO weights, learned gates, direct/random calibration, selected final controls | 10–30 hr across both datasets | included in combined estimate |
| **Expected full declared study** | **approximately 110–190 single-GPU hours across both datasets; higher if the precision rule expands the main games to ten seeds** | |

SHAPER-Select and removal displays reuse coalition models and require no new training. CPU execution remains possible but is not advertised for the full study. Replace every estimate with measured hardware-specific time and energy, and distinguish training cost from recommendation-time cost.

## A.14 Complexity

Let `B` be one full coalition-model training cost.

| Component | Complexity |
|---|---:|
| One coalition game | `O(2^K B)` per seed |
| Two co-primary canonical budget games | `O((2·2^K−2)B)` per seed; shared empty/grand |
| NLL-matched Game A companion | `O(2^K B)` per seed |
| Cosine-matched Game A diagnostic | `O(2^K B)` per diagnostic seed |
| Aggregate Shapley | `O(K 2^K)` |
| Per-user Shapley aggregation | `O(|U| K 2^K)` |
| Pair interactions | `O(K^2 2^K)` |
| SHAPER-Weight final training | `O(B)` per selected alpha/seed |

The valid selling point is that small `K` makes complete interventions feasible, not that attribution costs only a few head refits.

## A.15 Locked experiment scope

| Item | Locked scope |
|---|---|
| Canonical Game A (`1/|C|`) | both datasets; 5 confirmatory seeds, conditionally 10 |
| Canonical Game B (`1/K`) | both datasets; same seed count as Game A; shared empty/grand models |
| NLL-matched Game A | both datasets; follows Game-A seed count but never triggers expansion independently |
| Cosine-matched Game A | both datasets; seeds `2001–2003` only |
| Beauty K=4 Game A with dropout player | required RQ2 extension; Beauty only; seeds `3001–3005` |
| η/γ/β and protect-last diagnostics | frozen rec-only `V_tune` NLL/cosine only; no full coalition retraining |
| Ranking confirmation for severity extremes | pilot seed `1002`; singletons plus grand only; exploratory |
| Singleton `lambda_cl` stress test | `V_tune`, recipe seed `901`, three values; diagnostic only |
| Cached ranking-adapter surrogate | Beauty only, seed `4001` |
| Sampled-permutation Shapley audit | existing exact tables; no training |

The Beauty K=4 extension tests context richness only under Game A; it does not test K=4 policy dependence between Games A/B. If an experiment is absent from this table, it is not part of the archived study without a labeled amendment.

---

# PART B — REGISTERED HYPOTHESES

> **Staged preregistration note (2026-08-13).** This pre-pilot version fixes hypotheses, estimands, pilot exclusions, and the direction-blind seed rule. Archive it before the two excluded pilot seeds. After those pilots, archive one clearly labeled **pilot-informed amendment** containing Appendix J’s realized counts/variance table and any direction-independent threshold justification. No confirmatory model may run before that amendment.

## B.1 Protocol-level expectations

| Quantity | Registered expectation | Status |
|---|---|---|
| Processed dataset counts | generated from finalized artifacts; no raw-count substitution | validity check |
| Evaluation | full-catalog, one test target per user, deterministic filtering/ties | fixed protocol |
| Validation roles | disjoint 20%/60%/20% `V_tune`/`V_game`/`V_select` user partitions | fixed protocol |
| Co-primary estimands | canonical fixed-nominal-budget Game A and fixed-per-view-dose Game B | fixed protocol |
| Severity controls | NLL-matched Game A follows canonical Game-A seed count; cosine-matched Game A uses three seeds | required diagnostic |
| Grand-coalition uplift | directionally nonnegative on average, but may be small relative to seed noise | low-confidence hypothesis |
| Shapley exactness | efficiency holds exactly per realized seed-specific game | mathematical check |
| Training uncertainty | nonzero and reported across matched seeds | required |

No hard NDCG range is preregistered because loss choice and preprocessing materially affect scale. A result near 0.30 is not automatically invalid, but it triggers an audit for sampled-negative evaluation and preprocessing mismatch.

## B.2 View-credit hypotheses

Confirmatory directional hypotheses:

1. **Mask credit increases with available history length.** Longer histories contain more transitions from which masking can teach missing-event invariance while retaining enough context; short histories are more likely to lose a decisive recent signal. Test this on raw mask Shapley values across Q1–Q4.
2. **Mask is relatively more valuable on ML-1M than Beauty.** This is the only confirmatory cross-dataset ordering claim.

Crop and reorder rankings, signs, and segment trends are exploratory. Exact percentage ranges are exploratory. Directional mask claims carry interpretive weight only if their policy dependence and corruption-severity dependence are reported across canonical Games A/B and the NLL-matched companion.

Raw signed values are primary. Normalized shares are shown only when total uplift is stably away from zero.

## B.3 Redundancy and LOO hypotheses

- The controlled duplicate-view synthetic game must satisfy the exact **perfect substitutability** property, show zero grand-coalition LOO, equal Shapley values, and the analytically correct three-player behavior.
- Natural crop/reorder substitutability is exploratory. Report `epsilon_pq`, the interaction estimate, and uncertainty; do not claim the exact property holds.
- A practically directional interaction requires a 95% interval excluding zero and magnitude at least `delta_interaction=0.003`.
- Representation correlation is secondary and does not establish substitutability.
- Three-player removal displays ordered by `V_game` Shapley, grand-LOO, and random order are exploratory faithfulness diagnostics, not scalable removal curves.

## B.4 Segment heterogeneity hypotheses

Using Q1–Q4 pre-truncation training-length segments:

| Analysis | Status |
|---|---|
| Raw mask credit across Q1→Q4 | confirmatory increasing-trend hypothesis |
| Mask ML-1M vs Beauty | confirmatory cross-dataset hypothesis |
| Crop and reorder segment trends | exploratory; no directional prediction |
| Omnibus equality of all segment profiles | confirmatory test, but no guaranteed rejection claim |

No abstract claim about cold/heavy-user ordering is made before results. Attribution clustering is exploratory.

## B.5 Intervention hypotheses

| Intervention | Registered expectation |
|---|---|
| SHAPER-Weight | small positive test effect or parity versus uniform; medium-low confidence |
| SHAPER-Select | parity or small loss with lower augmentation cost; medium confidence |
| Shapley removal ordering | stronger degradation faithfulness than LOO/random; medium confidence |
| Direct weight search | may outperform Shapley; report performance and larger tuning cost honestly |
| Random simplex control | SHAPER should lie above the median; no arbitrary percentile is a validity requirement |

No fixed +1.5% to +10% gain is preregistered. Attribution-to-intervention transfer is an empirical question, not a theorem.

## B.6 Falsification and contingencies

| Outcome | Interpretation | Action |
|---|---|---|
| All nonempty coalition rankings are identical | ranking-path or training bug | stop; inspect coalition training |
| Coalition spread or a reported marginal does not exceed the seed-2001 repeat floor | differences are not separable from execution nondeterminism | label attribution/marginal inconclusive; do not interpret sign or ordering |
| Any canonical `phi_p` or shared grand-uplift CI exceeds its locked half-width | coalition precision is inadequate | expand canonical Games A/B and NLL companion to ten seeds; if still wide, report underpowered |
| Weight−uniform `V_select` CI exceeds `delta_action` | intervention precision is inadequate | expand final intervention seeds to ten before test; if still wide, report underpowered |
| Coalition spread is small relative to seed uncertainty | game is weak at this scale | report null; use prespecified smooth secondary metric only as diagnosis |
| Some `v(C∪{p}) < v(C)` | nonmonotonic view effect, negative interaction, dose-policy effect, or instability | report contextual marginal and uncertainty; diagnose rather than auto-discard |
| Grand uplift is zero/negative | contrast does not help this setup | report; do not normalize unstable shares or force weighting |
| Games A and B disagree in sign/order | attribution is budget-policy dependent | report both as a primary result; derive RQ4 only from Game A |
| NLL or cosine residual exceeds 10% | calibration failed to match the declared severity proxy | label partial alignment, report residuals, and do not claim type/severity separation |
| NLL/cosine severity controls disagree with canonical games | conclusions depend on corruption-severity convention | report mechanism dependence; do not select the favorable convention |
| Natural interactions are near zero | little augmentation redundancy | retain controlled synthetic result; soften RQ2 |
| Behavioural profiles are homogeneous | global weighting may suffice | report stability as a negative result |
| Weight/Select activation rule fails | contrast/action evidence is insufficient | still report unconditional forced candidates; activation-conditioned deployment is rec-only with `not activated` status |
| SHAPER-Weight does not beat uniform | attribution does not transfer to optimization | report; do not claim optimality |
| Direct search beats SHAPER | interpretability/evaluation-budget trade-off | report both performance and search cost |
| Some raw Shapley values are negative | view is harmful in the realized game | report raw; positive-clip only for training weights |
| Efficiency or per-user consistency fails | implementation error | stop interpretation until fixed |

Do not add a dataset or alter perturbation severity solely to recover a failed headline unless the extension was prespecified and clearly labeled exploratory.

## B.7 Milestone order

1. **Data artifact gate:** processed counts, timestamp ties, repeated targets, lengths, and leakage checks are correct.
2. **Single-model gate:** use the first excluded pilot/engineering seed to validate rec-only/grand metrics, fixed-step training, deterministic manifests, and same-config repeat behavior.
3. **Two-seed pilot gate:** complete two full canonical Game-A pilot seeds per dataset, verify rankings vary for substantive reasons, and pass tests 1–18. Pilot outcomes are excluded from confirmatory estimates and used only for the pilot-informed MDE table.
4. **Archive gate:** freeze the 20/60/20 roles, all seed integers, batch/optimizer recipe, locked scope table, MDE table, delta values, canonical/NLL/cosine parameters and residuals, budget policies, seed triggers, activation/tie rules, and Holm families.
5. **Coalition study:** run the locked K=3 games, plus the five-seed Beauty K=4 Game-A extension. Report complete tables, Shapley, contextual marginals, grand-LOO, interactions, precision intervals, and small-game removal displays. Extend canonical Games A/B and the NLL companion under the direction-blind rule.
6. **Behavioural heterogeneity:** fixed length segments and studentized label-permutation analysis.
7. **Interventions:** `V_game`-derived weights/orderings, `V_select` calibration, then locked all-user test evaluation.
8. **Optional appendix:** cached ranking-adapter surrogate only.

A weak or null result at steps 5–7 is reported; it is not repaired by post-hoc metric, dataset, player, or threshold changes.
