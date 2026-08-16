# SHAPER

**SHAP**ley-guided **PER**turbation learning — cooperative attribution of
training-time augmentation views in contrastive sequential recommendation.

> Research article: *SHAPER: Cooperative Attribution of Training-Time
> Augmentation Views in Contrastive Sequential Recommendation* (pre-results
> protocol). Authors: Mouad Louhichi*, Redwane Nesmaoui, Mohamed Lazaar —
> ENSIAS, Mohammed V University in Rabat, Morocco.

This repository is the implementation of the **registered SHAPER protocol**.
The two authoritative documents are:

- [`specs/SHAPER_Implementation_Spec.md`](specs/SHAPER_Implementation_Spec.md) — implementation source of truth
- [`specs/SHAPER_Paper_Structure.md`](specs/SHAPER_Paper_Structure.md) — paper/interpretation source of truth

## Scientific design (locked)

**Primary — exact K=3 Game A.** Players `crop`, `mask`, `reorder`. For every
coalition `C ⊆ P` a full ranking-relevant recommender is trained from a common
seed-specific initialization under the fixed nominal coefficient budget
`L_cl(C) = |C|⁻¹ Σ_p L_cl^p`; coalition value is
`v(C) = NDCG@10_game(C) − NDCG@10_game(∅)` on the fixed `V_game` users. All
8 coalitions are enumerated per confirmatory seed (2001–2005, conditionally
2006–2010 under the direction-blind precision rule), and Shapley values and
Grabisch–Roubens pair interactions are **exact** conditional on the realized
tables.

**Secondary — exact K=3 Game B** (fixed per-view dose `L_cl(C) = K⁻¹ Σ_p L_cl^p`),
seeds 2001–2003 only, six intermediate coalitions only (empty/grand reused from
Game A). Policy sensitivity; never pooled with Game A, never drives RQ4, never
expands.

**Diagnostics — frozen NLL/cosine severity calibration** on frozen rec-only
checkpoints (no additional coalition Shapley sweep); 10% matching rule with
`partial alignment` failure labels.

**Extension — Beauty K=4 Game A** with a `dropout` player (two extra keyed
stochastic forwards; ordinary backbone dropout stays active in every model),
seeds 3001–3005, **antithetic permutation Monte Carlo** with a hard cap of
**10 unique coalition models per seed** (loud failure at #11). Telescoping
`Σ_p marginal_p = v(P) − v(∅)` is preserved and regression-tested; the result
is labeled *Monte-Carlo approximate*; exact K=4 interactions are forbidden from
the incomplete table; ordering is interpreted only if the 95% MC half-width
≤ `delta_phi`, otherwise `INCONCLUSIVE`.

**RQ3** uses exact K=3 Game-A per-user values on frozen pre-truncation
training-length quartiles (studentized mask trend + omnibus profile statistic,
10,000 user-level label permutations, one permuted map across all seeds).
**RQ4** (SHAPER-Weight / SHAPER-Select) derives actions from Game A only, with
the locked activation rule (`NOT_ACTIVATED → rec-only` fallback, never counted
as adaptive success), fair controls, and a test set untouched until every
decision is frozen.

No FastSHAP / KernelSHAP / generic random-Shapley replacement is used for the
primary game: coalition training is the bottleneck, and the K=3 game is
enumerated exactly while the K=4 extension uses the prescribed antithetic
estimator.

## Repository layout

```text
configs/            ml1m.yaml, beauty.yaml, synthetic.yaml (test/verification
                    only), seeds.yaml (frozen registry), scope.yaml (locked
                    scope + no-go conditions), statistics.yaml (thresholds,
                    recipe/severity grids, Holm families, activation rules),
                    manifest_freeze.yaml (frozen after pilots + amendment)
data/               raw/ (downloads), processed/ (frozen artifacts),
                    manifests/ (build logs)
shaper/             scientific package: data, augment, schedules, backbone,
                    contrast, training, game, shapley, interactions,
                    monte_carlo, segments, adaptive, baselines, metrics, stats,
                    power, checkpoints, artifacts, report, provenance,
                    logging_utils (+ config/cost/compliance engineering modules)
scripts/            preflight, build_data, train_recipe, train_coalitions,
                    run_game, run_segments, run_power, run_interventions,
                    validate_run, run_all (staged orchestrator)
notebooks/          run_all.ipynb — 25-section, restartable/resume-aware
tests/              unit/ protocol/ integration/ regression/ resume/
results/runs/       <run_id>/ — manifests, logs, checkpoints, tables, figures,
                    spec_compliance.json/.md, reproducibility_report.md
docs/               paper_implementation_consistency.md
```

## Quick start

```bash
pip install -r requirements.txt        # exact lock: requirements.lock

# safe by default; never launches the whole study from one cell/command
python scripts/run_all.py --status
python scripts/run_all.py --validate                     # full test suite
python scripts/run_all.py --estimate-cost                # planning estimates only
python scripts/run_all.py --stage data --dataset synthetic
python scripts/run_all.py --stage recipe --dataset synthetic --run-id my-run
python scripts/run_all.py --stage pilot  --dataset synthetic --run-id my-run
python scripts/run_all.py --stage archive --dataset synthetic --run-id my-run
python scripts/run_all.py --stage game-a  --dataset synthetic --run-id my-run
python scripts/run_all.py --stage game-b  --dataset synthetic --run-id my-run
python scripts/run_all.py --stage k4-mc   --dataset synthetic --run-id my-run
python scripts/run_all.py --stage shapley --dataset synthetic --run-id my-run
# ... loo | interactions | severity | segments | power |
#     weight | select | controls | final-test | report
python scripts/run_all.py --resume --dataset synthetic --run-id my-run
```

**Measured timing probe** (before the full study, on any new machine):

    python scripts/run_all.py --stage data --dataset ml1m
    python scripts/timing_probe.py --dataset ml1m --steps 200     # empty + grand coalitions
    python scripts/timing_probe.py --dataset beauty --steps 200
    python scripts/timing_probe.py --dataset ml1m --combine       # both datasets

Reports seconds/step, extrapolated hours per 10,000-step coalition, and the
projected total for the declared scope (cost-estimator model counts x the
measured rate), saved under `results/timing/`. The paper reports measured
values; planning ranges (spec A.13: ~50–90 single-GPU-hours on an 8-GB
NVIDIA GPU) are never presented as results.

Additional pre-confirmatory artifacts:
- **Preregistration archive** — `python scripts/run_all.py --stage preregister`
  sha256-snapshots the spec documents, frozen configs, dependency lock and
  repository commit with a UTC timestamp BEFORE the pilot seeds (spec B.7).
- **Recommendation baselines** — `--stage baselines` trains GRU4Rec through
  the frozen recipe and records CL4SRec as a protocol-compatible grand-
  coalition reuse (DuoRec/CoSeRec flagged pending, per the literature freeze).
- **Approximation audit** — `shaper.monte_carlo.exact_vs_mc_audit` samples
  {2,4,8,16,32} permutations from the exact K=3 table (no training).
- **Protect-last diagnostics** — frozen rec-only NLL/cosine sweeps for
  protect-last-{0,1,2}; the main protocol stays protect-last-0.
- **CI** — `.github/workflows/ci.yml` runs the full test suite plus a real
  ml-100k staged-pipeline smoke on push/PR.

Confirmatory stages (`game-a`, `game-b`, `k4-mc`) refuse to run before the
pilot-informed archive freeze (spec B.7). Real datasets:

- **MovieLens-1M**: `python scripts/build_data.py --dataset ml1m --download`
  streams `ml-1m.zip` from GroupLens, verifies the published MD5
  (`c4d9eecf…`, locked in `configs/ml1m.yaml`), records a provenance sidecar,
  then converts rating ≥ 4 → positive, iterative 5-core, temporal LOO,
  max length 200, full-catalog evaluation (the raw 1,000,209 rating count is
  never reported as the processed count).
- **MovieLens-100K** (`--dataset ml100k`): a real-data **verification**
  dataset that is NOT part of the registered study. `--download` fetches
  `ml-100k.zip` from GroupLens (published SHA-1 `cd4dcac4…`) with a GitHub
  mirror fallback; the extracted `u.data` is checked against the canonical
  content fingerprint (100,000 rows, 943 users, 1,682 items, first rows).
  `configs/ml100k.yaml` uses a reduced model/budget and a 1-confirmatory-seed
  set so the complete staged pipeline runs in minutes — the full
  data→recipe→pilots→archive→Game A→Game B→K=4 MC→Shapley→LOO→interactions→
  severity→segments→power→Weight→Select→controls→final-test→audit chain has
  been executed against it.
- **Amazon-Beauty**: `python scripts/build_data.py --dataset beauty --download`
  streams `Beauty_5.json.gz` from the canonical UCSD host (fallback URL
  included), verifies archive integrity, and records the computed SHA-256 in
  the sidecar (no published checksum exists); every retained review positive;
  max length 50.
- Archives live under `data/raw/` (gitignored — downloaded for local research
  use and never committed, per the datasets' terms). The frozen artifact
  manifest records the download URL, hashes, retrieval time and license note
  as `raw_download`. Without `--download`, a missing archive produces an
  actionable message with the same instructions (or place the file under
  `data/raw/` manually). Notebook: `stage("data", download=True)`.

Every run writes `results/runs/<run_id>/manifest.json` (stages, hashes),
`spec_compliance.json`/`.md` (per-requirement PASS/FAIL/NOT_EXECUTED with
evidence — PASS is never fabricated), `reproducibility_report.md`, and the
structured JSONL event log.

## Feasibility (amendment + surrogate)

The registered protocol (10,000-step recipe, two-dataset exact enumeration)
is computationally heavy: measured at 8-11 s/step on a CPU-only Apple
Silicon machine, it projects to months of sequential compute. Two
spec-sanctioned mechanisms make it applicable:

1. **Feasibility amendment** — `configs/amendment.yaml` (draft, `status:
   proposed`) reduces the recipe/intervention grids and limits confirmatory
   scope to ML-1M, BEFORE any confirmatory execution and independent of any
   effect direction (spec B.7). Freeze it with
   `python scripts/run_all.py --stage freeze-amendment`; confirmatory stages
   refuse to run while it is proposed. Thresholds, seeds, players and
   hypothesis directions are untouched; freezing changes the config hashes
   (recorded everywhere downstream).
2. **Cached ranking-adapter surrogate** — `--stage surrogate` trains the
   spec A.6 appendix surrogate (frozen rec-only backbone + ranking-path
   adapter trained with the recommendation loss) for all 8 coalitions and
   reports its error against the full-retraining table (Spearman + Shapley
   MAE). It validates cheap attribution; it never replaces the primary game.

With the amendment frozen, the study costs roughly 2,000-step coalitions:
~60-70k total recipe+game steps ≈ **1 week on the Mac with 4 parallel
workers, or ~$10-40 / 2-5 days on a rented GPU**.

## Implementation verification vs experiment execution

These are different states, reported separately:

- **Implementation verified** — the full test suite passes on a synthetic
  end-to-end pipeline (210 tests: unit / protocol / integration / regression /
  resume, including the MC telescoping tests, worker/order invariance, failure
  injection, and the locked redundancy counterexample).
- **Experiment executed** — real ML-1M/Beauty runs are *not* executed in this
  checkout; a demonstration verification run (`results/runs/shaper-verify`,
  synthetic dataset) executes every stage, including the exact Shapley
  allocation (efficiency residual ≈ 3.5e-18), the nondeterminism-floor repeat
  (floor = 0 on CPU with deterministic kernels), and the locked RQ4 activation
  rule (correctly `NOT_ACTIVATED` on the toy data, with rec-only fallback).
- **Hypothesis supported** — a state that only executed study data can reach;
  no empirical claim is made here.

## Environment

Registered: Python 3.12, torch ≥ 2.4, numpy ≥ 2.4,<2.5, scipy ≥ 1.18,
scikit-learn ≥ 1.6, pandas ≥ 2.2, matplotlib ≥ 3.9, PyYAML, tqdm, pytest.
This checkout's verification suite ran on Python 3.11 with the newest
compatible SciPy (1.17.1); exact versions are in `requirements.lock` and in
every run's `environment.json`. Preflight enables
`torch.use_deterministic_algorithms(True)` with cuDNN benchmarking disabled
and deterministic cuDNN enabled; exceptions are recorded, never suppressed.
