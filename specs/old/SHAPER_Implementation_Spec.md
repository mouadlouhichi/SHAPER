# SHAPER — Technical Implementation Specification and Registered Predictions

**Companion to:** `SHAPER_Paper_Structure.md` (the paper blueprint). That file says *what the paper argues*; this file says *what to build and what to expect when it runs*.
**Status:** pre-implementation. Every number in Part B is a **prediction made before running anything**, not a result.
**Reuse:** `stats.py` and clustering/quality diagnostics from `ActionShap/code/` and `SignalShap/code/` port over with essentially no change. Nothing from DyHuCoG is used.

---

# PART A — IMPLEMENTATION

## A.1 Repository layout

```
SHAPER/code/
├── requirements.txt
├── configs/
│   ├── ml1m.yaml
│   └── beauty.yaml
├── shaper/
│   ├── __init__.py
│   ├── data.py              # loaders, 5-core filtering, temporal split, sequence building
│   ├── augment.py           # perturbation views: crop / mask / reorder / (dropout)
│   ├── backbone.py          # Transformer encoder (SASRec-style), frozen after grand coalition
│   ├── contrast.py          # InfoNCE head, coalition masking, refit logic
│   ├── normalize.py         # per-sequence per-view z-normalization + guard
│   ├── game.py              # characteristic function v(C), exact Shapley, per-user
│   ├── segments.py          # behavioural + attribution segmentation
│   ├── adaptive.py          # SHAPER-Weight, SHAPER-Denoise, optional Fuse
│   ├── baselines.py         # uniform, LOO-weighting, attention, forward selection
│   ├── metrics.py           # NDCG@K, Recall@K, HR@K, MRR, coverage
│   ├── stats.py             # PORTED from ActionShap/SignalShap
│   └── report.py            # LaTeX table + figure emitters
├── scripts/
│   ├── build_cache.py       # stages 1-2: data + cached encodings
│   ├── run_game.py          # stages 3-5: coalitions, Shapley, per-user
│   ├── run_segments.py      # stages 6-7: segments + Weight/Denoise
│   └── run_all.py
├── tests/
└── results/{raw,tables,figures}/
```

## A.2 Environment

```
python = 3.12
torch >= 2.4  (CPU build sufficient; CUDA optional)
numpy >= 2.4, < 2.5          # same pin as SignalShap; avoids macOS Accelerate segfault
scipy >= 1.18
scikit-learn >= 1.6
pandas >= 2.2
matplotlib >= 3.9
pyyaml, tqdm, pytest
```

No multi-GPU. Full game runs on a laptop CPU with PyTorch CPU, or single GPU if available — this is a claim the paper makes, so keep it true. Pin `torch` and `numpy` together; mismatched Accelerate builds cause the same segfault seen in SignalShap.

## A.3 Data layer (`data.py`)

| | MovieLens-1M | Amazon-Beauty |
|---|---|---|
| Source | GroupLens `ml-1m.zip` | Amazon Reviews 2018, Beauty 5-core |
| Raw interactions | 1,000,209 | ~198,502 |
| Implicit conversion | rating ≥ 4 → positive | all reviews → positive |
| Filtering | 5-core, **iterative to convergence** | 5-core, iterative |
| Sequence construction | Time-sort per user, truncate to $L=200$ (ML-1M) / $L=50$ (Beauty), left-pad | same, shorter $L$ for sparse |
| Timestamps | present | present |

Implementation requirements:

1. **Iterate 5-core to fixed point.** Loop until neither users nor items change, then record final counts for Table 2 — never pre-filter numbers.
2. **Temporal leave-one-out split per user.** Sort by timestamp; last = test, second-last = validation, rest = train. Break ties deterministically by row order.
3. **Freeze splits + sequence tensors to disk** with a config hash, so every downstream stage reads identical splits. Re-deriving per stage is how silent inconsistencies enter.
4. Emit `DatasetStats`: users, items, interactions, density, mean/median sequence length, length-quantile boundaries for segmentation, and sparsity ratio.
5. **Pad and mask.** Left-pad short sequences with `0` (padding idx), create attention mask. The Transformer must not attend to padding — a common leakage that inflates dense-dataset scores.

## A.4 Perturbation views (`augment.py`)

Each view is a **pure function** `Seq -> Seq` with no learned parameters. All are applied on **train sequences only**; validation/test are never perturbed.

| $p$ | Operator | Parameters | Implementation |
|---|---|---|---|
| **crop** | Random crop | keep ratio $\eta \sim U[0.5,1.0]$, contiguous | Sample start $s \sim U[0, n(1-\eta)]$, keep `seq[s:s+ηn]` |
| **mask** | Random mask | mask ratio $\gamma=0.2$ | Sample $\gamma n$ positions, replace with `[MASK]=n_items+1`, never mask last $2$ positions (keeps prediction learnable) |
| **reorder** | Span reorder | span ratio $\beta=0.2$ | Sample span length $\lfloor\beta n\rfloor$, sample start $s$, shuffle `seq[s:s+span]` uniformly |
| **(opt) dropout** | Model dropout | $p_{drop}=0.2$ | No input change; forward encoder twice with different dropout masks to create two views (only for $K=4$ appendix) |

Design choices that de-risk the paper:

- **Guard short sequences:** If $n_u < 5$, skip `reorder` for that sequence (span length 0 or 1 is a no-op that would create a degenerate view and dilute the game). Log skip rate.
- **Deterministic per coalition:** The *set* of available views per coalition is deterministic; stochasticity is only in *which* crop/mask is sampled per batch. This keeps $v(C)$ well-defined.
- **Ablation note:** `crop` and `reorder` are both length/order perturbations and are *engineered* to be substantially redundant — this is intentional, mirroring SignalShap's POP↔REC redundancy design, so RQ2 does not depend on luck.

```python
def augment_crop(seq, eta=None):
    n = len(seq)
    if n < 3: return seq
    eta = random.uniform(0.5, 1.0) if eta is None else eta
    keep = max(1, int(n * eta))
    start = random.randint(0, n - keep)
    return seq[start:start+keep]

def augment_mask(seq, gamma=0.2, mask_id=None):
    n = len(seq)
    if n < 3: return seq
    k = max(1, int(n * gamma))
    # never mask last 2 positions
    candidates = list(range(max(0, n-2)))
    if not candidates: return seq
    idx = random.sample(candidates, min(k, len(candidates)))
    out = seq.copy()
    for i in idx: out[i] = mask_id
    return out

def augment_reorder(seq, beta=0.2):
    n = len(seq)
    span = max(2, int(n * beta))
    if n < span + 1: return seq
    start = random.randint(0, n - span)
    segment = seq[start:start+span]
    random.shuffle(segment)
    return seq[:start] + segment + seq[start+span:]
```

## A.5 Backbone (`backbone.py`)

Standard 2-layer Transformer, SASRec-style:

- Item embedding: `nn.Embedding(n_items+2, d)` where `+1` = mask, `+1` = padding (0)
- Positional embedding: learnable `max_len = L`
- 2× `TransformerEncoderLayer` with `d=64`, `nhead=2`, `dim_feedforward=256`, `dropout=0.2`, causal mask on future positions + padding mask
- Output: sequence representation = last non-padded position's hidden state; for contrast, mean-pool over valid positions is also supported (ablation)

Crucial: **train the backbone once in the grand coalition** ($\mathcal{P}$ = all 3 views) with joint loss $\mathcal{L}=\mathcal{L}_{rec} + \lambda\mathcal{L}_{cl}$ ($\lambda$ tuned on validation, typically $0.1$–$0.3$). Then **freeze backbone** and cache encodings for coalition evaluation. Only the contrastive projection head (2-layer MLP, $d\to d$) and $\lambda$ scaling are refitted per coalition — this is what makes 8 coalitions trivial.

```python
# Pseudo: cache encodings after grand coalition training
backbone.eval()
with torch.no_grad():
    cache = {}
    for user, seq in train_loader:
        cache[user] = backbone.encode(seq)  # [d]
torch.save(cache, "cache/ml1m_encodings.pt")
```

## A.6 Contrastive head and coalition masking (`contrast.py`)

InfoNCE as in CL4SRec/DuoRec lineage (cite as baseline family, not as method):

$$\mathcal{L}_{cl}(C) = \frac{1}{|C|}\sum_{p\in C} -\log \frac{\exp(\text{sim}(h, h_p^+)/\tau)}{\sum_{j}\exp(\text{sim}(h, h_j)/\tau)}$$

where $h$ = encoding of original sequence, $h_p^+$ = encoding of its $p$-perturbed copy, negatives are other users in batch, $\tau=0.07$–$0.2$.

Coalition masking: for coalition $C$, the loss sums only over $p\in C$. Implementation hygiene — drop columns rather than zeroing, but both are separable under L2, as verified for SignalShap.

Head refit per coalition: re-initialize 2-layer projection MLP, train 10–20 epochs over cached encodings with Adam `lr=1e-3`, early stopping on validation NDCG. This is seconds, not minutes.

```python
def fit_head_for_coalition(cache, coalition, tau=0.1, epochs=15):
    """Refit contrastive projection head using only views in coalition."""
    head = ProjectionHead(d=64).train()
    opt = torch.optim.Adam(head.parameters(), lr=1e-3)
    for _ in range(epochs):
        for batch in loader(cache, coalition):  # coalition filters which perturbed copies are loaded
            loss = info_nce(head, batch, tau)
            loss.backward(); opt.step(); opt.zero_grad()
    return head
```

## A.7 Normalization (`normalize.py`)

Per-sequence, per-view, across that sequence's hidden dimensions (or across candidate scores if scoring head is used), analogous to SignalShap's per-user per-source z-norm. Guard `sigma=0` (constant encoding after heavy masking) by returning zeros, not `eps`-noised values — amplifying float noise creates a fake signal.

Log degenerate rate per view per dataset — it is evidence for which views are uninformative for short/cold sequences.

## A.8 The game (`game.py`)

```python
v(C) = ndcg_at_10(rank_by(f_theta^C)) - ndcg_at_10(pi0)
```

with $f_\theta^{\emptyset} \equiv \pi_0$ (non-contrastive Transformer, $\lambda=0$) so $v(\emptyset)=0$ by definition.

The non-contrastive baseline $\pi_0$ is the same backbone trained with $\mathcal{L}_{rec}$ only, with identical negative sampling and seed. Its NDCG@10 is reported so subtraction is auditable.

Exact Shapley over $2^K$ coalitions:

- $K=3$ → $8$ coalitions: `[], [crop], [mask], [reorder], [crop,mask], [crop,reorder], [mask,reorder], [crop,mask,reorder]`
- $K=4$ with dropout → $16$ coalitions (appendix)

Plus per-user values from same sweep by Proposition 3: $v_u(C)$ restricted to user $u$.

Expected uplift: $v(\mathcal{P})$ is the gain from adding contrastive learning at all — typically `0.02–0.05 NDCG` on ML-1M, `0.01–0.03` on Beauty (contrast helps but is not the main signal).

## A.9 Test suite (`tests/`)

Non-negotiable, priority order:

1. **Efficiency identity.** $\sum_p \varphi_p = v(\mathcal{P})$ to `1e-6`. Catches most bugs.
2. **Per-user consistency.** $\frac{1}{|\mathcal{U}|}\sum_u \varphi_p(u) = \varphi_p$ to `1e-6`.
3. **Empty coalition.** $v(\emptyset)=0$ exactly.
4. **Symmetry on synthetic data.** Two identical perturbation operators (e.g., duplicate `mask` with same seed) must receive equal $\varphi$, and their individual LOO ≈ 0. This is Prop. 2 as an executable test.
5. **Dummy view.** A pure-noise view (random shuffle of entire sequence) must receive $\varphi \approx 0$ (or negative).
6. **Degenerate normalization.** Constant encoding yields all-zero column, no NaN/inf.
7. **Backbone frozen.** Assert backbone weights are identical across all coalition refits (only head changed).
8. **Order invariance.** Shuffling player order in coalition enumeration does not change $\varphi$.

Tests 1,2,4 are the ones that would catch a wrong paper rather than a crashed run.

## A.10 SHAPER-Weight and SHAPER-Denoise (`adaptive.py`)

**SHAPER-Weight:** Convert $\bar\varphi_p$ to loss weights with shrinkage:

```python
phi_bar = phi / phi.sum()  # normalized share
phi_tilde = (1 - alpha) * (1/K) + alpha * phi_bar
# clip negatives to 0 before renorm if any phi < 0
phi_tilde = np.maximum(phi_tilde, 0)
phi_tilde /= phi_tilde.sum()
loss = sum(phi_tilde[p] * loss_p for p in coalition)
```

Sweep $\alpha \in [0,1]$ step $0.1$; $\alpha=0$ recovers uniform baseline (nested). Also support per-segment $\alpha_m$ as optional SHAPER-Fuse.

**SHAPER-Denoise:** For denoising, run a *sampled* per-position game per user (not exact). Use truncated Monte-Carlo with $M=32$ permutations; for each permutation, compute marginal gain of including position $k$ (leave-position-out NDCG delta). Keep top-$\rho$ positions ($\rho \in \{0.7,0.8,0.9\}$). Evaluate gain on validation.

Flag clearly in code and paper: per-view game is exact; per-position game is sampled, and sampling variance is reported.

## A.11 Runtime budget (single laptop, CPU or single GPU)

| Stage | ML-1M | Beauty |
|---|---|---|
| Load, filter, split, sequence build | < 1 min | < 1 min |
| Backbone training (grand coalition, ~100 epochs) | 15–30 min GPU / 40–60 min CPU | 8–15 min GPU / 20–30 min CPU |
| Cache encodings | 1–2 min | 1 min |
| 8 coalition head refits (Weight) | 2–5 min | 2–4 min |
| Monte-Carlo per-position sampling (Denoise, sampled) | 5–10 min | 3–6 min |
| Segments + adaptive sweep | 2–4 min | 2–4 min |
| **Full pipeline, one seed** | **~30 min GPU / ~70 min CPU** | **~20 min GPU / ~40 min CPU** |
| Five seeds, both datasets | ~2.5 hr GPU / ~6 hr CPU | |

If any stage is an order of magnitude over these, suspect dense operations on full user-item matrix or forgetting to freeze backbone.

---

# PART B — REGISTERED PREDICTIONS

> **Pre-registration.** Everything below is expected *before* running. When real numbers arrive, report them against this table and flag every miss explicitly.

## B.1 Pipeline-level quantities

| Quantity | ML-1M | Beauty | Confidence |
|---|---|---|---|
| Users after 5-core | ~6,040 | ~22,000 | high |
| Items after 5-core | ~3,400–3,700 | ~12,000 | high |
| Avg. sequence length | ~150–170 | ~8–10 | high |
| Non-contrastive baseline NDCG@10 ($\pi_0$) | 0.28–0.35 | 0.12–0.18 | medium |
| Grand coalition NDCG@10 ($f^{\mathcal{P}}$) | 0.31–0.38 | 0.14–0.21 | medium |
| Uplift $v(\mathcal{P})$ from contrast | **0.02–0.05** | **0.01–0.03** | medium |
| Candidate / next-item recall ceiling | >0.9 (next-item is ranking over items, not candidate retrieval) | >0.9 | high |

If $v(\mathcal{P}) < 0.01$, contrast is not helping and the game explains *why* but the weight intervention cannot beat a weak grand coalition — still report, and frame as negative result.

## B.2 View shares $\bar\varphi_p$ (headline result)

| View | ML-1M (dense, long) | Beauty (sparse, short) | Reasoning |
|---|---|---|---|
| **mask** | **35–50%** | 20–30% | Long histories tolerate masking; sparse histories are hurt by deleting any item |
| **crop** | 20–30% | **25–40%** | Cropping a short Beauty sequence deletes a large fraction → high variance, but also high signal when it keeps predictive prefix |
| **reorder** | 20–35% | 25–35% | ML-1M order matters (movie sequences have temporal taste drift); Beauty order is weak so reorder is near-noise |
| **(dropout)** | 5–12% if included | 5–12% | Model-level noise is largely redundant with input-level perturbations |

**Prediction that carries the paper:** the *ordering* inverts between datasets — mask leads on ML-1M, crop/reorder jointly exceed mask on Beauty. This is the density/length-contrast argument that justified the two-dataset choice. Joint share of the redundant pair (crop↔reorder) is the redundancy story for RQ2.

## B.3 Shapley versus leave-one-view-out / uniform

| Pair | Predicted rank corr. between view encodings | Predicted LOO / uniform delta | Predicted $\varphi$ | Confidence |
|---|---|---|---|---|
| **crop ↔ reorder** | **0.6–0.85** | both LOO < 0.005, uniform loses 1–2% NDCG | both 20–35% share | **high** — engineered redundancy |
| mask ↔ crop | 0.3–0.5 | moderate | mask moderate | medium |
| dropout ↔ any | 0.4–0.6 | small | small | medium |

Predicted headline: uniform weighting under-attributes the redundant pair by ~2×, and leaving either out alone shows near-zero drop while removing both drops $v(\mathcal{P})$ by 40–60% — visible efficiency gap $\sum \mathrm{LOO} \ll v(\mathcal{P})$. That gap is the cleanest single number and must be reported.

## B.4 Segment heterogeneity

Using length quartiles Q1 (shortest/cold) → Q4 (longest/heavy):

| Segment | Predicted dominant view | Predicted mask share |
|---|---|---|
| Q1 cold/short | **crop**, mask | 15–25% |
| Q2 | crop, reorder | 25–35% |
| Q3 | mask, reorder | 30–40% |
| Q4 heavy/long | **mask**, reorder | **40–55%** |

Prediction: **global ordering inverts in Q1 on at least one dataset** — crop overtakes mask on Beauty Q1, mask overtakes crop on ML-1M Q4. Permutation test $p < 0.01$ expected, high confidence. Adjusted Rand between behavioural (length) and attribution segments predicted **0.3–0.6**: related but not identical — the interesting outcome. Be prepared for ARI near zero; if so, drop the claim rather than defend.

## B.5 Weighted contrast and denoising gains

| Intervention | Predicted gain over uniform contrast | Confidence |
|---|---|---|
| **SHAPER-Weight** (global $\alpha$) | **+1.5% to +4% relative NDCG@10** overall | medium-high |
| SHAPER-Weight, Q1 cold | +4% to +10% relative | medium-high |
| SHAPER-Weight, Q4 heavy | −1% to +2% (parity) | medium |
| **SHAPER-Denoise** ($\rho=0.8$) | **+1% to +3% relative** | medium |
| **Combined Weight + Denoise** | **+2% to +5% relative** | medium |
| Best $\alpha$ | 0.5–0.8 | medium |
| Best $\rho$ | 0.7–0.8 | medium |

Shape matters more than aggregate: large gains where view importance diverges from uniform, parity where uniform was already near-optimal — that shape is the honest evidence.

## B.6 Falsification and contingencies

| If this happens | What it means | Contingency |
|---|---|---|
| $v(\mathcal{P}) < 0.005$ on both datasets | contrast not helping at all | Report as negative result; paper still stands on RQ1–RQ3 (attribution), but RQ4 weakens to *why* uniform fails |
| crop↔reorder correlation < 0.3 | engineered redundancy did not materialize | Report honestly; redundancy demo shifts to synthetic appendix where two views are forced identical |
| LOO and Shapley agree everywhere | no redundancy in this system | **Genuine negative result** — report it; framing in §1.3 softens; per-segment heterogeneity may still carry |
| Shares identical across datasets | length/density does not drive attribution | Two-dataset justification collapses; add LastFM-2K or vary $L$ truncation and reframe |
| Segments homogeneous ($p > 0.05$) | view credit is population-uniform | Drop behavioural-vs-attribution claim; paper still has three contributions |
| Adaptive gains ≤ 0 | attribution does not transfer to improvement | Drop combined claim; report negative result — gap between explanation and intervention is itself informative |
| Some $\varphi_p < 0$ | a view harms ranking | **Do not hide** — it is a finding; clip to 0 for weighting but report raw values |
| Efficiency check fails | implementation bug | **Stop**; do not interpret anything until tests 1–2 pass |

Two failures that wound the paper are homogeneous segments + no redundancy. Both are cheap to check early — **run RQ2 and RQ3 before writing prose** (first full pipeline run answers B.3 + B.4).

## B.7 Suggested milestone order

1. Data, splits, sequence build. **Gate: length distributions sane, no leakage.**
2. Backbone training (grand coalition). **Gate: $\pi_0$ vs grand coalition gap > 0.01.**
3. Cache encodings + coalition sweep (8 coalitions). **Gate: tests 1–4 pass.**
4. Shapley + B.2 check. **Gate: does ordering invert between datasets?**
5. LOO/uniform comparison + B.3 check. **Gate: does redundancy appear?**
6. Segments + B.4 check. **Gate: is heterogeneity significant?**
7. Weight/Denoise sweeps, statistics, LaTeX emitters.

Steps 4, 5, 6 are decision points. If all three land as predicted, the paper is essentially written. If any fails, the contingency table says what to do without improvising.

