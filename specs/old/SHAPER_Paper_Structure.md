# SHAPER — Full Paper Structure, TOC & Embedded Content

**New name:** **SHAPER** — **SHApley-weighted Perturbation Enhanced Recommender** (Contrastive Sequential Recommendation via Cooperative Games)
**Target journal:** *Discover Artificial Intelligence* (Springer Nature, open access, Q1 — Information Systems)
**Article type:** Research article
**Authors:** Mouad Louhichi¹*, Redwane Nesmaoui¹, Mohamed Lazaar¹
**Affiliation:** ¹ National Higher School of Computer Science and Systems Analysis (ENSIAS), Mohammed V University in Rabat, Morocco
**Corresponding author:** mouad_louhichi@um5.ac.ma

**Structure:** six-section layout of *Game Theory Meets Explainable AI* (IJACSA 2025), not the nine-section DyHuCoG layout.
**Target:** ≈ 8,500 words, 7 figures, 8 tables.

> **Why this is the easy-to-accept paper.** It needs no new dataset and no heavy theorem. It upgrades a *well-documented, widely-reproduced contrastive sequential baseline* (masked/crop/reorder augmentations + Transformer) by adding a cooperative game whose players are the **augmentation views themselves**. There are 3–4 players, so Shapley values are **computed exactly** — no sampling error to defend. Base encoders are trained once, only a lightweight weighting of contrastive views is refitted per coalition. The most expensive experiment is 16 refits of a contrastive head, seconds of compute. Novelty is in *what the players are* (perturbations, not features) and *how credit is turned into weighting and denoising*.

> **Critical dependency note.** This paper is **independent of DyHuCoG**. An extraction audit of DyHuCoG found 63 gaps that make faithful reimplementation impossible. *SignalShap* avoided the problem by using only standard signal sources. **SHAPER** does the same: it builds on a standard contrastive sequential recommender whose public PyTorch implementations run on CPU and are fully specified. Nothing from DyHuCoG is reused.

---

## Working Title (primary + alternates)

- **Primary (selected):** *SHAPER: Shapley-Guided Cooperative Perturbation for Contrastive Sequential Recommendation*
- Alt 1: *SynergySeq: Exact Shapley Credit Assignment over Contrastive Views in Sequential Recommendation*
- Alt 2: *Which Perturbation Teaches? Cooperative Attribution of Augmentation Views for Denoised Sequential Ranking*
- Alt 3: *From Equal Weighting to Earned Weighting: A Cooperative-Game View of Contrastive Sequential Learning*

*(SHAPER is the method and codebase name throughout. The title names the theme.)*

## One-paragraph thesis (the spine)

Contrastive sequential recommenders learn by contrasting **perturbed views** of the same user history — a random crop, a masked span, a reordered segment — and treating every view as equally useful. In practice the views are not equally useful and are often mutually redundant: cropping the last two items and masking a middle item can delete the same predictive signal, so averaging their losses double-counts noise and washes out the signal that actually matters. We recast contrastive sequential learning as a **cooperative game whose players are the perturbation views** and whose value is ranking quality (NDCG@10), and compute the **exact Shapley value** over that small player set. Because the Shapley operator is linear, the same global game yields **exact per-user and per-segment attributions at no extra cost**, revealing that the perturbation that teaches heavy users is not the one that teaches cold users. We close the loop two ways: (a) **Shapley-weighted contrastive loss** that lets earned credit set the learning weight, and (b) **Shapley-guided denoising** that retains high-contribution positions in the sequence. Both improve NDCG at zero additional inference cost.

## Research questions of *this paper*

| RQ | Question | Thesis link |
|---|---|---|
| **RQ1** | Can contrastive sequential learning be posed as a cooperative game whose players are perturbation views, such that the Shapley allocation exactly decomposes ranking-quality uplift and is computable exactly? | Extends the thesis spine from *source attribution* (SignalShap) to *perturbation attribution* |
| **RQ2** | Does Shapley weighting of views differ from equal weighting / leave-one-view-out, and is the difference explained by redundancy among perturbations? | The redundancy-failure argument that motivates cooperative attribution |
| **RQ3** | Is perturbation attribution homogeneous across users, or does a global weight conceal opposing segment-level stories? | Heterogeneity line from IJACSA 2025 + SignalShap — segments as explanatory unit |
| **RQ4** | Can attribution be turned into improvement — does Shapley-weighted contrast and Shapley-guided denoising beat the uniformly-weighted baseline? | Attribution → intervention closure |

---

# TABLE OF CONTENTS

```
Abstract / Keywords
1. Introduction
   1.1 Background: contrastive sequential recommendation
   1.2 The equal-weighting gap
   1.3 Why leave-one-view-out fails under redundant perturbations
   1.4 Contributions
   1.5 Organization
2. Literature Review
   2.1 Sequential recommendation: from Markov to Transformers
   2.2 Contrastive learning for sequences: augmentation, dropout, and view generation
   2.3 Shapley values in ML and XAI
   2.4 Cooperative games over model components, data, and augmentations
   2.5 Heterogeneity and personalization in sequential models
   2.6 Positioning and differentiation (comparison table)
3. Methodology
   3.1 Notation and problem formulation
   3.2 The contrastive sequential backbone
   3.3 The perturbation views (players)
   3.4 The perturbation-attribution game
   3.5 Exact Shapley computation and its cost
   3.6 Per-user and per-segment decomposition
   3.7 SHAPER-Weight: Shapley-weighted contrastive learning
   3.8 SHAPER-Denoise: Shapley-guided sequence pruning
   3.9 SHAPER-Fuse: segment-adaptive fusion (optional)
   3.10 Comparative analysis against alternative weightings
   3.11 Theoretical justification
   3.12 Complexity analysis
   3.13 Practical implementation
4. Experimental Results
   4.1 Datasets and preprocessing
   4.2 Protocol, metrics, and baselines
   4.3 RQ1 — exact perturbation attribution
   4.4 RQ2 — Shapley versus equal weighting / leave-one-view-out under redundancy
   4.5 RQ3 — segment heterogeneity of perturbation credit
   4.6 RQ4 — weighted contrast and denoising gains
   4.7 Sensitivity, stability, and ablations
   4.8 Statistical significance
5. Discussion and Broader Implications
   5.1 What perturbation attributions mean for augmentation design
   5.2 Redundancy as the hidden cost of more views
   5.3 Relation to feature-level and source-level attribution
   5.4 Limitations and threats to validity
6. Conclusion and Future Work
Declarations
Appendices
```

---

# ABSTRACT (draft, ~220 words)

Contrastive sequential recommenders improve ranking by contrasting randomly perturbed views of a user's history, yet they weight every view — a random crop, a masked span, a reordered subsequence — equally, despite the fact that different perturbations delete different amounts of predictive signal and are often mutually redundant. We recast contrastive sequential learning as a cooperative game in which the players are the **perturbation views** and the characteristic function is ranking quality, and we compute the **exact Shapley allocation**. With three to four views the game has 8–16 coalitions, so no Monte-Carlo approximation is required, and because the sequence encoder is trained once and only a lightweight contrastive head is refitted per coalition, the full game costs seconds. We prove that efficiency yields an exact additive decomposition of ranking-quality uplift, that equal weighting and leave-one-view-out collapse under perfect redundancy where Shapley splits credit evenly, and that linearity gives per-user Shapley values at no extra cost. Aggregating per-user values into behavioural segments shows that global perturbation credit conceals opposing segment-level stories: heavy users learn from reordering, cold users from masking. Exploiting this, Shapley-weighted contrast and Shapley-guided denoising improve NDCG@10 over the uniformly-weighted baseline at no additional inference cost. Experiments cover MovieLens-1M (dense) and Amazon-Beauty (sparse) and generalize across Transformer backbones.

**Keywords:** Shapley value; cooperative game theory; sequential recommendation; contrastive learning; data augmentation; credit assignment; user segmentation

---

# 1. INTRODUCTION

## 1.1 Background: contrastive sequential recommendation

Open flat, IJACSA voice. Sequential recommenders predict the next item from ordered history. Contrastive methods improve them by creating multiple perturbed copies of the same sequence and pulling their representations together while pushing others apart. Name the standard perturbations: **crop** (contiguous subsequence), **mask** (replace items with [MASK]), **reorder** (shuffle a short span), optionally **dropout** as implicit view. State that the literature optimizes *which perturbations to use* but always *averages their losses equally*.

## 1.2 The equal-weighting gap

The maintenance framing translates here to a learning-budget framing. Each perturbation defines a learning signal, and treating them equally is only defensible if each contributes equally — which beam-level intuition says they do not. A crop that deletes the last two (most predictive) items is far noisier than a mask over a middle padding region. Equal weighting lets noise set the gradient.

## 1.3 Why leave-one-view-out fails under redundant perturbations

Concrete intuition before formalism: if `crop` and `mask` happen to delete the same central item for a given user, removing either view alone changes the representation little, so ablation reports *neither matters*, while removing both is catastrophic. Ablation therefore violates additivity and can report zero for a view that is load-bearing whenever its redundant partner remains. Forward-reference Proposition 2.

## 1.4 Contributions

1. **A cooperative game over perturbation views.** We formalize contrastive sequential learning as a transferable-utility game whose players are augmentation views and whose value is NDCG@10 uplift over a non-contrastive baseline. The player set is 3–4 by construction, so Shapley is exact, with no sampling variance to bound.

2. **Three short results that make the allocation usable.** Efficiency gives an exact additive decomposition of uplift (Prop. 1). Under perfect redundancy, leave-one-out assigns zero to both members while Shapley splits evenly, identifying the failure mode (Prop. 2). Linearity over a per-user-mean value gives per-user attribution for free (Prop. 3).

3. **Segment-level perturbation profiles.** Aggregating per-user Shapley vectors over behavioural segments shows that global view weights average over heterogeneous, sometimes opposing populations. Quantified with a permutation test.

4. **Two attribution-to-improvement interventions.** **SHAPER-Weight:** Shapley-weighted InfoNCE; **SHAPER-Denoise:** retain high-Shapley positions, prune low-Shapley ones. Both improve NDCG@10 at no inference cost, since weights and pruning masks are precomputed.

5. **A reproducible artifact.** The backbone is a public PyTorch Transformer (SASRec-style) with public augmentation code; full game is reproducible on a laptop CPU/GPU. Code and configs released.

## 1.5 Organization

Roadmap paragraph naming Sections 2–6.

---

# 2. LITERATURE REVIEW

Thematic subsections, each closing with a gap sentence. ~1,400 words.

## 2.1 Sequential recommendation
From FPMC to GRU4Rec to SASRec/BERT4Rec. Establish that attention-based encoders are the modern backbone and that contrastive methods are model-agnostic wrappers around them. *Gap: the wrapper treats every wrapper-generated view equally.*

## 2.2 Contrastive learning for sequences
CL4SRec, DuoRec, CoSeRec lineage as conventional names — cite them as the *baseline family* without adopting their naming in the method. Augmentation taxonomy: crop/mask/reorder/dropout. *Gap: view generation is studied, view weighting is not.*

## 2.3 Shapley values in ML and XAI
SHAP, KernelSHAP, TreeSHAP; axiomatic case; computational objection and sampling estimators. Point: computational objection is a function of treating *features* as players (hundreds), and evaporates when players are *views* (3–4).

## 2.4 Cooperative games over model components, data, and views
Data Shapley, feature-group Shapley, ensemble-member attribution, augmentation Shapley in vision (brief). Position view attribution as an uninstantiated member of this family for recommendation.

## 2.5 Heterogeneity in sequential models
User activity heterogeneity, cold vs heavy, short vs long sessions. Prior segmentation work from the group. *Gap: heterogeneity of contrastive benefit is unmeasured.*

Cite the group's *Explanation Drift* paper in §2.3 and §2.5. That paper asks whether attributions are stable **across time**; this one asks whether **perturbation credit is stable across the population** — orthogonal axes, companion papers.

## 2.6 Positioning and differentiation
**Table 1.** Rows: prior contrastive sequential methods + weighting heuristics. Columns: *unit of attribution* (item / feature / view), *axiomatic guarantee*, *exact or approximate*, *heterogeneity-aware*, *closes loop to improvement*. SHAPER is the only row with view-level, exact, heterogeneity-aware, and loop-closing all marked.

---

# 3. METHODOLOGY

## 3.1 Notation

| Symbol | Meaning |
|---|---|
| $\mathcal{U}, \mathcal{I}$ | users, items |
| $\mathcal{S}_u = [i_1, ..., i_{n_u}]$ | user $u$'s time-ordered sequence |
| $\mathcal{P} = \{p_1,...,p_K\}$ | perturbation views, the players; $K=3$ or $4$ |
| $p(\mathcal{S}_u)$ | perturbed copy under $p$ |
| $C \subseteq \mathcal{P}$ | coalition of views |
| $f_\theta^C$ | encoder + contrastive head trained with only views in $C$ |
| $v(C)$ | characteristic function: NDCG@10 uplift of $f_\theta^C$ |
| $\varphi_p$ | Shapley value of view $p$ |
| $\varphi_p(u)$ | per-user Shapley value |
| $\mathcal{T}_1,...,\mathcal{T}_M$ | user segments |
| $\pi_0$ | non-contrastive baseline (same encoder, $\lambda_{cl}=0$) |

## 3.2 The contrastive sequential backbone

Backbone is a 2-layer Transformer encoder (SASRec-style) with positional embeddings, causal masking, and next-item prediction loss $ \mathcal{L}_{rec}$ (cross-entropy or BPR). Contrastive branch: for each sequence, generate $K$ perturbed copies, encode them, and apply InfoNCE over in-batch negatives:

$$\mathcal{L}_{cl} = -\log \frac{\exp(\text{sim}(z, z^+)/\tau)}{\sum_{j}\exp(\text{sim}(z, z_j)/\tau)}$$

Fusion of losses: $ \mathcal{L} = \mathcal{L}_{rec} + \lambda \mathcal{L}_{cl}$ with $\lambda$ tuned on validation. This is the *grand coalition*.

Critical design property: **the item embedding table and Transformer backbone are pre-trained once in the grand coalition and frozen for coalition evaluation**; a coalition $C$ is realized by masking the view set and refitting only the contrastive head / $\lambda$ scaling (or a tiny adapter). This makes the exact game cheap and should be stated twice — here and in complexity.

For item-level denoising (SHAPER-Denoise), also define per-position views: $p_k$ is the view that masks position $k$. For long sequences this is evaluated via sampling, not exact enumeration — flag the boundary clearly.

## 3.3 The perturbation views (players)

| $p$ | View | Operator | What it tests |
|---|---|---|---|
| crop | Random crop | Keep contiguous $\eta \cdot n_u$ items, $\eta \sim U[0.5,1.0]$ | Robustness to history truncation |
| mask | Random mask | Replace $\gamma$ fraction with `[MASK]`, $\gamma=0.2$ | Robustness to missing items |
| reorder | Random reorder | Shuffle a span of length $\lfloor\beta n_u\rfloor$, $\beta=0.2$ | Robustness to local order noise |
| (opt) dropout | Model dropout | Forward twice with different dropout masks, no input change | Robustness to representation noise |

$K=3$ (crop/mask/reorder) in the main paper; $K=4$ with dropout as a robustness check in appendix. Every player is a *procedure*, not a learned parameter — that makes masking well-defined.

## 3.4 The perturbation-attribution game

**Definition 1 (Perturbation-attribution game).** $(\mathcal{P}, v)$ with
$$v(C) = \mathrm{NDCG@10}(f_\theta^C) - \mathrm{NDCG@10}(\pi_0), \qquad v(\emptyset)=0.$$

Subtracting the non-contrastive baseline turns the value into *uplift attributable to contrast*, which is the decision-relevant quantity. Fix seeds and report $\pi_0$ NDCG so subtraction is auditable.

Define the empty coalition as the non-contrastive model: $f_\theta^{\emptyset} \equiv \pi_0$. Then $v(\emptyset)=0$ follows.

**Definition 2 (View Shapley value).**
$$\varphi_p = \sum_{C \subseteq \mathcal{P}\setminus\{p\}} \frac{|C|!\,(K-|C|-1)!}{K!} [v(C\cup\{p\}) - v(C)].$$

**Definition 3 (Normalized view share).** $\bar\varphi_p = \varphi_p / \sum_h \varphi_h$ as percentage, reported alongside raw $\varphi_p$. Treat negative $\varphi_p$ as a finding (a view that harms ranking) rather than hiding it.

## 3.5 Exact Shapley computation and its cost

$2^K = 8$ ($K=3$) or $16$ ($K=4$) coalitions, each requiring one contrastive-head refit over cached sequence encodings. Give wall-clock in results. Contrast with feature-level SHAP where $2^{|F|}$ is infeasible — the exactness is a *design choice to make views the players*.

## 3.6 Per-user and per-segment decomposition

Define $v_u(C)$ as NDCG@10 uplift restricted to user $u$, so $v(C)=\frac{1}{|\mathcal{U}|}\sum_u v_u(C)$. Proposition 3 gives $\varphi_p = \frac{1}{|\mathcal{U}|}\sum_u \varphi_p(u)$ with **no extra coalitions** — per-user values fall out of the same 8/16 refits.

> **Granularity warning.** With leave-one-out evaluation each user has exactly one held-out next item, so $v_u(C)$ takes ~11 values: $1/\log_2(r+1)$ if ranked $r\le10$, else $0$. Per-user values are *exact* w.r.t. the defined game but high-variance as estimates of underlying view dependence. **Report segment aggregates as primary results**; treat per-user vectors as intermediates.

Segments: (a) **behavioural segments** from activity (cold/light/heavy by history length, plus dwell-time if available), and (b) **attribution segments** from clustering per-user $\varphi(u)$ vectors. If they disagree, view importance is not reducible to activity.

> Specify the heterogeneity test concretely: use the **same permutation test already planned for §4.5** — cluster observed $\varphi(u)$, record between-cluster variance, recompute with shuffled labels many times, report $p$. Reusing machinery keeps methods short. Companion check: ARI across five seeds for cluster stability.

## 3.7 SHAPER-Weight: Shapley-weighted contrastive learning

Replace uniform averaging:

$$\mathcal{L}_{cl}^{uniform} = \frac{1}{K}\sum_{p\in\mathcal{P}} \mathcal{L}_{cl}^p$$

with Shapley weighting:

$$\mathcal{L}_{cl}^{SHAPER} = \sum_{p\in\mathcal{P}} \bar\varphi_p \, \mathcal{L}_{cl}^p, \qquad \sum_p \bar\varphi_p =1, \; \bar\varphi_p \ge 0 \; (\text{clip negatives to }0\text{ and renormalize if needed})$$

with shrinkage toward uniform to avoid overconfidence:

$$\tilde\varphi_p(\alpha) = (1-\alpha)\cdot \frac{1}{K} + \alpha \cdot \bar\varphi_p, \quad \alpha \in [0,1]$$

Sweep $\alpha\in[0,1]$ and report the curve; $\alpha=0$ recovers the baseline, so the comparison is nested. Optionally learn $\alpha$ per segment — see §3.9.

No inference cost: weights are fixed after training.

## 3.8 SHAPER-Denoise: Shapley-guided sequence pruning

For denoising, define per-position Shapley $\psi_k$ for position $k$ in $\mathcal{S}_u$ via a leave-position-out game (sampled, not exact, for long sequences). Retain top-$\rho$ positions:

$$\mathcal{S}_u^{pruned} = \{ i_k : \psi_k \text{ in top } \rho \text{ fraction} \}, \quad \rho \in \{0.7,0.8,0.9\}$$

Flag that per-position exact Shapley is $2^{n_u}$ and therefore **sampled** with truncated Monte-Carlo; only the per-view game is exact. Report sampling variance.

## 3.9 SHAPER-Fuse: segment-adaptive fusion (optional)

Fit $\alpha$ (or per-view weights) per segment rather than globally, with shrinkage $\theta_m = (1-\lambda)\theta_{global} + \lambda\theta_{segment(m)}$. Segments fitted on **training users only**; held-out users assigned by frozen segmenter. Inference is a table lookup.

## 3.10 Comparative analysis against alternative weightings

Compare on: uniform weighting, leave-one-view-out reweighting, forward-selection reweighting, permutation importance over view presence, learned attention over views (e.g., softmax), and Monte-Carlo Shapley. For each state what it measures, what axiom it violates, and its cost. Make redundancy failure visible as an empirical column.

## 3.11 Theoretical justification

**Calibration note.** Mirror the group's *Explanation Drift* paper (zero propositions) plus *SignalShap* (three propositions + remark). Keep it light: two propositions + one remark, proofs in Appendix A.

**Proposition 1 (Exact additive decomposition).** $\sum_p \varphi_p = v(\mathcal{P}) = \mathrm{NDCG@10}(f^{\mathcal{P}}) - \mathrm{NDCG@10}(\pi_0)$. Immediate from efficiency.

**Proposition 2 (Redundancy collapse of leave-one-view-out).** Let $p_1,p_2$ be perfectly redundant, i.e. $v(C\cup\{p_1\})=v(C\cup\{p_2\})=v(C\cup\{p_1,p_2\})$ for all $C\subseteq \mathcal{P}\setminus\{p_1,p_2\}$. Then $\mathrm{LOO}(p_1)=\mathrm{LOO}(p_2)=0$, while $\varphi_{p_1}=\varphi_{p_2}=\tfrac12 [v(\mathcal{P})-v(\mathcal{P}\setminus\{p_1,p_2\})]$. *Theoretical centre* — ablation reports zero for genuinely load-bearing redundant views.

**Proposition 3 (Free per-user decomposition).** Since $v=\frac1{|\mathcal{U}|}\sum_u v_u$ and Shapley is linear, $\varphi_p=\frac1{|\mathcal{U}|}\sum_u \varphi_p(u)$.

**Remark 1 (Scale invariance).** Per-sequence z-normalization of view encodings before contrast makes $\varphi_p$ invariant to affine rescaling $ax+b$, $a>0$, of a view's raw embedding magnitudes.

> Do not extend Remark 1 to nonlinear maps — contrast uses cosine similarity over normalized vectors, so only affine invariance holds.

## 3.12 Complexity analysis

Match the house convention of *SignalShap* and *Explanation Drift*. Let $B$ be cost of training the Transformer backbone, $\Phi$ cost of one contrastive-head refit over cached encodings, $P=|\mathcal{U}|\cdot L$ cached encodings.

| Stage | Cost | Note |
|---|---|---|
| Backbone training | $O(B)$ | paid **once** |
| Encoding cache | $O(P)$ | enabling choice |
| Coalition sweep | $O(2^K \Phi)$ | $8\Phi$ ($K=3$) or $16\Phi$ ($K=4$); $\Phi$ is tiny |
| Shapley aggregation | $O(K2^K)$ | negligible |
| Per-user decomposition | $O(1)$ extra | free by Prop. 3 |
| Per-position sampling (denoise) | $O(M \cdot n_u)$ | $M$ = MC permutations, only for denoising |

Headline: total $O(B + 2^K\Phi)$, dominated by $B$ — **attribution is cheaper than training the recommender it explains**. Contrast with feature-level exact $O(2^{|F|})$ infeasible route.

## 3.13 Practical implementation

Library versions, seeds, hardware, wall-clock, sequence lengths, negative sampling, hyperparameter grids, caching. Pin: empty coalition definition $f_\theta^\emptyset\equiv\pi_0$, per-view z-normalization, MC sampling details for denoising, and that full game is reproducible on laptop GPU/CPU.

Give a hyperparameter table plus **two numbered algorithm blocks** — one for cached encoding + coalition sweep, one for Shapley aggregation and weighting/denoising.

---

# 4. EXPERIMENTAL RESULTS

## 4.1 Datasets and preprocessing

| Dataset | Users | Items | Interactions | Density | Avg. len |
|---|---|---|---|---|---|
| MovieLens-1M | 6,040 | 3,706 | 1,000,209 | ~4.5% dense | ~165 |
| Amazon-Beauty | ~22,363 | ~12,101 | ~198,502 | ~0.07% sparse | ~9 |

Two datasets chosen for **density and length contrast** — long dense histories vs short sparse ones — the axis along which view attribution is predicted to move. 5-core iterative filtering to convergence. Temporal leave-one-out: last item = test, second-last = validation.

## 4.2 Protocol, metrics, and baselines

Metrics: NDCG@10 (primary, defines $v$), Recall@10, HR@10, MRR, plus coverage. Report $\pi_0$ NDCG explicitly.

Baselines for recommender: non-contrastive Transformer ($\pi_0$), uniform-weight contrastive ($K=3$ equally weighted), strong single-model references (SASRec, LightGCN) to establish competitiveness.

Baselines for weighting: leave-one-view-out, forward selection, permutation importance, learned attention over views, MC-Shapley.

## 4.3 RQ1 — exact perturbation attribution

**Table 4 + Figure 2:** $\varphi_p$ and $\bar\varphi_p$ per dataset with efficiency check $\sum_p\varphi_p=v(\mathcal{P})$ to machine precision.

Expected narrative (to be confirmed): mask dominates on dense ML-1M (long history tolerates masking), crop/reorder matter more on short Beauty where every item counts.

## 4.4 RQ2 — Shapley versus uniform / leave-one-view-out under redundancy

**Table 5:** $\varphi_p$ beside uniform, LOO, forward selection, with view-view redundancy diagnostic (mutual information between perturbed encodings) as final column. **Figure 3:** scatter LOO vs $\varphi$ annotated at maximal disagreement.

Report as **mean over five seeds with std** — both values inherit training noise.

Look for: **crop ↔ reorder strongly correlated** (both perturb order/length), LOO near zero, Shapley real share. Report efficiency gap $\sum \mathrm{LOO}$ vs $v(\mathcal{P})$.

Word claim as *consistent with* Prop. 2 direction, not *instance of* it — real data is approximate.

## 4.5 RQ3 — segment heterogeneity

**Figure 4:** stacked view shares per behavioural segment (cold/light/heavy by length). **Figure 5:** attribution vs behavioural segments (Sankey). **Table 6:** per-segment shares + permutation test $p$.

Headline to test: cold users derive uplift from **mask** (missingness robustness), heavy users from **reorder** (order robustness), so global ordering inverts in at least one segment.

## 4.6 RQ4 — weighted contrast and denoising gains

**Table 7:** NDCG@10, Recall@10, MRR for uniform, SHAPER-Weight ($\alpha$ sweep as **Figure 6**), SHAPER-Denoise ($\rho$ sweep), and combined. Report per-segment gains — expect large gain on cold/short segments, parity on heavy.

**Figure 6:** $\alpha$ sweep; $\alpha=0$ recovers baseline (nested). **Figure 7:** heatmap of coalition values $v(C)$.

## 4.7 Sensitivity, stability, and ablations

Seed stability of $\varphi$ across five seeds; sensitivity to $\lambda$ (contrast weight), $\tau$ (temperature), $K$ (3 vs 4 views), sequence truncation $N$, candidate set. NDCG@$k$ for $k\in\{5,10,20\}$ to show not cutoff artifact. Nonlinear rescaling robustness: apply percentile-rank transform to view encodings, re-run game.

## 4.8 Statistical significance

Paired tests over users (not runs), Holm–Bonferroni, Wilcoxon signed-rank, effect sizes (Cohen's $d_z$). Pair over **users**, state unit explicitly.

---

# 5. DISCUSSION AND BROADER IMPLICATIONS

## 5.1 What the attributions mean for augmentation design
A view with small $\bar\varphi$ and nonzero engineering cost (e.g., reorder needs careful span sampling) is a decommissioning candidate; a view that dominates on sparse but not dense suggests dataset-conditional augmentation.

## 5.2 Redundancy as the hidden cost of more views
More views is not monotonically better — redundant views double-count noise. The view-view correlation matrix is the diagnostic to report before adding a fourth view.

## 5.3 Relation to feature-level and source-level explanation
View attribution and source attribution (SignalShap) answer different questions and compose: once SignalShap identifies that the sequential source matters, SHAPER identifies *which perturbation within that source* teaches. SignalShap → SHAPER is the natural two-paper zoom.

## 5.4 Limitations and threats to validity
Be forthright: score-level contrast is one wrapper among several; masking a view at head-refit time is not identical to never having generated it; offline NDCG is a proxy; 3–4 views is a design choice and exactness degrades if a system has dozens; shares are affine-invariant but not invariant to nonlinear reshaping (§4.7); per-user values are exact but coarse, so segments are the reliable unit; two datasets are two datasets.

---

# 6. CONCLUSION AND FUTURE WORK

Restate four contributions against four RQs. Future work: Owen values when views are grouped (e.g., input-level vs model-level perturbations); temporal extension to streaming sessions (drift of $\varphi_p$ over time — bridge to *Explanation Drift*); online/interleaved validation; cascade wrappers where coalition structure is constrained (Myerson values); privacy-bearing views where credit-vs-exposure tradeoff is studied.

---

# DECLARATIONS

Funding · Competing interests · Ethics approval (public secondary data) · Data availability (links) · Code availability (repository) · Author contributions (CRediT) · **Use of AI tools** — include the same declaration as *SignalShap* and *Explanation Drift* for consistency.

---

# APPENDICES

- **A.** Proofs of Props. 1–3 and Remark 1 — one page
- **B.** Full coalition value tables $v(C)$ for all $2^K$ coalitions, per dataset — transparency showpiece (8 or 16 rows)
- **C.** Hyperparameter grids and selected values
- **D.** Per-position sampling details for denoising
- **E.** Per-segment attribution tables in full
- **F.** $K=4$ (with dropout) robustness results

---

# PLANNED FIGURES & TABLES

| # | Type | Content |
|---|---|---|
| Fig 1 | Diagram | Architecture: sequence → K perturbations → Transformer → Shapley-weighted contrast |
| Fig 2 | Bar | View shares $\bar\varphi_p$ per dataset |
| Fig 3 | Scatter | LOO vs Shapley, annotated |
| Fig 4 | Stacked bar | View shares per behavioural segment |
| Fig 5 | Sankey | Attribution vs behavioural segments |
| Fig 6 | Line | $\alpha$ (SHAPER-Weight) and $\rho$ (Denoise) sweeps |
| Fig 7 | Heatmap | Coalition value surface $v(C)$ |
| Tab 1 | Comparison | Positioning vs prior contrastive sequential work |
| Tab 2 | Descriptive | Dataset statistics |
| Tab 3 | Descriptive | The three (four) perturbation views and their costs |
| Tab 4 | Results | $\varphi_p$, $\bar\varphi_p$, efficiency check |
| Tab 5 | Results | Shapley vs LOO/uniform/attention, redundancy |
| Tab 6 | Results | Per-segment shares + heterogeneity test |
| Tab 7 | Results | Recommendation quality, SHAPER variants |
| Tab 8 | Results | Significance tests and effect sizes |

---

# PLANNING NOTES (NOT part of manuscript)

## Why this is likely to be accepted

Narrow claim, fully supported; no oversold theorem. Exact Shapley removes the most common reviewer attack. Components are standard and public, so reproducibility objections are weak. Two short propositions are easy to verify and non-trivial. Prop. 2 lifts this above an empirical note.

**Complexity calibrated to group's published level.** *Explanation Drift* has zero propositions; *SignalShap* has three + remark. This blueprint mirrors *SignalShap* (three props + remark cut to two + remark), with the same selling point: exact $2^K$ game over views vs infeasible $2^{|F|}$ over features.

## Build order

1. Data loaders and 5-core filtering; freeze splits.
2. Transformer backbone + augmentation operators, cached encodings.
3. Contrastive head + coalition-masking harness; verify $v(\emptyset)=0$.
4. Exact Shapley over $K$ views; **assert efficiency in unit test**.
5. Per-user decomposition; assert averaging.
6. SHAPER-Weight and SHAPER-Denoise; $\alpha$/$\rho$ sweeps.
7. Segmentation (reuse ActionShap pipeline) + optional Fuse.
8. Statistics and LaTeX emitters (port `stats.py`).

## What can be reused

`stats.py` (paired tests, Holm–Bonferroni, Cohen's $d_z$) and k-means diagnostics from ActionShap/SignalShap transfer unchanged. Candidate-generation code from SignalShap ports, but sequences use next-item retrieval, not candidate pools.

## Estimated effort

~1 week for someone with ActionShap codebase in hand, dominated by Transformer backbone rather than game theory. Game itself is 8–16 coalitions, trivial.

## Decisions taken

| Decision | Choice | Consequence |
|---|---|---|
| Players | Perturbation views (crop/mask/reorder), not features | $K=3$ exact, no sampling, attribution is about *how we learn* |
| $K$ | 3 main, 4 robustness | Keeps exactness; dropout as cheap 4th view in appendix |
| Backbone | 2-layer Transformer | Standard, public, CPU-runnable; SHAPER is wrapper-agnostic |

## Remaining open questions

- Whether backbone should be SASRec-style (causal, next-item) or BERT4Rec-style (masked LM) — SASRec is cleaner for $v(\emptyset)$ definition; BERT4Rec interleaves masking as both view and objective. Pre-commit to SASRec in main text, BERT4Rec in appendix.
- Whether behavioural segments by length quartiles vs k-means on profile stats — consider quantile split in main, k-means in appendix (same as SignalShap decision).

