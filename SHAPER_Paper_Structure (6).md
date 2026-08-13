# SHAPER — Revised Full Paper Structure and Embedded Draft Content

**Working expansion:** **SHA**pley-guided **PER**turbation learning  
**Primary title:** *SHAPER: Cooperative Attribution of Training-Time Augmentation Views in Contrastive Sequential Recommendation*  
**Target journal:** *Discover Artificial Intelligence* (venue indexing/quartile to be verified at submission)  
**Article type:** Research article  
**Authors:** Mouad Louhichi¹*, Redwane Nesmaoui¹, Mohamed Lazaar¹  
**Affiliation:** ¹ National Higher School of Computer Science and Systems Analysis (ENSIAS), Mohammed V University in Rabat, Morocco  
**Corresponding author:** mouad_louhichi@um5.ac.ma  
**Target length:** approximately 8,500 words  
**Status:** pre-results blueprint, revised on **2026-08-13 before experiments**  
**Companion implementation document:** `SHAPER_Implementation_Spec.md`

> **Scope.** This document is a manuscript blueprint, not a completed paper. Statements about empirical outcomes remain hypotheses until experiments are run. Internal acceptance strategy and unpublished numerical predictions are not manuscript content.

## Central design correction

The primary characteristic function uses a **separately trained ranking model for every coalition**. Within each seed, all coalitions begin from the same initialization and matched data/random schedules, but no smaller coalition inherits an all-view representation. A contrastive projection head outside the ranking path is never treated as sufficient to change NDCG.

With three views, exact enumeration requires eight complete training interventions per seed. The computational claim is therefore modest and valid: exact training-view attribution is feasible because the player set is small—not because a frozen all-view encoder makes the game free.

Subsequent design reviews additionally fixed the 20%/60%/20% validation roles, batch/optimizer/seed recipes, co-primary budget estimands, denominator-preserving no-op accounting, effective coefficient-mass logging, NLL/cosine alignment rules, pilot-informed precision triggers, seed-level Holm inference, activation-conditioned reporting, a required Beauty K=4 Game-A extension, and a complete compute budget.

---

## One-paragraph thesis

Contrastive sequential recommenders commonly learn from multiple perturbed histories, such as crops, masks, and local reorderings. These signals need not be equally useful and may substitute for or complement one another. SHAPER treats the augmentation procedures as players in a cooperative game. For each of the eight coalitions of three views, it trains a ranking-relevant recommender under a matched protocol and defines coalition value as full-catalog NDCG@10 uplift on a fixed game-calibration subset over the rec-only coalition. Two co-primary games distinguish inclusion under a fixed nominal coefficient budget from inclusion at a fixed per-view dose; a separate NLL-matched corruption-severity game diagnoses view-strength confounding. Exact Shapley values summarize each complete table, while context-averaged interaction indices diagnose substitutability or complementarity under the declared policy. Because NDCG is an average over a common user set, the same coalition models support segment-level decompositions without additional model training. The paper then tests—not assumes—whether validation-derived Shapley credit transfers to loss weighting and view selection on an untouched test set.

## Research questions

| RQ | Question |
|---|---|
| **RQ1** | Can augmentation-view contributions to ranking uplift be decomposed by an exact, correctly intervened cooperative game? |
| **RQ2** | When do Shapley, grand-coalition LOO, and interaction diagnostics disagree, and is the disagreement consistent with view substitutability or complementarity? |
| **RQ3** | Do independently defined behavioural segments exhibit different augmentation-credit profiles? |
| **RQ4** | Do validation-derived Shapley weights or view selection improve—or preserve while simplifying—test performance relative to uniform training and fair controls? |

---

# TABLE OF CONTENTS

```text
Abstract / Keywords
1. Introduction
   1.1 Contrastive sequential recommendation
   1.2 Unequal and interacting augmentation signals
   1.3 Contributions and scope
   1.4 Organization
2. Related Work
   2.1 Sequential recommendation
   2.2 Contrastive and adaptive augmentation
   2.3 Shapley values beyond feature attribution
   2.4 Heterogeneity in sequential models
   2.5 Positioning and differentiation
3. Methodology
   3.1 Notation and problem formulation
   3.2 Ranking backbone and augmentation views
   3.3 Coalition-specific training intervention
   3.4 Characteristic function and exact Shapley allocation
   3.5 LOO, interactions, and redundancy
   3.6 Per-user and behavioural-segment decomposition
   3.7 SHAPER-Weight
   3.8 SHAPER-Select
   3.9 Statistical estimand and uncertainty
   3.10 Complexity and implementation
4. Experiments and Results
   4.1 Datasets and preprocessing
   4.2 Protocol, metrics, and baselines
   4.3 RQ1 — exact augmentation attribution
   4.4 RQ2 — redundancy, interaction, and removal faithfulness
   4.5 RQ3 — behavioural heterogeneity
   4.6 RQ4 — weighting and view selection
   4.7 Sensitivity and robustness
   4.8 Statistical analysis
5. Discussion
   5.1 What augmentation credit means
   5.2 Attribution versus optimization
   5.3 Limitations and threats to validity
6. Conclusion
Declarations
Appendices
```

---

# ABSTRACT — pre-results version

Contrastive sequential recommenders use perturbed histories to improve representation learning, but crop, mask, and reorder views can provide unequal and interacting learning signals. We introduce SHAPER, a cooperative attribution framework in which augmentation procedures are the players and coalition value is full-catalog NDCG@10 uplift over a recommendation-only baseline. We distinguish two co-primary interventions—fixed nominal coefficient budget and fixed per-view dose—and add an NLL-matched corruption-severity control. With three views, all eight coalitions can be trained and evaluated for each declared game, so the Shapley allocation is exact with respect to coalition enumeration. To preserve the intervention’s meaning, every coalition trains a ranking-relevant recommender from the same seed-specific initialization; excluded views are never inherited through an all-view backbone. We complement Shapley values with context-averaged interaction indices and removal curves, and exploit the linearity of the user-averaged metric to obtain behavioural-segment decompositions without additional model fits. Finally, we evaluate whether validation-derived credit transfers to two interventions: Shapley-derived contrastive-loss weighting and low-credit view selection. Experiments are prespecified on MovieLens-1M and Amazon-Beauty using temporal evaluation, full-catalog ranking, disjoint 20%/60%/20% tuning/game/selection validation-user roles, matched random factors, and uncertainty over both users and training runs. This design separates exact cooperative allocation from stochastic model-training uncertainty and tests whether augmentation attribution is explanatory, actionable, or both.

**Keywords:** cooperative game theory; Shapley value; sequential recommendation; contrastive learning; data augmentation; interaction attribution

> Replace this abstract with a results-bearing version only after the locked analysis is complete. Do not insert predicted segment ordering or improvement claims as observed facts.

---

# 1. INTRODUCTION

## 1.1 Contrastive sequential recommendation

Sequential recommendation predicts a user’s next item from an ordered interaction history. Transformer-based recommenders learn this mapping through a recommendation objective, while contrastive extensions additionally align an original history with perturbed copies. Common perturbations include contiguous cropping, item masking, local reordering, and model dropout.

These perturbations are training procedures rather than inference-time features. The relevant explanatory question is therefore not only which input item affected one prediction, but also which training signal contributed to the ranking improvement produced by contrastive learning.

## 1.2 Unequal and interacting augmentation signals

Uniform averaging is a common, transparent default, but it does not imply equal utility. A crop can remove the recent suffix, a mask can erase isolated items, and a reorder can disturb local transitions. Their effect depends on sequence length, perturbation severity, and the dataset’s order signal.

Simple grand-coalition leave-one-view-out is incomplete when views substitute for one another. If either of two views can supply a similar benefit, removing only one may have little effect because the other remains. Conversely, complementary views may look weak individually but become valuable together. Cooperative allocation and interaction indices expose these context-dependent effects.

## 1.3 Contributions and scope

1. **A leakage-free measurement instrument for training views.** Each coalition trains a ranking-relevant recommender from a common seed-specific initialization and fixed update budget. With three views, all eight interventions are enumerated and published.
2. **Allocation and interaction diagnostics.** Shapley values summarize the complete table under standard allocation properties; the corrected redundancy property explains why grand-coalition LOO can be zero while symmetric substitutes retain equal, potentially nonzero credit. A named context-averaged interaction index measures substitutability and complementarity.
3. **Behavioural-segment decomposition without extra model fits.** User-level coalition utilities aggregate linearly, while independently defined training-length segments avoid circular clustering tests.
4. **Held-out transfer probes plus a reproducible artifact.** SHAPER-Weight and SHAPER-Select test whether validation attribution transfers to action; they are not claimed as optimal recommenders. The artifact includes manifests, matched random schedules, canonical and NLL-matched corruption-severity games, complete coalition tables, and seed/user uncertainty.

### Claims explicitly not made

- Shapley weights are not claimed to be optimal continuous loss coefficients.
- Exact coalition enumeration is not claimed to remove model-training randomness.
- Per-user attribution is not interpreted as a reliable individual explanation from one held-out target.
- Per-position denoising is outside the scope of this study.
- Generalization across multiple backbones is not claimed unless a second backbone is actually evaluated.
- K=3/K=4 removal and interaction panels are small-game diagnostics, not evidence of a scalable removal algorithm.
- Properties 1–3 are standard consequences of Shapley axioms used to define and validate the instrument; they are not presented as new cooperative-game theory.

## 1.4 Organization

Section 2 reviews sequential recommendation, adaptive augmentation, cooperative attribution, and heterogeneity. Section 3 defines the coalition intervention, allocation, interactions, segments, and interventions. Section 4 reports the locked experiments by research question. Section 5 discusses interpretation and limitations, and Section 6 concludes.

---

# 2. RELATED WORK

## 2.1 Sequential recommendation

Cover Markov and factorization approaches, GRU4Rec, SASRec, and BERT4Rec. Explain that SASRec is a design choice, not a theorem: causal next-item prediction gives a clean rec-only empty coalition, keeps masking solely an augmentation rather than also the backbone objective, and makes full coalition retraining feasible. A BERT4Rec cloze objective could define another valid game but would entangle the mask player with pretraining; portability remains future work.

## 2.2 Contrastive and adaptive augmentation

Cover CL4SRec, DuoRec, CoSeRec, ICLRec/RecDCL where protocol-compatible, and more recent view-generation, learned-augmentation, and adaptive-weighting methods. Engage explicitly with the broader “good views” and automated-augmentation literature, including InfoMin-style arguments and learned augmentation policies such as JOAO/AD-GCL in adjacent graph settings where relevant. Address the reproducibility argument that some contrastive gains arise from generic auxiliary regularization rather than the semantics of a particular view; this makes dose, interaction, and severity-controlled analyses central rather than optional.

**Gap statement:** prior work studies view construction, selection, and learned adaptation, but the paper investigates a narrower question: exact axiomatic attribution over a small set of augmentation-training interventions in sequential recommendation, with user-segment decomposition and removal faithfulness. Do not state that view weighting is unstudied.

## 2.3 Shapley values beyond feature attribution

Cover SHAP, Data Shapley, Beta/Distributional Shapley, GraSP-style gradient data scoring, grouped/component attribution, ensemble and multi-view attribution, and named augmentation-policy valuation work. Explain that Beta Shapley and GraSP value or prune training examples rather than estimating this three-view retraining game, so they are related estimands rather than forced empirical weighting baselines. Position coalition retraining against Data-Shapley-style estimands and their known cost/variance trade-offs. Emphasize that exactness is feasible here because the player set is deliberately small. Name the Grabisch–Roubens Shapley interaction convention and distinguish it from Banzhaf, Shapley–Taylor, and Faith-SHAP alternatives.

## 2.4 Heterogeneity in sequential models

Review cold/heavy users, sequence length, activity patterns, item popularity, and temporal heterogeneity. Distinguish independently defined behavioural segments from attribution-derived clusters. The former support confirmatory label-permutation tests; the latter require stability analysis rather than circular significance testing.

## 2.5 Positioning and differentiation

**Table 1** should compare representative methods on:

- unit of analysis: item, example, augmentation policy, or view;
- end-to-end training intervention versus post-hoc representation probe;
- exact versus sampled allocation;
- interaction analysis;
- behavioural decomposition;
- held-out attribution-to-action test.

Do not mark SHAPER as uniquely positive until the literature search through the submission date is complete.

---

# 3. METHODOLOGY

## 3.1 Notation and problem formulation

| Symbol | Meaning |
|---|---|
| \(\mathcal U,\mathcal I\) | users and retained item catalog |
| \(\mathcal S_u\) | training history of user \(u\) |
| \(\mathcal P=\{p_1,\ldots,p_K\}\) | augmentation procedures; \(K=3\) in the main study |
| \(C\subseteq\mathcal P\) | coalition of allowed augmentation procedures |
| \(g\in\{A,B\}\) | budget policy: fixed nominal coefficient budget or fixed per-view dose |
| \(\theta^{g}_{C,s}\) | ranking model trained for coalition \(C\), seed \(s\), and policy \(g\) |
| \(M_R^{g}(C,s)\) | full-catalog NDCG@10 for role \(R\in\{tune,game,select,test\}\) |
| \(v_s^{g}(C)\) | seed-specific aggregate uplift under policy \(g\) |
| \(v_{u,s}^{g}(C)\) | user \(u\)’s uplift in the same seed-specific game |
| \(\mathcal L_{cl}^{p}\) | batch-normalized symmetric NT-Xent loss for view \(p\), including zero loss for no-op examples |
| \(\varphi_{p,s}^{g}\) | exact Shapley value of view \(p\) in seed \(s\), policy \(g\) |
| \(I_{pq,s}^{g}\) | context-averaged pair interaction under policy \(g\) |
| \(V_{tune},V_{game},V_{select}\) | disjoint, hash-frozen validation-user roles |
| \(\mathcal T_m\) | independently defined behavioural segment |

The intervention, not only the player name, is part of the game definition. This includes loss normalization, fixed hyperparameters, random schedules, initialization, and evaluation protocol.

## 3.2 Ranking backbone and augmentation views

Use a two-block SASRec-style encoder with causal attention, item and positional embeddings, and next-item recommendation loss on original histories. The three main players are:

| Player | Transformation | Main question |
|---|---|---|
| crop | keep a contiguous 50–100% subsequence, forced non-identity where possible | robustness to history truncation |
| mask | replace 20% of valid positions by `[MASK]` | robustness to missing events |
| reorder | apply a non-identity local permutation to a 20% span | robustness to local order noise |

A required Beauty-only K=4 RQ2 extension adds a **contrastive dropout view**: two extra stochastic forwards form the positive pair. Ordinary backbone dropout remains active in every model, including empty; the player adds a contrastive procedure rather than toggling model dropout.

The augmented copies enter only the contrastive branch. A common-initialized two-layer `64→64→64` projection MLP and symmetric NT-Xent objective are used during training: each original/augmented representation is an anchor once, its paired representation is positive, and the other `2B−2` representations are in-batch negatives. Recommendation negatives are not used as contrastive negatives. Ranking is produced by the shared backbone and item embeddings. Coalition training updates the ranking parameters, so a view can affect NDCG through learned representations. For view \(p\), unchanged examples receive zero pair loss while remaining in the batch denominator: \(\mathcal L_{cl}^{p}=(B_p/B)\overline{\ell}_p\), where \(B_p\) is the number of changed examples. When \(B_p<B_{min}=8\), set the view term to zero and log the event, avoiding near-degenerate NT-Xent batches. Thus an inapplicable view does not transfer its nominal share to other views under either declared denominator. Applicability and effective-gradient rates are reported.

Report no-op rates, edit severity, crop suffix deletion, final-position masking, representation displacement, and gradient norms. Because view type is confounded with strength, RQ1 reports both canonical budget games beside a required fixed-nominal-budget **NLL-matched corruption-severity** game calibrated on `V_tune`. NLL matching uses frozen rec-only calibration checkpoints and equalizes rec-only target damage, not general contrastive difficulty. A match label requires every view to fall within 10% of the target severity; otherwise the paper reports partial alignment and residual gaps. A three-seed cosine-displacement-aligned Game A provides a contrastive-specific diagnostic under the same success rule. Protect-last-1/protect-last-2 masking is prespecified.

## 3.3 Coalition-specific training intervention

For seed \(s\), save one initial model state and clone it into every coalition. Match data order and recommendation negatives. A keyed random schedule ensures that a crop draw, for example, is identical whenever crop appears in different coalitions.

For budget policy \(g\in\{A,B\}\), train

\[
\theta^{g}_{C,s}=\operatorname{Train}\left(
\theta_{0,s},
\mathcal L_{rec}+\lambda\mathcal L_{cl}^{g}(C)
\right),
\qquad \mathcal L_{cl}^{g}(\emptyset)=0.
\]

### Player estimands and decision mapping

SHAPER reports two co-primary games because normalization changes the causal question.

- **Game A—fixed nominal coefficient budget:** \(\mathcal L_{cl}^{A}(C)=|C|^{-1}\sum_{p\in C}\mathcal L_{cl}^{p}\). Adding a view adds its signal while reducing incumbent coefficients. This is a budget-reallocation estimand and directly supports SHAPER-Weight and SHAPER-Select, whose active weights sum to one.
- **Game B—fixed per-view dose:** \(\mathcal L_{cl}^{B}(C)=K^{-1}\sum_{p\in C}\mathcal L_{cl}^{p}\). Adding a view leaves incumbent coefficients unchanged while increasing nominal total dose. This answers whether the view adds value at a fixed per-view coefficient.

“Fixed nominal coefficient budget” does not mean fixed realized gradient signal: applicability/no-op rates and loss magnitudes remain part of each procedure. For every coalition and seed, report per-view applicability \(a_p=B_p/B\), effective coefficient mass \(m_C^A=|C|^{-1}\sum a_p\) or \(m_C^B=K^{-1}\sum a_p\), and contrastive gradient norms.

Because \(|\mathcal P|=K\), Games A/B share the same empty and grand-coalition objectives; only intermediate coalition coefficients differ. Both games receive the full confirmatory seed set on both datasets. Their intermediate values and allocations are never pooled. Disagreement is reported as policy dependence, not treated as failed robustness. RQ4 derives actions only from Game A.

Before modeling, assign users by a stable, persisted, length-stratified hash to disjoint validation roles: 20% in \(V_{tune}\) for the common recipe and training-step budget, 60% in \(V_{game}\) for coalition values and Shapley attribution, and 20% in \(V_{select}\) for alpha and intervention selection. Architecture, \(\lambda\), \(\tau\), optimizer, and a fixed optimizer-step budget are selected only on \(V_{tune}\) using empty/grand models over prespecified calibration seeds and are then frozen. Per-coalition early stopping is not used.

Each example is one full truncated user history; deterministic epoch permutations are shared across coalitions. Batch sizes are 128 (ML-1M) and 256 (Beauty). Recommendation BCE averages all valid next-position losses per user and then across users; each contrastive term is also a batch mean. AdamW uses fixed betas, weight decay, gradient clipping, 10% warm-up, and cosine decay. Recipe calibration is single-pass: learning rate on empty seed 901, fixed steps from empty/grand NLL curves on seeds 901–903, then a screened lambda/temperature grid on the grand coalition. Appendix C lists exact seeds, grids, tie rules, and calibration cost. No coalition is warm-started from the grand coalition.

## 3.4 Characteristic function and exact Shapley allocation

### Validation calibration game

For each policy \(g\), let \(M_{game}^{g}(C,s)\) be NDCG@10 on fixed \(V_{game}\) users after the common recipe was determined on disjoint \(V_{tune}\). Define

\[
v_s^{g}(C)=M_{game}^{g}(C,s)-M_{game}^{g}(\emptyset,s),
\qquad v_s^{g}(\emptyset)=0,
\]

where the rec-only empty model is common to Games A/B within a seed. Every allocation and interaction below is computed separately for each \(g\); the superscript is suppressed only when unambiguous.

The Shapley value is

\[
\varphi_{p,s}=\sum_{C\subseteq\mathcal P\setminus\{p\}}
\frac{|C|!(K-|C|-1)!}{K!}
[v_s(C\cup\{p\})-v_s(C)].
\]

For \(K=3\), all eight coalition models are trained and the complete value table remains a primary reported object. Shapley does not replace that table; it provides the unique scalar allocation satisfying efficiency, symmetry, dummy, and additivity, avoids grand-coalition LOO’s substitutability pathology, aggregates linearly over users/segments, and supplies one prespecified input to the transfer probes. The allocation is exact for each realized seed-specific game, while the expected game remains estimated from finite stochastic runs.

By linearity across seeds,

\[
\varphi\!\left(\mathbb E_s[v_s]\right)=\mathbb E_s[\varphi(v_s)].
\]

Thus the mean seed-specific vector equals the Shapley vector of the mean observed value table; this does not remove finite-seed uncertainty.

### Property 1 — efficiency

For each seed-specific game,

\[
\sum_{p\in\mathcal P}\varphi_{p,s}=v_s(\mathcal P).
\]

This is an allocation identity, not a statement that every view helps, that the grand coalition is optimal, or that \(v(\mathcal P)>0\). Contrastive training may reduce ranking quality under the fixed recipe.

### Reporting signed values and shares

Raw signed \(\varphi_p\) is primary. A signed normalized share

\[
\bar\varphi_p=\varphi_p/v(\mathcal P)
\]

is shown only when total uplift is stably away from zero. If signs are mixed, do not present the result as an ordinary percentage allocation without explanation. Nonnegative training weights are a separate transformation in Section 3.7.

### Remark 1 — metric-scale invariance

If the metric is transformed as \(M'=aM+b\) with \(a>0\), baseline subtraction removes \(b\), all raw Shapley values scale by \(a\), and stable normalized shares are unchanged. This statement concerns metric scaling, not arbitrary nonlinear transformations or embedding rescaling.

## 3.5 LOO, interactions, and redundancy

The single LOO baseline is **grand-coalition leave-one-out**:

\[
\operatorname{LOO}^{grand}_p=v(\mathcal P)-v(\mathcal P\setminus\{p\}).
\]

The complete table also stores every contextual marginal \(v(C\cup\{p\})-v(C)\); Shapley aggregates those marginals, whereas the LOO baseline uses only the grand-coalition context.

### Property 2 — perfect substitutability and grand-coalition LOO

Let \(p_1,p_2\) satisfy, for every \(C\subseteq\mathcal P\setminus\{p_1,p_2\}\),

\[
v(C\cup\{p_1\})=v(C\cup\{p_2\})=v(C\cup\{p_1,p_2\}).
\]

Then

\[
\operatorname{LOO}^{grand}(p_1)=\operatorname{LOO}^{grand}(p_2)=0
\]

and, by Shapley symmetry,

\[
\varphi_{p_1}=\varphi_{p_2}.
\]

Their common value is not generally half of the grand-coalition pair-removal effect when other players exist. More precisely,

\[
\varphi_{p_1}=
\sum_{C\subseteq\mathcal P\setminus\{p_1,p_2\}}
\frac{|C|!(K-|C|-1)!}{K!}
[v(C\cup\{p_1\})-v(C)],
\]

with the same expression for \(p_2\). If the first redundant player contributes a context-independent \(\delta\), then each receives \(\delta/2\). Include a three-player counterexample to the rejected half-pair-removal formula in Appendix A.

**Locked three-player counterexample.** Let

| Coalition | \(\emptyset\) | \(p_1\) | \(p_2\) | \(p_1p_2\) | \(p_3\) | \(p_1p_3\) | \(p_2p_3\) | \(p_1p_2p_3\) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| \(v(C)\) | 0 | .10 | .10 | .10 | .05 | .10 | .10 | .10 |

Then \(p_1,p_2\) satisfy perfect substitutability, both grand-LOO values are zero, and \(\varphi_1=\varphi_2=.0417\), \(\varphi_3=.0167\). The rejected half-pair-removal expression gives \((.10-.05)/2=.025\), demonstrating why it fails with a third player.

**Interpretation:** grand-coalition LOO can assign zero to both perfect substitutes while Shapley gives equal, potentially nonzero credit based on contexts in which the first substitute enters. Positivity is not guaranteed. This exact condition is a synthetic property, not a claim about natural augmentations.

For empirical proximity, report

\[
\epsilon_{pq}=\max_{C\subseteq\mathcal P\setminus\{p,q\}}
\max\!\left(
|v(Cp)-v(Cq)|,
|v(Cpq)-v(Cp)|,
|v(Cpq)-v(Cq)|
\right),
\]

with `Cp` denoting \(C\cup\{p\}\). Call a pair approximately substitutable only descriptively, relative to the preregistered `delta_interaction=0.003` and its uncertainty; do not claim the exact property holds in real data.

### Grabisch–Roubens Shapley interaction index

Use the context-averaged Grabisch–Roubens convention and define

\[
I_{pq}=\sum_{S\subseteq\mathcal P\setminus\{p,q\}}
\frac{|S|!(K-|S|-2)!}{(K-1)!}
\big[v(S\cup\{p,q\})-v(S\cup\{p\})-v(S\cup\{q\})+v(S)\big].
\]

Negative values are consistent with substitutability and positive values with complementarity under the declared utility. Treat an interaction as practically directional only when its 95% interval excludes zero and its absolute mean is at least the preregistered `delta_interaction=0.003`; otherwise report it as inconclusive. Representation similarity is supporting evidence only.

## 3.6 Per-user and behavioural-segment decomposition

Let \(m_{u,s}(C)\) be user \(u\)’s NDCG contribution. Because the aggregate metric is a mean over the same users,

\[
v_s(C)=\frac1{|\mathcal U|}\sum_u v_{u,s}(C).
\]

### Property 3 — linear decomposition

\[
\varphi_{p,s}=\frac1{|\mathcal U|}\sum_u\varphi_{p,u,s}.
\]

This requires no additional model training, but does require per-user storage and \(O(|\mathcal U|K2^K)\) aggregation. These are user-specific evaluations of globally trained coalition models, not separately trained personal recommenders.

With one held-out item, individual NDCG is coarse. Do not present individual explanations as reliable estimates. Primary RQ3 evidence consists of averages over Q1–Q4 pre-truncation training-history segments whose boundaries were frozen before attribution was observed.

### Heterogeneity test

- Behavioural labels are independent of Shapley outcomes.
- Test the confirmatory increasing mask trend with \(T_{mask}=\sum_{m=1}^{4}(m-2.5)\mu_{mask,m}\), and test omnibus profile dispersion with the prespecified studentized statistic.
- Form both permutation distributions by shuffling frozen behavioural labels at user level and applying the same map across seeds.
- Report raw segment uplift and raw Shapley values before shares.
- Use bootstrap confidence intervals and suppress unstable shares when segment uplift is near zero.

Attribution-derived clustering is exploratory. Evaluate it with bootstrap/seed stability and held-out cluster-quality diagnostics, not a circular shuffled-label test.

## 3.7 SHAPER-Weight

Uniform contrastive training uses

\[
\mathcal L_{cl}^{uniform}=K^{-1}\sum_p\mathcal L_{cl}^p.
\]

Convert validation Shapley values to a nonnegative target distribution:

\[
q_p=\frac{\max(\varphi_p,0)}{\sum_h\max(\varphi_h,0)},
\]

with a numerical uniform fallback when the denominator is below a fixed tolerance. If all raw values or Game-A grand uplift are nonpositive, SHAPER-Weight is declared not activated and the deployment fallback is rec-only rather than relabeling uniform contrast as an adaptive success. Otherwise apply shrinkage:

\[
w_p(\alpha)=(1-\alpha)K^{-1}+\alpha q_p.
\]

Average the canonical fixed-nominal-budget Game-A seed-specific \(V_{game}\) Shapley vectors to form one dataset-level target distribution. NLL/cosine severity-control-derived weights are exploratory and cannot replace this target after outcomes are observed. The weighted model uses

\[
\mathcal L=\mathcal L_{rec}+\lambda\sum_p w_p(\alpha)\mathcal L_{cl}^p,
\qquad \sum_p w_p=1,
\]

with no additional `1/K` factor. Select one \(\alpha\in\{0,.25,.5,.75,1\}\) on disjoint \(V_{select}\) users using the same three recipe-calibration initializations, then evaluate five separate final seeds on test. Report raw signed allocation separately. Clipping means the training weights no longer satisfy the Shapley efficiency identity.

This is a hypothesis-driven transfer probe. Fairness axioms do not imply weight optimality. Compare with uniform weighting, LOO-derived weights, learned training-time gates, a prespecified 15-point simplex calibration, and a ten-vector Dirichlet validation reference under explicit budgets.

## 3.8 SHAPER-Select

Select a lower-cost view set from validation attribution:

SHAPER-Select activates only when canonical Game-A grand uplift satisfies the same positive-uncertainty rule as Weight; otherwise deployment falls back to rec-only. When active, let \(\varphi_{(1)}\le\varphi_{(2)}\) be the two lowest values. If \(\varphi_{(2)}-\varphi_{(1)}<\delta_\varphi\), Select takes no removal action and retains the grand coalition. Otherwise,

\[
C_{select}=\mathcal P\setminus\{\arg\min_p\varphi_p\},
\]

with display tie order crop, mask, reorder, dropout. Because all coalitions were already trained, validation performance of the locked action is auditable. Evaluate the locked selected coalition on test. Plot remove-and-retrain curves for Shapley, LOO, and random orderings using test outcomes only after orderings are fixed.

SHAPER-Select is a transfer/faithfulness probe directly supported by the discrete game, not a new scalable selection algorithm. With three players, its removal display has only a few points and reuses enumerated coalition models. Report this limitation, augmentation-generation and training savings, and ranking quality.

## 3.9 Statistical estimand and uncertainty

For each seed, calculate a complete coalition table and exact Shapley vector. Exact enumeration does not remove stochastic uncertainty from initialization, optimization, dropout, negatives, or view draws.

Report:

- per-seed aggregate values and Shapley vectors;
- matched-seed distributions and hierarchical confidence intervals as the confirmatory uncertainty;
- seed-level paired tests as the unit for Holm adjustment;
- user-paired effects, Wilcoxon, and rank-biserial/Cliff’s delta as descriptive conditional-on-model analyses;
- absolute and relative metric changes with practical-effect thresholds.

Two full pilot seeds per dataset are excluded from confirmatory estimates, and the final archive is labeled pilot-informed. Appendix J combines finalized `V_game` sizes, pilot variance estimates, and a prespecified variance grid using the conservative planning formula \(h=t_{.975,S-1}\sqrt{\sigma^2_{seed}/S+\sigma^2_{user}/N_{game}}\), alongside hierarchical-bootstrap estimates. The smallest practically relevant effects are fixed at 0.003 for raw NDCG Shapley, pair interactions, and intervention NDCG; Appendix J reports whether the pilot-informed five/ten-seed designs can resolve that common utility-scale threshold. After five seeds, canonical Games A/B extend to ten if any raw canonical Shapley interval exceeds the first half-width or shared grand uplift exceeds the second; the NLL companion follows Game A but does not trigger. Final Weight models have a separate `V_select` precision trigger. Activation additionally requires positive grand-uplift uncertainty and at least 80% seedwise sign stability for positively weighted views.

A prespecified smooth full-catalog target-negative-log-likelihood game is secondary for RQ1/RQ2 and co-primary with NDCG for RQ3:

\[
v^{NLL}(C)=-\operatorname{NLL}(C)+\operatorname{NLL}(\emptyset),
\]

on `V_game`. It cannot replace NDCG for RQ1/RQ4; RQ3 reports both co-primary segment outcomes and labels disagreement metric-dependent.

For segment heterogeneity, use a studentized dispersion statistic and apply the same user-label permutation across seeds. Separate Holm families cover: six intervention comparisons; 12 confirmatory mask tests (six dataset/budget tests under each of NDCG and utility NLL); and, within each dataset, six canonical pair-by-budget interaction tests. Trends from the NLL/cosine **severity-calibration games** remain exploratory.

## 3.10 Complexity and implementation

Let \(B\) be the cost of one full coalition training.

| Stage | Cost | Interpretation |
|---|---:|---|
| One coalition game | \(O(2^K B)\) per seed | eight complete trainings for K=3 |
| Two co-primary canonical budget games | \(O((2\cdot2^K-2)B)\) per seed | shared empty/grand; distinct intermediates |
| NLL-matched Game A companion | \(O(2^K B)\) per seed | rec-only corruption-severity control |
| Cosine-matched Game A diagnostic | \(O(2^K B)\) per diagnostic seed | representation-displacement control |
| Beauty K=4 Game A | \(O(16B)\) per seed | required RQ2 extension |
| Aggregate Shapley | \(O(K2^K)\) | negligible after model training |
| User decomposition | \(O(|\mathcal U|K2^K)\) | no extra model fits |
| Pair interactions | \(O(K^2 2^K)\) | negligible for K=3 |
| Final weighted model | \(O(B)\) per selected setting/seed | training-time intervention |

The full declared study—including two co-primary canonical budget games, NLL/cosine severity controls, and calibrated interventions—is expected to require approximately 110–190 single-GPU hours, with additional cost if the precision rule expands co-primary games to ten seeds. Replace this planning range with measured runtime and energy. A cached ranking-adapter surrogate may be evaluated in an appendix against the full game; an inference-irrelevant projection-head refit is not a valid surrogate.

---

# 4. EXPERIMENTS AND RESULTS

> This section is a results template. Replace every placeholder with executed values. Do not convert registered expectations into observed claims.

## 4.1 Datasets and preprocessing

**Table 2** is generated from the finalized data artifact.

| Dataset | Positive interactions before core | Users after core | Items after core | Interactions after core | Density | Mean train length | Truncated % |
|---|---:|---:|---:|---:|---:|---:|---:|
| MovieLens-1M | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Amazon-Beauty | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

MovieLens statistics must reflect the rating-≥4 conversion; do not report the raw 1,000,209 ratings as positive interactions. Describe iterative 5-core filtering, deterministic temporal splitting, train/validation/test histories, quartile edges frozen on the full validation-eligible pool before role assignment, and the 20%/60%/20% length-stratified assignment of validation users to \(V_{tune}\), \(V_{game}\), and \(V_{select}\). Persist `raw_row_id` for every retained event. Report Amazon same-day validation/test tie rates; primary ordering uses raw row order, while robustness ordering uses a persisted salted hash of `(user,item,timestamp,raw_row_id)`. Acknowledge that iterative core eligibility is computed from the full record before holdout, a standard transductive preprocessing choice that uses held-out interactions for eligibility.

## 4.2 Protocol, metrics, and baselines

### Evaluation

- full retained-item-catalog ranking, justified against sampled-metric evaluation literature;
- disjoint validation-user roles for tuning, game construction, and intervention selection;
- all eligible users in final test tables after decisions are locked;
- all prefix items filtered; if the target repeats an earlier item, retain it as a candidate and report the repeat-target rate;
- deterministic score ties;
- NDCG@10 primary;
- HR@10 and MRR@10 secondary;
- coverage and item-popularity/Gini analysis if retained in the final tables.

Recall@10 is omitted because it equals HR@10 with one relevant target.

### Recommendation baselines

- rec-only SASRec coalition \(C=\emptyset\);
- uniform three-view contrastive SASRec;
- relevant sequential references such as GRU4Rec, CL4SRec, DuoRec, and CoSeRec;
- LightGCN may appear only as a clearly labeled non-sequential reference, not as the main sequential comparator.

Before confirmatory training, archive the baseline implementations, versions, tuning budgets, deterministic flags, and hardware with a literature-search cutoff date. Report measured peak allocated/reserved GPU memory and wall-clock time; the design target is a single 8-GB GPU, not an unverified guarantee. Update the narrative through submission, but label any later-added empirical baseline exploratory unless the confirmatory plan is formally amended.

### Attribution/action baselines

- grand-coalition LOO;
- context-averaged interactions;
- learned training-time gates;
- LOO-derived weights;
- limited direct simplex search;
- random removal and random weight controls;
- sampled-permutation Shapley convergence as an approximation audit against the exact answer;
- all coalition-specific marginal contributions, not only grand-LOO.

GraSP and Beta Shapley are discussed in related work but are not empirical weighting baselines: GraSP scores training examples through gradients, while Beta Shapley changes data-valuation coalition weighting. Neither estimates the same three-view intervention game without a new method definition.

## 4.3 RQ1 — exact augmentation attribution

**Table 3:** complete `V_game` coalition values for co-primary canonical Games A/B and NLL-matched Game A; cosine-matched Game A appears as a three-seed diagnostic panel.  
**Table 4:** raw Shapley values for every declared game, stable shares where valid, maximum efficiency residual, and seed uncertainty.  
**Figure 2:** fixed-budget, fixed-dose, NLL-matched, and cosine-matched raw view credit by dataset with confidence intervals.

Required statements:

- all nonempty coalition models produce ranking-relevant differences;
- efficiency holds for every seed and the maximum absolute numerical residual is reported;
- every observed monotonicity violation is listed with uncertainty and diagnosed rather than labeled automatically as training failure;
- exactness is conditional on each realized game;
- whether grand-coalition uplift is distinguishable from zero;
- no normalized-share headline where the denominator is unstable;
- how the co-primary fixed-nominal-budget and fixed-per-view-dose games differ and which decision each supports;
- whether NLL- and cosine-matched severity diagnostics preserve the canonical conclusions.

## 4.4 RQ2 — redundancy, interaction, and removal faithfulness

**Table 5:** Shapley, LOO, pair interaction, joint-removal effect, and representation-similarity diagnostic.  
**Figure 3:** three-player removal displays ordered by `V_game` Shapley, `V_game` LOO, and random order, evaluated on locked test outcomes using the already enumerated subset models; do not draw or describe them as smooth scalable removal curves.  
**Figure 4:** pair interaction values across datasets/seeds, with the required five-seed Beauty K=4 Game-A panel providing more than two contexts per pair and longer removal orderings.

The K=3 panels remain exact small-game diagnostics; the K=4 Beauty extension is the primary stress test for whether Shapley/LOO disagreement and interactions remain informative beyond three players. Use “consistent with substitutability” or “consistent with complementarity,” not “proved redundant,” for empirical observations. The forced-duplicate synthetic game validates the theorem and implementation separately from natural-data claims.

## 4.5 RQ3 — behavioural heterogeneity

**Figure 5:** raw NDCG- and NLL-game view credit by Q1–Q4 training-length segment.  
**Table 6:** both co-primary RQ3 outcomes, segment uplift, raw values, stable shares, confidence intervals, and studentized label-permutation tests.

The only confirmatory segment direction is increasing raw mask credit from Q1 to Q4, motivated by longer histories providing more transitions and sufficient residual context for missing-event invariance. Crop/reorder trends are exploratory. The omnibus segment-profile test is confirmatory but does not preregister rejection. Report homogeneous results without reframing them as implementation failures.

Exploratory attribution clusters, if stable, belong in the appendix or a clearly labeled exploratory subsection.

## 4.6 RQ4 — weighting and view selection

**Table 7A—unconditional transfer:** test NDCG@10, HR@10, MRR@10, training cost, and augmentation cost for rec-only, uniform, the locked forced Weight candidate, the forced lowest-Shapley removal candidate, LOO controls, learned gates, and direct/random controls—regardless of activation.

**Table 7B—activation-conditioned deployment:** the model selected by the locked activation/no-action rule. A failed activation appears as rec-only with status `not activated`, never as a Weight/Select success.

**Figure 6:** `V_select` \(\alpha\) curve based on the fixed `V_game` target weights; test reports only the setting selected without test feedback.  
**Figure 7:** validation-versus-test relationship for prespecified weighting controls or a coalition-value/interactions heatmap.

Table 7 labels direct search explicitly as **15-point, one-initialization search plus five-seed confirmation**, not a five-seed-tuned oracle. If SHAPER does not improve test ranking, report the null. If direct search performs better, quantify the performance-versus-search-budget trade-off.

## 4.7 Sensitivity and robustness

Prespecified checks:

- frozen-encoder η/γ/β and protect-last-1/2 diagnostics, with only the locked pilot singleton/grand ranking confirmation;
- co-primary canonical fixed-nominal-budget `1/|C|` and fixed-per-view-dose `1/K` games;
- required main-text NLL-matched Game A and three-seed cosine-matched Game A diagnostic;
- \(\lambda\) and temperature sensitivity, including singleton diagnostics;
- required five-seed Beauty K=4 Game-A dropout-view extension;
- NDCG@5/10/20;
- no-op/easy-positive rates by dataset and segment;
- optional cached ranking-adapter surrogate against full retraining;
- smooth validation-NLL game versus ranking-quality game;
- identical-configuration grand-coalition repeat to measure residual nondeterminism.

Do not add a dataset or alter severity after seeing a failed headline unless labeled exploratory.

## 4.8 Statistical analysis

Report seed-level means/distributions and hierarchical intervals as confirmatory uncertainty. Apply Holm only to seed-level paired tests in the declared families. Report user-level paired tests and effect sizes as descriptive conditional-on-model analyses. Lead with absolute effects, practical thresholds, and intervals rather than significance.

---

# 5. DISCUSSION

## 5.1 What augmentation credit means

A view’s Shapley value is its average marginal contribution under the declared training interventions and evaluation metric. The player includes its natural applicability/no-op frequency; no-op examples receive zero loss without reallocating the coalition denominator. Credit is not an intrinsic property of “masking” independent of severity, architecture, data, budget policy, or no-op rule. Because contrast uses the last valid hidden state, suffix crop and final-position masking are mechanistically privileged; the result values these exact procedures, not abstract augmentation semantics. Negative credit is valid.

Interaction indices add a different statement: whether two procedures tend to substitute for or complement one another across coalition contexts. Embedding similarity alone cannot answer this question.

## 5.2 Attribution versus optimization

Fair allocation and optimal continuous weighting are different problems. SHAPER-Weight tests whether an attribution signal is useful as a regularized heuristic. SHAPER-Select is more directly aligned with the discrete game because it acts on the same inclusion/exclusion decisions. A failure to improve does not invalidate the attribution identity; it demonstrates a gap between explanation and intervention.

## 5.3 Limitations and threats to validity

Include at least:

- three views define the cross-dataset game and one SASRec-style backbone is used; Beauty alone has a required K=4 extension under Game A, so K=4 policy dependence between Games A/B and larger-player generalization remain untested;
- exact coalition enumeration but finite seed uncertainty;
- one held-out target makes individual user utilities coarse;
- two datasets do not establish universal length/density effects;
- view definitions, suffix corruption, and severity influence credit;
- fixed-nominal-budget `1/|C|` and fixed-per-view-dose `1/K` define different games;
- architecture, `lambda`, `tau`, and the step budget are calibrated using empty/grand models and may under-tune singleton coalitions;
- offline NDCG is not online utility;
- validation targets are partitioned by user into tuning, game, and intervention-selection roles, which reduces leakage but also reduces each calibration sample; test remains feedback-free;
- attribution clusters are exploratory;
- segment-adaptive methods may interact with popularity and demographic biases not observed in these public datasets;
- every budget/severity game costs eight full trainings per seed, with additional calibrated intervention models; attribution is not a seconds-only procedure.

The optional cached surrogate, if used, is a separate estimand and must be validated against full retraining.

---

# 6. CONCLUSION

The conclusion should answer the four RQs with observed evidence and uncertainty. It should distinguish:

1. the mathematical efficiency identity;
2. empirical view credit;
3. empirical interactions and segment differences;
4. held-out intervention performance.

Future work may consider Owen values for grouped input/model perturbations, online drift in view credit, richer histories with multiple evaluation targets, constrained coalition structures, learned augmentation policies, and privacy/utility interactions.

---

# DECLARATIONS

Funding · Competing interests · Ethics/public secondary data · Data availability · Code availability · Author contributions · AI-tool disclosure · Preregistration archive DOI/timestamp and pilot-informed amendment link.

Do not state code availability until the repository and immutable experiment manifests are actually public.

---

# APPENDICES

- **A.** Derivations for Properties 1–3, seed-linearity identity, metric-scale remark, and the three-player counterexample to the rejected redundancy formula
- **B.** Complete coalition tables by seed, dataset, and split
- **C.** Hyperparameter grids, calibration budgets, fixed-step selection, and selected settings
- **D.** Interaction-index definition and alternative interaction conventions
- **E.** Full behavioural-segment values and uncertainty
- **F.** Required Beauty K=4 dropout-view coalition tables and extended RQ2 diagnostics
- **G.** NLL- and cosine-matched corruption-severity diagnostics, no-op accounting, and parameter grids
- **H.** Cached ranking-adapter surrogate versus full-retraining game
- **I.** Preregistration compliance: hypothesis, direction, result, match/miss, and exploratory amendments
- **J.** Pilot-informed MDE and CI-width tables for five and ten seeds

---

# PLANNED FIGURES AND TABLES

| # | Type | Content |
|---|---|---|
| Fig. 1 | Diagram | coalition-specific training: common initialization → eight independent models → value table → allocation/action |
| Fig. 2 | Bars/intervals | canonical Games A/B plus NLL/cosine severity-control Shapley by dataset |
| Fig. 3 | Curves | test removal faithfulness for Shapley, LOO, random order |
| Fig. 4 | Interaction plot | pairwise context-averaged interactions by dataset |
| Fig. 5 | Segment plot | raw view credit and uplift by behavioural length segment |
| Fig. 6 | Line | `V_select` coarse alpha path; selected point marked |
| Fig. 7 | Heatmap/scatter | coalition values or validation-test intervention relationship |
| Tab. 1 | Related work | positioning with verified literature claims |
| Tab. 2 | Data | generated post-filter statistics |
| Tab. 3 | Coalition values | all eight values with seed uncertainty |
| Tab. 4 | Attribution | raw Shapley, stable shares, efficiency residual |
| Tab. 5 | RQ2 | Shapley, LOO, interaction, joint removal, similarity |
| Tab. 6 | RQ3 | segment uplift and raw credit with valid test |
| Tab. 7A/B | RQ4 | unconditional transfer and activation-conditioned deployment, with cost |
| Tab. 8 | Statistics | confidence intervals, effects, adjusted tests |

---

# INTERNAL BUILD PLAN — remove before submission

1. Finalize data protocol, validation-role hash, timestamp/repeat diagnostics, and generated Table 2.
2. Use the first excluded pilot/engineering seed to select and verify the fixed-step recipe, full-catalog evaluation, and same-config repeat behavior.
3. Complete two excluded canonical Game-A pilot seeds per dataset, pass tests 1–18, and generate provisional variance estimates.
4. Archive Appendix J, all seed integers, batch/optimizer recipe, scope table, deltas, budget estimands, canonical/NLL/cosine residuals, Holm families, activation/tie rules, and seed triggers.
5. Run the locked K=3 games and five-seed Beauty K=4 Game-A extension; apply the direction-blind seed rule exactly as specified.
6. Calculate complete value tables, Shapley, LOO, interactions, and modest three-view removal orderings.
7. Run the fixed, studentized behavioural-segment analysis.
8. Calibrate Weight/controls on `V_select`, lock one dataset-level recipe, and evaluate all eligible test users.
9. Add only the prespecified mask-policy and cached-surrogate analyses beyond the declared games.
10. Replace all future-tense placeholders with measured results; retain nulls and amended hypotheses. Then run a prose pass that favors active voice, defines each acronym once, and removes repeated hedging.

## No-go conditions

- A projection-only refit does not change ranking.
- Smaller coalitions inherit an all-view checkpoint.
- Test outcomes influence weights, selected views, segments, or thresholds.
- Property 2 uses the rejected half-pair-removal formula without the stronger context-independent marginal assumption.
- Raw MovieLens counts are presented as post-filter statistics.
- Exact enumeration is described as eliminating training uncertainty.
- Coalitions receive different optimizer-step budgets within either co-primary game.
- An identity contrastive view is called a mathematical dummy; adding its loss can change training and coalition normalization.

## Expected effort

Across the two datasets, the canonical Games A/B, NLL companion, cosine diagnostic, and required Beauty K=4 extension total approximately 332 distinct coalition trainings after valid empty/grand reuse, before pilots, recipe calibration, and intervention controls. Budget roughly 110–190 single-GPU hours, with prespecified expansion if co-primary precision is inadequate, then replace estimates with measurements. The project is not described as a one-week CPU/head-refit exercise.
