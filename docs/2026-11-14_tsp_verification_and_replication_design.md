# TSP pilot: independent verification, root-cause analysis and replication design (2026-11-14)

Status: **verification COMPLETED (validation, analysis only; no training); replication design DRAFT, NOT FROZEN,
awaiting approval.** Nothing in this document changes the pre-registered pilot verdict (NOT SUPPORTED, commit
`214fc7a`). The pilot folder, its pre-registration, checkpoints and outputs were not modified.

Evidence produced for this report: `results/geometry_of_truth/truth_subspace_preservation/verification/`
(script `scripts/geometry_of_truth/28_tsp_independent_verification.py`, 1,966 s on the A10G, 0 TEST rows).
All post-hoc analyses below are labelled **post hoc**; they inform the next design and are not results.

---

## 1. Independently verified TSP results

**Method of verification.** Script 28 re-implements every number the verdict rests on without importing the
pilot's evaluation code (scripts 24/26/27, M2, `information_levels`, the ladder, the splicing module or the SAE
class):
* its own GoT loader, reading the raw activation files and split CSVs;
* its own TopK forward pass from the state dict;
* sklearn probes with its own grouped C selection;
* its own grouped bootstrap (2,000 resamples of validation cities, shared by all SAEs);
* its own GPT-2 splicing hook;
* its own statistics.

**Agreement with the pilot outputs** (all 18 SAEs, `verification/agreement_with_pilot.csv`):

| Quantity | Max \|ours − pilot\| | Comment |
|---|---|---|
| Frozen logistic AUROC gap | 0.0003 | probe solver seed differs (0 vs 42) |
| Re-fit logistic gap | 0.0002 | |
| Re-fit MLP gap (3 × 64-unit nets) | 0.0046 (mean 0.0004) | MLP training noise, one SAE |
| Error-only AUROC | 0.0003 | |
| GoT NMSE, GoT L0 | < 1e-8, 0 | |
| OWT NMSE, OWT L0 | < 1e-8, 0 | |
| Spliced ΔCE, KL | < 1e-5 | |

The 12 checkpoints load, their SHA-256s are recorded, and the configurations and the bit-identical recovery
reproduction were already verified by script 25b. **The pilot's numbers are correct.**

**Verified results** (VALIDATION; GoT readers fitted on group B; mean ± sd over 3 paired seeds; lower gap = better):

| Metric | R2 base | ctrl | tsp | rand | pca |
|---|---|---|---|---|---|
| Frozen logistic gap | 0.210 ± 0.008 | 0.051 ± 0.014 | **0.012 ± 0.017** | 0.091 ± 0.005 | 0.060 ± 0.007 |
| Re-fit logistic gap (C chosen by CV) | 0.063 ± 0.024 | 0.019 ± 0.004 | **−0.001 ± 0.007** | 0.024 ± 0.025 | 0.026 ± 0.008 |
| Re-fit MLP gap, pilot reader (3 nets) | 0.032 ± 0.014 | 0.037 ± 0.016 | 0.027 ± 0.009 | 0.028 ± 0.014 | 0.027 ± 0.004 |
| Re-fit MLP gap, 10 nets (new) | 0.024 ± 0.008 | 0.028 ± 0.004 | 0.015 ± 0.010 | 0.019 ± 0.005 | 0.021 ± 0.002 |
| Re-fit MLP gap, 5 × 256 units (new) | 0.029 ± 0.009 | 0.021 ± 0.006 | 0.005 ± 0.009 | 0.018 ± 0.008 | 0.016 ± 0.008 |
| Error-only AUROC (frozen probe on x − x̂) | 0.708 ± 0.041 | 0.708 ± 0.019 | 0.609 ± 0.039 | 0.743 ± 0.016 | 0.716 ± 0.024 |
| GoT NMSE (validation) | 0.840 | 0.037 | 0.043 | 0.053 | 0.037 |
| OWT NMSE / EV | 0.0959 / 0.9041 | 0.0974 / 0.9026 | 0.0974 / 0.9026 | 0.0971 / 0.9029 | 0.0974 / 0.9026 |
| Spliced ΔCE (nats) | 0.0603 | 0.0614 | 0.0618 | 0.0617 | 0.0614 |
| Spliced KL | 0.0680 | 0.0691 | 0.0691 | 0.0689 | 0.0691 |
| M4 T_rec/T_full (pilot ladder) | 0.726 | 0.706 | 0.711 | 0.719 | 0.712 |
| L0 on OWT / GoT / RepE | 128 / 128 / 128 for every SAE | | | | |

The original-activation readers score: logistic 0.819, 3-net MLP 0.846, 10-net MLP 0.851, 256-unit MLP 0.853.
These were computed on 448 validation statements (112 cities) with probes trained on 1,048 group-B statements.

**Pre-registered rule, recomputed** (tsp − ctrl; se = √(se_seed² + v_stmt); t with df = 2; Holm over 2):

| Co-primary | Per seed | Mean [95 % CI] | p | Holm p | Pass |
|---|---|---|---|---|---|
| Frozen logistic gap | −0.040 / −0.049 / −0.027 | −0.039 [−0.085, +0.008] | 0.070 | 0.140 | no |
| Re-fit logistic gap | −0.021 / −0.023 / −0.016 | −0.020 [−0.058, +0.018] | 0.151 | 0.151 | no |

The pilot reported Holm p 0.135 / 0.147; the small differences come from the different bootstrap draws (2,000 vs
1,000). Specificity (3/3 seeds against rand and pca on both co-primaries) and all five guards are reproduced.
**Verdict confirmed: NOT SUPPORTED.**

**M4 causal share** was not re-computed; it would require re-implementing the patching decomposition. It was
checked from the raw ladder rows: tsp − ctrl = +0.006 [−0.010, +0.021], 3/3 seeds.

## 2. Discrepancies and methodological concerns

None of these changes the verdict. Items 1, 3 and 4 correct claims I made in the previous report.

1. **Anti-conservative secondary intervals in the pilot (analysis bug in scope, not in the primary).**
   * The pilot saved bootstrap replicates only for the frozen/re-fit logistic and mass-mean gaps (`BOOT_KEYS`).
   * Its secondary CIs for error-only AUROC, MLP, sparse-z and kNN therefore omit statement sampling.
   * Example: error-only AUROC tsp − ctrl. The pilot reported [−0.154, −0.044], p = 0.016. With statement sampling
     included it is [−0.189, −0.009], p = 0.042; 63 % of its variance is statement sampling.
   * The primary metrics are unaffected.
2. **Construct validity of the frozen-probe metric.**
   * The frozen group-B probe direction has **58 % of its squared norm inside the truth subspace S**. Chance for a
     rank-8 subspace is 1 %; the random and PCA subspaces hold 1.0 % and 0.01 %.
   * S and the probe come from disjoint rows, so there is no row-level circularity. But TSP explicitly protects
     the evaluator's direction, so a frozen-gap improvement is close to guaranteed by construction.
   * The relative error along the probe direction falls 0.30 → 0.19 (R2 base: 7.6).
   * The error-only AUROC drop is the same effect seen from the error side.
   * The frozen gap and error-only AUROC should not be primary evidence of *information* preservation.
3. **Specificity was overstated.** My earlier summary said the gain "comes from the truth directions, not from any
   subspace penalty". That holds for the frozen gap only:
   * frozen gap: tsp − rand −0.079, p = 0.028; tsp − pca −0.047, p = 0.051 (df = 2);
   * re-fit logistic: tsp − rand −0.025, p = 0.24; tsp − pca −0.027, p = 0.12;
   * re-fit MLP (any reader): 2/3 seeds, p ≥ 0.29.

   The controls are also weak. Neither random nor PCA subspaces overlap the truth direction, so they test "any
   penalty" but not "any *supervised* GoT subspace". A label-permuted S, built with the same estimator but carrying
   no truth information, is the missing control.
4. **"No cost" holds against ctrl, not against R2 base.**
   * tsp − ctrl: OWT NMSE 0.0000, ΔCE +0.0004 [−0.0015, +0.0023], KL +0.0000.
   * GoT task replay itself costs, compared with R2 base, 3/3 seeds each:
     * OWT NMSE +1.6 % (CI excludes 0);
     * KL +1.6 % (p = 0.030);
     * ΔCE +1.9 % (p = 0.063). tsp vs base gives ΔCE +2.5 % (p = 0.045).
   * The pilot's guards compared point estimates with ctrl; a non-inferiority guard should use a one-sided CI bound.
5. **Re-fit gaps below zero.**
   * At the chosen C, tsp's re-fit logistic gap is −0.001; at C = 0.01 it is −0.017 ± 0.011.
   * So with strong regularisation x̂ is *more* linearly separable than x. A label-derived subspace (from group A)
     makes the SAE slightly *enhance* the truth contrast rather than merely preserve it, a mild version of the
     truth-aware distortion.
   * Small, but "preservation" should be scored two-sided (|gap|) or with an enhancement flag.
6. **The re-fit MLP reader was noisy.**
   * The pilot's 3 × 64-unit ensemble varies by up to 0.016 between seeds of the same arm, and 0.005 between two
     runs of the same SAE.
   * Larger ensembles show a consistent MLP improvement: 10 nets −0.013 (3/3), 256 units −0.016 (3/3, Satterthwaite
     p = 0.0008, post hoc). So the "MLP does not improve" reading came partly from reader noise.
7. **The df = 2 choice made the rule far more conservative than its nominal level** (see §4).
   * The statement-sampling variance (v_stmt) is estimated from 1,000 resamples, but it was given the seed component's
     df = 2.
   * Post hoc, with Satterthwaite df: frozen p = 0.0025 (df ≈ 16), re-fit p = 0.023 (df ≈ 760). Both would pass Holm.
   * This is **not** grounds to change the verdict. The rule was frozen, and choosing it after seeing results would
     be p-hacking. It is the main lesson for the next pre-registration.
8. **No held-out template exists inside the evaluated task.**
   * GoT-A training rows and the validation rows share the same two templates ("The city of X is [not] in Y.").
   * Countries recur everywhere; only the cities are disjoint.
   * A probe trained on affirmative statements scores AUROC 0.16 on negated ones (inverted), so even "wording
     transfer" within cities is not a usable held-out test.
9. Cosmetic: the pilot's `comparisons.csv` prints p = 0 for metrics with zero variance (L0).

## 3. Root cause of the inconclusive result

| Question | Finding | Evidence |
|---|---|---|
| Does TSP improve truth preservation? | **In distribution, probably yes, but small.** Every reader favours tsp over ctrl in 3/3 seeds: frozen −0.039, re-fit logistic −0.020, MLP −0.010 to −0.016. **Out of distribution, no evidence** | numerical-domain transfer below |
| Were 3 seeds adequate? | **No, and the pilot could not have passed.** At the observed effects, the probability of passing the frozen rule was 2 % (frozen) and < 0.1 % (re-fit) | `verification/power.csv` |
| Does statement sampling dominate? | **Yes.** 95 % of the re-fit variance and 65 % of the frozen variance come from sampling 112 validation cities (se_stmt 0.0086 vs se_seed 0.0020) | `comparisons.csv`, `stmt_share_of_var` |
| Is it mainly frozen readout recovery? | **Largely.** The frozen effect is twice the re-fit effect, and S contains 58 % of the frozen probe direction. A smaller re-fit and MLP effect remains | §2.2 |
| Why is the MLP effect smaller? | Partly reader noise (§2.6); partly because the original MLP reader is already better (0.85 vs 0.82), leaving less loss to recover; S is a linear object | |
| Does it depend on shared templates? | **Probably.** S captures no numerical truth: the numerical probe direction has 1.3 % of its norm in S, which is chance. On GoT numerical (new template and entity type), tsp − ctrl is frozen +0.014 (1/3 seeds), re-fit +0.004 (2/3), n.s. Caveat: numerical has a surface shortcut (second number alone gives AUROC 0.84; results registry) and every SAE reconstructs it badly (NMSE 3.4–3.8 within the set) | `per_sae.csv` `num__*` |
| Do the random and PCA controls establish truth specificity? | **Only for the frozen readout**, and they cannot tell "truth" from "any label-derived GoT subspace" | §2.3 |
| Are reconstruction, behaviour and causal share non-inferior? | **Against ctrl, yes** (point and interval). **Against R2 base, no:** GoT replay costs 1.6 % NMSE and about 2 % ΔCE/KL | §2.4 |

**Separating the three kinds of evidence:**
* **Numerical gains** are consistent: 3/3 seeds on every reader.
* **Statistical evidence** under the frozen rule is absent. Post hoc, under a df choice we did not pre-register, it
  would be moderate.
* **Genuine truth preservation** is not established. The largest effect is on the metric TSP targets by
  construction, specificity is shown only for that metric, and the only out-of-template check shows nothing.

The biggest single effect in the pilot is **task replay** (ctrl vs base: frozen −0.159, p = 0.016, against
1.6 % OWT NMSE). It is in-distribution and template-shared too.

## 4. Statistical power and uncertainty

**Model** (`power.csv`, 20,000 simulations per cell):
* each paired seed difference is d_s = δ + ε_stmt + ε_seed;
* ε_stmt is shared across seeds and has se_stmt · √(112/G) for G validation groups;
* ε_seed has the observed sd of the seed differences;
* the rule is the frozen one generalised to n seeds: all n favour tsp, t(n−1) CI with Bonferroni α/2.

| P(pass), re-fit logistic gap | 112 groups | 224 | 500 | 1,000 |
|---|---|---|---|---|
| 3 seeds, observed δ = −0.020 | 0.000 | 0.002 | 0.06 | 0.40 |
| 5 seeds | 0.11 | 0.38 | 0.86 | 0.99 |
| 8 seeds | 0.30 | 0.65 | 0.97 | 1.00 |
| 15 seeds | 0.43 | 0.77 | 0.99 | 1.00 |
| 8 seeds, δ/2 = −0.010 | 0.04 | 0.11 | 0.32 | 0.64 |
| 15 seeds, δ/2 | 0.09 | 0.19 | 0.46 | 0.77 |

The frozen gap is easier, because its seed component is larger: 5 seeds × 500 groups gives 0.98.

**Conclusions:**
1. **Five seeds are not sufficient on their own.** With the current 112 validation cities, even 15 seeds give
   43 % power for the information metric. The binding constraint is the number of **independent statements
   (groups)**, not seeds.
2. Observed pilot effects are probably inflated (winner's curse). A design should be powered for about half the
   observed effect (smallest effect of interest 0.01 AUROC). That is at the edge of what is feasible: 1,000 groups
   with 8 seeds give ≈ 0.64 for the re-fit metric (0.68 for the frozen metric; 0.85 without the all-seeds condition),
   and 15 seeds give ≈ 0.77. Effects of ≈ 0.02 are well within reach with ≥ 500 groups and ≥ 5–8 seeds.
3. The "all n seeds favour" condition becomes restrictive as n grows. At δ/2 with 15 seeds it alone fails half the
   time. It should be a descriptive consistency check, not part of the pass rule.
4. The interval should pre-register Satterthwaite df, or a two-way bootstrap over seeds × groups, instead of
   giving the statement variance the seed df.
5. Statement-sampling intervals cover only cities drawn from the *same template*. **Template-level variance is not
   estimable from one template.** It needs several held-out task families, with family as a random effect.

## 5. Literature-supported improvement opportunities

| Idea | Support | Assessment for TSP |
|---|---|---|
| **Universal truth subspace across families and polarity**: estimate S from several statement families with negations, separating a general truth direction from a polarity-sensitive one | Bürger et al. 2024 ([arXiv 2407.12831](https://arxiv.org/abs/2407.12831)): a 2-D truth subspace (general + polarity) explains why probes fail on negations; Marks & Tegmark 2023 ([arXiv 2310.06824](https://arxiv.org/abs/2310.06824)): cross-dataset probe transfer | **Most justified change.** Our S is cities-specific (numerical overlap at chance; affirmative probe inverts on negations). A multi-family S is the only way the method could generalise |
| **Rank** | Bürger: rank 2 suffices for general truth; our rank 8 mixes polarity-specific contrasts | Ablate r ∈ {2, 4, 8} on DEV only, report as secondary; do not tune on held-out families |
| **Label-permuted S control** | Standard randomisation control | Needed for specificity (§2.3) |
| **Subdomain fine-tuning / task replay** | Muhamed et al. 2025, Specialized SAEs ([arXiv 2411.00743](https://arxiv.org/abs/2411.00743)): subdomain data plus tilted ERM recovers tail concepts | Matches our largest effect (ctrl vs base). The replication must keep ctrl and report the OWT cost |
| **Information remains in the SAE error** | Engels et al. 2025 ([arXiv 2410.14670](https://arxiv.org/abs/2410.14670)): much of the error is linearly predictable | Consistent with error-only AUROC 0.61–0.74. Error-side metrics stay secondary |
| **Strong-baseline evaluation of SAE probing** | Kantamneni et al. 2025 ([arXiv 2502.16681](https://arxiv.org/abs/2502.16681)); GDM negative results 2025 ([Alignment Forum](https://alignmentforum.org/posts/4uXCAJNuPKtKBsi28/negative-results-for-saes-on-downstream-tasks)); SAEBench ([arXiv 2503.09532](https://arxiv.org/abs/2503.09532)) | Evaluate under covariate shift (held-out families) against the original-activation probe, not in distribution |
| **Gradient interference** (PCGrad, GradNorm) | Yu et al. 2020 ([arXiv 2001.06782](https://arxiv.org/abs/2001.06782)) | **Not justified yet.** The penalty left OWT NMSE unchanged (tsp − ctrl 0.0000), so there is no sign of conflict at λ = 1. Log the cosine between the reconstruction and subspace gradients first; act only if it is negative |
| **Multi-probe training** | — | **Not recommended.** It repeats the truth-aware circularity risk (docs/2026-11-07 audit). The subspace is the principled generalisation |
| **Nonlinear preservation: minimal-pair difference loss** ‖(x_t − x_f) − (x̂_t − x̂_f)‖² on same-entity true/false pairs | Uses labels only for pairing, fits no direction, and targets local truth geometry | Candidate for a **separate** exploratory pilot, not this replication |
| **Variance-aware evaluation** | Bouthillier et al. 2021 ([arXiv 2103.03098](https://arxiv.org/abs/2103.03098)): randomise data sampling as well as initialisation; Agarwal et al. 2021 ([arXiv 2108.13264](https://arxiv.org/abs/2108.13264)): interval estimates and stratified bootstrap for few runs | Basis for §4's recommendations |

## 6. Recommended experimental design: TSP-R (draft; not frozen)

**Two questions, answered by one set of runs:**
* **Q1, in-distribution replication.** Does the exact pilot method (S from cities group A, rank 8, μ = λ = 1)
  reduce the re-fit logistic gap with adequate power?
* **Q2, generalisation.** Does it reduce the gap on truth families with different templates and entities, which
  played no part in S, replay or probe selection?

**Phase 0 (no SAE training; about 1 h; needs approval to download public data).**
1. Download the public GoT files from `github.com/saprmarks/geometry-of-truth/datasets`. None is in the repo, so
   the project has never seen them:
   * `sp_en_trans` + `neg_sp_en_trans` (354 + 354);
   * `companies_true_false` (1,200; 3 templates);
   * `common_claim_true_false` (4,450; free-form);
   * `counterfact_true_false` (31,960; many relation templates);
   * `cities_cities_conj` / `_disj` (new template, city entities).

   Record SHA-256s.
2. **Before computing any activation**, split each file by a salted hash of its entity / relation group into DEV
   (30 %) and **TEST (70 %, locked)**. In counterfact, hold out whole relations, so TEST templates never occur in
   DEV. Freeze the split CSVs and their hashes.
3. Extract GPT-2 L8 (and L11) final-token activations.
4. **Eligibility on DEV only:** keep a family if the original group-cross-validated logistic AUROC on DEV is ≥ 0.70.
   GPT-2 Small may not represent truth for some families. The threshold is frozen before the check.
5. Cities cross-fitting: re-partition the 635 cities train + validation groups (the 113 test groups stay excluded;
   they are already exposed) into 3 folds. Each fold rotates its roles through A (S and replay), B (probes) and
   V (evaluation), so every city is evaluated once: **≈ 635 groups, 2,540 statements**.

**Phase 1 (training and evaluation).**

| Item | Specification |
|---|---|
| Model / site | GPT-2 Small frozen, `blocks.8.hook_resid_pre` (primary); L11 secondary if the budget allows |
| SAE | R2 TopK-12288, k = 128, 100M OWT tokens. **8 base seeds:** 42 / 1042 / 2042 (existing) + 5 new seeds trained with the unchanged `17_streaming_sae_training.py` recipe |
| Arms (5) | `ctrl` (OWT + GoT-A replay); `tsp` (+ S, rank 8, the exact pilot method); `rand` (Haar rank 8); `pca` (top-8 PCA of A); **`shuf`** (S from the same estimator with truth labels permuted within group and polarity) |
| References | R2 base (8 seeds) |
| Fine-tune recipe | Unchanged from script 25 (Adam 2e-5, clip 1, 1,230 steps × 32 × 127 OWT tokens, shard 79 chunks ≥ 10,000; μ = λ = 1); per fold |
| Runs | 5 arms × 8 seeds × 3 folds = 120 fine-tunes |
| Readers (fitted on original activations or on x̂ of the probe split only) | Frozen logistic and mass-mean; re-fit logistic (grouped-CV C, plus the fixed-C grid as sensitivity); re-fit MLP, 10 nets × 64 and 5 × 256; sparse-z |
| Held-out families | Probes fitted on the family's own train part. In Phase 1 they are scored on DEV only, and touched on TEST once at the end |
| Sparsity | L0 = 128 exactly (TopK) on OWT, every GoT family and RepE; reported separately |
| Behaviour / causal | Ladder on fold 1: OWT NMSE/EV, spliced ΔCE and KL, persona/neutral readers, M4 T_rec/T_full |

**Seeds and statistics:**
* Seeds are paired by base seed and fold.
* se = √(se_seed² + v_stmt) with **Satterthwaite df**.
* Sensitivity: two-way bootstrap over seeds × groups.
* Held-out families are pooled by random-effects meta-analysis (family as a random effect), with I² reported.

## 7. Exact datasets, splits, architectures, seeds and metrics

| | Training (S, replay) | Probe fitting | Evaluation | Never used |
|---|---|---|---|---|
| GoT cities / neg_cities | fold-A groups | fold-B groups | fold-V groups (≈ 635 in total) | 113 test groups (exposed) |
| GoT numerical | — | — | secondary, validation (already seen here; caveat: shortcut) | its test split |
| Fresh families (Phase 0) | — | family train part (DEV) | DEV in Phase 1; **TEST once** | — |
| OWT | shard 79 (5M tokens per run) | — | validation cache (12,700 tokens) | test |
| RepE-LM | — | persona/neutral validation (ladder) | validation | test (already exposed, H1-TEST) |

**Primary metrics (2, Holm):**
* **P1:** re-fit logistic gap, cross-fitted cities; tsp vs ctrl.
* **P2:** re-fit logistic gap, random-effects mean over eligible fresh families on **TEST**; tsp vs ctrl.

**Key secondary metrics:**
* the same comparisons for the 10-net MLP;
* the frozen logistic gap, labelled as a readout metric (§2.2);
* \|gap\| and the enhancement flag;
* the error-only AUROC with statement sampling.

## 8. Estimated GPU time and storage (single A10G, 4 vCPU)

Timings are measured on this instance:
* fine-tune ≈ 116 s;
* R2 base SAE ≈ 506 s (R2 log: 39 SAEs in 19,722 s);
* ladder ≈ 80 s per SAE;
* stage-1 readers ≈ 100 s per SAE per task family (CPU-bound).

| Step | Time |
|---|---|
| Phase 0: download, split, activations (~40k statements), DEV eligibility | ≈ 0.5–1 h |
| 5 new R2 base SAEs | ≈ 0.7–1 h |
| 120 fine-tunes | ≈ 3.9 h |
| Readers: 128 SAEs × (cities + about 4 families) | ≈ 6–8 h CPU, overlappable with training |
| Ladder, fold 1 (48 SAEs) | ≈ 1.1 h |
| **Total wall clock** | **≈ 10–13 h**, run under tmux with per-stage logs and resumable, refuse-to-overwrite outputs |

Storage: ~125 checkpoints × 75 MB ≈ 9.4 GB, plus R2 intermediates ≈ 1.5 GB, plus activations < 1 GB, so **≈ 12 GB**.
75 GB is free. Checkpoints stay gitignored.

Cheaper variant: 5 seeds, no cross-fitting, fresh families only. ≈ 4 h, but Q1 cannot be answered (power ≤ 0.11 on
112 groups).

## 9. Pre-registered success and failure criteria (draft)

* **Smallest effect of interest:** 0.01 AUROC.
* **Q1 supported:** P1 tsp − ctrl 95 % CI upper bound < 0 and Holm p < 0.05.
* **Q2 supported:** the same for P2.
* **Truth-specific:** additionally, tsp − shuf and tsp − rand on the same metric have CI upper bound < 0
  (Holm over the 2 controls). Otherwise the label is "subspace-penalty effect, not truth-specific".
* **Information vs readout:** if only the frozen gap passes, the label is "READOUT-ONLY".
* **Non-inferiority vs ctrl** (one-sided 95 % bound):
  * OWT NMSE ≤ +2 % (relative);
  * spliced ΔCE ≤ +5 %; KL ≤ +5 %;
  * persona re-fit gap ≤ +0.02;
  * M4 T_rec/T_full ≥ −0.02;
  * L0 = 128 exactly.

  A failure here gives the label "TRADE-OFF".
* **Absolute cost vs R2 base** is reported for every arm.
* **Enhancement guard:** if tsp's re-fit gap CI lies entirely below −0.01, report "distortion toward the label
  direction", not "preservation".
* **Failure, reported as such:** if P1 passes and P2 does not, TSP is in-distribution only. If neither passes at
  ≥ 0.9 power for δ = 0.02, TSP does not improve truth information at a size worth pursuing.
* **Stopping rules:**
  * non-finite loss;
  * L0 ≠ 128;
  * a change in the GPT-2 hash;
  * a TEST split read before the final step.

  No tuning after any DEV result except the frozen eligibility check.

## 10. Recommendation

**Do not run a plain "≥ 5 seeds" replication on cities.** The power analysis shows seeds are not the bottleneck
(15 seeds on 112 cities: 43 % power for the information metric). Such a run cannot address the main threat: the
effect may depend on the shared template and on S containing the evaluator's direction.

**Recommended:**
1. **Approve Phase 0 first** (≈ 1 h, no SAE training). Fresh public GoT families, hash-split into DEV/TEST before
   any activation is computed; DEV eligibility; cities cross-fitting folds. This settles whether GPT-2 Small at L8
   gives enough truth signal on held-out families for Q2 to be answerable at all.
2. **Then TSP-R as specified** (≈ 10–13 h), with the exact pilot method plus the label-permuted control. With
   ≈ 635 cross-fitted cities groups and 8 seeds it has high power (≈ 0.95+) for δ ≈ 0.02 on Q1 and only moderate
   power (≈ 0.4–0.6) for δ ≈ 0.01. For Q2, power depends on how many fresh-family TEST groups survive the DEV
   eligibility check; counterfact alone provides thousands.
3. **Do not change the architecture or loss yet.**
   * There is no evidence of gradient conflict.
   * Multi-probe objectives carry the known circularity risk.
   * A multi-family "universal" truth subspace (Bürger et al.) is the scientifically justified modification. Run it
     as a separately pre-registered follow-up only if TSP-R shows Q1 without Q2, so the replication stays a test of
     the method we already ran.
4. If budget is the constraint, prioritise Q2 (held-out families) over more cities seeds. That question decides
   whether TSP is a truth-preservation method at all.

Nothing has been trained. Waiting for approval.
