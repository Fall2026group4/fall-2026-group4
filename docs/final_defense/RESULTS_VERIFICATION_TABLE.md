# Results verification table (independently re-checked 2026-11-10)

Every number below was read or recomputed from a committed result file in this working tree. The source path is
given for each table. Numbers marked **[recomputed]** were recalculated from the raw per-SAE rows by
`scripts/shared/m5_independent_verification.py` (read-only), and they agree with the committed summaries.

**Conventions:**
* Improvement % = 100 × (reference − new) / reference for ↓ metrics. For ↑ metrics it is 100 × (new − reference) / reference.
* Means are over seeds 42 / 1042 / 2042; sd is in brackets; n = 3 per family unless stated.
* **Evidence labels:** TEST = M5 single pre-registered held-out pass. VALIDATION = validation split. HISTORICAL =
  kept for provenance only.
* **Do not compare across tables.** Each table has one protocol, one split and one SAE set.

---

## 0. What can and cannot be put into one before → after chain

| Layer set | Original (pre-review) SAE | Corrected baseline | Improved SAE (R2) | R4 KL fine-tune |
|---|---|---|---|---|
| 2, 6, 10 | yes (OWT test, seed 42) | yes (3 seeds) | **none trained** | none |
| 1, 8, 11 | only the 1000-step sweep (validation, seed 42; convergence-speed only) | yes (3 seeds) | yes (3 seeds) | L8, L11 only |

**No layer has a controlled original → corrected → improved chain.**
* The professor-review correction (original → corrected) was done at layers 2/6/10.
* The R2 improvement (corrected → improved) was done at layers 1/8/11.
* The two steps are reported separately below. Never read them as one three-step chain.

---

## 1. Stage B: original vs corrected baseline (OWT TEST, 12,700 tokens, L0 target 128, width 3072)

Source: `docs/archive/2026-11-04_methodology_review_corrections_report.md` §D1 (`reconstruction_comparison.csv`).
* **Original:** seed 42, with a mixed optimiser procedure.
* **Corrected:** one uninterrupted 15k-step protocol, 3 seeds.

| Layer | Arch | Original EV | Corrected EV (mean ± sd) | Change | Original → corrected rank |
|---|---|---|---|---|---|
| 2 | ReLU | 0.9362 | 0.9358 ± 0.0003 | −0.0004 | 1 → 1 |
| 2 | TopK | 0.9251 | 0.9259 ± 0.0003 | +0.0008 | 3 → 3 |
| 2 | JumpReLU | 0.9293 | 0.9355 ± 0.0017 | +0.0062 | 2 → 2 (tie with ReLU) |
| 6 | ReLU | 0.8311 | 0.8315 ± 0.0008 | +0.0004 | 3 → 3 |
| 6 | TopK | 0.8573 | 0.8572 ± 0.0002 | −0.0001 | 2 → 2 |
| 6 | JumpReLU | 0.8658 | 0.8649 ± 0.0004 | −0.0009 | 1 → 1 |
| 10 | ReLU | 0.7730 | 0.7727 ± 0.0004 | −0.0003 | 3 → 3 |
| 10 | TopK | 0.8217 | 0.8225 ± 0.0004 | +0.0008 | 2 → 2 |
| 10 | JumpReLU | 0.8253 | 0.8227 ± 0.0003 | −0.0026 | 1 → 1 (tie with TopK) |

**Reading:**
* The correction was **not** an improvement. It was a consistency fix: |ΔEV| ≤ 0.006 and no rank changes.
* Two earlier "wins" became ties: L2 ReLU vs JumpReLU, and L10 JumpReLU vs TopK.
* Evidence: TEST (OWT test was used for this table only).

**Historical descriptive reference at layers 1/8/11** (1000-step sweep; L0 target 128; seed 42; **VALIDATION**;
source `results/openwebtext/sae_sparsity_sweep/layer_{1,8,11}/comparison.csv`). It is not comparable to the tables
below: the budget is different (1000 vs 15,000 steps) and so is the split.

| Layer | ReLU EV | TopK EV | JumpReLU EV |
|---|---|---|---|
| 1 | 0.790 | 0.823 | 0.660 (L0 138) |
| 8 | 0.608 | 0.710 | 0.615 |
| 11 | 0.607 | 0.683 | 0.607 |

---

## 2. Stage D/E: corrected baseline vs R2 vs R4: OWT reconstruction and behaviour (M5 TEST) [recomputed]

Source: `results/shared/m5_confirmatory_test/m5_rows.csv`, `comparisons.csv`.
* **Data:** OWT test, 100 documents, 12,700 non-BOS tokens per layer. Original GPT-2 CE is 3.302 nats/token.
* **SAEs:**
  * "Old" = the corrected uninterrupted baseline (width 3072; 101,600 training tokens; 15k steps × 256 batch).
  * "R2" = width 12,288 (TopK-3072 is the width ablation); 100M fresh training tokens.
* **Matching:** all compared SAEs are within L0 128 ± 10 % on OWT test (127.6–133.4).
* **Seeds:** n = 3, except the frozen-decoder TopK and res-jb (n = 1; descriptive).

### 2a. Absolute values (mean (sd))

| Layer | SAE | NMSE ↓ | EV ↑ | Cosine ↑ | OWT L0 | Dead frac. ↓ | Spliced ΔCE (nats) ↓ | Spliced KL ↓ | SAE/noise ΔCE ratio ↓ |
|---|---|---|---|---|---|---|---|---|---|
| 1 | old ReLU | 0.0630 (0.0003) | 0.9370 | 0.9896 | 132.2 | 0.047 | 0.0141 (0.0017) | 0.0199 | 1.88 |
| 1 | old TopK | 0.0787 (0.0005) | 0.9213 | 0.9867 | 128.0 | 0.444 | 0.0221 (0.0016) | 0.0259 | 2.12 |
| 1 | old JumpReLU | 0.0675 (0.0004) | 0.9325 | 0.9877 | 129.3 | 0.683 | 0.0088 (0.0020) | 0.0139 | 1.31 |
| 1 | R2 ReLU | 0.0225 (0.0001) | 0.9775 | 0.9961 | 131.2 | 0.001 | 0.0038 (0.0003) | 0.0054 | 1.61 |
| 1 | R2 TopK | 0.0248 (0.0001) | 0.9752 | 0.9959 | 128.0 | 0.000 | 0.0056 (0.0011) | 0.0063 | 1.77 |
| 1 | R2 BatchTopK | **0.0178** (0.0001) | **0.9822** | 0.9969 | 131.5 | 0.000 | **0.0019** (0.0008) | 0.0034 | 1.02 |
| 1 | R2 TopK-3072 | 0.0512 (0.0000) | 0.9488 | 0.9916 | 128.0 | 0.000 | 0.0132 (0.0008) | 0.0149 | 1.97 |
| 1 | R2 TopK random frozen decoder (n=1) | 0.3217 | 0.6783 | 0.9412 | 128.0 | — | 0.1097 | 0.1349 | — |
| 1 | res-jb public (n=1, **L0 15, unmatched**) | 0.0506 | 0.9494 | — | 15.0 | — | 0.0172 | 0.0207 | — |
| 8 | old ReLU | 0.1951 (0.0005) | 0.8049 | 0.9457 | 131.9 | 0.004 | 0.2175 (0.0048) | 0.2360 | 2.00 |
| 8 | old TopK | 0.1598 (0.0005) | 0.8402 | 0.9543 | 128.0 | 0.085 | 0.1330 (0.0027) | 0.1402 | 1.50 |
| 8 | old JumpReLU | 0.1560 (0.0001) | 0.8440 | 0.9545 | 130.3 | 0.077 | 0.1212 (0.0011) | 0.1311 | 1.44 |
| 8 | R2 ReLU | 0.1320 (0.0001) | 0.8680 | 0.9626 | 127.6 | 0.001 | 0.1310 (0.0006) | 0.1399 | 1.81 |
| 8 | R2 TopK | 0.0941 (0.0000) | 0.9059 | 0.9733 | 128.0 | 0.002 | 0.0620 (0.0015) | 0.0661 | 1.20 |
| 8 | R2 BatchTopK | **0.0897** (0.0000) | **0.9103** | 0.9742 | 129.3 | 0.003 | 0.0542 (0.0008) | 0.0594 | 1.13 |
| 8 | R2 TopK-3072 | 0.1246 (0.0004) | 0.8754 | 0.9644 | 128.0 | 0.000 | 0.0843 (0.0024) | 0.0936 | 1.24 |
| 8 | R4 MSE control (TopK-12288) | 0.0938 (0.0000) | 0.9062 | 0.9734 | 128.0 | 0.002 | 0.0623 (0.0021) | 0.0659 | 1.21 |
| 8 | R4 KL (TopK-12288) | 0.0958 (0.0000) | 0.9042 | 0.9728 | 128.0 | 0.002 | **0.0514** (0.0016) | **0.0564** | 0.98 |
| 8 | random frozen decoder (n=1) | 0.4055 | 0.5945 | — | 128.0 | — | 0.6097 | 0.6198 | — |
| 8 | res-jb (n=1, **L0 66, unmatched**) | 0.1435 | 0.8565 | — | 66.0 | — | 0.1677 | 0.1757 | — |
| 11 | old ReLU | 0.2432 (0.0008) | 0.7568 | 0.9532 | 133.2 | 0.050 | 0.3417 (0.0008) | 0.3485 | 2.37 |
| 11 | old TopK | 0.1848 (0.0003) | 0.8152 | 0.9619 | 128.0 | 0.317 | 0.2041 (0.0021) | 0.2062 | 1.86 |
| 11 | old JumpReLU | 0.1851 (0.0001) | 0.8149 | 0.9619 | 132.1 | 0.162 | 0.2155 (0.0011) | 0.2178 | 1.96 |
| 11 | R2 ReLU | 0.1683 (0.0001) | 0.8317 | 0.9658 | 129.5 | 0.001 | 0.2205 (0.0036) | 0.2233 | 2.24 |
| 11 | R2 TopK | 0.1112 (0.0001) | 0.8888 | 0.9772 | 128.0 | 0.002 | 0.0962 (0.0025) | 0.0973 | 1.48 |
| 11 | R2 BatchTopK | **0.1073** (0.0002) | **0.8927** | 0.9781 | 131.7 | 0.004 | 0.0927 (0.0023) | 0.0924 | 1.48 |
| 11 | R2 TopK-3072 | 0.1394 (0.0001) | 0.8606 | 0.9713 | 128.0 | 0.000 | 0.1356 (0.0020) | 0.1352 | 1.64 |
| 11 | R4 MSE control | 0.1109 (0.0001) | 0.8891 | 0.9773 | 128.0 | 0.002 | 0.0964 (0.0019) | 0.0968 | 1.49 |
| 11 | R4 KL | 0.1133 (0.0001) | 0.8867 | 0.9768 | 128.0 | 0.003 | **0.0807** (0.0015) | **0.0813** | 1.22 |
| 11 | random frozen decoder (n=1) | 0.4142 | 0.5858 | — | 128.0 | — | 0.5004 | 0.5359 | — |
| 11 | res-jb (n=1, **L0 60, unmatched**) | 0.1883 | 0.8117 | — | 60.1 | — | 0.2616 | 0.2723 | — |

**Notes:**
* The dead fraction is measured over the 12,700 test tokens. A latent is counted dead if it never fires on them.
* The noise ratio is the SAE's ΔCE divided by the ΔCE of a norm-matched random error. Above 1 means the SAE error is
  more harmful than random noise of the same size.
* The cosines of the frozen decoder at L8/L11 are in `m5_rows.csv`. They were not printed here.

### 2b. Pre-registered comparisons (TEST): change, CI and class

**Rules:**
* **C1–C3, A1, A2:** IMPROVEMENT requires all 9 new-vs-old seed pairs to be better **and** the Welch 95 % CI of
  the mean difference to exclude 0. These seeds are not paired.
* **K1/K4:** paired by seed; better on all 3 seeds and |mean d| > 2 sd(d).

| Layer | ID | Comparison | Metric | Reference | New | Δ [95 % CI] | Improvement | Class | Seed pairs better |
|---|---|---|---|---|---|---|---|---|---|
| 1 | C1 | old TopK-3072 → R2 TopK-12288 | NMSE | 0.0787 | 0.0248 | −0.0540 [−0.0553, −0.0527] | 68.6 % | IMPROVEMENT | 9/9 |
| 1 | C1 | | ΔCE | 0.0221 | 0.0056 | −0.0166 [−0.0198, −0.0133] | 74.9 % | IMPROVEMENT | 9/9 |
| 1 | C2 | old ReLU → R2 ReLU-12288 | NMSE | 0.0630 | 0.0225 | −0.0405 [−0.0412, −0.0397] | 64.2 % | IMPROVEMENT | 9/9 |
| 1 | C2 | | ΔCE | 0.0141 | 0.0038 | −0.0103 [−0.0143, −0.0062] | 73.0 % | IMPROVEMENT | 9/9 |
| 1 | C3 | old JumpReLU → R2 BatchTopK-12288 | NMSE | 0.0675 | 0.0178 | −0.0497 [−0.0506, −0.0488] | 73.7 % | IMPROVEMENT | 9/9 |
| 1 | C3 | | ΔCE | 0.0088 | 0.0019 | −0.0070 [−0.0112, −0.0027] | 78.8 % | IMPROVEMENT | 9/9 |
| 1 | A1 | old TopK-3072 → R2 TopK-3072 (data + recipe) | NMSE | 0.0787 | 0.0512 | −0.0276 [−0.0289, −0.0262] | 35.0 % | IMPROVEMENT | 9/9 |
| 1 | A1 | | ΔCE | 0.0221 | 0.0132 | −0.0089 [−0.0122, −0.0056] | 40.3 % | IMPROVEMENT | 9/9 |
| 1 | A2 | R2 TopK-3072 → R2 TopK-12288 (width) | NMSE | 0.0512 | 0.0248 | −0.0264 [−0.0266, −0.0263] | 51.6 % | IMPROVEMENT | 9/9 |
| 1 | A2 | | ΔCE | 0.0132 | 0.0056 | −0.0077 [−0.0098, −0.0055] | 58.0 % | IMPROVEMENT | 9/9 |
| 8 | C1 | TopK | NMSE | 0.1598 | 0.0941 | −0.0658 [−0.0670, −0.0646] | 41.1 % | IMPROVEMENT | 9/9 |
| 8 | C1 | | ΔCE | 0.1330 | 0.0620 | −0.0710 [−0.0767, −0.0653] | 53.4 % | IMPROVEMENT | 9/9 |
| 8 | C2 | ReLU | NMSE | 0.1951 | 0.1320 | −0.0632 [−0.0643, −0.0620] | 32.4 % | IMPROVEMENT | 9/9 |
| 8 | C2 | | ΔCE | 0.2175 | 0.1310 | −0.0865 [−0.0981, −0.0748] | 39.8 % | IMPROVEMENT | 9/9 |
| 8 | C3 | JumpReLU → BatchTopK | NMSE | 0.1560 | 0.0897 | −0.0663 [−0.0665, −0.0662] | 42.5 % | IMPROVEMENT | 9/9 |
| 8 | C3 | | ΔCE | 0.1212 | 0.0542 | −0.0670 [−0.0692, −0.0648] | 55.3 % | IMPROVEMENT | 9/9 |
| 8 | A1 | data + recipe | NMSE | 0.1598 | 0.1246 | −0.0352 [−0.0362, −0.0342] | 22.0 % | IMPROVEMENT | 9/9 |
| 8 | A1 | | ΔCE | 0.1330 | 0.0843 | −0.0487 [−0.0545, −0.0428] | 36.6 % | IMPROVEMENT | 9/9 |
| 8 | A2 | width | NMSE | 0.1246 | 0.0941 | −0.0305 [−0.0314, −0.0296] | 24.5 % | IMPROVEMENT | 9/9 |
| 8 | A2 | | ΔCE | 0.0843 | 0.0620 | −0.0223 [−0.0271, −0.0175] | 26.5 % | IMPROVEMENT | 9/9 |
| 11 | C1 | TopK | NMSE | 0.1848 | 0.1112 | −0.0736 [−0.0744, −0.0729] | 39.8 % | IMPROVEMENT | 9/9 |
| 11 | C1 | | ΔCE | 0.2041 | 0.0962 | −0.1080 [−0.1132, −0.1027] | 52.9 % | IMPROVEMENT | 9/9 |
| 11 | C2 | ReLU | NMSE | 0.2432 | 0.1683 | −0.0748 [−0.0768, −0.0728] | 30.8 % | IMPROVEMENT | 9/9 |
| 11 | C2 | | ΔCE | 0.3417 | 0.2205 | −0.1212 [−0.1295, −0.1129] | 35.5 % | IMPROVEMENT | 9/9 |
| 11 | C3 | JumpReLU → BatchTopK | NMSE | 0.1851 | 0.1073 | −0.0778 [−0.0781, −0.0775] | 42.0 % | IMPROVEMENT | 9/9 |
| 11 | C3 | | ΔCE | 0.2155 | 0.0927 | −0.1228 [−0.1277, −0.1179] | 57.0 % | IMPROVEMENT | 9/9 |
| 11 | A1 | data + recipe | NMSE | 0.1848 | 0.1394 | −0.0454 [−0.0462, −0.0446] | 24.6 % | IMPROVEMENT | 9/9 |
| 11 | A1 | | ΔCE | 0.2041 | 0.1356 | −0.0686 [−0.0732, −0.0640] | 33.6 % | IMPROVEMENT | 9/9 |
| 11 | A2 | width | NMSE | 0.1394 | 0.1112 | −0.0282 [−0.0285, −0.0279] | 20.2 % | IMPROVEMENT | 9/9 |
| 11 | A2 | | ΔCE | 0.1356 | 0.0962 | −0.0394 [−0.0446, −0.0341] | 29.1 % | IMPROVEMENT | 9/9 |

Spliced KL follows ΔCE in every cell, with improvements of 33–76 %, all IMPROVEMENT (`comparisons.csv`).

**Per-seed values (seeds 42 / 1042 / 2042), TEST ΔCE in nats:**

| Layer | Old TopK | R2 TopK | Old ReLU | R2 ReLU | Old JumpReLU | R2 BatchTopK |
|---|---|---|---|---|---|---|
| 1 | .0233 / .0228 / .0203 | .0064 / .0059 / .0044 | .0132 / .0131 / .0160 | .0041 / .0037 / .0036 | .0096 / .0066 / .0103 | .0022 / .0025 / .0010 |
| 8 | .1337 / .1353 / .1300 | .0627 / .0630 / .0603 | .2230 / .2145 / .2150 | .1305 / .1309 / .1317 | .1224 / .1205 / .1206 | .0544 / .0533 / .0549 |
| 11 | .2064 / .2037 / .2023 | .0938 / .0988 / .0960 | .3412 / .3412 / .3426 | .2210 / .2167 / .2238 | .2166 / .2143 / .2156 | .0918 / .0910 / .0954 |

The worst new seed beats the best old seed in every cell. The improvement is consistent across seeds.

### 2c. R4 KL fine-tune (TEST; paired by seed; TopK-12288; L0 = 128 exactly)

| Layer | ID | Metric | Reference | KL arm | Mean paired Δ ± sd | Improvement | Class | Per-seed reference → KL |
|---|---|---|---|---|---|---|---|---|
| 8 | K1 vs MSE control | ΔCE | 0.0623 | 0.0514 | −0.0108 ± 0.0005 | 17.4 % | IMPROVEMENT | .0635/.0635/.0599 → .0524/.0523/.0496 |
| 8 | K1 | KL | 0.0659 | 0.0564 | −0.0095 ± 0.0004 | 14.4 % | IMPROVEMENT | |
| 8 | K1 | NMSE (guard) | 0.0938 | 0.0958 | +0.0020 ± 0.00002 | −2.1 % | **REGRESSION** | all 3 seeds worse |
| 8 | K1 | GoT re-fit gap | 0.0585 | 0.0484 | −0.0101 ± 0.0087 | — | NO CLEAR CHANGE | |
| 8 | K4 vs R2 base | ΔCE | 0.0620 | 0.0514 | −0.0106 ± 0.0002 | 17.0 % | IMPROVEMENT | |
| 11 | K1 | ΔCE | 0.0964 | 0.0807 | −0.0157 ± 0.0007 | 16.3 % | IMPROVEMENT | .0946/.0985/.0960 → .0791/.0820/.0810 |
| 11 | K1 | KL | 0.0968 | 0.0813 | −0.0155 ± 0.0006 | 16.0 % | IMPROVEMENT | |
| 11 | K1 | NMSE (guard) | 0.1109 | 0.1133 | +0.0024 ± 0.00001 | −2.2 % | **REGRESSION** | all 3 seeds worse |
| 11 | K1 | GoT re-fit gap | 0.0780 | 0.0784 | +0.0004 ± 0.0076 | — | NO CLEAR CHANGE | |
| 11 | K4 | ΔCE | 0.0962 | 0.0807 | −0.0155 ± 0.0011 | 16.1 % | IMPROVEMENT | |

* The NMSE regression is the expected cost of trading reconstruction for behaviour.
* The **pre-registered guard** was on EV: KL ≥ MSE − 0.02. It passes (−0.002).
* "KL" in the R4 objective is a **training-loss term** with a coefficient α of 1.14–1.51, set by first-batch scale
  matching. The "spliced KL" rows above are **measured** KL(p_orig ‖ p_spliced) on test tokens. They are not the
  same quantity.

---

## 3. GoT cities: truth information (M5 TEST, layers 8 and 11) [recomputed]

Source: `m5_rows.csv`, `comparisons.csv`.
* **Data:** GoT cities + neg_cities test, 452 statements, final-token activations.
* **Readers:** fitted on TRAIN rows of **original** activations; city groups do not cross splits (asserted in code).
* **Original AUROC on test:** L8 0.8386, L11 0.7916.
* **Gaps:** original AUROC − reconstruction AUROC, so ↓ is better.

| Layer | SAE | GoT-prompt L0 | Frozen logistic gap | Re-fit logistic gap | SAE − PCA re-fit gap (< 0 = SAE beats PCA) |
|---|---|---|---|---|---|
| 8 | old ReLU | 76.6 | 0.275 (0.028) | **0.034** (0.009) | 0.008 (0.009) |
| 8 | old TopK | 128 | 0.276 (0.017) | 0.061 (0.020) | 0.036 (0.020) |
| 8 | old JumpReLU | 62.7 | 0.280 (0.030) | 0.058 (0.020) | 0.034 (0.020) |
| 8 | R2 ReLU | 89.0 | 0.281 (0.017) | **0.028** (0.012) | 0.010 (0.012) |
| 8 | R2 TopK | 128 | 0.247 (0.039) | 0.056 (0.004) | 0.042 (0.004) |
| 8 | R2 BatchTopK | 79.5 | 0.307 (0.021) | 0.065 (0.024) | 0.053 (0.024) |
| 8 | R2 TopK-3072 | 128 | 0.255 (0.014) | 0.044 (0.003) | 0.027 (0.003) |
| 8 | R4 KL | 128 | 0.241 (0.025) | 0.048 (0.008) | 0.035 (0.008) |
| 8 | **random frozen decoder** (n=1) | 128 | 0.281 | 0.067 | **−0.003** |
| 8 | res-jb (n=1, unmatched) | 60.7 | 0.294 | 0.041 | 0.020 |
| 11 | old ReLU | 111.9 | 0.221 (0.005) | 0.077 (0.017) | 0.040 (0.017) |
| 11 | old TopK | 128 | 0.228 (0.017) | 0.101 (0.018) | 0.073 (0.018) |
| 11 | old JumpReLU | 92.5 | 0.210 (0.014) | 0.107 (0.026) | 0.079 (0.026) |
| 11 | R2 ReLU | 111.1 | 0.237 (0.016) | **0.053** (0.005) | 0.030 (0.005) |
| 11 | R2 TopK | 128 | 0.197 (0.043) | 0.078 (0.012) | 0.072 (0.012) |
| 11 | R2 BatchTopK | 105.8 | 0.186 (0.017) | 0.087 (0.008) | 0.081 (0.008) |
| 11 | R4 KL | 128 | 0.205 (0.033) | 0.078 (0.022) | 0.073 (0.022) |
| 11 | **random frozen decoder** (n=1) | 128 | 0.228 | 0.086 | **−0.013** |
| 11 | res-jb (n=1, unmatched) | 55.0 | 0.273 | 0.094 | 0.064 |

**Pre-registered classes on GoT (TEST):**

| Layer | ID | Re-fit gap Δ [95 % CI] | Class | SAE − PCA Δ [95 % CI] | Class |
|---|---|---|---|---|---|
| 8 | C1 | −0.005 [−0.053, +0.043] | NO CLEAR CHANGE | +0.006 [−0.042, +0.054] | NO CLEAR CHANGE |
| 8 | C2 | −0.006 [−0.031, +0.019] | NO CLEAR CHANGE | +0.003 [−0.022, +0.028] | NO CLEAR CHANGE |
| 8 | C3 | +0.007 [−0.043, +0.057] | NO CLEAR CHANGE | +0.019 [−0.031, +0.070] | NO CLEAR CHANGE |
| 8 | A1 | −0.017 [−0.066, +0.032] | NO CLEAR CHANGE (9/9 pairs better, CI includes 0) | −0.009 [−0.057, +0.040] | NO CLEAR CHANGE |
| 8 | A2 | +0.012 [+0.004, +0.020] | **REGRESSION** | +0.015 [+0.007, +0.023] | **REGRESSION** |
| 11 | C1 | −0.023 [−0.060, +0.013] | NO CLEAR CHANGE | −0.001 [−0.037, +0.035] | NO CLEAR CHANGE |
| 11 | C2 | −0.025 [−0.062, +0.013] | NO CLEAR CHANGE (9/9 pairs better, CI includes 0) | −0.011 [−0.049, +0.027] | NO CLEAR CHANGE |
| 11 | C3 | −0.020 [−0.080, +0.040] | NO CLEAR CHANGE | +0.002 [−0.058, +0.062] | NO CLEAR CHANGE |
| 11 | A1 | −0.013 [−0.050, +0.025] | NO CLEAR CHANGE | +0.004 [−0.033, +0.041] | NO CLEAR CHANGE |
| 11 | A2 | −0.011 [−0.041, +0.019] | NO CLEAR CHANGE | −0.005 [−0.035, +0.026] | NO CLEAR CHANGE |

**Verified observations:**
1. **Better reconstruction did not produce better GoT truth preservation on TEST.**
   * 0 of 34 GoT cells improved: 30 cells from C1–C3/A1/A2 over frozen, re-fit and SAE − PCA, plus 4 R4 re-fit
     cells.
   * 2 cells regressed (A2 at L8).
2. **The best GoT re-fit gaps belong to ReLU**, which has the **worst** OWT NMSE at L8 and L11. Within a layer,
   reconstruction error and truth preservation are not ordered the same way. ReLU's GoT-prompt L0 (77–112) also
   differs from TopK's (128), so this is a confounded, descriptive observation.
3. **A random frozen decoder has SAE − PCA ≈ 0 or below 0 at both layers. Every trained SAE has a positive value.**
   * On this metric, trained SAEs do **not** beat a random-decoder control.
   * The GoT re-fit gap is a weak discriminator at these layers.
   * This is a single seed (n = 1), so it is descriptive.
4. **No trained SAE beats PCA** at matched OWT validation NMSE on GoT test (all SAE − PCA values > 0).

---

## 4. RepE-LM persona: representation and causal (VALIDATION only; no test split exists)

Sources: `results/openwebtext/r2_scaled_saes/ladder/*.csv`, `r4_kl_finetune/ladder/*.csv` [recomputed means],
`results/repe/sae_patch_decomposition/`, `results/repe/r5_causal_subspace_skyline/summary.csv`.

| Layer (M4 site) | SAE | Persona re-fit gap ↓ | Persona SAE − PCA ↓ | Pair-direction cos(Δ, Δ̂) ↑ | T_rec / T_full ↑ (causal share carried by reconstruction) |
|---|---|---|---|---|---|
| L1 (persona slot) | old ReLU / TopK / JumpReLU | .095 / .083 / .097 | .023 / .007 / .025 | .519 / .373 / .384 | .938 / .931 / .968 |
| L1 | R2 ReLU / TopK / BatchTopK | .089 / .067 / .067 | .046 / .021 / .030 | .602 / .507 / .526 | .992 / .988 / .994 |
| L1 | random frozen decoder | .104 | −.023 | .128 | .660 |
| L8 (final token) | old ReLU / TopK / JumpReLU | .212 / .156 / .162 | .075 / .033 / .040 | .558 / .394 / .436 | .621 / .625 / .665 |
| L8 | R2 ReLU / TopK / BatchTopK | .193 / .108 / .126 | .092 / .027 / .053 | .607 / .488 / .504 | .643 / .726 / .754 |
| L8 | R4 MSE / R4 KL | .109 / .106 | .028 / .023 | .487 / .474 | .720 / .710 |
| L8 | random frozen decoder | .195 | −.018 | .200 | .023 |
| L11 (final token) | old ReLU / TopK / JumpReLU | .180 / .149 / .118 | .104 / .083 / .052 | .557 / .432 / .478 | .677 / .620 / .678 |
| L11 | R2 ReLU / TopK / BatchTopK | .194 / .109 / .101 | .137 / .070 / .067 | .622 / .513 / .517 | .712 / .692 / .701 |
| L11 | R4 MSE / R4 KL | .107 / .111 | .068 / .071 | .512 / .500 | .694 / .671 (**REGRESSION**, −0.024 ± 0.002 paired) |
| L11 | random frozen decoder | .154 | .034 | .250 | .109 |

**R5 causal skyline: share of the persona effect carried by a rank-k patch** (VALIDATION, descriptive; DAS mean of 3 seeds):

| Site | k | DAS (supervised upper reference) | PCA of TRAIN Δ | Random (mean of 20) | Old-SAE top-k features | Whole old-SAE reconstruction |
|---|---|---|---|---|---|---|
| L1 slot | 4 | 1.016 | 1.000 | 0.004 | 0.11–0.16 | 0.94–0.97 |
| L8 final | 4 | 0.913 | 0.518 | 0.008 | 0.10–0.16 | 0.62–0.67 |
| L8 final | 64 | 1.289 | 0.826 | 0.110 | 0.34–0.48 | |
| L11 final | 4 | 0.619 | 0.587 | 0.008 | 0.08–0.14 | 0.62–0.68 |
| L11 final | 64 | 0.888 | 0.839 | 0.127 | 0.32–0.46 | |

---

## 5. Stage C: unsuccessful and inconclusive improvement attempts (VALIDATION unless stated)

| Experiment | Layer(s) | Key number | Pre-registered verdict | Source |
|---|---|---|---|---|
| Historical recon-only fine-tune | 6 | GoT test MLP gap .327/.154/.323 → .161/.110/.146 (ReLU/TopK/JumpReLU); **L0 not matched** (ReLU GoT L0 65 → 144) | none (historical; GoT test scored twice) | `results/geometry_of_truth/sae_finetuning/heldout_comparison_all_datasets.csv` |
| Historical truth-aware fine-tune | 6 | logistic gap .019/.076/.063 is **CIRCULAR** (the teacher = the evaluator); MLP gap .225/.184/.218, worse than recon-only in 6/6 cells | INVALID as truth evidence | `docs/2026-11-07_truth_aware_finetuning_audit.md` |
| Corrected truth-aware pilot (TopK, A/B split) | 6 | B-MLP gap: base .204, recon-only **.107**, truth-aware .189 | NOT MET | `results/geometry_of_truth/truth_aware_pilot/` |
| Method A (GoT recon + OWT replay) | 6 | B-MLP gap ReLU .321 → .245, TopK .204 → .107, JumpReLU .292 → .238; 18/18 paired better | NOT MET (no cell > 2 sd) | `results/geometry_of_truth/method_a_layer6/` |
| E2E-inspired B/C (output KL) | 6 | TopK OWT KL .115 → .102; ReLU/JumpReLU L0 +5–11 % | TopK behaviour MET; ReLU/JumpReLU guard FAIL; truth FAIL everywhere | `results/geometry_of_truth/e2e_inspired_layer6/` |
| KL-weight ablation, TopK | 6 | MLP gap .107 / .111 / .112 / .114 (w = 0 / .25 / .5 / 1) | INCONCLUSIVE | `results/geometry_of_truth/kl_weight_ablation_layer6_topk/` |
| R1 RepE honesty fine-tune (historical task) | 6 | ReLU R-honesty − R-recon −.0135 ± .0048; RepE-prompt L0 +7–8 % | ReLU INCONCLUSIVE; TopK/JumpReLU NOT SUPPORTED | `results/repe/r1_honesty_preservation_layer6/` |
| R2-clamp SAE-in-the-loop honesty features (historical task) | 6 | E(α=1) ≈ −0.10 to −0.12, below random-feature controls | NOT SUPPORTED | `results/repe/r2_sae_in_the_loop_steering_layer6/` |
| IMP-L Method A | 3, 8, 9 | frozen B-MLP gap fell in 9/9 cells (4–47 %); re-trained reader: no change in any cell | 4/9 PRELIMINARY IMPROVEMENT (frozen reader only) | `results/geometry_of_truth/multilayer_finetune/` |
| IMP-L output-KL | 3, 8, 9 | OWT KL −8 to −18 % vs Method A | behaviour 5/9 creditable; 4/9 NOT CREDITABLE (L0 +5.3 to +8.1 %); truth 0/9, 2 regressions | same |
| E3 causal after fine-tune | 8 | T_rec/T_full base .621/.625/.665; Method A .622/.615/.669 | NO IMPROVEMENT; output-KL TopK NO IMPROVEMENT; ReLU/JumpReLU NOT CREDITABLE | `results/repe/e3_finetuned_causal_preservation/` |
| RLM-RP | 0–11 | Method A 0/12 cells changed; output-KL ≤ 0.007 AUROC in 3/12 | NO IMPROVEMENT / tiny | `results/repe/repe_lm_representation_preservation/` |

---

## 6. Earlier measurement results worth citing (VALIDATION unless stated)

| Result | Number | Status | Source |
|---|---|---|---|
| EV falls with depth (108 SAEs, OWT test) | ~0.93–0.95 (L0–2) → 0.757 / 0.815 / 0.815 (ReLU / TopK / JumpReLU, L11) | TEST | `results/openwebtext/reconstruction_evaluation/uninterrupted_protocol/` |
| Frozen GoT probe collapse is mostly fragility (M2) | frozen 0.28 vs same-norm noise 0.25 vs PCA 0.22; re-trained gap 0.07–0.12 | VALIDATION | `results/shared/m2_four_level_faithfulness/` |
| SAE never better than PCA at equal OWT error (M2) | 0 cells better; worse in 52/81 (GoT), 59/99 (persona), 81/90 (neutral) | VALIDATION | same |
| Within-layer: reconstruction error predicts splice damage, not the truth gap (M0) | β = 0.165, Holm p = 0.0008 (ΔCE); truth gap p = 0.044, fragile (p = 0.89 without architecture FE) | VALIDATION | `results/shared/m0_within_layer_reanalysis/` |
| Causal share of the persona effect carried by reconstruction (M4) | L1 slot 0.93–0.97; L8/L11 final 0.62–0.68 | VALIDATION | `results/repe/sae_patch_decomposition/` |
| Historical causal test, layer 6 | positive control 0.234; raw direction 0.0048 (p = 0.010 vs 200 random); SAE directions not distinguishable | TEST, HISTORICAL (length-confounded task) | `results/repe/steering/causal_random_controls/` |
| Gemma-2-2B L11 steering | random directions ≈ 73 % TRUE at ±0.5; context direction 23 % → 100 % TRUE | VALIDATION + TEST, descriptive, 20 controls, **no SAE** | `results/gemma/steering_sweep_layer11/` |
