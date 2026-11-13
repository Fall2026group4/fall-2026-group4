# AUDIT-SENS: sensitivity analyses after the 2026-11-13 scientific audit. Status: PRE-REGISTERED (plan below); results in §3 when run

**Evidence status of everything in this folder: SENSITIVITY / DIAGNOSTIC (exploratory).**
* None of these analyses is a new confirmatory test.
* No frozen protocol, frozen result or committed number is changed.
* No TEST row is read. S1 re-uses only the *committed* TEST seed statistics of H1-TEST and M5.

## 1. Why

The audit (`docs/RESEARCH_PROGRESS.md` §44) found problems that can change what the existing results support:

| ID | Audit finding | Analysis |
|---|---|---|
| S1 | The R2 rule's Welch CI covers seed variance only; statement and document sampling is ignored (M5, H1-TEST) | Paired cluster bootstrap on VALIDATION; transfer to the committed TEST seed statistics |
| S2 | "Part of the persona loss is persona-specific" rests on raw gaps; the neutral contrast has far less headroom | Headroom-normalised loss, persona vs neutral, paired statement bootstrap |
| S3 | L0 is matched on OWT only; the arms differ strongly on task prompts | Inference-time truncation to a common task L0; C1–C3 and the old-architecture ordering re-evaluated |
| S4 | Non-linear GoT probes read "truth" at layers 1–3 where the logistic probe is at chance | Diagnostics on original activations: per polarity, permutation null, cross-polarity transfer, text-only baselines |
| S5 | The only noise reference is isotropic, which is especially damaging to a standardised logistic probe | Row-shuffled SAE error and error-covariance Gaussian references |
| S6 | Logistic C grid {0.01, 0.1, 1}; the original persona probe sits at its edge | Extended 11-value grid; C1–C3 re-fit-gap classes re-checked |

All six analyses connect to the main research question in the same way. They test whether the stated
truth/honesty-preservation and causal-preservation conclusions survive a fairer comparison and a fuller account of
uncertainty.

## 2. Design (fixed before the full runs)

Scripts (`scripts/shared/audit_s{1..6}_*.py`) re-use the project's unchanged functions:
* the M2 probes;
* the RLM-RP pair geometry;
* the M4 decomposition;
* the OWT splice evaluation;
* the R2 rule.

Each script checks that it reproduces the committed per-SAE values before using them. Split, seed, bootstrap and
decision-rule details are in each script's docstring and in `preregistration.json`.

* **S1.**
  * For each comparison: se = sqrt(se_seed² + v_stmt).
    * v_stmt is the variance of the paired cluster-bootstrap difference of seed means (B = 2000; clusters are
      statements for RepE-LM and documents for OWT).
    * df = min(Satterthwaite, Welch), so the interval is never narrower than the frozen seed-only CI (fixed after a unit test showed Satterthwaite alone can narrow it; before any full run).
  * Applied to the 13 H1-TEST cells and the 18 M5 C1–C3 NMSE/ΔCE cells.
    * Test values use v_stmt scaled by n_val/n_test: 89/86 statements and 100/100 documents.
    * This assumes the per-cluster variance transfers from validation to test.
  * A cell is "supported under seed + statement uncertainty" if:
    * all 9 pairs are better;
    * the adjusted CI excludes 0;
    * Holm p < 0.05 (over 13 cells, or 18 for M5).
* **S2.**
  * Normalised loss NL = gap / (AUROC_orig − 0.5); ΔNL = NL_persona − NL_neutral.
  * Paired statement bootstrap, B = 1000.
  * A layer is "persona-specific" if the CI is above 0 and BH q < 0.05 over layers 1–11.
  * The reverse direction is reported too.
* **S3.**
  * k* = the minimum natural TRAIN-row L0 over the six arm types, per layer and task.
  * Each SAE gets a global threshold calibrated on TRAIN rows; the achieved VALIDATION L0 must be within ±5 %.
  * A class change between natural and matched L0 is flagged "L0-sensitive".
* **S4.**
  * Pooling artefact: pooled non-linear ≥ 0.65 while per-polarity non-linear ≤ 0.60.
  * Lexical shortcut: text baseline ≥ 0.65.
  * Pipeline leak: permutation null outside [0.45, 0.55].
* **S5.** A cell shows "label-aligned loss" if the CI of gap(SAE) − gap(shuffled SAE error) is above 0, with BH over
  cells.
* **S6.**
  * Extended grid {1e-4 … 10}.
  * A class change against the committed ladder is flagged "grid-sensitive".

## 3. Disclosure: what was seen before this plan was frozen

* **Smoke runs on VALIDATION**, all into the scratch directory and none of them committed:
  * S1, layer 8, seed 42: reproduction only, max |diff| 4e-5;
  * S2, layer 8, seed 42: the neutral loss was larger than the persona loss;
  * S3, layer 8, seed 42: all arms matched;
  * S5, layer 8, seed 42: the shuffled-error gap was smaller than the SAE gap;
  * S6, layer 8, seed 42;
  * S4, layer 1 (full diagnostics).
* **The S4 layer-1 smoke showed:**
  * per-polarity *logistic* probes at 0.75–0.76, with cross-polarity transfer at 0.76;
  * pooled logistic 0.52.

  After seeing this, one descriptive flag was added to S4: "pooled-linear suppression", meaning per-polarity
  logistic ≥ 0.65 while pooled ≤ 0.60. It is labelled POST HOC in the script and in every report.
* The audit itself had already read the committed M2, ladder, H1-TEST and M5 outputs.

## 4. Results (VALIDATION-based; sensitivity only)

### S1: statement and document uncertainty (`s1_statement_uncertainty/`)

**Reproduction:** every recomputed per-SAE value equals the committed ladder (max |diff| 1.9e-6).

**How much of the variance comes from sampling statements or documents?** On validation, cluster sampling accounts for
33–99 % of the total variance of a comparison (`validation_combined.csv`, `stmt_share_of_variance`). The seed-only
R2-rule CIs are therefore too narrow, often by a factor of 2–5 in standard error.

**H1-TEST cells** (committed TEST seed statistics plus the validation-estimated statement variance; primary df =
min(Satterthwaite, Welch); Holm over 13; `h1_test_cells_adjusted.csv`):

| Cell | TEST diff | Seed-only CI (frozen) | Seed + statement CI | Holm p | Frozen verdict | Under seed + statement |
|---|---|---|---|---|---|---|
| HT1 C1 L8 (T_rec/T_full) | +0.127 | [+0.079, +0.175] | [+0.014, +0.240] | 0.156 | confirmed | **not supported** |
| HT1 C3 L8 | +0.072 | [+0.015, +0.130] | [−0.005, +0.150] | 0.156 | not confirmed | not supported |
| HT2 (9 cells, pair cosine) | +0.035 to +0.141 | all exclude 0 | all exclude 0 | 0.0006–0.023 | confirmed | **supported (9/9)** |
| HT3 C1 L8 (persona re-fit gap) | −0.048 | [−0.079, −0.016] | [−0.093, −0.002] | 0.156 | confirmed | **not supported** |
| HT3 C3 L8 | −0.022 | [−0.055, +0.011] | [−0.070, +0.026] | 0.245 | not confirmed | not supported |

* With the Satterthwaite df alone (secondary), HT1-C1 (p = 0.0002) and HT3-C1 (p = 0.0075) would pass.
* Their survival therefore depends on how few seeds there are (Welch df ≈ 2–4), and they are reported as **fragile**.
* The frozen H1-TEST verdicts are not changed: they remain the pre-registered result. This analysis qualifies how
  far they generalise beyond the 86 test statements.

**M5 C1–C3 reconstruction and behaviour cells** (Holm over 18; `m5_test_cells_adjusted.csv`):
* 16 of 18 survive.
* The two that do not are the **layer-1 ΔCE gains for C2 (ReLU) and C3 (BatchTopK vs JumpReLU)**: −0.010 and −0.007
  nats, Holm p = 0.062.
* Every NMSE cell and every L8/L11 ΔCE cell survives with a wide margin.

### S4: GoT probe anomaly (`s4_got_probe_anomaly/`)

* **Lexical shortcut: none.** Bag-of-words probes reach at most 0.55; the country-prior lookup gives 0.48.
* **Pre-registered "pooling artefact" (non-linear): not met at any layer.** Per-polarity non-linear probes are at
  least as good as pooled ones.
* **Post-hoc flag, pooled-linear suppression: layers 1 and 2.**
  * Within one polarity, a *linear* probe reads the city–country truth at L1–2 (0.75–0.77) and transfers across
    polarity (cities → neg_cities, labels flipped: 0.71–0.76).
  * Pooled, the logistic probe sits at 0.52–0.59, because the pooled label is truth XOR negation.
  * The M2 "uninformative" labels for GoT at L0–L4 (pooled logistic < 0.60) therefore say nothing about whether
    truth information exists there. Per polarity it does from L1 on.
  * The "truth emerges at middle layers" reading of the pooled table is an artefact of pooling.
* **Pre-registered "pipeline leak" rule: fired at layers 5–9 and 11. Investigation: the rule was miscalibrated, not a
  leak.**
  * The rule required all 15 permutation-null AUROCs per layer to lie in [0.45, 0.55].
  * The null is centred: mean 0.497 over 180 draws.
  * Its SD (0.0274) equals the theoretical SD of an AUROC under the null for 224/224 rows (0.0273). [0.45, 0.55] is
    only about ±1.8 SD, so some exceedances are expected.
  * The flag is reported, not hidden; the probes are not leaking.

### S2: persona vs neutral, normalised by headroom (`s2_persona_headroom/`)

**Reproduction:** the re-fit gaps equal M2's per_cell values to within 7.6e-4.

The measure is NL = gap / (AUROC_orig − 0.5), the share of the decodable signal lost. ΔNL = NL_persona − NL_neutral;
CIs come from a paired statement bootstrap, with BH over layers 1–11.

| Layer | Original AUROC, persona / neutral | Raw re-fit gap, persona / neutral | NL persona / neutral (old SAEs) | ΔNL [95 % CI] | Verdict |
|---|---|---|---|---|---|
| 1 | 0.66 / 0.63 | 0.092 / 0.084 | 0.58 / 0.66 | −0.08 [−0.17, +0.03] | no difference |
| 2–4 | 0.73–0.80 / 0.60–0.63 | 0.14–0.17 / 0.06–0.08 | 0.56–0.59 / 0.57–0.59 | −0.01 to −0.00 | no difference (L4 low precision) |
| 5 | 0.79 / 0.61 | 0.161 / 0.070 | 0.55 / 0.65 | −0.10 [−0.17, −0.03] | **neutral loss larger** |
| 6 | 0.83 / 0.62 | 0.192 / 0.079 | 0.58 / 0.65 | −0.07 [−0.13, +0.01] | no difference |
| 7–11 | 0.83–0.87 / 0.63–0.83 | 0.15–0.19 / 0.08–0.17 | 0.41–0.52 / 0.51–0.64 | −0.10 to −0.12, all CIs < 0 | **neutral loss larger** |

R2 SAEs: L1 shows no difference; at L8 and L11 the neutral loss is larger (q ≤ 0.018).

* **No layer shows a persona-specific loss** (q ≥ 0.999 for "persona > neutral" everywhere).
* The paper's "part of the mid-layer loss is persona-specific" rests on raw gaps that are about twice as large for
  persona only because the persona contrast is about twice as decodable. Once that is accounted for, the SAEs lose
  the same or a *larger* share of the neutral one-word contrast.
* **Corrected reading:** the SAEs lose information about which instruction word was present in general. Nothing here
  shows honesty-specific information loss.
* This also bears on construct validity: RepE-LM measures one-token instruction retention, not honesty.

### S3: comparisons at matched task L0 (`s3_task_l0_matched/`)

**k\* (minimum natural TRAIN L0 over the six arm types):**
* persona: 128 at L1, L8 and L11 (set by old TopK);
* GoT: 60.3 at L8 and 88.8 at L11 (set by old JumpReLU).

All 54 SAEs were truncated with TRAIN-calibrated thresholds. The achieved VALIDATION L0 is within ±5 % for every GoT
cell. For persona, **R2 BatchTopK reached only 119–121** at L1 and L8 (−6 %), so those cells are UNMATCHED.

* **R2 gains (C1–C3): 54 of 57 classes are unchanged at matched L0.** The three that change are not headline results:
  * L1 C2, persona re-fit gap: no clear change → improvement;
  * L1 C2, persona frozen gap: improvement → no clear change;
  * L8 C1, GoT frozen gap: improvement → no clear change. At k\* = 60, R2 TopK is truncated further from its trained
    k = 128.
* **The C3 L8 causal gain holds at matched L0** (T_rec/T_full +0.087, 9/9 pairs). On validation it is not an artefact
  of BatchTopK using fewer latents than JumpReLU.
* **Disclosed flag bug (found while reading the output):** the matched flag of the RLM pair-cosine and M4 metrics looked
  for an `rlm__matched` column and defaulted to True. These metrics use the persona truncation, so C3 pair-cosine at L1
  and L8 and C3 M4 at L8 should have been flagged UNMATCHED. The deviation leaves the *new* arm slightly sparser, which
  works against it, so their IMPROVEMENT classes stand conservatively. The output was not re-generated; the scripts
  refuse to overwrite.
* **Old-architecture orderings on the task metrics are NOT robust to sparsity.** Matching changes the order in 13 of
  19 metric × layer cells (`old_architecture_ordering.csv`):
  * L8 M4 preservation: JumpReLU > TopK > ReLU becomes TopK > ReLU > JumpReLU;
  * L1 pair cosine: ReLU > JumpReLU > TopK becomes TopK > ReLU > JumpReLU;
  * L8 persona task NMSE: JumpReLU best becomes TopK best.

  These orderings are means over 3 seeds and were not tested statistically. They show that rankings between
  architectures on task prompts must not be reported without task-level L0 matching.

### S5: noise references for the frozen-probe gaps (`s5_noise_references/`)

**Reproduction:** within 1.2e-3 of M2's frozen gaps; mass-mean within 2e-5.

| Task, reader | SAE gap | Isotropic | Shuffled SAE error | Error-covariance Gaussian | Label-aligned cells |
|---|---|---|---|---|---|
| GoT, frozen logistic (L6–11, old SAEs) | 0.26–0.32 | 0.23–0.30 | 0.15–0.20 | 0.21–0.26 | 92 of 99 (layers 3–11, old + R2) |
| GoT, frozen mass-mean (L6–11) | 0.019–0.039 | 0.002–0.011 | 0.001–0.009 | 0.004–0.019 | 72 of 99 |
| Persona, frozen logistic (L1–11) | 0.11–0.20 | 0.04–0.09 | 0.04–0.08 | 0.03–0.08 | 126 of 126 |
| Persona, frozen mass-mean | 0.03–0.10 | ≤ 0.02 | ≤ 0.02 | ≤ 0.02 | 126 of 126 |

* **The M2 statement "the frozen-probe truth collapse is largely reproduced by same-norm noise" depends on the
  isotropic reference.**
  * Isotropic noise reproduces 84–96 % of the GoT logistic drop at layers 6–11.
  * The SAE's own errors, re-assigned at random to other rows, reproduce only 50–67 % at layers 6–11.
  * The remaining 33–50 % of the drop comes from the error being aligned with the input and its label.
* For the mass-mean reader, the SAE loses 2.0–5.0× more than the strongest reference at each layer and
  3.3–12.5× more than isotropic noise at layers 6–11.
* **Corrected reading:**
  * Part of the frozen-probe drop is generic sensitivity of the standardised logistic reader to perturbations of that
    size.
  * A substantial, label-aligned part is specific to the SAE's error.
  * The isotropic reference understates SAE-specific loss.
* For persona, the earlier "2–3× noise" reading holds under every reference.

### S6: extended probe-regularisation grid (`s6_probe_grid/`)

* **No conclusion depends on the grid.**
  * 0 of 15 C1–C3 re-fit-gap classes change (persona at L1/8/11; GoT at L8/11).
  * Differences shift by ≤ 0.007 AUROC.
* **No probe selects an edge of the extended grid** (1e-4 … 10).
  * The original persona probe moves from 0.01 to 0.003 at L1 and to 0.03 at L11, both interior values.
  * Its AUROC changes by −0.010 and −0.006. The extended grid is chosen by training-CV, so it can score slightly lower
    on validation.
* The audit concern (C at the boundary of {0.01, 0.1, 1}) is resolved: the default grid was adequate.

## 5. Summary: what changes

| Earlier statement | Status after AUDIT-SENS |
|---|---|
| H1-TEST HT2: pair-direction gain confirmed (9/9) | **Holds** with statement uncertainty (9/9) |
| H1-TEST HT1-C1: TopK L8 causal gain confirmed on TEST | **Fragile.** Confirmed under the frozen seed-only rule; not supported once statement sampling is included (conservative df; Holm p = 0.156). Passes only with Satterthwaite df |
| H1-TEST HT3-C1: TopK L8 persona re-fit gap confirmed | **Fragile** (same pattern; Holm p = 0.156) |
| M5: R2 reconstruction and behaviour gains confirmed | **Hold** in 16/18 cells; the L1 ΔCE gains for C2 and C3 are not robust |
| "Part of the mid-layer persona loss is persona-specific" | **Not supported.** After headroom normalisation the neutral control loses an equal or larger share at every layer (significantly larger at L5 and L7–11) |
| "Frozen-probe truth collapse is largely perturbation sensitivity" | **Overstated.** True against isotropic noise; against the SAE's own shuffled error, 33–50 % of the GoT drop at L6–11 is label-aligned and SAE-specific |
| "Truth becomes decodable at middle layers" (pooled GoT) | **Artefact of pooling.** Per polarity, linear probes read city–country truth from L1 (0.75–0.77) |
| Architecture orderings on persona/GoT metrics | **Not robust** to task-level L0 (13 of 19 orderings change) |
| R2 gains C1–C3 on task metrics | **Robust** to matched task L0 (54/57 classes unchanged, including the C3 L8 causal gain) |
| Probe C grid | **Robust** (0/15 classes change) |
